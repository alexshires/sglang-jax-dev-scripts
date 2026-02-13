# Investigation: Segment Mask TPU Kernel Lowering Issue

**Date:** 2026-02-12
**Status:** Root cause identified, fix pending
**Blocks:** RFC-013 Track 2 (Segment Kernel Fix)
**Workaround:** Use `--multi-item-mask-impl=dense`

## Summary

The segment mask mode (`MULTI_ITEM_MASK_MODE_SEGMENT=2`) fails on TPU with:
```
ValueError: Cannot do int indexing on TPU during kernel lowering
```

**Root cause:** Dynamic integer array indexing (gather operation) inside Pallas kernel is not supported by TPU lowering.

## Affected Code

### Location: `ragged_paged_attention.py:930-937`

```python
def load_mask(q_span, k_span):
    if multi_item_mask_mode == 2:
        q_token_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_token_indices = q_token_start + lax.iota(jnp.int32, actual_bq_sz)
        row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]  # <-- FAILS HERE
        row_seg_starts = jnp.repeat(row_seg_by_token, num_q_heads_per_kv_head)

        q_row_positions = q_span[:, 0]
        prefix_end = multi_item_prefix_end_ref[seq_idx]
        # ... rest of mask computation
```

### The Problem

Line 933: `multi_item_row_seg_starts_ref[q_token_indices]`

- `q_token_indices` is a JAX array of shape `(actual_bq_sz,)` with dtype `int32`
- This is **advanced integer indexing** (gather operation)
- Pallas TPU kernel lowering does NOT support dynamic gather operations
- The error occurs during JAX's XLA compilation, not at runtime

### When This Triggers

Segment mode is selected when:
1. `--multi-item-mask-impl=segment` (explicit)
2. `--multi-item-mask-impl=auto` AND `padded_token_len <= 32768` (default threshold)

Since `auto` is the default and the threshold is 32768, **most multi-item scoring requests will trigger this bug**.

## Mode Comparison

| Mode | Kernel Behavior | Memory | TPU Status |
|------|-----------------|--------|------------|
| `dense` | Precompute O(n²) mask on host, pass to kernel | O(n²) | Works |
| `segment` | Pass segment metadata, kernel does gather | O(n) | **Fails** |
| `auto` | Choose segment if tokens ≤ 32768 | varies | **Usually fails** |

## Workaround

### Immediate: Force dense mode

```bash
# Option 1: Explicit dense
python -m sgl_jax.launch_server ... --multi-item-mask-impl=dense

# Option 2: Set threshold to 0 (auto always picks dense)
python -m sgl_jax.launch_server ... --multi-item-segment-fallback-threshold=0
```

### Temporary: Environment variable

```python
# In test setup
global_server_args_dict["multi_item_mask_impl"] = "dense"
```

## Fix Options

### Option A: Pre-scatter segment data (Recommended)

Transform `row_seg_starts[padded_tokens]` into `row_seg_starts[num_blocks, block_size]` during metadata construction. Kernel loads contiguous block using BlockSpec.

```python
# Current: 1D array indexed by token
multi_item_row_seg_starts: jax.Array  # shape: [padded_num_tokens]

# Proposed: 2D array indexed by (block_idx, offset_in_block)
multi_item_row_seg_starts: jax.Array  # shape: [num_q_blocks, bq_sz]

# In kernel:
row_seg_by_token = multi_item_row_seg_starts_ref[:]  # Load entire block, no gather
```

**Pros:** No kernel control flow changes, O(n) memory
**Cons:** Metadata construction more complex, slight memory overhead from padding

### Option B: Scalar loop with pl.load

Replace array gather with scalar loop using `pl.load`:

```python
# Instead of:
row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]

# Use:
row_seg_by_token = jnp.zeros(actual_bq_sz, dtype=jnp.int32)
for i in range(actual_bq_sz):
    idx = q_token_start + i
    row_seg_by_token = row_seg_by_token.at[i].set(
        pl.load(multi_item_row_seg_starts_ref, (idx,))
    )
```

**Pros:** Minimal changes
**Cons:** Sequential loads, potential performance impact

### Option C: Compute mask on-the-fly from delimiter positions

Instead of storing per-token segment starts, store delimiter positions and compute segment membership inside kernel:

```python
# Pass delimiter positions: [d0, d1, d2, ...] where d_i = position of i-th delimiter
# In kernel:
def get_segment(token_pos, delimiter_positions):
    return jnp.searchsorted(delimiter_positions, token_pos)

q_segments = get_segment(q_positions, delimiter_positions)
k_segments = get_segment(k_positions, delimiter_positions)
mask = (q_segments == k_segments) | (k_segments == 0)  # Same segment or prefix
```

**Pros:** Minimal metadata, mathematically clean
**Cons:** `searchsorted` in kernel may have issues, need to verify TPU support

### Option D: Use dense mode as permanent fallback for segment

Accept that segment mode has limitations and keep dense as the production path while pursuing prefill+extend (RFC-013 Strategy 5) for performance gains.

**Pros:** No kernel changes needed, focus on higher-impact optimization
**Cons:** O(n²) memory remains a constraint

## Recommended Path Forward

1. **Immediate:** Lock to dense mode for stability (RFC-013 Track 1)
2. **Short-term:** Implement Option A (pre-scatter) for segment mode fix
3. **Medium-term:** Pursue prefill+extend (RFC-013 Track 3) for bigger gains
4. **Evaluate:** Compare fixed segment mode vs prefill+extend, keep winner

## Test Coverage

The failing test is in `test/srt/test_multi_item_segment_mask.py`:
- `test_get_forward_metadata_selects_segment_mode_in_auto()` - passes metadata construction
- **Actual failure:** Occurs during forward pass when kernel executes

To add regression test:
```python
def test_segment_mode_forward_pass():
    """Verify segment mode works end-to-end on TPU."""
    # This will fail until the kernel is fixed
    ...
```

## References

- RFC-013: Multi-Item Scoring Performance Optimization
- RFC-008: Multi-Item Scoring Design
- File: `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py`
- File: `python/sgl_jax/srt/layers/attention/flashattention_backend.py`
