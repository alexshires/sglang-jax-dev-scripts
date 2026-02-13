# Track 2: Segment Kernel Fix - Agent Handoff

**Date:** 2026-02-12
**Status:** Ready to Start
**Priority:** High
**Estimated Scope:** 2-3 days focused work

---

## Mission

Fix the TPU kernel lowering issue in segment mask mode so that `--multi-item-mask-impl=segment` works on TPU. This unblocks O(n) memory usage for multi-item scoring (vs O(n²) for dense mode).

## Problem Statement

The segment mask path fails on TPU with:
```
ValueError: Cannot do int indexing on TPU during kernel lowering
```

**Root cause:** Line 933 in `ragged_paged_attention.py` does dynamic integer array indexing (gather operation) which Pallas TPU lowering doesn't support:

```python
row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]
```

Where `q_token_indices` is a dynamically computed JAX array.

## Success Criteria

1. `--multi-item-mask-impl=segment` works on TPU v6e-1
2. All existing tests pass: `pytest test/srt/test_multi_item_*.py`
3. Segment mode produces identical scores to dense mode (max_diff ≤ 0.0001)
4. Memory usage is O(n) not O(n²)
5. Throughput within 20% of dense mode (acceptable if faster)

## Key Files

| File | Purpose |
|------|---------|
| `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` | **Primary target** - Pallas kernel with the bug |
| `python/sgl_jax/srt/layers/attention/flashattention_backend.py` | Metadata construction, mode selection |
| `test/srt/test_multi_item_segment_mask.py` | Existing unit tests for segment mode |
| `test/srt/test_multi_item_regression.py` | E2E regression tests |

## Relevant Code Context

### The Failing Code (ragged_paged_attention.py:929-951)

```python
def load_mask(q_span, k_span):
    if multi_item_mask_mode == 2:  # SEGMENT mode
        q_token_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_token_indices = q_token_start + lax.iota(jnp.int32, actual_bq_sz)
        row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]  # <-- BUG
        row_seg_starts = jnp.repeat(row_seg_by_token, num_q_heads_per_kv_head)

        q_row_positions = q_span[:, 0]
        prefix_end = multi_item_prefix_end_ref[seq_idx]
        is_prefix_row = q_row_positions < prefix_end

        causal_allow = k_span <= q_row_positions[:, None]
        shared_prefix_allow = k_span < prefix_end
        segment_allow = jnp.logical_and(
            k_span >= row_seg_starts[:, None],
            k_span <= q_row_positions[:, None],
        )
        allow = jnp.where(
            is_prefix_row[:, None],
            causal_allow,
            jnp.logical_or(shared_prefix_allow, segment_allow),
        )
        return jnp.logical_not(allow)
```

### Metadata Construction (flashattention_backend.py:320-361)

```python
# Segment metadata arrays created here:
prefix_end = np.zeros_like(seq_lens, dtype=np.int32)  # [batch_size]
row_seg_starts = np.zeros((padded_token_len,), dtype=np.int32)  # [padded_num_tokens]

# Populated per-request:
req_prefix_end, req_row_seg_starts = self._build_multi_item_segment_layout(...)
prefix_end[req_idx] = req_prefix_end
row_seg_starts[req_start:req_end] = req_row_seg_starts

# Transferred to device:
metadata.multi_item_prefix_end = device_array(prefix_end)
metadata.multi_item_row_seg_starts = device_array(row_seg_starts)
```

### Kernel Input Specs (ragged_paged_attention.py:1573-1597)

```python
if multi_item_row_seg_starts is None:
    multi_item_row_seg_starts = jnp.zeros((queries.shape[0],), dtype=jnp.int32)

# BlockSpec for segment data:
pl.BlockSpec(memory_space=pltpu.ANY),  # multi_item_row_seg_starts
```

## Recommended Fix: Pre-Scatter Approach

Transform the 1D token-indexed array into a 2D block-indexed array during metadata construction. The kernel then loads a contiguous block instead of doing a gather.

### Step 1: Change metadata shape

**In flashattention_backend.py:**
```python
# Before: 1D array indexed by global token position
row_seg_starts = np.zeros((padded_token_len,), dtype=np.int32)

# After: 2D array indexed by (block_idx, offset_in_block)
num_q_blocks = (padded_token_len + bq_sz - 1) // bq_sz
row_seg_starts_2d = np.zeros((num_q_blocks, bq_sz), dtype=np.int32)
```

### Step 2: Reshape during metadata construction

```python
# After populating 1D array, reshape to 2D:
row_seg_starts_1d = ...  # existing code
row_seg_starts_2d = row_seg_starts_1d.reshape(num_q_blocks, bq_sz)
# Or pad and reshape if not evenly divisible
```

### Step 3: Update BlockSpec

**In ragged_paged_attention.py:**
```python
# Before:
pl.BlockSpec(memory_space=pltpu.ANY),  # [padded_num_tokens]

# After - load one block at a time:
pl.BlockSpec(
    index_map=lambda seq_idx, bq_idx, bkv_idx: (bq_idx, 0),
    block_shape=(1, bq_sz),
    memory_space=pltpu.ANY,
),
```

### Step 4: Update kernel to use block-local indexing

```python
def load_mask(q_span, k_span):
    if multi_item_mask_mode == 2:
        # Before: gather from global array
        # row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]

        # After: load contiguous block (already indexed by BlockSpec)
        row_seg_by_token = multi_item_row_seg_starts_ref[0, :actual_bq_sz]
        row_seg_starts = jnp.repeat(row_seg_by_token, num_q_heads_per_kv_head)
        # ... rest unchanged
```

## Alternative Approaches (If Pre-Scatter Doesn't Work)

### Option B: Scalar Loop with pl.load

```python
# Replace gather with explicit scalar loads
row_seg_by_token = jnp.zeros(actual_bq_sz, dtype=jnp.int32)
def body_fn(i, arr):
    idx = q_token_start + i
    val = multi_item_row_seg_starts_ref[idx]  # scalar index OK
    return arr.at[i].set(val)
row_seg_by_token = lax.fori_loop(0, actual_bq_sz, body_fn, row_seg_by_token)
```

### Option C: Compute On-the-Fly from Delimiters

Pass delimiter positions instead of per-token segment starts. Compute segment membership in kernel using searchsorted-like logic.

## Testing Strategy

### 1. Unit test the fix

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-jax
pytest test/srt/test_multi_item_segment_mask.py -v
```

### 2. Verify segment == dense equivalence

```python
# Add to test_multi_item_segment_mask.py:
def test_segment_matches_dense_scores():
    """Segment mode produces same scores as dense mode."""
    # Run same request with both modes
    # Assert max_diff <= 0.0001
```

### 3. Run full regression suite

```bash
pytest test/srt/test_multi_item_regression.py -v
```

### 4. Memory validation

```python
# Verify segment mode uses O(n) not O(n²) memory
# Dense: O(seq_len²) = O(n²) for mask
# Segment: O(seq_len) = O(n) for segment arrays
```

## Environment Setup

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-jax

# Activate venv if needed
source .venv/bin/activate

# For local TPU testing (if available):
export TPU_NAME="your-tpu-name"
export ZONE="us-east5-b"

# Force segment mode for testing:
export MULTI_ITEM_MASK_IMPL="segment"
```

## Constraints

1. **No breaking changes** to dense mode - it must continue working
2. **No new dependencies** - use existing JAX/Pallas primitives
3. **Maintain backward compatibility** with existing API
4. **Keep kernel changes minimal** - prefer metadata reshaping over kernel rewrites

## Reference Documents

- [Investigation: Segment Mask TPU Lowering Issue](../investigations/segment-mask-tpu-lowering-issue.md)
- [RFC-013: Multi-Item Scoring v1.0 Optimization](../rfcs/013-multi-item-scoring-v1-optimization.md)
- [RFC-008: Multi-Item Scoring Design](../rfcs/008-multi-item-scoring.md)

## Deliverables

1. Fixed `ragged_paged_attention.py` with working segment mode
2. Updated `flashattention_backend.py` if metadata shape changes
3. New/updated tests proving segment == dense equivalence
4. Brief PR description explaining the fix

## Out of Scope

- Performance optimization beyond "working correctly"
- Prefill+extend implementation (Track 3)
- Runtime mode selector (Track 4)
- Dense mode changes

---

## Quick Start Commands

```bash
# 1. Read the failing code
cat python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py | head -n 1000 | tail -n 100

# 2. Read metadata construction
cat python/sgl_jax/srt/layers/attention/flashattention_backend.py | head -n 420 | tail -n 100

# 3. Run existing segment tests (will fail until fixed)
pytest test/srt/test_multi_item_segment_mask.py -v

# 4. After fix - run full validation
pytest test/srt/test_multi_item_*.py -v
```
