# Investigation: Multi-Item Scoring Attention Mechanism

| | |
|------------|------|
| **Date** | 2026-02-06 |
| **Status** | Complete — recommendation ready |
| **Related** | [RFC-008](../rfcs/008-multi-item-scoring.md), Decision 8 |
| **Blocks** | RFC-008 Phase 2 (Attention Mechanism) |

## Question

Which JAX API/mechanism should implement shared-prefix + block-diagonal attention masking for multi-item scoring on TPU?

### Required Masking Pattern

For sequence `[query <d> item1 <d> item2 <d> item3 <d>]`:

```
              query  <d>  item1  <d>  item2  <d>  item3  <d>
query           C
<d>             C     C
item1           1     1    C
<d>             1     1    C      C
item2           1     1    X      X    C
<d>             1     1    X      X    C      C
item3           1     1    X      X    X      X    C
<d>             1     1    X      X    X      X    C      C

C = causal (attend)    1 = attend (shared prefix)    X = blocked
```

Properties:
- All tokens attend to query (prefix) — standard causal within prefix
- Each item attends to itself — causal within item
- Items do NOT attend to other items — block-diagonal isolation
- Delimiters belong to the preceding item's block

---

## Candidates Evaluated

### 1. Existing `custom_mask` in `ragged_paged_attention` (Pallas kernel)

**Already integrated** in sglang-jax for speculative decoding (EAGLE).

**How it works:**
- `custom_mask` is a flattened 1D `int32` array: `[sum(q_len_i * kv_len_i)]` across all sequences in batch
- Values: `1` = attend, `0` = blocked
- When `custom_mask` is provided, `causal` is set to `0` (line 458 of `flashattention_backend.py`)
- Mask is stored in HBM, streamed to VMEM via DMA in tile-sized blocks
- The Pallas kernel prefetches mask tiles alongside KV blocks — no full materialization in VMEM

**Code path:**
```
Construct mask (host/numpy) → jax.Array (HBM)
  → FlashAttentionMetadata.custom_mask
  → ragged_paged_attention(custom_mask=..., causal=0)
  → Pallas kernel DMA's mask tiles to VMEM per block
```

**Key files:**
- `flashattention_backend.py:43` — metadata field
- `flashattention_backend.py:457-458` — `causal=0` when mask present
- `ragged_paged_attention.py:124` — parameter definition
- `ragged_paged_attention.py:177-186` — mask application in kernel
- `ragged_paged_attention.py:474-509` — DMA mask fetch
- `ragged_paged_attention.py:926-934` — mask loading in attention loop

**Memory cost (HBM) for typical scoring workloads:**

| Query tokens | Items | Tokens/item | Total seq_len | Mask size (int32) |
|-------------|-------|-------------|---------------|-------------------|
| 300 | 10 | 10 | 410 | 672 KB |
| 300 | 50 | 10 | 810 | 2.6 MB |
| 300 | 100 | 10 | 1,310 | 6.9 MB |
| 300 | 128 | 50 | 6,700 | 180 MB |
| 300 | 128 | 100 | 13,100 | 686 MB |

**Verdict:** Practical for seq_len < ~8K. The 128-items × 50-tokens case (180 MB) is at the edge. The 128 × 100 case (686 MB) is too expensive.

### 2. Splash Attention with `NumpyMask`

**Not currently used by sglang-jax.** Available in `jax.experimental.pallas.ops.tpu.splash_attention`.

**How it works:**
- Construct a 2D numpy boolean mask on the host
- Wrap in `NumpyMask`, pass to `make_splash_mha()`
- Splash preprocesses the mask at kernel build time into block-sparse structure
- Fully-masked blocks are **skipped entirely** (no compute)
- Only partially-masked blocks load mask data
- Kernel uses O(seq_len) device memory for attention computation

**Code pattern:**
```python
from jax.experimental.pallas.ops.tpu.splash_attention import (
    NumpyMask, MultiHeadMask, make_splash_mha
)

mask_np = build_multi_item_mask(prefix_len, item_lens)  # numpy [seq, seq] bool
kernel = make_splash_mha(
    MultiHeadMask([NumpyMask(mask_np)] * num_heads),
    head_shards=1, q_seq_shards=1
)
output = kernel(q, k, v)
```

**Memory cost:**
- Host: O(seq²) numpy array (CPU memory, not HBM)
- Device: O(seq_len) — flash attention tiling, no full mask materialization
- The block-sparse preprocessing means even the host cost is amortized if the kernel is reused

**Verdict:** Best memory profile. But requires new integration path — sglang-jax currently does NOT use splash attention. The ragged paged attention kernel is the entire attention stack.

### 3. `jax.nn.dot_product_attention` with mask/bias

**Standard JAX API.**

- Supports `mask` (boolean) or `bias` (additive) parameter
- XLA backend materializes full O(seq²) attention matrix on device
- No flash attention optimization with custom masks on TPU
- `cudnn` backend is GPU-only

**Verdict:** O(seq²) on device. Useful for correctness testing on CPU only.

### 4. Pallas Flash Attention (`jax.experimental.pallas.ops.tpu.flash_attention`)

**TPU-optimized, supports `segment_ids`.**

- `segment_ids` creates strict block-diagonal: same ID can attend, different IDs cannot
- **Cannot express shared-prefix pattern**: if prefix = segment 0 and item1 = segment 1, item1 cannot see prefix
- `ab` parameter (additive bias) can express arbitrary patterns but is O(seq²) on device

**Verdict:** `segment_ids` too restrictive. `ab` is O(seq²). Not suitable.

### 5. Native Backend Extension (for testing)

**Already has block-diagonal masking** in `_apply_extend_mask` (`native_backend.py:271-319`).

The existing code computes batch IDs from sequence lengths:
```python
q_batch_ids = jnp.cumsum(q_batch_indicators) - 1
final_mask = q_batch_ids[:, None] == k_batch_ids[None, :]  # block-diagonal
```

For multi-item scoring, we'd modify the rule to:
```python
# segment 0 = query, segment 1+ = items
final_mask = (q_segment_ids[:, None] == k_segment_ids[None, :]) | (k_segment_ids[None, :] == 0)
final_mask = final_mask & causal_mask
```

This is O(seq²) but the native backend already is. Suitable for CPU correctness testing.

**Verdict:** Trivial to implement for testing. Not for production.

### 6. Custom Pallas Kernel (from scratch)

**Maximum flexibility.** The team already has extensive Pallas experience (ragged_paged_attention is ~1800 lines of Pallas).

Could compute the mask procedurally from `prefix_len` and `item_boundaries` without materializing the full mask:
```
mask(q_pos, k_pos) = causal(q_pos, k_pos)
    AND (same_item(q_pos, k_pos) OR k_pos < prefix_len)
```

This would be O(seq_len) memory.

**Verdict:** Ideal memory profile but high development effort. Worth it only if `custom_mask` proves too expensive.

---

## Decision Matrix

| Criterion | Weight | `custom_mask` (existing) | Splash `NumpyMask` | Native extension | Custom Pallas |
|-----------|--------|--------------------------|---------------------|------------------|---------------|
| **Correctness** | Must-have | Yes | Yes | Yes (CPU) | Yes |
| **Already integrated** | High | Yes | No | Partially | No |
| **Device memory** | High | O(seq²) HBM | O(seq) HBM | O(seq²) | O(seq) |
| **Host memory** | Low | Negligible | O(seq²) numpy | Negligible | Negligible |
| **Dev effort** | High | **Low** — mask construction only | Medium — new integration path | Very low | High |
| **TPU optimized** | High | Yes (Pallas DMA) | Yes (block-sparse) | No (CPU only) | Yes |
| **Handles seq_len > 8K** | Medium | No (180MB+ mask) | Yes | N/A | Yes |

---

## Recommendation

### MVP: Existing `custom_mask` in `ragged_paged_attention`

**This is the clear winner for initial implementation.**

Rationale:
1. **Zero kernel changes.** The entire Pallas kernel, DMA pipeline, and metadata flow already exist and are tested in production (speculative decoding).
2. **Low development cost.** The only new code is a mask construction function (~50 lines) that builds a flattened int32 array encoding the shared-prefix + block-diagonal pattern.
3. **Memory is acceptable for real workloads.** Scoring typically uses seq_len < 4K (300-token query + moderate items). Even 100 items × 10 tokens = 7 MB mask — trivial.
4. **Add a server-side guard** on maximum total sequence length when multi-item is enabled (e.g., `max_multi_item_seq_len = 8192`), rejecting requests that would create >256MB masks.

**What to build:**

```python
def build_multi_item_attention_mask(
    prefix_len: int,
    item_lens: list[int],  # includes delimiter tokens
) -> np.ndarray:
    """Build flattened custom_mask for multi-item scoring.

    Returns int32 array of shape [seq_len * seq_len] where:
    - 1 = attend, 0 = blocked
    - All positions can attend to prefix (positions 0..prefix_len-1)
    - Each item can attend to itself (causal within item)
    - Items cannot attend to other items
    """
    seq_len = prefix_len + sum(item_lens)
    mask = np.zeros((seq_len, seq_len), dtype=np.int32)

    # Prefix: standard causal
    for i in range(prefix_len):
        mask[i, :i+1] = 1

    # Items: each sees prefix + causal within itself
    offset = prefix_len
    for item_len in item_lens:
        for i in range(item_len):
            q_pos = offset + i
            mask[q_pos, :prefix_len] = 1         # attend to all prefix
            mask[q_pos, offset:offset+i+1] = 1   # causal within item
        offset += item_len

    return mask.flatten()
```

**Integration point:** Construct the mask in the scheduler or batch preparation code, set `forward_batch.attn_backend.forward_metadata.custom_mask = jnp.array(mask)`, and the existing kernel handles the rest.

### Production Optimization (if needed): Splash Attention or Procedural Pallas

Only pursue this if:
1. Users hit the seq_len > 8K case frequently (128 items × 50+ tokens)
2. Memory profiling shows the mask is a bottleneck in practice

**Splash Attention path:** Replace `ragged_paged_attention` with `splash_attention` for multi-item scoring requests only. This is a medium-effort integration that gives O(seq) device memory. The `NumpyMask` handles block-sparse optimization automatically.

**Procedural Pallas path:** Modify the existing `ragged_paged_attention` kernel to compute the mask procedurally from `prefix_len` + `item_boundaries` instead of loading a materialized mask. This avoids the O(seq²) HBM cost entirely. Higher effort but stays within the existing kernel framework.

### Testing: Native Backend Extension

Extend `_apply_extend_mask` in `native_backend.py` with multi-item segment logic. This gives a reference implementation for CPU correctness testing that's independent of the Pallas kernel.

---

## Open Risks

1. **Static shape interaction:** The `custom_mask` must have a static shape for JIT compilation. The mask size is `seq_len²`, where `seq_len` depends on query length + total item tokens. This interacts with the existing token-length padding buckets. Need to verify that the mask is constructed after padding and matches the padded seq_len.

2. **Batch size > 1:** The `custom_mask` format supports multiple sequences in a batch (`cu_q_lens` defines boundaries). For multi-item scoring where each request is a single sequence, batch_size=1 is the common case. If multiple multi-item requests are batched together, each needs its own mask segment — the existing infrastructure handles this via `cu_seq_mask_lens`.

3. **`causal=0` side effects:** Setting `causal=0` when `custom_mask` is present (line 458) disables causal masking globally. The custom mask must encode the causal constraint itself (which our construction does). Verify no other code path depends on `causal=1` for scoring requests.

---

## Files Referenced

| File | Lines | What |
|------|-------|------|
| `sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` | 124, 177-186, 474-509, 926-934, 1376 | custom_mask parameter, kernel usage, DMA |
| `sgl_jax/srt/layers/attention/flashattention_backend.py` | 43, 457-458, 523 | Metadata field, causal override, kernel call |
| `sgl_jax/srt/layers/attention/native_backend.py` | 271-319 | Block-diagonal masking (reference for testing) |
| `sgl_jax/srt/models/umt5.py` | 65-97 | Block-diagonal mask construction pattern |
| `sgl_jax/srt/speculative/eagle_worker.py` | 354-362 | Existing custom_mask usage (construction) |
| `sgl_jax/srt/speculative/eagle_util.py` | 235-265, 680 | Mask construction function, data structure |
