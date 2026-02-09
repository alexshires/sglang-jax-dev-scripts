# Investigation: Multi-Item Scoring Compilation Overhead

| | |
|------------|------|
| **Date** | 2026-02-06 |
| **Status** | Complete — recommendation ready |
| **Related** | [RFC-008](../rfcs/008-multi-item-scoring.md), Decision 7 |
| **Blocks** | RFC-008 Phase 0 (Compilation assessment) |

## Question

How many additional JIT compilations does multi-item scoring introduce, and should they be precompiled at startup?

---

## Background: How JIT Compilation Works in sglang-jax

### The JIT Boundary

The outer JIT wraps the entire model forward pass (`model_runner.py:191-208`):

```python
@partial(
    jax.jit,
    donate_argnames=["token_to_kv_pool"],
    static_argnames=["model_state_def"],
    compiler_options=jit_compiler_options,
)
def jitted_run_model(
    model_def, model_state_def, model_state_leaves,
    forward_batch, token_to_kv_pool, logits_metadata,
):
    ...
```

JAX JIT traces based on:
1. **Pytree structure** of all dynamic arguments (number and arrangement of leaves)
2. **Shapes and dtypes** of all array leaves
3. **Values** of static arguments (`model_state_def`)

A new compilation is triggered when ANY of these change.

### Padding Buckets

sglang-jax pads inputs to fixed sizes to limit the number of distinct shapes:

- **Token paddings** (default): `[64, 128, 256, 512, 1024, 2048, 4096, 8192]` — 8 values
  (`common_utils.py:37`)
- **Batch size paddings** (default): `[1, 2, 4, 8, 16, 32, 64, 128, 256]` — 9 values
  (`common_utils.py:38`)

### EXTEND Mode Constraint

EXTEND mode forces batch_size to the maximum padding value (`schedule_batch.py:1162`):

```python
bs_paddings = bs_paddings[-1:]  # e.g., [256]
```

This means EXTEND always uses a single, fixed batch_size — only token count varies.

---

## The Pytree Chain for `custom_mask`

Multi-item scoring introduces `custom_mask` (a per-token attention mask) into the forward pass. The pytree path from the JIT boundary to the mask is:

```
forward_batch: ForwardBatch (pytree, @register_pytree_node_class)
  └── children[6]: attn_backend: FlashAttention (pytree via nnx.Module)
      └── children[0]: forward_metadata: FlashAttentionMetadata (pytree, @register_pytree_node_class)
          └── children[6]: custom_mask: None | jax.Array
```

**Key files:**
- `forward_batch_info.py:185-213` — ForwardBatch.tree_flatten
- `flashattention_backend.py:395-404` — FlashAttention.tree_flatten
- `flashattention_backend.py:45-57` — FlashAttentionMetadata.tree_flatten

### Why `None` → `jax.Array` Triggers Recompilation

In JAX's pytree system:
- `None` is an **empty node** (flattens to zero leaves)
- `jax.Array` is a **leaf** (flattens to one leaf)

When `custom_mask` transitions from `None` to `jax.Array`, the total number of leaves in the flattened pytree changes. This is a **structural** difference — JAX JIT treats it as a completely different function signature and must recompile from scratch.

---

## Current Compilation Budget (Baseline)

### Normal (non-speculative) server

| Mode | batch_size values | token values | Compilations |
|------|-------------------|--------------|--------------|
| EXTEND | 1 (max bs) | 8 (token buckets) | ≤ 8 |
| DECODE | 9 (bs buckets) | = bs | 9 |
| **Total** | | | **~17** |

All compilations have `custom_mask=None` because `get_forward_metadata` (`flashattention_backend.py:100-165`) never sets it.

### Speculative (EAGLE) server

EAGLE adds TARGET_VERIFY and SPEC_EXTEND/SPEC_DECODE modes. TARGET_VERIFY uses `custom_mask` via `get_eagle_forward_metadata` (`flashattention_backend.py:167-187`), but this has a different `forward_mode` (different `aux_data` in ForwardBatch pytree), so it's already a separate compilation variant from regular EXTEND.

EAGLE's `custom_mask` compilations are **independent** of multi-item scoring's — no overlap or sharing.

---

## Multi-Item Scoring Compilation Impact

### What changes

Multi-item scoring uses **EXTEND mode** with `custom_mask = jax.Array[mask_size]` instead of `None`.

### How many additional compilations

| Factor | Analysis |
|--------|----------|
| **Pytree structure** | Changes (None → Array) — new compilation per bucket |
| **Batch size axis** | Fixed at max_bs for EXTEND — no additional axis |
| **Token axis** | 8 buckets — one compilation each |
| **Mask shape** | Deterministic per token bucket: `[T²]` for bucket T |
| **Item count** | Does NOT affect mask shape — only mask values |
| **DECODE mode** | Not needed — scoring is a single EXTEND pass |

**Additional compilations: exactly +8 (one per token bucket)**

### Why item count does NOT create extra compilations

The RFC previously claimed item-count buckets `[1, 5, 10, 50, 128]` would add 5 compilations. This was incorrect:

- Item count affects the **values** inside the mask (which positions attend to which)
- Item count does NOT affect the mask's **shape** (determined by padded total token count)
- Same token bucket → same mask shape → same compilation key
- 5 items and 50 items in the same 512-token bucket both use `custom_mask` shape `[262144]`

The relationship is: `mask_size = padded_total_tokens²`, where `padded_total_tokens` is the token padding bucket. Item count is invisible to JIT.

### Mask shape per token bucket

For a single multi-item sequence (batch_size=1) where q_len ≈ kv_len ≈ T:

| Token bucket (T) | Mask shape `[T²]` | Mask size (int32) |
|-------------------|-------------------|-------------------|
| 64 | [4,096] | 16 KB |
| 128 | [16,384] | 64 KB |
| 256 | [65,536] | 256 KB |
| 512 | [262,144] | 1 MB |
| 1,024 | [1,048,576] | 4 MB |
| 2,048 | [4,194,304] | 16 MB |
| 4,096 | [16,777,216] | 64 MB |
| 8,192 | [67,108,864] | 256 MB |

Note: The kernel internally expands the mask to `[mask_size, head_dim]` via `jnp.repeat` (`ragged_paged_attention.py:1514`), but this happens inside the JIT boundary and doesn't affect the compilation key.

### Compilation cost estimates

| Metric | Estimate |
|--------|----------|
| Time per compilation | ~10-30s on TPU v6e |
| Total time for 8 compilations | ~80-240s |
| HBM per cached compilation | ~100-500MB (model-dependent) |
| Total HBM for 8 variants | ~0.8-4GB |

---

## Precompilation Strategy

### Option A (Recommended): Lazy compilation — no precompile

- JIT compiles on first multi-item request at each token bucket
- Cost: ~10-30s latency on first request per bucket
- Benefit: zero startup overhead; most servers hit only 2-3 buckets in practice
- Suitable for MVP

### Option B: Opt-in precompilation flag

- Add `--precompile-multi-item-scoring` server argument
- Precompiles all 8 EXTEND variants with `custom_mask`
- Cost: +80-240s startup, +0.8-4GB HBM
- Benefit: predictable latency from first request

### Recommendation

**Option A for MVP.** Multi-item scoring is an opt-in feature (`--multi-item-scoring-delimiter`). Most servers won't use it. For those that do, natural JIT caching handles it: the first request at each size compiles, all subsequent requests at that size are instant.

Add Option B as a follow-up only if users report unacceptable latency spikes on first multi-item requests.

---

## Correction to RFC-008

The RFC's Decision 7 claimed "5 item-count compilations from buckets [1, 5, 10, 50, 128]". This is **incorrect**:

- Item count does not affect compilation (see analysis above)
- The actual overhead is +8 EXTEND compilations from the token bucket axis
- These are **additive** (17 baseline + 8 = 25 total), not multiplicative

The RFC should be updated to reflect:

> Multi-item scoring adds at most 8 new EXTEND-mode compilations (one per token padding bucket), triggered by the pytree structure change from `custom_mask=None` to `custom_mask=jax.Array`. Item count has no effect on compilation — only on mask values. Precompilation is not needed for MVP; standard JIT caching handles the lazy compilation.

---

## Open Risks

1. **Mask padding and static shapes:** The `custom_mask` array must have a shape that matches the token padding bucket (e.g., `[T²]` for bucket T). This requires padding the mask with zeros (= block) for the padding region. The mask construction function must accept the padded size as an argument, not just the actual content lengths.

2. **HBM pressure on small TPUs:** 8 additional compilation variants at ~100-500MB each could consume significant HBM on a TPU v6e-4 (32GB total). If this is a problem, limit the number of token buckets used for multi-item scoring (e.g., only [256, 512, 1024, 2048]).

3. **Batch size > 1:** If multiple multi-item requests are batched together, the mask format uses `cu_seq_mask_lens` boundaries (`ragged_paged_attention.py:1500-1502`) to handle per-sequence mask segments. The total mask size is `sum(q_len_i * kv_len_i)` across sequences, which still must fit within the padded shape.

---

## Files Referenced

| File | Lines | What |
|------|-------|------|
| `sgl_jax/srt/model_executor/model_runner.py` | 191-208 | JIT boundary (`jitted_run_model`) |
| `sgl_jax/srt/model_executor/forward_batch_info.py` | 132-245, 286-404 | ForwardBatch pytree, init_new |
| `sgl_jax/srt/layers/attention/flashattention_backend.py` | 28-71, 100-165, 395-418, 440-526 | FlashAttentionMetadata pytree, get_forward_metadata, FlashAttention pytree, forward with custom_mask |
| `sgl_jax/srt/managers/tp_worker.py` | 244-338, 371-430 | Precompile loops, generate_model_worker_batch |
| `sgl_jax/srt/managers/schedule_batch.py` | 1148-1265 | Batch padding selection, EXTEND bs constraint |
| `sgl_jax/srt/utils/common_utils.py` | 37-38 | Default padding values |
| `sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` | 1370-1376, 1500-1522 | custom_mask parameter, cu_seq_mask_lens computation |
| `sgl_jax/srt/speculative/eagle_worker.py` | 354-362, 696-730 | EAGLE custom_mask usage, spec precompile |
