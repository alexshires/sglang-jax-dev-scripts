# RFC-013: Multi-Item Scoring Performance Optimization (v0.1 → v1.0)

| | |
|------------|------|
| **Date** | 2026-02-11 |
| **Status** | Draft |
| **Author** | Engineering Team |
| **Depends On** | [RFC-008](008-multi-item-scoring.md) (v0.1 baseline) |
| **Implementation** | **[v1.0 Implementation Spec](../specs/multi-item-scoring-v1-impl.md)** |
| **Related** | [Investigation: PyTorch Multi-Item Isolation](../investigations/pytorch-multi-item-isolation-semantics.md) |

## Summary

This RFC defines the optimization roadmap to evolve multi-item scoring from **v0.1** (feature-complete MVP) to **v1.0** (production-optimized). The goal is to match or exceed PyTorch GPU throughput on TPU while maintaining correctness.

## Motivation

Multi-item scoring v0.1 ([RFC-008](008-multi-item-scoring.md), [PR #15](https://github.com/alexshires/sglang-jax/pull/15)) achieved:
- **16.5x speedup** over serial scoring (chunk_size=64)
- **Zero isolation drift** (items don't affect each other's scores)
- **Feature-gated MVP** with comprehensive validation

However, performance is limited by the O(seq²) attention mask memory:
- OOM at chunk_size=128 (84MB mask)
- Maximum practical throughput ~80 items/sec on TPU v6e-1

**Target for v1.0:** Match or beat PyTorch GPU throughput with production-ready stability.

### Throughput Targets (Target workload: query=2000, items=500, item_tokens=20)

| Scenario | Target | vs v0.1 |
|----------|--------|---------|
| Conservative (tile-skip only) | 150-200 items/s | 2-2.5x |
| Expected (prefill+extend works) | 300-400 items/s | 4-5x |
| Optimistic (everything optimal) | 500-600 items/s | 6-7x |

See [Implementation Spec](../specs/multi-item-scoring-v1-impl.md#performance-targets) for detailed analysis.

---

## Current State: v0.1 Baseline

### Performance Metrics (TPU v6e-1, Qwen3-0.6B)

| Chunk Size | Throughput | Speedup vs Serial | Mask Memory | Status |
|------------|------------|-------------------|-------------|--------|
| Serial | 4.8 items/s | 1.0x | N/A | Baseline |
| 32 | 52.2 items/s | 10.8x | 27 MB | Default |
| 64 | 79.6 items/s | 16.5x | 43 MB | Recommended |
| 128 | OOM | - | 84 MB | Fails |

### Architecture Summary

```
Request → TokenizerManager (chunking) → Scheduler (mask construction)
       → ragged_paged_attention (custom_mask) → Score extraction
```

Key components:
1. **Chunking** in TokenizerManager (default chunk_size=32)
2. **Custom attention mask** construction in flashattention_backend.py
3. **Position reset** at delimiter boundaries in schedule_batch.py
4. **Score extraction** via scipy.softmax in main process

### Known Blockers

| Blocker | Status | Impact | Workaround |
|---------|--------|--------|------------|
| **Segment mask TPU lowering** | Open | Strategy 1 blocked | Use `--multi-item-mask-impl=dense` |

**Segment mask issue:** The segment mode (`MULTI_ITEM_MASK_MODE_SEGMENT=2`) fails on TPU with `ValueError: Cannot do int indexing on TPU during kernel lowering`. Root cause: dynamic integer array indexing (gather) at `ragged_paged_attention.py:933` is not supported in Pallas kernels.

See: [Investigation: Segment Mask TPU Lowering Issue](../investigations/segment-mask-tpu-lowering-issue.md)

### Known Bottlenecks

There are **two orthogonal bottlenecks** that must both be addressed:

#### Bottleneck A: O(N²) Compute Waste (Algorithmic)

The current implementation constructs a single sequence `Query + Item1 + Item2 + ...` and applies a `custom_mask` to prevent cross-item attention. However, the kernel computes the **full N×N attention matrix** before applying the mask.

```
For M items of length L:
  Actual compute:   O((M × L)²)     # Full attention matrix
  Useful compute:   O(M × L²)       # Each item attends to query + itself
  Efficiency:       ~1/M            # Gets WORSE as items increase!
```

**Impact:** With 64 items, we're doing **64x more FLOPs than necessary**, even if memory weren't a constraint. The kernel calculates attention scores between Item_i and Item_j only to zero them out.

#### Bottleneck B: O(N²) Memory + Transfer Overhead

The mask is constructed on CPU and transferred to TPU:

```
Current flow:
  Python loops (slow) → Dense mask (84MB for 128 items) → CPU→TPU transfer → HBM storage
```

| Bottleneck | Impact | Location |
|------------|--------|----------|
| **Python loop mask construction** | Slow preprocessing | `flashattention_backend.py:113-161` |
| **O(seq²) mask memory** | OOM at chunk>64 | `flashattention_backend.py:248-285` |
| **CPU→TPU mask transfer** | Bandwidth bottleneck | Every forward pass |
| +8 JIT compilations | 40-50s first request | pytree structure change |
| Python softmax | CPU-bound for large N | `tokenizer_manager.py:1292` |

#### Why Both Matter

| Bottleneck | Symptom | Fix Required |
|------------|---------|--------------|
| Compute waste | Poor scaling with item count | Tile-skipping in kernel |
| Memory overhead | OOM at chunk>64 | Procedural mask or on-device generation |

Even if we solve the memory problem, we'll still waste compute. Even if we solve compute, we may still OOM. **Both must be addressed for v1.0.**

---

## Critical Investigation: PyTorch Isolation Semantics

Before committing to optimization strategies, we must answer:

**Does PyTorch SGLang enforce item isolation in multi-item scoring?**

Preliminary analysis suggests PyTorch may use pure causal attention without custom masks, meaning items CAN see previous items:

```
# With isolation (JAX current):
score(item2) = P(item2 | query)           # Correct

# Without isolation (PyTorch suspected):
score(item2) = P(item2 | query, item1)    # Contaminated
```

**Why this matters:**
- If PyTorch lacks isolation, JAX's custom mask is a **correctness advantage**
- If users don't need strict isolation, we could offer a "causal mode" for 2-4x additional speedup
- The optimization strategy depends on whether isolation is negotiable

**Action:** See [Investigation: PyTorch Multi-Item Isolation Semantics](../investigations/pytorch-multi-item-isolation-semantics.md)

---

## Optimization Strategies

### Strategy 1: Tile-Skipping Kernel with Segment IDs (Recommended if isolation required)

**Addresses:** Both Bottleneck A (compute waste) AND Bottleneck B (memory)

**Current flow:**
```
Python loops → Dense mask (84MB) → CPU→TPU transfer → Full O(N²) kernel → Apply mask
```

**Proposed flow:**
```
Segment IDs (O(M)) → Pass to kernel → Skip non-matching tiles → O(M×L²) kernel
```

**Key insight:** Instead of computing full attention then masking, **skip tiles entirely** when query and key belong to different items.

```python
# Current: Pass dense mask, compute everything, then mask
custom_mask = build_dense_mask(...)  # O(seq²) memory
output = ragged_paged_attention(custom_mask=custom_mask, ...)  # O(seq²) compute

# Proposed: Pass segment boundaries, skip in kernel
segment_ids = [0]*query_len + [1]*item1_len + [2]*item2_len + ...  # O(seq) memory
# OR just pass: prefix_len, delimiter_positions

# Inside Pallas kernel's KV block loop:
def compute_with_bkv(q_segment, kv_segment, ...):
    if q_segment != kv_segment and kv_segment != 0:  # 0 = shared prefix
        return  # SKIP ENTIRELY - no compute, no memory access
    # ... normal attention computation ...
```

| Metric | Current | Proposed |
|--------|---------|----------|
| Compute | O((M×L)²) | O(M×L²) |
| Memory | O(seq²) | O(seq) |
| Max chunk | ~64 | 256+ |
| Efficiency | ~1/M | ~100% |
| Kernel changes | None | Pallas modification |

**Implementation:**
1. Add `segment_ids: jax.Array` or `delimiter_positions: jax.Array` parameter to `ragged_paged_attention`
2. Modify Pallas kernel's `compute_with_bkv` loop to check segment membership
3. Early-exit when query segment ≠ key segment (and key is not prefix)
4. Remove dense mask construction in `get_forward_metadata()`

**Effort:** Medium-High (2-3 weeks)

**This is the key optimization.** Without tile-skipping, we cannot scale efficiently to many items.

---

### Strategy 2: Causal Mode (If isolation not required)

**Offer a flag to disable item isolation and use pure causal attention.**

```bash
# Server arg
--multi-item-scoring-mode [isolated|causal]
```

| Mode | Isolation | Custom Mask | Memory | Max Chunk | Speedup |
|------|-----------|-------------|--------|-----------|---------|
| `isolated` | Yes | Required | O(seq²) | ~64 | 16.5x |
| `causal` | No | None | O(seq) | 256+ | ~40-50x |

**When causal mode is acceptable:**
- Reranking independent candidates (search, recommendations)
- Workloads where slight score contamination doesn't affect ranking
- Users who prioritize throughput over strict semantics

**Implementation:**
1. Add `--multi-item-scoring-mode` server arg
2. In `causal` mode: skip `_build_multi_item_attention_mask()`, keep `custom_mask=None`
3. Position reset still applies (each item sees query at position 0-N)

**Effort:** Low (2-3 days)

**Prerequisite:** Verify PyTorch behavior in isolation investigation.

---

### Strategy 3: Splash Attention Integration

**Use `jax.experimental.pallas.ops.tpu.splash_attention` for block-sparse attention.**

Splash attention is designed for exactly this pattern:
- O(seq) device memory (vs O(seq²) for current approach)
- Block-sparse preprocessing skips fully-masked blocks
- TPU-optimized tiling

```python
from jax.experimental.pallas.ops.tpu.splash_attention import (
    NumpyMask, MultiHeadMask, make_splash_mha
)

mask_np = build_multi_item_mask(prefix_len, item_lens)
kernel = make_splash_mha(
    MultiHeadMask([NumpyMask(mask_np)] * num_heads),
    head_shards=1, q_seq_shards=1
)
output = kernel(q, k, v)
```

**Pros:**
- Best memory profile
- TPU-native optimization
- Handles block-diagonal patterns efficiently

**Cons:**
- Requires new integration path (sglang-jax doesn't use splash attention)
- API stability uncertain (`jax.experimental`)
- Would need to coexist with ragged_paged_attention

**Effort:** High (2-4 weeks)

---

### Strategy 4: Incremental Optimizations

Quick wins that improve v0.1 without architectural changes:

#### 4a. Vectorize Mask Generation (Immediate Fix)

**Addresses:** Bottleneck B (preprocessing overhead)

The current `_build_multi_item_attention_mask()` uses Python loops to construct the mask token-by-token. For seq_len=8192, this is extremely slow.

```python
# Current: Python loops (SLOW)
def _build_multi_item_attention_mask(tokens, delimiter):
    mask = np.zeros((seq_len, seq_len), dtype=np.int32)
    for i in range(seq_len):
        for j in range(i + 1):
            if same_segment(i, j):
                mask[i, j] = 1
    return mask

# Proposed: NumPy vectorization (FAST)
def _build_multi_item_attention_mask_vectorized(tokens, delimiter):
    # Assign segment IDs using searchsorted
    delimiter_positions = np.where(tokens == delimiter)[0]
    segment_ids = np.searchsorted(delimiter_positions, np.arange(len(tokens)))

    # Vectorized mask: same segment OR key is in prefix (segment 0)
    q_segments = segment_ids[:, None]  # (seq, 1)
    k_segments = segment_ids[None, :]  # (1, seq)

    same_segment = (q_segments == k_segments)
    key_is_prefix = (k_segments == 0)
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    mask = ((same_segment | key_is_prefix) & causal).astype(np.int32)
    return mask
```

**Benefit:** Orders of magnitude faster mask construction on CPU.

**Effort:** Low (1 day) — Safe, localized change.

#### 4b. On-Device Mask Generation (Intermediate)

**Addresses:** Bottleneck B (CPU→TPU transfer)

Move mask generation from CPU (NumPy) to TPU (JAX):

```python
# Current: CPU mask, then transfer
mask_np = build_mask_numpy(...)  # CPU
mask_jax = jnp.array(mask_np)    # Transfer to TPU

# Proposed: Generate directly on TPU
@jax.jit
def build_mask_on_device(tokens, delimiter, seq_len):
    delimiter_positions = jnp.where(tokens == delimiter, size=max_items)[0]
    segment_ids = jnp.searchsorted(delimiter_positions, jnp.arange(seq_len))

    q_segments = segment_ids[:, None]
    k_segments = segment_ids[None, :]

    same_segment = (q_segments == k_segments)
    key_is_prefix = (k_segments == 0)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    return ((same_segment | key_is_prefix) & causal).astype(jnp.int32)
```

**Benefit:** Eliminates CPU→TPU transfer for the mask (84MB for 128 items).

**Effort:** Low-Medium (2-3 days)

**Note:** Still O(seq²) memory on device, so doesn't solve OOM. But removes transfer bottleneck.

#### 4c. Dynamic Chunk Sizing

Adapt chunk size based on item lengths to maximize throughput within memory budget:

```python
def optimal_chunk_size(query_len, item_lens, hbm_budget_mb=64):
    avg_item_len = sum(item_lens) / len(item_lens)
    # Find max chunk where mask fits in budget
    for chunk in [128, 64, 32, 16, 8, 4, 2]:
        seq_len = query_len + (chunk * (avg_item_len + 1))  # +1 for delimiter
        mask_mb = (seq_len ** 2 * 4) / 1e6
        if mask_mb <= hbm_budget_mb:
            return chunk
    return 2
```

**Effort:** Low (1 day)

#### 4d. Async Chunk Pipelining

Overlap computation of chunk N with score extraction of chunk N-1:

```python
async def score_multi_item_pipelined(items, chunk_size):
    chunks = split_into_chunks(items, chunk_size)
    prev_future = None
    scores = []

    for chunk in chunks:
        compute_future = launch_forward_pass_async(chunk)
        if prev_future:
            scores.extend(await prev_future)  # Extract while computing
        prev_future = compute_future

    scores.extend(await prev_future)
    return scores
```

**Effort:** Medium (3-5 days)

#### 4e. Batch Softmax for Large N

Current scipy.softmax is CPU-bound. For large item counts:

```python
if len(scores) > 100 and jax.default_backend() != 'cpu':
    scores_array = jnp.array(scores)
    probs = jax.nn.softmax(scores_array)
    return np.array(probs)
else:
    return scipy.special.softmax(scores)
```

**Effort:** Low (1 day)

#### 4f. Multi-Item Block Size Tuning

Add tuned configs for multi-item patterns to `tuned_block_sizes.py`:

```python
MULTI_ITEM_BLOCK_CONFIGS = {
    # (dtype, heads, head_dim, page_size, max_tokens, num_items)
    (bfloat16, 16, 128, 16, 512, 32): (4, 32),
    (bfloat16, 16, 128, 16, 1024, 64): (8, 64),
    # ...
}
```

**Effort:** Medium (profile + tune, 1 week)

#### 4g. Multi-Item Precompile/Warmup

Precompile kernels for common token buckets to eliminate first-request JIT penalty:

```python
def warmup_multi_item_kernels(model, common_shapes):
    """Precompile for common multi-item workload shapes."""
    shapes = [
        (2000, 32, 20),   # query_len, num_items, item_len
        (2000, 64, 20),
        (2000, 128, 20),
        (500, 64, 50),
    ]
    for query_len, num_items, item_len in shapes:
        dummy_request = build_dummy_multi_item_request(query_len, num_items, item_len)
        _ = model.score(dummy_request)  # Trigger JIT compilation
```

**Benefit:** First-request latency drops from 40-50s to <1s for precompiled shapes.

**Effort:** Low (1-2 days)

---

### Strategy 5: Prefill+Extend with Prefix Cache (Best for Long-Query/Short-Item)

**Addresses:** Both bottlenecks for specific workload geometry

For workloads with **long queries and many short items**, a different algorithm may outperform packed multi-item:

```
Workload: 2000-token query + 500 × 20-token items

Packed Multi-Item (current):
  - One sequence: 2000 + 500×20 = 12,000 tokens
  - Attention: O(12000²) even with tile-skipping
  - Memory: segment_ids for 12K tokens

Prefill+Extend (proposed):
  - Prefill query ONCE: O(2000²) → cache KV
  - 500 batched extends of 20 tokens, reusing prefix cache
  - Attention per extend: O(20 × 2020) = tiny
  - Total: O(2000²) + O(500 × 20 × 2020) ≈ O(2000²) + O(20M)
  - Much smaller than O(144M) for packed approach!
```

**Implementation:**

```python
async def score_prefill_extend(query: str, items: List[str], batch_size: int = 32):
    # 1. Prefill query once, cache KV
    prefix_cache_id = await prefill_and_cache(query)

    # 2. Batch items into extend requests
    scores = []
    for batch in chunk(items, batch_size):
        # Each extend reuses the cached query KV
        batch_scores = await batched_extend_score(
            prefix_cache_id=prefix_cache_id,
            items=batch,
        )
        scores.extend(batch_scores)

    return scores
```

**Requirements:**
- Re-enable radix cache for scoring (currently disabled)
- Add `prefill_and_cache()` API to cache query KV
- Add `batched_extend_score()` for parallel item scoring

**When Prefill+Extend wins:**

| Workload | Packed Multi-Item | Prefill+Extend | Winner |
|----------|-------------------|----------------|--------|
| 2000 query, 500×20 items | O(144M) | O(4M + 20M) | **Prefill+Extend** |
| 500 query, 100×100 items | O(100M) | O(0.25M + 20M) | **Prefill+Extend** |
| 100 query, 10×500 items | O(26M) | O(0.01M + 5M) | **Prefill+Extend** |
| 100 query, 500×10 items | O(26M) | O(0.01M + 5M) | **Prefill+Extend** |

**Key insight:** Prefill+Extend wins when `query_len >> item_len` because the query attention is computed once instead of repeated.

**Effort:** High (2-3 weeks) — requires cache integration

---

### Strategy 6: Runtime Policy Selector

**Automatically choose the best algorithm based on workload geometry.**

```python
def select_scoring_algorithm(
    query_len: int,
    num_items: int,
    avg_item_len: int,
    memory_headroom_mb: float,
) -> ScoringAlgorithm:
    """
    Choose between:
    - PACKED_MULTI_ITEM: Current approach with tile-skipping
    - PREFILL_EXTEND: Query prefill once + batched extends
    - SERIAL: Fallback for edge cases
    """
    packed_seq_len = query_len + num_items * (avg_item_len + 1)
    packed_memory_mb = estimate_packed_memory(packed_seq_len, num_items)

    # Heuristic: Prefill+Extend wins when query dominates
    query_dominates = query_len > 4 * avg_item_len
    many_items = num_items > 32

    if query_dominates and many_items:
        return ScoringAlgorithm.PREFILL_EXTEND

    if packed_memory_mb < memory_headroom_mb:
        return ScoringAlgorithm.PACKED_MULTI_ITEM

    # Fallback to chunked packed or serial
    return ScoringAlgorithm.PACKED_MULTI_ITEM_CHUNKED


# Log selection for observability
logger.info(f"Selected {algorithm.name} for query_len={query_len}, "
            f"num_items={num_items}, avg_item_len={avg_item_len}")
```

**Server args:**

```bash
--multi-item-scoring-algorithm [auto|packed|prefill-extend|serial]
```

- `auto` (default): Runtime selection based on geometry
- `packed`: Force packed multi-item (current approach)
- `prefill-extend`: Force prefill+extend
- `serial`: Force serial scoring (baseline)

**Effort:** Medium (1 week) — after Strategies 1 and 5 are implemented

---

## Implementation Phases

**Execution Model:** Phases 0-1 are sequential prerequisites. Phases 2-4 run in **parallel tracks** with strict interface boundaries. Phase 5 gates all promotions.

```
Phase 0 (Stability)     ─┬─→ Phase 2 (Segment Fix)    ─┬─→ Phase 5 (Hardening)
                         │                              │
Phase 1 (Quick Wins)    ─┼─→ Phase 3 (Investigation)  ─┤
                         │                              │
                         └─→ Phase 4 (Prefill+Extend) ─┘
```

### Phase 0: Stability/Baseline (Immediate)

**Objective:** Lock runtime to `dense` mode, establish clean baseline for parallel optimization work.

| Task | Action | Status |
|------|--------|--------|
| Default to dense | Set `--multi-item-mask-impl=dense` or threshold=0 | Pending |
| Run TPU regression | Full test suite with dense mode | Pending |
| Capture baseline artifacts | Latency, throughput, memory on canonical workload | Pending |
| Document dense as production fallback | Update user docs | Pending |

**Exit criteria:** All regression tests pass with dense mode on TPU v6e-1. Baseline artifacts committed.

**Rationale:** Dense mode works. Segment mode has a TPU kernel lowering bug. Lock to dense so optimization work (Phases 2-4) doesn't block production stability.

### Phase 1: Quick Wins

| Task | Strategy | Effort | Expected Gain |
|------|----------|--------|---------------|
| **Vectorize mask generation** | **4a** | **1 day** | **10-100x faster preprocessing** |
| Change default to chunk=64 | Config | 1 hour | Already 16.5x |
| Dynamic chunk sizing | 4c | 1 day | Better defaults |
| Batch softmax for large N | 4e | 1 day | ~10x for 500+ items |
| On-device mask generation | 4b | 2 days | Eliminate CPU→TPU transfer |
| **Precompile common shapes** | **4g** | **1-2 days** | **First-request: 40s → <1s** |

**Exit criteria:** 20%+ improvement in large-batch throughput, measurable reduction in preprocessing latency, first-request penalty <5s for precompiled shapes.

### Phase 2: Investigation (1 week, parallel with Phase 1)

| Task | Deliverable |
|------|-------------|
| Verify PyTorch isolation behavior | Investigation doc with test results |
| Benchmark PyTorch multi-item throughput | Baseline numbers for comparison |
| Decide: isolation required or optional? | ADR or RFC amendment |

**Exit criteria:** Clear decision on whether to pursue Strategy 2 (causal mode).

### Phase 3: Core Optimization - Packed Multi-Item (2-3 weeks)

Based on Phase 2 outcome:

**If isolation required:**
- Implement Strategy 1 (tile-skipping kernel with segment IDs)
- Modify Pallas kernel to skip non-matching tiles
- Target: chunk_size=256, ~150 items/s, O(M×L²) compute

**If isolation optional:**
- Implement Strategy 2 (causal mode)
- Target: chunk_size=512+, ~300 items/s

### Phase 4: Prefill+Extend Algorithm (2-3 weeks, can parallel with Phase 3)

| Task | Purpose |
|------|---------|
| Re-enable radix cache for scoring | Prerequisite for prefix caching |
| Implement `prefill_and_cache()` API | Cache query KV |
| Implement `batched_extend_score()` | Parallel item scoring |
| Implement Strategy 5 (Prefill+Extend) | Alternative algorithm |
| Implement Strategy 6 (Runtime Selector) | Auto-select best algorithm |

**Exit criteria:** Prefill+Extend beats Packed Multi-Item by >20% on 2000-query/500×20-item workload.

### Phase 5: Production Hardening (1 week)

| Task | Purpose |
|------|---------|
| Async chunk pipelining (4d) | Hide latency |
| Multi-item block size tuning (4f) | TPU optimization |
| Comprehensive benchmarking | Validate targets across workload shapes |
| Flag-first rollout | `--multi-item-scoring-algorithm` flags |
| Documentation update | User guide with algorithm selection guidance |

**Rollout process:**
1. Ship with flags, default to `auto`
2. Run 3 stable comparison runs with identical winner selection
3. Promote measured winners to defaults

---

## Success Metrics for v1.0

### Performance Targets

| Metric | v0.1 Baseline | v1.0 Target | Stretch Goal |
|--------|---------------|-------------|--------------|
| Max chunk size (packed) | 64 | 256 | 512 |
| Throughput (500 items) | 80 items/s | 200 items/s | 400 items/s |
| Speedup vs serial | 16.5x | 40x | 80x |
| First-request latency | 40-50s | <5s (precompiled) | <1s |
| Memory efficiency | O(seq²) | O(seq) | O(seq) |
| **Compute efficiency** | **~1/M (wasteful)** | **~100% (tile-skip)** | **~100%** |
| Mask preprocessing | Python loops | Vectorized | On-device |

### Algorithm Coverage

| Workload Shape | v0.1 | v1.0 |
|----------------|------|------|
| Long query, many short items | Packed only | **Prefill+Extend** (auto-selected) |
| Short query, few long items | Packed only | Packed with tile-skipping |
| Mixed workloads | Manual chunk tuning | **Auto-selection** |

### Correctness Gates (Hard Requirements)

Every optimization must pass before eligibility:
- `max_abs_diff <= 0.02` vs reference
- `mean_abs_diff <= 0.01` vs reference
- All v0.1 isolation tests pass (unless causal mode explicitly selected)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyTorch does have isolation | Strategy 2 invalid | Fall back to Strategy 1 |
| Procedural mask degrades kernel perf | Lower than expected gains | Benchmark early, have splash attention as backup |
| Splash attention API unstable | Integration breaks | Pin JAX version, have fallback path |
| TPU memory varies by version | OOM on smaller TPUs | Dynamic chunk sizing handles this |

---

## Alternatives Considered

### ~~Alternative 1: Prefix Caching + Single-Item Batching~~ → Promoted to Strategy 5

**Description:** Cache query KV, process items in parallel batches (not concatenated).

**Original rejection reasoning (now reconsidered):**
- ~~Radix cache currently incompatible with multi-item mode~~ → Can be re-enabled for scoring
- ~~Would require significant architectural changes~~ → Worth it for long-query workloads
- ~~Less efficient than true multi-item~~ → Actually MORE efficient when `query_len >> item_len`

**Update (2026-02-11):** After further analysis, this approach is now **Strategy 5: Prefill+Extend**. For workloads with long queries and many short items, prefill+extend can significantly outperform packed multi-item by computing query attention only once.

### Alternative 2: Custom Pallas Kernel from Scratch

**Description:** Write a purpose-built kernel for multi-item scoring.

**Why not chosen:**
- High effort, low ROI vs modifying existing kernel
- Maintenance burden of separate code path
- Existing ragged_paged_attention is well-tested

### Alternative 3: Wait for Upstream PyTorch Improvements

**Description:** Let PyTorch optimize first, then port.

**Why not chosen:**
- JAX/TPU has different constraints than PyTorch/GPU
- Proactive optimization gives competitive advantage
- Can always adopt good ideas from PyTorch later

---

## References

- [RFC-008: Multi-Item Scoring (v0.1)](008-multi-item-scoring.md)
- [Investigation: Multi-Item Attention Mechanism](../investigations/multi-item-attention-mechanism.md)
- [Investigation: Multi-Item Chunk Size Benchmark](../investigations/multi-item-chunk-size-benchmark.md)
- [Investigation: PyTorch Multi-Item Isolation Semantics](../investigations/pytorch-multi-item-isolation-semantics.md) (pending)
- [Report: Multi-Item Scoring TPU Validation](../reports/multi-item-scoring-tpu-validation-2026-02-07.md)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-12 | **Parallel track model:** Added Phase 0 (Stability) as prerequisite. Documented segment mask TPU lowering blocker. Added execution model diagram showing parallel tracks. Dense mode now explicit production fallback until segment fix lands. |
| 2026-02-11 | **Major update:** Added Strategy 5 (Prefill+Extend) and Strategy 6 (Runtime Policy Selector). Promoted "Prefix Caching" from rejected alternative to first-class strategy. Added Phase 4 for prefill+extend implementation. Added precompile/warmup (4g). Updated success metrics with algorithm coverage and explicit correctness gates. Added flag-first rollout process. |
| 2026-02-11 | Enhanced with dual-bottleneck analysis (compute + memory). Added vectorization (4a) and on-device mask generation (4b) strategies. Clarified Strategy 1 as "tile-skipping kernel" with early-exit semantics. Reordered incremental optimizations. |
| 2026-02-11 | Initial draft |
