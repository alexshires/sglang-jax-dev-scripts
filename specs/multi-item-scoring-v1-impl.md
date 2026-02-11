# Implementation Spec: Multi-Item Scoring v1.0

| | |
|------------|------|
| **Date** | 2026-02-11 |
| **Status** | Ready for Implementation |
| **Target** | Match or beat PyTorch GPU on TPU v6e |
| **Workload Contract** | query=2000, items=500, item_tokens=20, labels=9454,2753 |
| **RFC** | [RFC-013](../rfcs/013-multi-item-scoring-v1-optimization.md) |

## Objective

1. Match or beat frozen PyTorch `sglang` GPU performance for multi-item scoring
2. Keep `/v1/score` external API unchanged
3. Preserve correctness semantics with hard parity gates
4. Enable parallel development across independent workstreams

---

## Performance Targets

### Baseline (v0.1 on TPU v6e-1, Qwen3-0.6B)

| Config | Throughput | Status |
|--------|------------|--------|
| Serial | 4.8 items/s | Baseline |
| chunk=64 | 79.6 items/s | Current best (16.5x) |
| chunk=128 | OOM | Blocked by O(seq²) mask |

### v1.0 Targets (Target workload: query=2000, items=500, item_tokens=20)

| Scenario | Target Throughput | vs v0.1 | Rationale |
|----------|-------------------|---------|-----------|
| **A only** (tile-skip) | 150-200 items/s | 2-2.5x | Enables chunk=256+, but kernel still O(T²) iteration |
| **B only** (prefill+extend) | 300-500 items/s | 4-6x | 6x compute reduction: O(24M) vs O(144M) |
| **A + B** (runtime selector) | 400-600 items/s | 5-7x | Best algorithm per workload geometry |

### Compute Analysis

```
Packed approach (current):
  Sequence length: 2000 + 500×21 = 12,500 tokens
  Attention ops: O(12500²) = 156M

Prefill+Extend approach:
  Prefill: O(2000²) = 4M
  500 extends: O(500 × 20 × 2020) = 20M
  Total: 24M ops

Compute reduction: 156M / 24M ≈ 6.5x
```

### Decision Gates

| Gate | Threshold | Decision |
|------|-----------|----------|
| B0 spike vs A | B0 > A by 20% | Proceed to B1 |
| B0 spike vs A | A > B0 by 20% | Skip B1, A is sufficient |
| B0 spike vs A | Within 20% | Implement both for runtime selection |
| v1.0 vs PyTorch GPU | JAX ≥ PyTorch | Success |
| v1.0 vs PyTorch GPU | JAX < PyTorch by >20% | Investigate further |

### Optimistic vs Conservative

| Estimate | Throughput | Assumption |
|----------|------------|------------|
| **Conservative** | 150-200 items/s | Only tile-skip works, prefill+extend has overhead issues |
| **Expected** | 300-400 items/s | Prefill+extend works, moderate batching efficiency |
| **Optimistic** | 500-600 items/s | Everything works perfectly, excellent extend batching |

### Risk Mitigations (from Review Feedback)

| Risk | Workstream | Mitigation |
|------|------------|------------|
| **Radix cache eviction edge cases** | B0 | Test cache eviction under memory pressure. Verify LRU logic works when re-enabled for scoring after being "dormant" in disabled mode. Add explicit eviction test to B0 exit criteria. |
| **Tile-skipping efficacy** | A | TPU/XLA control flow may not skip memory loads even with `if q_seg != k_seg`. **Verify speedup early** - if only 1.5x instead of ~3x, reprioritize B over A. |
| **Memory limit surprisingly low** | A | ~2700 tokens OOM (2000 OK) suggests kernel allocates large intermediates (perhaps `[B,H,Q,K]` in float32). Investigate kernel memory footprint if OOM persists after tile-skip. Splash Attention (Strategy 3) may be needed sooner than expected. |
| **Python loop overhead at scale** | B1 | For 500 items, 16 async calls is fine. For 5000+ items, Python dispatch latency becomes visible. Ensure `_batched_extend_score` keeps JAX runtime fed. Consider `jax.lax.scan` for v2.0 if needed. |

**Key insight from review:** Workstream B (Prefill+Extend) is likely the winner for the target workload. Workstream A provides correctness and robustness, but B is where the order-of-magnitude speedup lives.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         /v1/score Request                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WORKSTREAM D: Orchestration                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Runtime Policy Selector                         │    │
│  │  select_algorithm(query_len, num_items, item_len, memory)   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                          │                    │
              ┌───────────┘                    └───────────┐
              ▼                                            ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│   WORKSTREAM A: Kernel       │          │   WORKSTREAM B: Prefill+Extend│
│   (Packed + Tile-Skip)       │          │   (Cache + Batched Extends)   │
│                              │          │                              │
│  ┌────────────────────────┐  │          │  ┌────────────────────────┐  │
│  │ Segment Mask Logic     │  │          │  │ prefill_and_cache()    │  │
│  │ (no dense T² mask)     │  │          │  │ batched_extend_score() │  │
│  └────────────────────────┘  │          │  └────────────────────────┘  │
│                              │          │                              │
│  ┌────────────────────────┐  │          │  ┌────────────────────────┐  │
│  │ Tuned Block Sizes      │  │          │  │ Radix Cache Integration│  │
│  │ (v6e-specific)         │  │          │  │ (re-enable for scoring)│  │
│  └────────────────────────┘  │          │  └────────────────────────┘  │
└──────────────────────────────┘          └──────────────────────────────┘
              │                                            │
              └───────────────────┬────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WORKSTREAM C: Startup                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Precompile + Warmup for common token buckets                │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Workstream Summary

| Workstream | Focus | Owner | Dependencies | Effort |
|------------|-------|-------|--------------|--------|
| **0: Interface Freeze** | Shared contracts | Tech Lead | None | 2 days |
| **A: Kernel** | Tile-skipping segment mask | Dev 1 | 0 | 2 weeks |
| **B0: Prefill+Extend Spike** | Feasibility + radix cache re-enable | Dev 2 | 0 | 1 week |
| **B1: Prefill+Extend Prod** | Full implementation | Dev 2 | B0, A | 1 week |
| **C: Startup** | Precompile/warmup | Dev 3 | A | 1 week |
| **D: Orchestration** | Runtime selector | Dev 4 | A, B1 | 1 week |
| **E: Validation** | Testing + benchmarking | Dev 5 | A, B1, C, D | 1 week |

**Critical Path:** 0 → A + B0 in parallel → B1 (if spike succeeds) → C + D → E

**Note:** A and B touch shared surfaces (`server_args`, `tokenizer_manager`, orchestration contract). Workstream 0 freezes these interfaces before parallel work begins.

---

## Workstream 0: Interface Freeze

### Objective
Define and freeze shared interfaces before parallel development begins. This prevents merge conflicts and ensures A and B can proceed independently.

### Deliverables

1. **Shared Enums and Types** (in `sgl_jax/srt/managers/io_struct.py`):
   ```python
   class ScoringAlgorithm(Enum):
       SERIAL = "serial"
       PACKED_DENSE = "packed_dense"
       PACKED_SEGMENT = "packed_segment"
       PREFILL_EXTEND = "prefill_extend"

   class MultiItemMaskMode(Enum):
       CAUSAL = 0      # Standard causal (no multi-item)
       DENSE = 1       # Dense custom_mask (v0.1)
       SEGMENT = 2     # Tile-skipping segment mask (Workstream A)
   ```

2. **Server Args Contract** (in `server_args.py`):
   ```python
   # Multi-item scoring v1.0 args (placeholder values, implementation fills in)
   multi_item_mask_impl: str = "auto"  # "auto" | "dense" | "segment"
   multi_item_enable_prefill_extend: bool = False  # Workstream B gate
   multi_item_algorithm: str = "auto"  # Orchestration: "auto" | "packed" | "prefill_extend" | "serial"
   ```

3. **TokenizerManager Interface** (method signatures only):
   ```python
   # Workstream A adds:
   def _score_packed_segment(self, req: ScoreRequest) -> List[float]: ...

   # Workstream B adds:
   def _score_prefill_extend(self, req: ScoreRequest) -> List[float]: ...

   # Orchestration (D) adds:
   def _select_scoring_algorithm(self, req: ScoreRequest) -> ScoringAlgorithm: ...
   ```

4. **ModelWorkerBatch Contract** (new fields):
   ```python
   # Workstream A adds:
   multi_item_mask_mode: int = 0
   multi_item_prefix_ends: np.ndarray | None = None
   multi_item_row_seg_starts: np.ndarray | None = None

   # Workstream B adds:
   scoring_cache_handle: Optional[str] = None
   is_scoring_extend: bool = False
   ```

### Ownership Boundaries

| Surface | Workstream A Owns | Workstream B Owns | Shared (freeze first) |
|---------|-------------------|-------------------|----------------------|
| `server_args.py` | `multi_item_mask_impl` | `multi_item_enable_prefill_extend` | Arg parsing, validation |
| `tokenizer_manager.py` | `_score_packed_segment()` | `_score_prefill_extend()` | `score_request()` dispatch |
| `schedule_batch.py` | `multi_item_mask_mode`, segment fields | `scoring_cache_handle` | `ModelWorkerBatch` base |
| `flashattention_backend.py` | Segment mask logic | None | `FlashAttentionMetadata` |
| `scheduler.py` | None | Cache handle support | None |

### Exit Criteria
- [ ] All enums and types committed to `io_struct.py`
- [ ] Server args placeholders committed (values TBD by implementations)
- [ ] Interface stubs committed (raise NotImplementedError)
- [ ] Both Dev 1 and Dev 2 confirm they can proceed independently

---

## Workstream A: Kernel (Tile-Skipping Segment Mask)

### Objective
Replace dense O(T²) mask with procedural segment-based masking in the Pallas kernel.

### Data Structures

```python
# New fields in ModelWorkerBatch (schedule_batch.py)
@dataclass
class ModelWorkerBatch:
    # ... existing fields ...

    # Multi-item segment mask fields
    multi_item_mask_mode: int = 0  # 0=causal, 1=dense, 2=segment
    multi_item_prefix_ends: np.ndarray | None = None  # shape [bs_padded]
    multi_item_row_seg_starts: np.ndarray | None = None  # shape [token_padded]

# Mirror in FlashAttentionMetadata (flashattention_backend.py)
@dataclass
class FlashAttentionMetadata:
    # ... existing fields ...

    multi_item_mask_mode: int = 0
    multi_item_prefix_ends: jax.Array | None = None
    multi_item_row_seg_starts: jax.Array | None = None
```

### Kernel Mask Rule

```python
def is_visible(q_pos: int, k_pos: int, prefix_end: int, row_seg_start: int) -> bool:
    """
    Determines if query position q can attend to key position k.

    Args:
        q_pos: Query token position
        k_pos: Key token position
        prefix_end: End of shared prefix (first delimiter + 1)
        row_seg_start: Start of current item's segment for this query

    Returns:
        True if attention is allowed, False if blocked
    """
    if q_pos < prefix_end:
        # Query is in prefix: standard causal
        return k_pos <= q_pos
    else:
        # Query is in item: can see prefix OR own segment (causal within)
        return (k_pos < prefix_end) or (row_seg_start <= k_pos <= q_pos)
```

### Server Args

```python
# server_args.py - add to ServerArgs class
multi_item_mask_impl: str = "auto"  # "dense" | "segment" | "auto"
multi_item_segment_fallback_threshold: int = 4096  # Fall back to dense above this
```

### Files to Modify

| File | Changes |
|------|---------|
| `sgl_jax/srt/server_args.py` | Add `multi_item_mask_impl` arg |
| `sgl_jax/srt/managers/schedule_batch.py` | Add segment metadata fields to `ModelWorkerBatch` |
| `sgl_jax/srt/layers/attention/flashattention_backend.py` | Build segment metadata instead of dense mask; add to `FlashAttentionMetadata` |
| `sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` | Add segment mask logic in kernel; skip tiles when segments don't match |
| `sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py` | Add v6e-tuned entries for segment mode |

### Implementation Steps

**⚠️ CRITICAL: Verify tile-skipping speedup early (after step 5)**

TPU/XLA control flow may not skip memory loads even with `if q_seg != k_seg`. Before completing all steps, run a microbenchmark:
- If speedup is ~3x: proceed as planned
- If speedup is only 1.5x or less: escalate, may need to reprioritize B over A

1. **Add server args** for mask implementation selection
2. **Add segment metadata fields** to `ModelWorkerBatch` and `FlashAttentionMetadata`
3. **Build segment metadata** in `get_forward_metadata()`:
   ```python
   def _build_segment_metadata(tokens, delimiter, seq_len):
       delimiter_positions = np.where(tokens == delimiter)[0]
       prefix_end = delimiter_positions[0] + 1 if len(delimiter_positions) > 0 else seq_len

       # Compute segment start for each position
       row_seg_starts = np.zeros(seq_len, dtype=np.int32)
       segment_idx = 0
       for i in range(seq_len):
           if i < prefix_end:
               row_seg_starts[i] = 0  # Prefix positions
           else:
               # Find which segment this position belongs to
               while segment_idx < len(delimiter_positions) - 1 and i >= delimiter_positions[segment_idx + 1]:
                   segment_idx += 1
               row_seg_starts[i] = delimiter_positions[segment_idx] + 1

       return prefix_end, row_seg_starts
   ```
4. **Modify kernel** to use segment metadata instead of dense mask
5. **Add tile-skipping logic**: Early-exit when query segment ≠ key segment
6. **Tune block sizes** for segment mode on v6e
7. **Keep dense fallback** behind `multi_item_mask_impl="dense"`

### Acceptance Criteria

- [ ] **EARLY GATE (after step 5):** Tile-skip microbenchmark shows ≥2x speedup. If <1.5x, escalate.
- [ ] Segment mode produces identical scores to dense mode (max_abs_diff < 1e-6)
- [ ] Memory usage is O(seq) not O(seq²)
- [ ] No OOM at chunk_size=128 (currently fails)
- [ ] Throughput improves by >50% at chunk_size=64
- [ ] Dense fallback works when flag set

---

## Workstream B0: Prefill+Extend Feasibility Spike

### Objective
Validate that Prefill+Extend is viable before committing to full implementation. The core risk is the radix cache constraint: current v0.1 **requires** `--disable-radix-cache` for multi-item scoring.

### Blocking Constraint

From `server_args.py:1122`:
```python
if self.multi_item_scoring_delimiter is not None:
    assert self.disable_radix_cache, (
        "Multi-item scoring requires radix cache to be disabled..."
    )
```

This constraint exists because:
1. Multi-item packed mode constructs a single sequence `Query + D + Item1 + D + Item2 + ...`
2. Radix cache would cache this compound sequence
3. Next request with different items would get wrong cached prefix

**Prefill+Extend takes a different approach:**
1. Prefill query alone → cache just the query KV
2. Each item is a separate extend request reusing query cache
3. This is radix-cache compatible (query prefix is shared, items are distinct)

### Spike Deliverables

1. **Analysis doc**: Document radix cache interaction with scoring
2. **Prototype**:
   - Add `--enable-scoring-cache` flag that enables cache for prefill+extend path
   - Implement minimal `_prefill_and_cache()` + `_single_extend_score()`
   - Run on target workload (2000 query, 500 items)
3. **Decision gate**: Compare spike throughput vs Workstream A segment mode
   - If B0 beats A by >20%: proceed to B1
   - If A beats B0 by >20%: skip B1, A is sufficient
   - If within 20%: implement both for runtime selection

### Exit Criteria
- [ ] Radix cache analysis doc complete
- [ ] Prototype runs without errors
- [ ] **Cache eviction under memory pressure tested** (fill cache with query prefixes, verify eviction works)
- [ ] Throughput comparison vs Workstream A draft
- [ ] Go/no-go decision documented

### Duration: 1 week

---

## Workstream B1: Prefill+Extend Production (Conditional)

**Prerequisite:** B0 spike shows Prefill+Extend is viable and beneficial.

### Objective
Full production implementation of prefill+extend scoring algorithm.

### API Design

```python
# New internal API in tokenizer_manager.py

async def score_prefill_extend(
    self,
    query_tokens: List[int],
    item_tokens_list: List[List[int]],
    label_token_ids: List[int],
    extend_batch_size: int = 32,
) -> List[float]:
    """
    Score items using prefill+extend strategy.

    1. Prefill query once, cache KV
    2. Batch items into extend requests
    3. Each extend reuses cached query KV
    4. Extract scores from extend outputs

    Best for: long queries with many short items
    """
    # Step 1: Prefill query and get cache handle
    cache_handle = await self._prefill_and_cache(query_tokens)

    try:
        # Step 2: Process items in batches
        all_scores = []
        for batch in chunk(item_tokens_list, extend_batch_size):
            batch_scores = await self._batched_extend_score(
                cache_handle=cache_handle,
                items=batch,
                label_token_ids=label_token_ids,
            )
            all_scores.extend(batch_scores)

        return all_scores
    finally:
        # Step 3: Release cache
        await self._release_cache(cache_handle)
```

### Server Args

```python
# server_args.py - add to ServerArgs class
multi_item_enable_prefill_extend: bool = True
multi_item_extend_batch_size: int = 32
multi_item_prefill_extend_cache_timeout: float = 60.0  # seconds
```

### Files to Modify

| File | Changes |
|------|---------|
| `sgl_jax/srt/server_args.py` | Add prefill+extend args |
| `sgl_jax/srt/managers/tokenizer_manager.py` | Add `score_prefill_extend()`, `_prefill_and_cache()`, `_batched_extend_score()` |
| `sgl_jax/srt/managers/scheduler.py` | Support cache handle for scoring; add extend-with-cache path |
| `sgl_jax/srt/managers/schedule_batch.py` | Add scoring cache metadata |
| `sgl_jax/srt/managers/io_struct.py` | Add internal request types for prefill-cache and extend-score |

### Implementation Steps

1. **Add server args** for prefill+extend configuration
2. **Re-enable radix cache for scoring** (currently disabled via `--disable-radix-cache`)
   - Add `--enable-scoring-cache` flag that allows cache for scoring requests only
   - Keep cache disabled for multi-item packed mode (incompatible)
3. **Implement `_prefill_and_cache()`**:
   ```python
   async def _prefill_and_cache(self, query_tokens: List[int]) -> CacheHandle:
       """Prefill query and return handle to cached KV."""
       req = GenerateReqInput(
           input_ids=query_tokens,
           max_new_tokens=0,  # Prefill only
           return_logprob=False,
           cache_for_scoring=True,  # New flag
       )
       result = await self._send_request(req)
       return result.cache_handle
   ```
4. **Implement `_batched_extend_score()`**:
   ```python
   async def _batched_extend_score(
       self,
       cache_handle: CacheHandle,
       items: List[List[int]],
       label_token_ids: List[int],
   ) -> List[float]:
       """Score items by extending from cached prefix."""
       requests = [
           GenerateReqInput(
               input_ids=item_tokens,
               max_new_tokens=0,
               return_logprob=True,
               logprob_start_len=0,
               token_ids_logprob=label_token_ids,
               extend_from_cache=cache_handle,  # New field
           )
           for item_tokens in items
       ]
       # Submit as batch
       results = await self._send_batch(requests)
       return [self._extract_score(r) for r in results]
   ```
5. **Add cache management** for scoring (timeout, cleanup)
6. **Integrate with scheduler** to handle extend-from-cache requests

### Acceptance Criteria

- [ ] Prefill+extend produces identical scores to packed mode (max_abs_diff < 1e-5)
- [ ] Cache is properly released after scoring completes
- [ ] Timeout handling works (cache evicted after timeout)
- [ ] Batched extends run in parallel
- [ ] Memory usage stable under sustained load

---

## Workstream C: Startup (Precompile + Warmup)

### Objective
Eliminate first-request JIT penalty by precompiling kernels for common token bucket shapes.

### Server Args

```python
# server_args.py - add to ServerArgs class
multi_item_enable_startup_warmup: bool = True
multi_item_precompile_buckets: str = "2048,4096,8192"  # Token counts
multi_item_warmup_chunk_sizes: str = "32,64,128"  # Chunk sizes to warm
```

### Files to Modify

| File | Changes |
|------|---------|
| `sgl_jax/srt/server_args.py` | Add warmup configuration args |
| `sgl_jax/srt/managers/tp_worker.py` | Add `warmup_multi_item_kernels()` |
| `sgl_jax/srt/entrypoints/http_server.py` | Call warmup during startup |

### Implementation Steps

1. **Add server args** for warmup configuration
2. **Implement warmup function**:
   ```python
   def warmup_multi_item_kernels(
       model,
       token_buckets: List[int],
       chunk_sizes: List[int],
       mask_modes: List[int] = [1, 2],  # dense and segment
   ):
       """Precompile kernels for common multi-item shapes."""
       logger.info(f"Warming up multi-item kernels for buckets={token_buckets}, chunks={chunk_sizes}")

       for bucket in token_buckets:
           for chunk_size in chunk_sizes:
               for mask_mode in mask_modes:
                   # Build dummy request matching bucket shape
                   query_len = 500  # Typical
                   item_len = (bucket - query_len) // chunk_size

                   dummy_tokens = build_dummy_multi_item_tokens(
                       query_len=query_len,
                       num_items=chunk_size,
                       item_len=item_len,
                   )

                   # Trigger JIT compilation
                   _ = model.forward_extend(
                       input_ids=dummy_tokens,
                       multi_item_mask_mode=mask_mode,
                   )

       logger.info("Multi-item kernel warmup complete")
   ```
3. **Call warmup at startup** after model load, before accepting requests
4. **Add warmup timing metrics** for observability

### Acceptance Criteria

- [ ] First multi-item request latency < 5s (vs 40-50s without warmup)
- [ ] Warmup completes in < 120s for default bucket/chunk combinations
- [ ] Warmup can be disabled via flag for faster dev iteration
- [ ] Warmup logs timing for each bucket/chunk combination

---

## Workstream D: Orchestration (Runtime Policy Selector)

### Objective
Automatically select optimal algorithm (packed vs prefill+extend) based on workload geometry.

### Algorithm Selection Logic

```python
# tokenizer_manager.py

class ScoringAlgorithm(Enum):
    PACKED_DENSE = "packed_dense"       # Legacy dense mask
    PACKED_SEGMENT = "packed_segment"   # Tile-skipping segment mask
    PREFILL_EXTEND = "prefill_extend"   # Query cache + batched extends
    SERIAL = "serial"                   # Fallback: one item at a time

def select_scoring_algorithm(
    query_len: int,
    num_items: int,
    avg_item_len: int,
    memory_headroom_mb: float,
    server_args: ServerArgs,
) -> ScoringAlgorithm:
    """
    Select optimal scoring algorithm based on workload geometry.

    Heuristics:
    - Prefill+Extend wins when query >> items (amortize query cost)
    - Packed+Segment wins when query ≈ items (single forward pass efficient)
    - Fall back to dense/serial when segment mode unavailable or OOM risk
    """
    # Check if algorithms are available
    segment_available = server_args.multi_item_mask_impl in ("segment", "auto")
    prefill_extend_available = server_args.multi_item_enable_prefill_extend

    # Estimate memory for packed approach
    # IMPORTANT: Primary OOM cause is attention intermediates (softmax, matmul buffers),
    # NOT the mask metadata. Use empirical safety caps from v0.1 OOM observations.
    packed_seq_len = query_len + num_items * (avg_item_len + 1)

    # Empirical OOM boundaries from v0.1 testing on TPU v6e-1 (Qwen3-0.6B):
    #
    # v0.1 benchmark context (short queries, ~100 tokens):
    #   - chunk_size=64  → seq ~1500 tokens → OK
    #   - chunk_size=128 → seq ~2700 tokens → OOM
    #
    # For target workload (query=2000, item_tokens=20):
    #   - chunk_size=64  → seq = 2000 + 64×21 = 3344 tokens → likely OOM!
    #   - chunk_size=32  → seq = 2000 + 32×21 = 2672 tokens → borderline
    #
    # OOM threshold depends on model size, batch size, and attention intermediates.
    # Effective memory limit is lower than raw HBM due to model weights,
    # KV cache, and activation memory. Don't rely on theoretical calculations.
    #
    # EMPIRICALLY MEASURED on TPU v6e-1 with Qwen3-0.6B:
    #   - seq=2700: OOM
    #   - seq=2000: OK
    # This is the measured boundary, not a theoretical calculation.
    EMPIRICAL_SEQ_LIMIT_V6E = 2500  # Conservative: 2000 measured OK, 2700 OOM

    packed_segment_memory_mb = packed_seq_len * 8 / 1e6  # Segment metadata (small)
    packed_dense_memory_mb = (packed_seq_len ** 2) * 4 / 1e6  # Dense mask

    # Attention intermediate estimate: ~4 bytes * seq² * num_heads (rough)
    # This is the actual OOM driver, not the mask itself
    packed_attention_intermediates_mb = (packed_seq_len ** 2) * 32 * 4 / 1e6

    # Heuristic: Prefill+Extend wins when query dominates
    query_dominates = query_len > 4 * avg_item_len
    many_items = num_items > 32

    # OOM risk check using empirical boundary (not just mask size)
    oom_risky = packed_seq_len > EMPIRICAL_SEQ_LIMIT_V6E

    # Decision tree
    # 1. If OOM-risky for packed, must use prefill+extend or serial
    if oom_risky:
        if prefill_extend_available:
            return ScoringAlgorithm.PREFILL_EXTEND
        else:
            return ScoringAlgorithm.SERIAL

    # 2. If prefill+extend available and workload suits it
    if prefill_extend_available and query_dominates and many_items:
        return ScoringAlgorithm.PREFILL_EXTEND

    # 3. Segment mode if available
    if segment_available:
        return ScoringAlgorithm.PACKED_SEGMENT

    # 4. Dense mode if within memory
    if packed_dense_memory_mb < memory_headroom_mb:
        return ScoringAlgorithm.PACKED_DENSE

    # Fallback to serial
    return ScoringAlgorithm.SERIAL
```

### Server Args

```python
# server_args.py - add to ServerArgs class
multi_item_algorithm: str = "auto"  # "auto" | "packed" | "prefill_extend" | "serial"
multi_item_algorithm_query_threshold: int = 4  # query_len > N * item_len → prefill_extend
multi_item_algorithm_items_threshold: int = 32  # num_items > N → consider prefill_extend
```

### Auto-Chunk Selection

```python
# Empirically measured seq limits per TPU type (with Qwen3-0.6B)
# These are MEASURED boundaries, not theoretical calculations.
# OOM is driven by attention intermediates, not mask size.
EMPIRICAL_SEQ_LIMITS = {
    "v6e-1": 2500,   # Measured: 2000 OK, 2700 OOM
    "v6e-4": 4000,   # Estimate: scale with HBM
    "v6e-8": 6000,   # Estimate: scale with HBM
    "default": 2000, # Conservative fallback
}

def select_chunk_size(
    query_len: int,
    num_items: int,
    avg_item_len: int,
    chunk_candidates: List[int],
    tpu_type: str = "v6e-1",
) -> int:
    """
    Select largest safe chunk size using EMPIRICAL seq limits.

    NOTE: We use measured OOM boundaries, not mask-size calculations,
    because OOM is driven by attention intermediates (O(seq²) activations),
    not the mask itself.
    """
    max_seq = EMPIRICAL_SEQ_LIMITS.get(tpu_type, EMPIRICAL_SEQ_LIMITS["default"])

    for chunk in sorted(chunk_candidates, reverse=True):
        seq_len = query_len + chunk * (avg_item_len + 1)

        if seq_len < max_seq * 0.9:  # 10% safety margin
            return chunk

    return chunk_candidates[-1]  # Smallest fallback
```

### Files to Modify

| File | Changes |
|------|---------|
| `sgl_jax/srt/server_args.py` | Add algorithm selection args |
| `sgl_jax/srt/managers/tokenizer_manager.py` | Add `select_scoring_algorithm()`, `select_chunk_size()`, route to appropriate impl |

### Implementation Steps

1. **Add server args** for algorithm and chunk selection
2. **Implement selection functions** with heuristics
3. **Route requests** in `score_request()`:
   ```python
   async def score_request(self, query, items, ...):
       algorithm = select_scoring_algorithm(
           query_len=len(query_tokens),
           num_items=len(items),
           avg_item_len=avg([len(i) for i in item_tokens]),
           memory_headroom_mb=self.memory_headroom_mb,
           server_args=self.server_args,
       )

       logger.info(f"Selected algorithm={algorithm.value} for query_len={len(query_tokens)}, items={len(items)}")

       if algorithm == ScoringAlgorithm.PREFILL_EXTEND:
           return await self.score_prefill_extend(query_tokens, item_tokens, ...)
       elif algorithm in (ScoringAlgorithm.PACKED_SEGMENT, ScoringAlgorithm.PACKED_DENSE):
           return await self.score_packed_multi_item(query_tokens, item_tokens, ..., mask_mode=algorithm)
       else:
           return await self.score_serial(query_tokens, item_tokens, ...)
   ```
4. **Add metrics/logging** for algorithm selection observability

### Acceptance Criteria

- [ ] Auto-selection picks prefill+extend for 2000-query/500×20-items workload
- [ ] Auto-selection picks packed+segment for 100-query/10×100-items workload
- [ ] Manual override via `--multi-item-algorithm` works
- [ ] Selection logged for every request
- [ ] Chunk auto-selection picks largest safe chunk

---

## Workstream E: Validation (Testing + Benchmarking)

### Objective
Comprehensive testing and benchmarking to validate v1.0 meets targets.

### Test Categories

#### 1. Unit Tests (per workstream)

| Test File | Coverage |
|-----------|----------|
| `test_multi_item_segment_mask.py` | Segment metadata construction, mask equivalence |
| `test_multi_item_prefill_extend.py` | Cache lifecycle, batched extends, score extraction |
| `test_multi_item_algorithm_selection.py` | Selection heuristics, edge cases |
| `test_multi_item_warmup.py` | Warmup timing, bucket coverage |

#### 2. Integration Tests

| Test | Purpose |
|------|---------|
| `test_multi_item_regression.py` | Isolation invariants (changing item N doesn't affect M) |
| `test_multi_item_parity.py` | Packed vs Prefill+Extend score equivalence |
| `test_multi_item_fallback.py` | Graceful degradation when segment/cache unavailable |

#### 3. Benchmark Tests

| Test | Purpose |
|------|---------|
| `test_bench_multi_item_score.py` | Throughput by chunk size and algorithm |
| `test_bench_cold_start.py` | First-request latency with/without warmup |
| `test_bench_memory.py` | Memory usage by sequence length |

#### 4. Cross-Backend Comparison

Run existing JAX vs PyTorch comparison harness:
- `runbooks/running-jax-vs-pytorch-multi-item-comparison.md`
- Fill in `reports/jax-vs-pytorch-multi-item-comparison-2026-02-11.md`

### Files to Create/Modify

| File | Purpose |
|------|---------|
| `test/srt/test_multi_item_segment_mask.py` | New: Segment mask unit tests |
| `test/srt/test_multi_item_prefill_extend.py` | New: Prefill+extend unit tests |
| `test/srt/test_multi_item_algorithm_selection.py` | New: Selection logic tests |
| `test/srt/test_multi_item_regression.py` | Update: Add segment mode cases |
| `test/srt/test_bench_multi_item_score.py` | Update: Add algorithm comparison |

### Acceptance Criteria (v1.0 Release Gates)

| Gate | Threshold | Status |
|------|-----------|--------|
| Correctness: max_abs_diff | ≤ 0.02 | ⬜ |
| Correctness: mean_abs_diff | ≤ 0.01 | ⬜ |
| Isolation: same-length mutation | 0.0 | ⬜ |
| Isolation: changed-length mutation | 0.0 | ⬜ |
| Stability: success rate | 100% | ⬜ |
| Throughput: vs PyTorch (best-native) | ≥ 100% | ⬜ |
| First-request latency (with warmup) | < 5s | ⬜ |
| Memory: no OOM at chunk=128 | Pass | ⬜ |

---

## Integration Points

### Workstream Dependencies

```
     ┌─────────┐     ┌─────────┐
     │    A    │     │    B    │
     │ Kernel  │     │ Prefill │
     │         │     │ Extend  │
     └────┬────┘     └────┬────┘
          │               │
          │   ┌───────┐   │
          └──►│   C   │◄──┘
              │Startup│
              └───┬───┘
                  │
              ┌───▼───┐
              │   D   │
              │Orchest│
              └───┬───┘
                  │
              ┌───▼───┐
              │   E   │
              │Valid. │
              └───────┘
```

### Merge Order

**Dependency chain:** `0 → A+B0 (parallel) → B1 (if B0 succeeds) → C+D → E`

1. **Merge 0** (interface freeze) - Define shared contracts before parallel work
2. **Merge A** (kernel) - Can be tested independently with `multi_item_mask_impl="segment"`
3. **Merge B0** (prefill+extend spike) - Feasibility validation, compare throughput vs A
4. **Merge B1** (prefill+extend prod) - Only if B0 shows benefit; requires A for comparison
5. **Merge C** (startup) - Requires A for segment warmup
6. **Merge D** (orchestration) - Requires A and B1 (or just A if B1 skipped)
7. **Merge E** (validation) - Final integration tests

### Interface Contracts Between Workstreams

#### A → D (Kernel → Orchestration)

```python
# Kernel exposes mask mode enum
class MultiItemMaskMode(Enum):
    CAUSAL = 0
    DENSE = 1
    SEGMENT = 2

# Orchestration calls kernel with mode
def score_packed_multi_item(..., mask_mode: MultiItemMaskMode):
    ...
```

#### B1 → D (Prefill+Extend → Orchestration)

```python
# Prefill+Extend exposes scoring function
async def score_prefill_extend(
    query_tokens: List[int],
    item_tokens_list: List[List[int]],
    ...
) -> List[float]:
    ...

# Orchestration routes to it
if algorithm == ScoringAlgorithm.PREFILL_EXTEND:
    return await self.score_prefill_extend(...)
```

#### A, B → C (Kernel, Prefill+Extend → Startup)

```python
# Startup warms both paths
def warmup_multi_item_kernels(...):
    # Warm segment kernel
    for bucket in token_buckets:
        warm_packed_segment(bucket)

    # Warm prefill+extend path
    for query_len in query_lens:
        warm_prefill_extend(query_len)
```

---

## Rollout Plan

### Stage 1: Flag-Gated (Week 1 after merge)

```bash
# All new features behind flags, defaults unchanged
--multi-item-mask-impl=dense        # Keep dense default
--multi-item-algorithm=packed       # Keep packed default
--multi-item-enable-startup-warmup=false
```

### Stage 2: Matrix Testing (Week 2)

Run full comparison matrix:
```bash
# Test all combinations
for mask_impl in dense segment; do
  for algorithm in packed prefill_extend auto; do
    run_benchmark --multi-item-mask-impl=$mask_impl --multi-item-algorithm=$algorithm
  done
done
```

### Stage 3: Flip Defaults (Week 3)

After 3 stable runs with identical winner selection:
```bash
# Promote measured winners
--multi-item-mask-impl=auto         # Segment when beneficial
--multi-item-algorithm=auto         # Auto-select by geometry
--multi-item-enable-startup-warmup=true
```

### Stage 4: Monitor (Week 4+)

- Track algorithm selection distribution in production
- Monitor for regressions
- Extend tuning to other TPU topologies

---

## File Summary (All Workstreams)

| File | Workstream | Changes |
|------|------------|---------|
| `sgl_jax/srt/server_args.py` | 0, A, B, C, D | All new server args |
| `sgl_jax/srt/managers/io_struct.py` | 0, B | Enums, types, request structs |
| `sgl_jax/srt/managers/schedule_batch.py` | A, B | Segment metadata fields, cache handle |
| `sgl_jax/srt/layers/attention/flashattention_backend.py` | A | Segment mask builder |
| `sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py` | A | Segment mask logic in kernel |
| `sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py` | A | v6e tuned entries |
| `sgl_jax/srt/managers/tokenizer_manager.py` | 0, B0, B1, D | Interface stubs, prefill+extend, selection |
| `sgl_jax/srt/managers/scheduler.py` | B0, B1 | Cache-for-scoring support |
| `sgl_jax/srt/managers/tp_worker.py` | C | Warmup function |
| `sgl_jax/srt/entrypoints/http_server.py` | C | Startup warmup call |
| `test/srt/test_multi_item_*.py` | E | Test files |

---

## Appendix: Workload Analysis

### Target Workload

```
Query: 2000 tokens
Items: 500 × 20 tokens each
Total: 2000 + 500 × 21 = 12,500 tokens (with delimiters)
```

### Complexity Comparison

| Algorithm | Attention FLOPs | Memory | Notes |
|-----------|-----------------|--------|-------|
| Packed + Dense Mask | O(12500²) ≈ 156M | O(12500²) = 625MB | Computes all tiles, masks out cross-item |
| Packed + Segment (tile-skip) | O(12500²) → **~50M with skip** | O(12500) = 100KB metadata | Skips tiles where segment_id[q] ≠ segment_id[k], but still O(T²) shape |
| Prefill+Extend | O(2000²) + O(500×20×2020) ≈ 24M | O(2000×head_dim) per layer | True asymptotic win via KV cache |

**Important:** Tile-skipping reduces *executed* FLOPs by skipping entire tiles, but the kernel still iterates O(T²) tiles. Actual speedup depends on tile size and sparsity pattern. Empirical: expect 2-3x improvement, not 30x.

**True compute reduction** requires Prefill+Extend which changes the algorithm fundamentally.

**Winner depends on workload geometry:**
- **Packed+Segment:** Best when query is small relative to total, or items are few
- **Prefill+Extend:** Best when query >> items (amortizes O(query²) across many items)

**That's why we need both + runtime selection.**

---

## Appendix: Developer Assignment Template

### Tech Lead: Workstream 0 (Interface Freeze)

```
Focus: Freeze shared interfaces before parallel work
Files: io_struct.py (enums), server_args.py (placeholders), tokenizer_manager.py (stubs)
Tests: None (compile check only)
Exit criteria: All interfaces defined, A and B can proceed independently
Duration: 2 days
```

### Dev 1: Workstream A (Kernel)

```
Focus: Tile-skipping segment mask implementation
Files: server_args.py, schedule_batch.py, flashattention_backend.py,
       ragged_paged_attention.py, tuned_block_sizes.py
Tests: test_multi_item_segment_mask.py
Exit criteria: Segment mode works, no OOM at chunk=128, >50% throughput gain
Dependencies: Workstream 0 complete
```

### Dev 2: Workstream B0 (Prefill+Extend Spike)

```
Focus: Feasibility spike - validate radix cache compatibility
Files: server_args.py (--enable-scoring-cache), tokenizer_manager.py (prototype)
Deliverable: Analysis doc + minimal prototype + go/no-go decision
Exit criteria: Throughput comparison vs A, documented decision
Duration: 1 week
Dependencies: Workstream 0 complete
```

### Dev 2: Workstream B1 (Prefill+Extend Production) [CONDITIONAL]

```
Focus: Full prefill+extend implementation (if B0 shows benefit)
Files: server_args.py, tokenizer_manager.py, scheduler.py, managers/io_struct.py
Tests: test_multi_item_prefill_extend.py
Exit criteria: Prefill+extend works, scores match packed mode, cache lifecycle correct
Dependencies: B0 go decision, Workstream A complete (for comparison)
```

### Dev 3: Workstream C (Startup)

```
Focus: Precompile and warmup
Files: server_args.py, tp_worker.py, entrypoints/http_server.py
Tests: test_multi_item_warmup.py
Exit criteria: First-request < 5s, warmup < 120s
Dependencies: Needs A (segment kernel) to be partially complete
```

### Dev 4: Workstream D (Orchestration)

```
Focus: Runtime algorithm selection
Files: server_args.py, tokenizer_manager.py
Tests: test_multi_item_algorithm_selection.py
Exit criteria: Auto-selection picks optimal algorithm, manual override works
Dependencies: Needs A and B1 complete (or just A if B1 skipped)
```

### Dev 5: Workstream E (Validation)

```
Focus: Testing and benchmarking
Files: test/srt/test_multi_item_*.py, runbooks, reports
Exit criteria: All gates pass, JAX ≥ PyTorch on target workload
Dependencies: Needs A, B1 (if applicable), C, D complete
```
