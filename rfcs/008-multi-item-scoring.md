# RFC-008: Multi-Item Scoring

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-01 |
| **Updated** | 2026-02-01 |
| **Related** | [RFC-000](000-score-api-design.md), [ADR-001](../decisions/001-pure-python-softmax.md) |

## Summary

Add multi-item scoring mode to the JAX Score API, enabling N items to be scored in a single forward pass instead of N separate forward passes. This matches the PyTorch implementation and provides significant performance improvements for batch scoring workloads.

## Motivation

### Current State

The JAX Score API processes each item independently:

```python
# Scoring 3 items requires 3 forward passes
scores = await score_request(
    query="The capital of France is",
    items=[" Paris", " London", " Berlin"],
    label_token_ids=[...],
)

# Internally:
# Forward pass 1: "The capital of France is Paris"
# Forward pass 2: "The capital of France is London"
# Forward pass 3: "The capital of France is Berlin"
```

### The Problem

For N items, we do N forward passes. This is inefficient because:

1. **Redundant computation**: The query is processed N times
2. **Poor batching**: Each forward pass has batch_size=1
3. **High latency**: Total time ≈ N × single_item_latency
4. **Underutilized hardware**: TPU/GPU sits idle between passes

### PyTorch Solution

PyTorch SGLang added multi-item scoring which concatenates all items into a single sequence:

```
query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
```

This enables:
- **1 forward pass** instead of N
- **Query processed once**
- **Better hardware utilization**
- **~Nx throughput improvement** for large N

### Performance Impact (Estimated)

| Items | Current (JAX) | Multi-Item | Speedup |
|-------|---------------|------------|---------|
| 1 | 12ms | 12ms | 1x |
| 10 | 120ms | 15ms | 8x |
| 100 | 1200ms | 50ms | 24x |
| 1000 | 12000ms | 200ms | 60x |

*Estimates based on typical prefill scaling. Actual numbers depend on sequence length and hardware.*

## Proposed Design

### API Changes

**No API changes required.** Multi-item scoring is an internal optimization controlled by server configuration.

```python
# User code remains the same
scores = await score_request(
    query="The capital of France is",
    items=[" Paris", " London", " Berlin"],
    label_token_ids=[...],
)
```

### Server Configuration

New server argument to enable multi-item scoring:

```python
# Enable multi-item scoring with a delimiter token
--multi-item-scoring-delimiter 128009  # Token ID for delimiter
```

When set:
- Score requests use multi-item mode
- `item_first` parameter is ignored (fixed format)
- Delimiter token should be a special token unlikely to appear in content

### Internal Sequence Format

**Text inputs:**
```
{query}{delimiter_text}{item1}{delimiter_text}{item2}{delimiter_text}{item3}{delimiter_text}
```

**Token inputs:**
```
[query_tokens...][delimiter_id][item1_tokens...][delimiter_id][item2_tokens...][delimiter_id][item3_tokens...][delimiter_id]
```

### Logprob Extraction

With multi-item scoring, logprobs are extracted at delimiter positions:

```
Position:  query  <d>  item1  <d>  item2  <d>  item3  <d>
                   ↑          ↑          ↑          ↑
                   |          |          |          └── Score for item3
                   |          |          └── Score for item2
                   |          └── Score for item1
                   └── (skipped - query/item1 boundary)
```

The logprobs at each delimiter position give the model's prediction for what comes next, which is used to score the preceding item.

### Implementation Components

#### 1. GenerateReqInput Changes

```python
batch_request = GenerateReqInput(
    text=[combined_prompt],  # Single sequence
    token_ids_logprob=label_token_ids,
    return_logprob=True,
    logprob_start_len=0,  # Capture logprobs at ALL positions
    stream=False,
    sampling_params={"max_new_tokens": 0},
)
```

Key: `logprob_start_len=0` tells the model to return logprobs starting from position 0, not just the output position.

#### 2. Helper Methods

```python
def _build_multi_item_token_sequence(
    self, query: List[int], items: List[List[int]], delimiter_token_id: int
) -> List[int]:
    """Build combined sequence: query<d>item1<d>item2<d>...<d>"""
    combined = query[:]
    for item in items:
        combined.append(delimiter_token_id)
        combined.extend(item)
    combined.append(delimiter_token_id)  # Final delimiter
    return combined

def _process_multi_item_scoring_results(
    self, results, items, label_token_ids, apply_softmax
) -> List[List[float]]:
    """Extract scores from input_token_ids_logprobs at delimiter positions."""
    input_logprobs = results[0]["meta_info"].get("input_token_ids_logprobs", [])

    scores = []
    # Skip first delimiter (query/item1 boundary)
    for i, item in enumerate(items):
        logprobs_at_delimiter = input_logprobs[i + 1]
        score_list = self._convert_logprobs_to_scores(
            logprobs_at_delimiter, label_token_ids, apply_softmax
        )
        scores.append(score_list)

    return scores
```

#### 3. Main score_request Changes

```python
async def score_request(self, query, items, label_token_ids, ...):
    use_multi_item = (
        self.server_args.multi_item_scoring_delimiter is not None
        and self.multi_item_delimiter_text is not None
    )

    if use_multi_item:
        # Build single combined sequence
        # Extract logprobs at delimiter positions
        return self._process_multi_item_scoring_results(...)
    else:
        # Current behavior: separate forward pass per item
        return self._process_single_item_scoring_results(...)
```

### Softmax Consideration

Per ADR-001, softmax must be pure Python in TokenizerManager (device-agnostic). This applies to multi-item scoring as well:

```python
def _convert_logprobs_to_scores(self, logprobs, label_token_ids, apply_softmax):
    score_list = [logprobs.get(tid, float("-inf")) for tid in label_token_ids]

    if apply_softmax:
        # Pure Python softmax (not JAX)
        max_logprob = max(score_list)
        exp_scores = [math.exp(x - max_logprob) if x != float("-inf") else 0.0
                      for x in score_list]
        sum_exp = sum(exp_scores)
        score_list = [x / sum_exp if sum_exp > 0 else 0.0 for x in exp_scores]
    else:
        score_list = [math.exp(x) if x != float("-inf") else 0.0 for x in score_list]

    return score_list
```

## Design Decisions

### Decision 1: Server-Level Configuration

**Choice:** Enable multi-item scoring via server arg, not per-request.

**Rationale:**
- Simpler API (no new request parameters)
- Consistent behavior across requests
- Delimiter token is model-specific, not request-specific

**Trade-off:** Can't mix single-item and multi-item in same server instance.

### Decision 2: Ignore `item_first` in Multi-Item Mode

**Choice:** Multi-item mode always uses `query<d>items...` format.

**Rationale:**
- `item_first=True` would require `item1<d>item2<d>...<d>query<d>` which is awkward
- Simpler implementation
- Matches PyTorch behavior

**Trade-off:** Users needing `item_first` must use single-item mode.

### Decision 3: Use `input_token_ids_logprobs`

**Choice:** Extract scores from `input_token_ids_logprobs`, not `output_token_ids_logprobs`.

**Rationale:**
- Multi-item scoring needs logprobs at interior positions (delimiters)
- `output_token_ids_logprobs` only has the final position
- `input_token_ids_logprobs` with `logprob_start_len=0` captures all positions

**Trade-off:** Requires scheduler support for `logprob_start_len`.

### Decision 4: Final Delimiter Required

**Choice:** Sequence ends with delimiter: `query<d>item1<d>item2<d>`

**Rationale:**
- Logprob at final delimiter scores the last item
- Without it, last item has no scoring position
- Consistent format

**Trade-off:** One extra token per sequence.

## Compatibility

### Backward Compatibility

- **API:** No changes - existing requests work unchanged
- **Default behavior:** Without `--multi-item-scoring-delimiter`, behavior is identical to current
- **Response format:** Unchanged

### PyTorch Parity

This design matches PyTorch SGLang's implementation:
- Same server arg name
- Same sequence format
- Same logprob extraction logic
- Same `item_first` handling

## Implementation Plan

### Phase 1: Infrastructure

- [ ] Add `multi_item_scoring_delimiter` to server args
- [ ] Add `multi_item_delimiter_text` initialization
- [ ] Verify `logprob_start_len` works in JAX scheduler

### Phase 2: Core Implementation

- [ ] Add `_build_multi_item_token_sequence()` helper
- [ ] Add `_process_multi_item_scoring_results()` helper
- [ ] Add `_extract_logprobs_for_tokens()` helper
- [ ] Add `_convert_logprobs_to_scores()` helper
- [ ] Modify `score_request()` to support both modes

### Phase 3: Testing

- [ ] Unit tests for helper methods
- [ ] Integration test: multi-item vs single-item equivalence
- [ ] Performance benchmark: multi-item speedup
- [ ] Edge cases: empty items, single item, many items

### Phase 4: Documentation

- [ ] Update RFC-000 with multi-item mode
- [ ] Add usage examples
- [ ] Document delimiter token selection

## Testing Strategy

### Correctness Tests

```python
def test_multi_item_equals_single_item():
    """Multi-item scores should match single-item scores."""
    query = "The answer is"
    items = [" yes", " no", " maybe"]

    # Single-item mode (current)
    single_scores = score_single_item(query, items, label_token_ids)

    # Multi-item mode
    multi_scores = score_multi_item(query, items, label_token_ids)

    # Should match within tolerance
    assert_allclose(single_scores, multi_scores, rtol=1e-5)
```

### Performance Tests

```python
def test_multi_item_speedup():
    """Multi-item should be faster for large N."""
    query = "Context..."
    items = [f" item{i}" for i in range(100)]

    t_single = benchmark(score_single_item, query, items)
    t_multi = benchmark(score_multi_item, query, items)

    # Expect significant speedup
    assert t_multi < t_single * 0.2  # At least 5x faster
```

## Open Questions

1. **Delimiter token selection:** What's the best delimiter token for Llama models? PyTorch uses a configurable token ID - should we provide guidance?

2. **Max items limit:** Should we limit the number of items to prevent OOM on very long sequences?

3. **`logprob_start_len` support:** Does the JAX scheduler already support this parameter? Need to verify.

4. **Prefix caching interaction:** Does multi-item scoring work with prefix caching? The query portion could be cached.

5. **Priority:** Given the performance benefits, should this be prioritized over test expansion?

## Alternatives Considered

### Alternative 1: Batched Single-Item

**Description:** Keep separate sequences but batch them together.

```python
# Batch of 3 sequences
["query + item1", "query + item2", "query + item3"]
```

**Pros:**
- Simpler implementation
- No delimiter needed
- `item_first` works naturally

**Cons:**
- Query still processed N times
- Less efficient than single sequence
- Higher memory (N × query_length)

**Why not chosen:** Multi-item is more efficient and matches PyTorch.

### Alternative 2: Per-Request Multi-Item Toggle

**Description:** Add `use_multi_item: bool` to request.

**Pros:**
- Flexible per-request control
- Can mix modes

**Cons:**
- More complex API
- Delimiter still needs to be server-configured
- PyTorch doesn't do this

**Why not chosen:** Server-level config is simpler and sufficient.

### Alternative 3: Automatic Mode Selection

**Description:** Automatically use multi-item when beneficial.

```python
if len(items) > threshold and delimiter_configured:
    use_multi_item = True
```

**Pros:**
- Best of both worlds
- No user decision needed

**Cons:**
- Unpredictable behavior
- Hard to test/debug
- May surprise users with different results

**Why not chosen:** Explicit configuration is more predictable.

## Success Metrics

- [ ] Multi-item scores match single-item within tolerance
- [ ] 10x+ speedup for 100 items
- [ ] No increase in memory usage per item
- [ ] All existing tests pass
- [ ] PyTorch parity verified

## References

- [RFC-000: Score API Design and Architecture](000-score-api-design.md)
- [ADR-001: Pure Python Softmax Decision](../decisions/001-pure-python-softmax.md)
- PyTorch implementation: `sglang/python/sglang/srt/managers/tokenizer_manager_multiitem_mixin.py`
- [Investigation: Score API PyTorch vs JAX Comparison](../investigations/score-api-pytorch-vs-jax.md) (needs update)
