# RFC-000: Score API Design and Architecture

**Status:** Accepted
**Author:** Engineering Team
**Created:** 2026-01-29
**Updated:** 2026-02-01
**Related:** [ADR-001](../decisions/001-pure-python-softmax.md), [RFC-001](001-score-api-comprehensive-tests.md), [RFC-008](008-multi-item-scoring.md)

## Summary

This document describes the design rationale, architecture, and intended behavior of the `/v1/score` Scoring API in SGLang JAX. It serves as the foundational reference for understanding why the API works the way it does.

## What is the Score API?

The Score API computes probability scores for a set of candidate items given a query context. It's designed for classification, ranking, and selection tasks where you need to evaluate how well different candidates match a given context.

### Core Use Case

**Given:**
- A query/context (e.g., "The capital of France is")
- A list of candidate items (e.g., [" Paris", " London", " Berlin"])
- A list of label token IDs to score

**Returns:**
- Probability scores for each item × label combination

### Example

```python
scores = engine.score(
    query="I pledge allegiance",
    items=[" to the flag", " of the republic"],
    label_token_ids=[311, 315, 369],  # Token IDs for specific labels
    apply_softmax=True
)
# Returns: [[0.85, 0.10, 0.05], [0.30, 0.60, 0.10]]
#           ↑ item 1 scores     ↑ item 2 scores
```

## Design Principles

### 1. Prefill-Only Optimization

The Score API only needs the model's output logits at a specific position—it doesn't need to generate new tokens. This enables a significant optimization:

```python
# Scoring only needs prefill (forward pass)
# NOT decode (autoregressive generation)
sampling_params = {"max_new_tokens": 0}  # Prefill only
```

**Why this matters:**
- Prefill is ~10-100x faster than decode for scoring
- No KV cache updates needed beyond prefill
- No sampling/generation overhead

### 2. Selective Logprob Extraction

Instead of returning logprobs for the entire vocabulary (128K+ tokens), the Score API only extracts logprobs for the specified `label_token_ids`:

```python
# Only extract logprobs for tokens we care about
token_ids_logprob = [311, 315, 369]  # 3 tokens, not 128K
```

**Why this matters:**
- Memory efficient (3 floats vs 128K floats per position)
- Faster extraction and transfer
- Cleaner API for users

### 3. Flexible Concatenation Order

The `item_first` parameter controls whether the model sees `query + item` or `item + query`:

```python
# item_first=False (default): "I pledge allegiance" + " to the flag"
# item_first=True:            " to the flag" + "I pledge allegiance"
```

**Why this matters:**
- Some tasks work better with query-first (e.g., completion scoring)
- Some tasks work better with item-first (e.g., "Is Tokyo a city?")
- Flexibility without multiple API endpoints

### 4. Softmax as Post-Processing

The `apply_softmax` parameter controls whether raw logprobs or normalized probabilities are returned:

```python
# apply_softmax=False: Returns raw logprobs [-2.3, -4.5, -6.7]
# apply_softmax=True:  Returns probabilities [0.85, 0.10, 0.05] (sum to 1.0)
```

**Why this matters:**
- Raw logprobs useful for custom normalization
- Probabilities easier to interpret and use directly
- Softmax done in Python to avoid device conflicts (see [ADR-001](../decisions/001-pure-python-softmax.md))

## Architecture

### Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                           │
│  POST /v1/score                                                 │
│  {query, items, label_token_ids, apply_softmax, item_first}    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OpenAI API Server                           │
│  - Parse request                                                │
│  - Validate parameters (RFC-006)                                │
│  - Route to TokenizerManager                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TokenizerManager                             │
│  (Main Process - Device Agnostic)                               │
│                                                                 │
│  1. Tokenize query and items                                    │
│  2. Concatenate based on item_first                             │
│  3. Build GenerateReqInput with:                                │
│     - max_new_tokens=0 (prefill only)                           │
│     - token_ids_logprob=label_token_ids                         │
│     - return_logprob=True                                       │
│  4. Send to Scheduler via IPC                                   │
│  5. Receive logprobs from Scheduler                             │
│  6. Apply softmax if requested (pure Python)                    │
│  7. Return scores                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                           IPC (pipe)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Scheduler                                 │
│  (Subprocess - Has TPU/GPU Access)                              │
│                                                                 │
│  1. Receive GenerateReqInput                                    │
│  2. Run model forward pass (prefill only)                       │
│  3. Extract logprobs for token_ids_logprob                      │
│  4. Return via IPC                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Split?

The TokenizerManager runs in the main process and must be **device-agnostic**:

1. **Process isolation:** Scheduler subprocess has exclusive TPU/GPU access
2. **No device conflicts:** Main process can't use JAX operations (see [ADR-001](../decisions/001-pure-python-softmax.md))
3. **Clean separation:** Tokenization/formatting separate from inference

This is why softmax is implemented in pure Python, not JAX.

## API Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str \| list[int]` | The context/query to score against |
| `items` | `list[str] \| list[list[int]]` | Candidate items to score |
| `label_token_ids` | `list[int]` | Token IDs to extract logprobs for |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_softmax` | `bool` | `False` | Normalize logprobs to probabilities |
| `item_first` | `bool` | `False` | Concatenate as `item + query` instead of `query + item` |
| `model` | `str` | Server default | Model to use for scoring |

### Input Type Flexibility

The API accepts both text and token inputs:

```python
# Text input (tokenized internally)
query = "The answer is"
items = [" yes", " no"]

# Token input (bypass tokenization)
query = [450, 1234, 338]
items = [[4874], [694]]
```

**Constraint:** Query and items must be the same type (both text or both tokens).

## Response Format

```json
{
    "object": "scoring",
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "scores": [
        [0.85, 0.10, 0.05],
        [0.30, 0.60, 0.10]
    ],
    "usage": {
        "prompt_tokens": 42,
        "completion_tokens": 0,
        "total_tokens": 42
    },
    "created": 1706540400
}
```

### Scores Array Structure

```
scores[i][j] = score for item i, label j

scores = [
    [score_item0_label0, score_item0_label1, ...],
    [score_item1_label0, score_item1_label1, ...],
    ...
]
```

## Design Decisions

### Decision 1: Prefill-Only Mode

**Choice:** Use `max_new_tokens=0` to skip decode phase.

**Rationale:**
- Scoring doesn't need token generation
- Prefill gives us all the logprobs we need
- 10-100x faster than running decode

**Trade-off:** Requires special handling in scheduler to not fail on zero tokens.

### Decision 2: Pure Python Softmax

**Choice:** Implement softmax in Python, not JAX.

**Rationale:**
- TokenizerManager runs in main process
- Main process cannot access TPU/GPU (subprocess has exclusive access)
- JAX operations in main process cause device conflicts

**Trade-off:** Slightly slower than JAX softmax, but negligible for small label sets.

**See:** [ADR-001](../decisions/001-pure-python-softmax.md) for detailed analysis.

### Decision 3: Batch All Items Together

**Choice:** Send all items in a single batch request.

**Rationale:**
- More efficient than individual requests
- Scheduler can optimize batch processing
- Reduces IPC overhead

**Trade-off:** Large batches may hit memory limits.

### Decision 4: Label Token IDs, Not Text

**Choice:** Require token IDs, not text labels.

**Rationale:**
- Unambiguous (text can tokenize to multiple tokens)
- More efficient (no tokenization at scoring time)
- User has control over exact tokens

**Trade-off:** User must know token IDs (can use tokenizer to get them).

## Comparison with PyTorch Version

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Implementation location | `tokenizer_manager.py` | `tokenizer_manager.py` |
| Softmax implementation | JAX (in process) | Pure Python (device-agnostic) |
| Test coverage | 17 tests | 4+ tests (expanding) |
| HTTP endpoint | `/v1/score` | `/v1/score` |
| Request format | Identical | Identical |
| Response format | Identical | Identical |

### Known Differences

1. **Numerical precision:** JAX uses bf16 by default, may have slight differences from PyTorch fp32
2. **Softmax location:** Python vs in-framework (functionally identical)
3. **Error messages:** May differ in wording

## Performance Characteristics

### Factors Affecting Throughput

1. **Batch size:** Larger batches = higher throughput (up to memory limit)
2. **Sequence length:** Longer query+item = slower prefill
3. **Number of labels:** Minimal impact (only affects logprob extraction)
4. **Hardware:** TPU > GPU > CPU

### Typical Performance (TPU v6e)

| Batch Size | Items/Second | Latency (p50) |
|------------|--------------|---------------|
| 1 | ~85 | ~12ms |
| 8 | ~590 | ~14ms |
| 16 | ~900 | ~18ms |
| 32 | ~1200 | ~27ms |

*Note: Actual numbers depend on model, sequence length, and hardware.*

### TPU-Specific Behavior

The Score API in sglang-jax runs on TPU. Users should be aware of these characteristics:

#### 1. First-Request Latency (XLA Compilation)

JAX compiles to XLA on first execution for each unique input shape. This means:

```
Request 1 (seq_len=50):  ~2-5 seconds  (compilation + inference)
Request 2 (seq_len=50):  ~12ms         (cached, inference only)
Request 3 (seq_len=75):  ~2-5 seconds  (new shape, recompile)
Request 4 (seq_len=75):  ~12ms         (cached)
```

**Implications:**
- First request is slow—this is expected, not a bug
- Warmup with representative shapes before serving
- The engine may bucket sequences to reduce unique shapes (check engine docs)

#### 2. bfloat16 Precision

TPU natively uses bfloat16 (bf16), which has less precision than float32:

- **Model inference:** bf16 (TPU-optimized)
- **Softmax computation:** float32 via pure Python (see [ADR-001](../decisions/001-pure-python-softmax.md))
- **Expected variance:** Scores may differ by ~0.01 vs float32 reference

For most scoring tasks, bf16 precision is sufficient. If exact reproducibility with CPU/GPU float32 is required, this is a known limitation.

#### 3. Process Architecture

```
Main Process (CPU)              Subprocess (TPU)
┌─────────────────────┐        ┌─────────────────────┐
│  TokenizerManager   │  IPC   │     Scheduler       │
│  - Tokenization     │◄──────►│  - Model inference  │
│  - Softmax (Python) │        │  - Logprob extract  │
│  - Score formatting │        │  - TPU operations   │
└─────────────────────┘        └─────────────────────┘
```

The Score API code in TokenizerManager is **device-agnostic**—it runs on CPU and communicates with the TPU via IPC. This design:
- Avoids JAX device conflicts in multi-process setup
- Allows softmax to use stable float32 math
- Keeps TPU dedicated to model inference

#### 4. Memory and Batch Limits

TPU HBM (High Bandwidth Memory) limits maximum batch size:

| TPU Type | HBM | Approximate Max Batch* |
|----------|-----|------------------------|
| v6e-1 | 16 GB | ~64-128 items |
| v6e-4 | 64 GB | ~256-512 items |
| v6e-8 | 128 GB | ~512-1024 items |

*Depends on model size and sequence length. These are rough estimates for Llama-3.2-1B.

If you hit OOM errors, reduce batch size or use multi-item scoring ([RFC-008](008-multi-item-scoring.md)) when available.

#### 5. Multi-Device Sharding

For TPU pods (multiple chips), the model is sharded across devices:
- Tensor parallelism splits model weights
- Score API works transparently—no user changes needed
- Results should be identical to single-device (verified by sharding tests)

#### What Score API Does NOT Handle

These are engine-level concerns, not Score API:
- XLA compilation caching strategy
- Sequence length bucketing
- Device placement
- Memory management

If you need to tune these, see the sglang-jax engine documentation.

## Common Use Cases

### 1. Classification

Score candidate class labels:

```python
scores = engine.score(
    query="This movie was absolutely terrible and boring.",
    items=[" positive", " negative", " neutral"],
    label_token_ids=get_token_ids([" positive", " negative", " neutral"]),
    apply_softmax=True
)
# Returns: [[0.02, 0.95, 0.03]]  # Strongly negative
```

### 2. Multiple Choice

Score answer options:

```python
scores = engine.score(
    query="What is 2+2? The answer is",
    items=[" 3", " 4", " 5"],
    label_token_ids=get_token_ids([" 3", " 4", " 5"]),
    apply_softmax=True
)
# Returns: [[0.05, 0.90, 0.05]]  # Answer is 4
```

### 3. Ranking

Rank candidates by score:

```python
candidates = [" Paris", " London", " Berlin", " Tokyo"]
scores = engine.score(
    query="The capital of France is",
    items=candidates,
    label_token_ids=get_token_ids(candidates),
    apply_softmax=True
)
# Sort by score to get ranking
```

### 4. Next Token Prediction

Score likely continuations:

```python
scores = engine.score(
    query="The quick brown fox",
    items=[" jumps", " runs", " sleeps"],
    label_token_ids=get_token_ids([" jumps", " runs", " sleeps"]),
    apply_softmax=True
)
```

## Error Handling

See [RFC-006](006-error-handling-api-contract.md) for complete error handling specification.

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `empty_items` | `items=[]` | Provide at least one item |
| `empty_label_token_ids` | `label_token_ids=[]` | Provide at least one token ID |
| `negative_token_id` | Negative value in labels | Use valid token IDs (≥ 0) |
| `token_id_exceeds_vocab` | Token ID ≥ vocab size | Use valid token IDs (< vocab_size) |
| `mixed_input_types` | Text query with token items | Use consistent types |

## Future Considerations

### Potential Enhancements

1. **Multi-item scoring ([RFC-008](008-multi-item-scoring.md)):** Score N items in single forward pass instead of N passes. Major performance optimization—10-60x speedup for large N. Design complete, implementation pending.
2. **Streaming scores:** Return scores as items are processed (for very large batches)
3. **Prefix caching:** Cache prefill for repeated queries
4. **Multi-label scoring:** Score multiple label sets in one request
5. **Batch queries:** Different queries in same request

### Not Planned

1. **Generation:** Use `/v1/completions` or `/v1/chat/completions`
2. **Embeddings:** Use `/v1/embeddings` (separate endpoint)
3. **Fine-tuning:** Out of scope for inference API

## References

- [ADR-001: Pure Python Softmax Decision](../decisions/001-pure-python-softmax.md)
- [RFC-001: Comprehensive Testing for Score API](001-score-api-comprehensive-tests.md)
- [RFC-003: Comprehensive Score API Test Suite](003-score-api-comprehensive-test-suite.md)
- [RFC-005: OpenAI Client Compatibility](005-openai-client-compatibility.md)
- [RFC-006: Error Handling and API Contract](006-error-handling-api-contract.md)
- [RFC-008: Multi-Item Scoring](008-multi-item-scoring.md) (performance optimization)
- [Investigation: TokenizerManager Architecture](../investigations/tokenizer-manager-architecture.md)
- [Investigation: Score API PyTorch vs JAX Comparison](../investigations/score-api-pytorch-vs-jax.md)
