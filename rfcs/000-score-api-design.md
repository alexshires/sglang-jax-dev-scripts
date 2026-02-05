# RFC-000: Score API Design and Architecture

| | |
|------------|------|
| **Status** | Accepted |
| **Author** | Engineering Team |
| **Created** | 2026-01-29 |
| **Updated** | 2026-02-01 (v3) |
| **Related** | [ADR-001](../decisions/001-pure-python-softmax.md), [RFC-001](001-score-api-comprehensive-tests.md), [RFC-008](008-multi-item-scoring.md) |

**Revision History:**
- v3 (2026-02-01): Clarified items vs labels terminology, consistent use of `get_single_token_id` helper
- v2 (2026-02-01): Clarified scoring semantics, softmax axis, label constraints, fixed example patterns
- v1 (2026-01-29): Initial version

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

## Scoring Semantics

This section precisely defines what the Score API computes.

### What Position Is Scored?

For each item, the API:
1. Concatenates `query + item` (or `item + query` if `item_first=True`)
2. Runs a model forward pass on this sequence
3. Extracts logprobs at the **next-token position** (position after the last token)

```
Sequence:    "I pledge allegiance to the flag"
Positions:    0    1       2       3   4   5    [6] ← next-token position
                                                 ↑
                                          Logprobs extracted here
```

The score answers: **"What is the probability of each label token being the next token after this sequence?"**

### How Do Items and Labels Relate?

- **Items** are context extensions appended to the query (can be empty `[""]`)
- **Labels** are the candidate token IDs we're scoring as potential next tokens
- The API computes a score for **every item × label combination**

**Two common usage modes:**

| Mode | Items | Labels | Use Case |
|------|-------|--------|----------|
| **Candidate scoring** | `[""]` (empty) | Candidate tokens | Classification, multiple choice, ranking |
| **Context comparison** | Different contexts | Fixed target token(s) | Comparing which context leads to target |

**Example: Context comparison mode**
```
query = "I pledge allegiance"
items = [" to the flag", " of the republic"]  # Different context extensions
label_token_ids = [311, 315, 369]  # Tokens to score after each context

For item 0 (" to the flag"):
  - Sequence: "I pledge allegiance to the flag"
  - At next-token position: extract logprobs for tokens 311, 315, 369
  - Result: [logprob_311, logprob_315, logprob_369]

For item 1 (" of the republic"):
  - Sequence: "I pledge allegiance of the republic"
  - At next-token position: extract logprobs for tokens 311, 315, 369
  - Result: [logprob_311, logprob_315, logprob_369]

Final scores array: [
    [score_item0_label0, score_item0_label1, score_item0_label2],
    [score_item1_label0, score_item1_label1, score_item1_label2]
]
```

**Example: Candidate scoring mode** (most common)
```
query = "The capital of France is"
items = [""]  # Empty - no context extension
label_token_ids = [PARIS_ID, LONDON_ID, BERLIN_ID]  # Candidate tokens

Sequence: "The capital of France is"
At next-token position: extract logprobs for Paris, London, Berlin tokens
Result: [[logprob_Paris, logprob_London, logprob_Berlin]]
```

### Label Token ID Constraints

**Each `label_token_id` must be a single token ID.** The API does not support multi-token labels.

```python
# ✓ Correct: Single token IDs
label_token_ids = [311, 315, 369]

# ✗ Incorrect: These are NOT supported
label_token_ids = [[311, 312], [315, 316]]  # Multi-token sequences
```

**Verifying single-token labels:**

```python
def get_single_token_id(tokenizer, text):
    """Get token ID, asserting it's a single token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f"'{text}' tokenizes to {len(ids)} tokens: {ids}"
    return ids[0]

# Safe: These are typically single tokens
A_ID = get_single_token_id(tokenizer, " A")      # Letter with space
YES_ID = get_single_token_id(tokenizer, " yes")  # Common word with space

# Risky: These may be multi-token depending on tokenizer
# " Paris" → might be [" Par", "is"] in some tokenizers
# Always verify!
```

**Common single-token choices:**
- Letters with leading space: `" A"`, `" B"`, `" C"` (good for multiple choice)
- Single digits: `" 0"` through `" 9"`
- Common short words: `" yes"`, `" no"`, `" true"`, `" false"`

If you need to score multi-token labels, you must:
1. Use only the first token of each label, OR
2. Make multiple API calls with different label tokens, OR
3. Use a different scoring approach (e.g., compute full sequence logprobs)

### Common Scoring Patterns

Remember: **scores are always for the next token AFTER the full `query + item` sequence.**

**Pattern 1: Classification** - Score class labels as next token
```python
# "Is this positive or negative?"
query = "This movie was great! Sentiment:"
items = [""]  # Empty item - score what comes after query
label_token_ids = [POS_TOKEN, NEG_TOKEN]  # Single token IDs for " positive", " negative"
# scores[0] = [P(" positive" | query), P(" negative" | query)]
```

**Pattern 2: Multiple Choice** - Score answer options as next token
```python
# Which answer is most likely?
query = "What is 2+2? The answer is"
items = [""]  # Empty item - we want next token after the question
label_token_ids = [TOKEN_3, TOKEN_4, TOKEN_5]  # Single token IDs for " 3", " 4", " 5"
# scores[0] = [P(" 3" | query), P(" 4" | query), P(" 5" | query)]
# Highest score = most likely answer
```

**Pattern 3: Ranking Candidates** - Score candidates as next token
```python
# Which continuation is most likely?
query = "The capital of France is"
items = [""]  # Empty - score candidates as next token
label_token_ids = [PARIS_TOKEN, LONDON_TOKEN, BERLIN_TOKEN]
# scores[0] = [P(" Paris" | query), P(" London" | query), P(" Berlin" | query)]
# Compare scores to rank candidates
```

**Pattern 4: Continuation Quality** - Score how well each continuation leads to a target
```python
# Which context leads to target token most confidently?
query = "The capital of"
items = [" France is", " Germany is", " Japan is"]  # Different continuations
label_token_ids = [PARIS_TOKEN]  # Score probability of " Paris" after each
# scores = [[P(" Paris" | "The capital of France is")],
#           [P(" Paris" | "The capital of Germany is")],
#           [P(" Paris" | "The capital of Japan is")]]
# France context should score highest for Paris
```

**Important:** For Patterns 1-3, use `apply_softmax=False` (raw logprobs) to compare across items. Softmax normalizes per-item, which is only meaningful when comparing labels within a single item.

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

**Softmax Axis:** Softmax is applied **per-item, across the provided labels only**. Each item's scores are normalized independently:

```python
# With 2 items and 3 labels:
scores = [
    [0.85, 0.10, 0.05],  # Item 0: sums to 1.0
    [0.30, 0.60, 0.10]   # Item 1: sums to 1.0 (independent of item 0)
]
```

**Important distinction:**
- `apply_softmax=False`: Returns true model logprobs (log P(token | context) from full vocabulary)
- `apply_softmax=True`: Returns **relative probabilities within the label set**, not true model probabilities. Useful for comparing labels, but the values only sum to 1.0 over your provided labels.

**When to use each:**
- Use `apply_softmax=False` when comparing scores across different items (e.g., ranking)
- Use `apply_softmax=True` when comparing labels within a single item (e.g., classification)

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
│  6. Apply softmax if requested (SciPy)                          │
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

This is why softmax is implemented using SciPy, not JAX.

## API Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str \| list[int]` | The context/query to score against |
| `items` | `list[str] \| list[list[int]]` | Candidate items to score |
| `label_token_ids` | `list[int]` | Single token IDs to extract logprobs for (must be individual tokens, not sequences) |

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
    "created": 1769644800
}
```

### Scores Array Structure

```
scores[i][j] = score for label_token_ids[j] given query + items[i]

scores = [
    [score_item0_label0, score_item0_label1, ...],  # Scores for item 0
    [score_item1_label0, score_item1_label1, ...],  # Scores for item 1
    ...
]
```

**Interpretation depends on `apply_softmax`:**

| `apply_softmax` | `scores[i][j]` represents | Use case |
|-----------------|---------------------------|----------|
| `False` | True model logprob: log P(label[j] \| query + item[i]) | Cross-item comparison, ranking |
| `True` | Relative probability within label set (sums to 1.0 per row) | Within-item label comparison |

**Note:** With `apply_softmax=True`, scores are normalized over your provided labels only, not the full vocabulary. This is useful for "which label is most likely?" but the values are not true probabilities unless your labels cover all possible next tokens.

## Design Decisions

### Decision 1: Prefill-Only Mode

**Choice:** Use `max_new_tokens=0` to skip decode phase.

**Rationale:**
- Scoring doesn't need token generation
- Prefill gives us all the logprobs we need
- 10-100x faster than running decode

**Trade-off:** Requires special handling in scheduler to not fail on zero tokens.

### Decision 2: SciPy Softmax

**Choice:** Implement softmax using SciPy, not JAX.

**Rationale:**
- TokenizerManager runs in main process
- Main process cannot access TPU/GPU (subprocess has exclusive access)
- JAX operations in main process cause device conflicts

**Trade-off:** Uses CPU via SciPy, but negligible for small label sets.

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
| Softmax implementation | Pure Python | SciPy (device-agnostic) |
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

When running sglang-jax on TPU (the primary deployment target), users should be aware of these characteristics. Note that JAX also supports GPU and CPU backends, though TPU is the focus of this section:

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
- **Softmax computation:** float32 via SciPy (see [ADR-001](../decisions/001-pure-python-softmax.md))
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

**Reminder:** Scores are computed for the next token AFTER `query + item`. For candidate scoring (most common), use `items=[""]` and put candidate tokens in `label_token_ids`.

```python
# Helper function (defined once, used in all examples below)
def get_single_token_id(tokenizer, text):
    """Get token ID, asserting it's a single token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f"'{text}' tokenizes to {len(ids)} tokens: {ids}"
    return ids[0]
```

### 1. Classification

Score candidate class labels as the next token:

```python
# Get verified single-token IDs for class labels
POS_ID = get_single_token_id(tokenizer, " positive")
NEG_ID = get_single_token_id(tokenizer, " negative")

scores = engine.score(
    query="This movie was absolutely terrible and boring. Sentiment:",
    items=[""],  # Empty - score next token after query
    label_token_ids=[POS_ID, NEG_ID],
    apply_softmax=True  # Compare within label set
)
# Returns: [[0.05, 0.95]]  # Index 1 (negative) has highest probability
```

### 2. Multiple Choice

Score answer options as the next token:

```python
# Letters are typically single tokens
A_ID = get_single_token_id(tokenizer, " A")
B_ID = get_single_token_id(tokenizer, " B")
C_ID = get_single_token_id(tokenizer, " C")

scores = engine.score(
    query="What is 2+2?\nA) 3\nB) 4\nC) 5\nAnswer:",
    items=[""],  # Empty - score next token after query
    label_token_ids=[A_ID, B_ID, C_ID],
    apply_softmax=True
)
# Returns: [[0.05, 0.90, 0.05]]  # B (answer 4) is most likely
```

### 3. Ranking Candidates

Rank candidates by their likelihood as next token:

```python
# Verify these are single tokens for your tokenizer
PARIS_ID = get_single_token_id(tokenizer, " Paris")
LONDON_ID = get_single_token_id(tokenizer, " London")
BERLIN_ID = get_single_token_id(tokenizer, " Berlin")

scores = engine.score(
    query="The capital of France is",
    items=[""],  # Empty - score next token after query
    label_token_ids=[PARIS_ID, LONDON_ID, BERLIN_ID],
    apply_softmax=False  # Raw logprobs for ranking
)
# Returns: [[-0.5, -3.2, -4.1]]  # Paris has highest logprob
# Rank by scores[0]: Paris > London > Berlin
```

### 4. Comparing Contexts

Score how different contexts lead to a target token:

```python
TARGET_ID = get_single_token_id(tokenizer, " Paris")

scores = engine.score(
    query="The capital of",
    items=[" France is", " Germany is", " Italy is"],  # Different contexts
    label_token_ids=[TARGET_ID],  # Same target for all
    apply_softmax=False  # Raw logprobs for comparison
)
# Returns: [[-0.5], [-5.2], [-4.8]]
# "The capital of France is" → " Paris" has highest probability
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

- [ADR-001: SciPy Softmax Decision](../decisions/001-pure-python-softmax.md)
- [RFC-001: Comprehensive Testing for Score API](001-score-api-comprehensive-tests.md)
- [RFC-003: Comprehensive Score API Test Suite](003-score-api-comprehensive-test-suite.md)
- [RFC-005: OpenAI Client Compatibility](005-openai-client-compatibility.md)
- [RFC-006: Error Handling and API Contract](006-error-handling-api-contract.md)
- [RFC-008: Multi-Item Scoring](008-multi-item-scoring.md) (performance optimization)
- [Investigation: TokenizerManager Architecture](../investigations/tokenizer-manager-architecture.md)
- [Investigation: Score API PyTorch vs JAX Comparison](../investigations/score-api-pytorch-vs-jax.md)
