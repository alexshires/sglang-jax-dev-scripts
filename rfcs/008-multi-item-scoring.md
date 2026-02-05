# RFC-008: Multi-Item Scoring

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-01 |
| **Updated** | 2026-02-05 |
| **Related** | [RFC-000](000-score-api-design.md), [ADR-001](../decisions/001-pure-python-softmax.md) |
| **PyTorch PR** | [sgl-project/sglang#10979](https://github.com/sgl-project/sglang/pull/10979) |

## Summary

Add multi-item scoring mode to the JAX Score API, enabling N items to be scored in a single forward pass instead of N separate forward passes. This matches the PyTorch implementation (PR #10979) and provides significant performance improvements for batch scoring workloads.

**Critical Implementation Note:** Multi-item scoring is NOT simply concatenating items with delimiters. It requires **custom attention masking** to prevent items from attending to each other. Without this, later items would be conditioned on earlier items, producing incorrect scores.

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

### PyTorch Solution (PR #10979)

PyTorch SGLang added multi-item scoring which concatenates all items into a single sequence with **custom attention masking**:

```
query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>
```

With attention boundaries that ensure:
- All tokens can attend to the query
- Each item can only attend to itself (not other items)
- Delimiter tokens mark scoring positions

This enables:
- **1 forward pass** instead of N
- **Query processed once**
- **Better hardware utilization**
- **~Nx throughput improvement** for large N

### Performance Impact (from PR #10979 Description)

The following numbers are cited from the PR description. They have not been independently verified in the codebase and should be treated as indicative benchmarks:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| P99 Latency | 8,276 ms | 511 ms | **16.2x faster** |
| Throughput | 950 items/sec | 1,200 items/sec | **26.3% higher** |

*Configuration: Qwen3-0.6B, 300-token queries, 10 items/request, 120 QPS, H100 GPU.*

*Note: JAX/TPU performance will differ. These numbers are GPU-specific and require independent validation.*

---

## Critical Architecture: Attention Mask / Item Boundary Isolation

> **This section describes the most important implementation detail that makes multi-item scoring work correctly.**

### The Problem Without Attention Masking

If we simply concatenate `query<d>item1<d>item2<d>item3<d>` and run standard causal attention:

```
Attention Pattern (WRONG - standard causal):
                query  <d>  item1  <d>  item2  <d>  item3  <d>
query             ✓
<d>               ✓     ✓
item1             ✓     ✓    ✓
<d>               ✓     ✓    ✓      ✓
item2             ✓     ✓    ✓      ✓    ✓
<d>               ✓     ✓    ✓      ✓    ✓      ✓
item3             ✓     ✓    ✓      ✓    ✓      ✓    ✓
<d>               ✓     ✓    ✓      ✓    ✓      ✓    ✓      ✓
```

**Problem:** `item2` attends to `item1`, and `item3` attends to both `item1` and `item2`. This means:
- Score for `item2` is **conditioned on seeing `item1`**
- Score for `item3` is **conditioned on seeing `item1` AND `item2`**
- Results will NOT match single-item scoring!

### The Solution: Custom Attention Mask

Multi-item scoring requires a custom attention pattern:

```
Attention Pattern (CORRECT - multi-item):
                query  <d>  item1  <d>  item2  <d>  item3  <d>
query             ✓
<d>               ✓     ✓
item1             ✓     ✓    ✓
<d>               ✓     ✓    ✓      ✓
item2             ✓     ✓    ✗      ✗    ✓
<d>               ✓     ✓    ✗      ✗    ✓      ✓
item3             ✓     ✓    ✗      ✗    ✗      ✗    ✓
<d>               ✓     ✓    ✗      ✗    ✗      ✗    ✓      ✓
```

**Key properties:**
- All tokens attend to query (prefix)
- Each item attends only to itself
- Items do NOT attend to other items
- Delimiters mark boundaries

### PyTorch Implementation: FlashInfer Parameters

PR #10979 implements this via FlashInfer's specialized multi-item parameters:

```python
@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring attention computation."""

    # Length of query (prefix) that all tokens can attend to
    prefix_len_ptr: torch.Tensor          # uint32, shape [batch_size]

    # Relative position within each item (resets to 0 at each delimiter)
    # This controls the attention window for each item
    token_pos_in_items_ptr: torch.Tensor  # uint16, shape [total_item_tokens]

    # Padded length for batch processing
    token_pos_in_items_len: int

    # Maximum item length per prompt (for memory allocation)
    max_item_len_ptr: torch.Tensor        # uint16, shape [batch_size]
```

**Example computation:**

```
Text: "What is the capital of France? <d> London <d> Paris <d> Berlin <d>"
Tokens: [What, is, the, capital, of, France, ?, <d>, London, <d>, Paris, <d>, Berlin, <d>]
Index:  [ 0,   1,  2,   3,      4,  5,      6,  7,   8,      9,   10,    11,  12,     13]

prefix_len_ptr: [7]  # Query ends at position 7 (before first delimiter)

token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0]
                         │  │  │  │  │  │  └── final <d> (pos 0 = delimiter)
                         │  │  │  │  │  └── Berlin (pos 1 = first token in item)
                         │  │  │  │  └── <d> after Paris (pos 0 = delimiter)
                         │  │  │  └── Paris (pos 1)
                         │  │  └── <d> after London (pos 0)
                         │  └── London (pos 1)
                         └── first <d> (pos 0)

max_item_len_ptr: [1]  # All items are single tokens
```

The position reset at each delimiter creates the attention boundaries.

### Backend Restriction: FlashInfer Required

> **CRITICAL:** In PyTorch, multi-item attention isolation is implemented **only** in the FlashInfer backend. Other backends (e.g., Triton, native PyTorch) do NOT support the `MultiItemScoringParams` and will silently produce incorrect results.

```python
# FlashInfer backend check (from flashinfer_backend.py)
if self.multi_item_scoring_delimiter is not None:
    # Only FlashInfer supports multi-item params
    # Other backends would compute standard causal attention
    multi_item_params = self._process_multi_item_scoring(forward_batch)
```

**Implications for JAX:**
- JAX must implement an equivalent attention isolation mechanism
- Without it, multi-item mode will give wrong results (items attending to each other)
- This is not a "nice to have" - it's a correctness requirement

### JAX Implementation Requirements

For JAX/TPU, we need an equivalent mechanism. Options to investigate:

1. **Pallas custom attention kernel** with item-boundary-aware masking
2. **JAX attention with explicit mask tensor** (may be memory-intensive)
3. **Segment-based attention** if JAX attention libraries support it
4. **Custom position embeddings** that reset at delimiters

**This is the primary implementation challenge for JAX parity.**

---

## Runtime Constraints

> **Multi-item scoring requires specific server configuration. These are NOT optional.**

### Required Server Flags

When `--multi-item-scoring-delimiter` is set, the following constraints apply:

| Constraint | Requirement | Rationale |
|------------|-------------|-----------|
| Radix Cache | **Must be disabled** (`--disable-radix-cache`) | Cache keys don't account for item boundaries |
| Chunked Prefill | **Must be disabled** (`--chunked-prefill-size -1`) | Chunking could split across item boundaries |
| Sliding Window | **Automatically disabled** (FlashInfer only) | Window could cross item boundaries |
| Ragged Prefill | **Automatically disabled** (FlashInfer only) | Incompatible with multi-item params |
| Attention Backend | **FlashInfer required** (PyTorch) | Only backend implementing multi-item params |

**Note:** Sliding window and ragged prefill are disabled *within FlashInfer* when multi-item params are enabled. Other backends don't support multi-item at all.

### Validation at Startup

```python
# In server initialization
if server_args.multi_item_scoring_delimiter is not None:
    if not server_args.disable_radix_cache:
        raise ValueError(
            "Multi-item scoring requires --disable-radix-cache. "
            "Radix cache is incompatible with item boundary isolation."
        )
    if server_args.chunked_prefill_size != -1:
        raise ValueError(
            "Multi-item scoring requires --chunked-prefill-size -1. "
            "Chunked prefill could split sequences at item boundaries."
        )
```

### Request-Level Constraints

| Constraint | Behavior |
|------------|----------|
| Prefill-only | Multi-item only activates for scoring (no generation) |
| `max_new_tokens` | Must be 0 for multi-item requests |
| `item_first` | Ignored in multi-item mode |
| Speculative decoding | **Incompatible** - see warning below |

> **WARNING: Speculative Decoding Interaction**
>
> When `speculative_algorithm` is set, multi-item scoring is **implicitly disabled** in the scheduler/logits path, but the TokenizerManager may still build the multi-item prompt format. This mismatch can cause:
> - Shape assumption violations in logprob arrays
> - Incorrect scoring results
> - Silent failures
>
> **Recommendation:** Validate at startup that `speculative_algorithm` is not set when `multi_item_scoring_delimiter` is configured, or explicitly document this as unsupported.

### `next_token_logits` Handling

For prefill-only scoring requests, `next_token_logits` can be `None`:

```python
@dataclass
class LogitsProcessorOutput:
    # Can be None for prefill-only requests (e.g., multi-item scoring)
    next_token_logits: Optional[torch.Tensor]
```

This affects worker behavior - workers must handle `None` logits gracefully.

> **JAX Implementation Required Changes:**
>
> The current JAX codebase treats `next_token_logits` as a required tensor:
>
> | File | Issue |
> |------|-------|
> | `logits_processor.py:27-33` | Type annotation assumes non-None |
> | `tp_worker.py:529-571` | Uses logits unconditionally for sampling and debug logging |
>
> **Changes needed for multi-item support:**
> 1. Update `LogitsProcessorOutput` to allow `Optional[Tensor]`
> 2. Add prefill-only detection in `tp_worker.py`
> 3. Gate sampling logic: skip when `next_token_logits is None`
> 4. Gate debug log dumping that accesses logits
> 5. Consider adding `is_prefill_only` flag to `LogitsMetadata` (as PyTorch does)

---

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

```bash
# Enable multi-item scoring with a delimiter token
python -m sglang.launch_server \
    --model meta-llama/Llama-3-8B \
    --multi-item-scoring-delimiter 128009 \
    --disable-radix-cache \
    --chunked-prefill-size -1
```

When set:
- Score requests use multi-item mode (prefill-only)
- `item_first` parameter is ignored (fixed format)
- Delimiter token **MUST NOT appear** in query or item content (see below)

> **CRITICAL: Delimiter Collision Hazard**
>
> The PR treats **any occurrence** of the delimiter token as a boundary when building attention params and slicing logits. If the delimiter token appears inside the query or item text, scoring becomes incorrect:
>
> ```python
> # In logits_processor.py - finds ALL delimiter occurrences
> multi_item_indices = (input_ids == delimiter_token).nonzero(as_tuple=True)[0] - 1
> ```
>
> **This is a HARD REQUIREMENT, not a suggestion:**
> - Choose a delimiter token that cannot appear in user content
> - Recommended: Use model-specific special tokens (e.g., `<|eot_id|>` for Llama-3)
> - Validation must check **tokenized IDs**, not substring matching (BPE may encode delimiter text differently in context)
>
> ```python
> # Correct: Token-ID based validation
> query_ids = tokenizer.encode(query)
> if delimiter_token_id in query_ids:
>     raise ValueError("Delimiter token found in query")
>
> # Wrong: Substring check (misses BPE edge cases)
> if delimiter_text in query:  # Don't do this!
>     ...
> ```

### Internal Sequence Format

**Text inputs:**
```
{query}{delimiter_text}{item1}{delimiter_text}{item2}{delimiter_text}{item3}{delimiter_text}
```

**Delimiter text derivation:** The server decodes the delimiter token ID to get the delimiter text:

```python
# During initialization - MUST use skip_special_tokens=False
self.multi_item_delimiter_text = tokenizer.decode(
    [delimiter_token_id],
    skip_special_tokens=False  # Critical! Otherwise special tokens return empty string
)
```

> **Why `skip_special_tokens=False`?** Many tokenizers return empty string for special tokens when `skip_special_tokens=True` (the default in some APIs). Since delimiters are typically special tokens (e.g., `<|eot_id|>`), omitting this flag breaks delimiter insertion.

**Note:** PyTorch does NOT validate that the delimiter text re-tokenizes to a single token. For robustness, JAX implementation SHOULD add this validation:

```python
# Recommended validation (not in PyTorch)
delimiter_tokens = tokenizer.encode(delimiter_text, add_special_tokens=False)
if len(delimiter_tokens) != 1 or delimiter_tokens[0] != delimiter_token_id:
    logger.warning(
        f"Delimiter text '{delimiter_text}' tokenizes to {delimiter_tokens}, "
        f"expected [{delimiter_token_id}]. This may cause incorrect scoring."
    )
```

**Token inputs:**
```
[query_tokens...][delimiter_id][item1_tokens...][delimiter_id][item2_tokens...][delimiter_id][item3_tokens...][delimiter_id]
```

---

## Logprob Extraction Dataflow

> **IMPORTANT: This section corrects a fundamental misunderstanding in the original RFC.**

### Original (Incorrect) Assumption

The original RFC assumed:
1. Model returns logprobs for ALL positions
2. TokenizerManager computes delimiter indices via cumulative lengths
3. Extract logprobs at those computed indices

### Actual PyTorch Dataflow (Correct)

The PR implements a **different, more efficient dataflow**:

1. **Logits computed only at delimiter positions** (in `LogitsProcessor`)
2. **Scheduler emits only delimiter-position logprobs**
3. **TokenizerManager receives pre-sliced arrays** (no index computation needed)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Forward Pass                              │
│  input_ids: [query..., <d>, item1..., <d>, item2..., <d>]       │
│  hidden_states: [h0, h1, ..., h_query, h_d1, h_item1, h_d2, ...]│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LogitsProcessor (compute_logprobs_for_multi_item)  │
│                                                                  │
│  1. Find delimiter positions: [d1_pos, d2_pos, d3_pos, ...]     │
│  2. Slice hidden states at (delimiter_pos - 1):                 │
│     sliced_hidden = hidden_states[delimiter_indices - 1]        │
│  3. Compute logits only for sliced positions                    │
│  4. Return logprobs array of length num_delimiters              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scheduler Output                              │
│                                                                  │
│  input_token_ids_logprobs: [logprobs_d1, logprobs_d2, ...]     │
│  Length: num_items + 1 (includes query/item1 boundary)          │
│  First entry (query/item1 boundary) is typically skipped        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TokenizerManager                              │
│                                                                  │
│  Receives pre-sliced logprobs - NO index computation needed     │
│  Simply iterate: scores[i] = process(logprobs[i+1])             │
│  (Skip index 0 which is query/item1 boundary)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: Hidden State Slicing

The scoring position is **one token before** the delimiter (the last token of the item):

```
Position:  query  ...  item1_last  <d>  item2_last  <d>  item3_last  <d>
                           ↑             ↑               ↑
                           │             │               └── Extract logprobs here
                           │             └── Extract logprobs here
                           └── Extract logprobs here (scores item1)
```

The logprobs at position `delimiter - 1` represent the model's prediction for what comes next after seeing the item, which is exactly what we need for scoring.

### LogitsProcessor Implementation

```python
def compute_logprobs_for_multi_item_scoring(
    self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
    delimiter_token: int,
) -> LogitsProcessorOutput:
    """Compute logprobs at delimiter positions for multi-item scoring.

    Instead of computing logprobs for all positions, we:
    1. Find delimiter positions in input_ids
    2. Slice hidden_states at (delimiter_pos - 1)
    3. Compute logits only for those positions
    4. Return compact logprob arrays
    """
    # Find positions right before each delimiter
    multi_item_indices = (input_ids == delimiter_token).nonzero(as_tuple=True)[0] - 1

    # Extract hidden states at scoring positions
    sliced_hidden = hidden_states[multi_item_indices]

    # Compute logits only for these positions
    sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)
    sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)

    # Return compact output (only delimiter positions)
    return LogitsProcessorOutput(
        next_token_logits=None,  # Prefill-only, no generation
        input_token_ids_logprobs_val=...,  # Length = num_delimiters
        ...
    )
```

### Metadata Adjustment: `extend_logprob_pruned_lens_cpu`

> **IMPORTANT:** The PR recomputes `extend_logprob_pruned_lens_cpu` based on delimiter counts, not original sequence lengths. This is required for correct indexing of `top_logprobs` and `token_ids_logprobs`.

```python
# From logits_processor.py:1021-1042
# Original: extend_logprob_pruned_lens_cpu contains sequence lengths
# Multi-item: Recompute to contain delimiter counts per request

if logits_metadata.token_ids_logprobs or logits_metadata.extend_return_top_logprob:
    logits_metadata.extend_logprob_pruned_lens_cpu = []

    if logits_metadata.extend_seq_lens_cpu is not None:
        # Multi-request batch: count delimiters per request
        input_pt = 0
        for req_seq_len in logits_metadata.extend_seq_lens_cpu:
            req_input_ids = input_ids[input_pt : input_pt + req_seq_len]
            delimiter_count = (req_input_ids == delimiter_token).sum().item()
            logits_metadata.extend_logprob_pruned_lens_cpu.append(delimiter_count)
            input_pt += req_seq_len
    else:
        # Single request: all delimiters belong to one request
        total_delimiters = (input_ids == delimiter_token).sum().item()
        logits_metadata.extend_logprob_pruned_lens_cpu = [total_delimiters]
```

**JAX Implication:** If JAX exposes `top_logprobs` or `token_ids_logprobs` fields, it must also recompute the pruned lengths based on delimiter counts.

### Output Shape Changes: All Logprob Arrays

The PR changes **multiple** output arrays to contain only delimiter positions, not just `input_token_ids_logprobs`:

| Array | Standard Mode | Multi-Item Mode |
|-------|---------------|-----------------|
| `input_token_logprobs` | Length = seq_len | Length = num_delimiters |
| `input_top_logprobs_val` | Length = seq_len | Length = num_delimiters |
| `input_top_logprobs_idx` | Length = seq_len | Length = num_delimiters |
| `input_token_ids_logprobs_val` | Length = seq_len | Length = num_delimiters |
| `input_token_ids_logprobs_idx` | Length = seq_len | Length = num_delimiters |

This reshaping happens in `scheduler_output_processor_mixin.py:590-683`. Any downstream code (e.g., detokenizer) must be aware of this shape change.

### TokenizerManager Processing (Simplified)

Since logprobs arrive pre-sliced, processing is straightforward:

```python
def _process_multi_item_scoring_results(
    self, results, items, label_token_ids, apply_softmax
) -> List[List[float]]:
    """Process pre-sliced logprobs from scheduler.

    NOTE: Unlike the original RFC, we do NOT compute delimiter indices.
    The scheduler already extracted logprobs at delimiter positions.
    """
    # Logprobs are already at delimiter positions
    # Length = num_items + 1 (first is query/item1 boundary, skip it)
    input_logprobs = results[0]["meta_info"].get("input_token_ids_logprobs", [])

    scores = []
    for i in range(len(items)):
        # Skip index 0 (query/item1 boundary), use index i+1 for item i
        logprobs_for_item = input_logprobs[i + 1]

        score_list = self._convert_logprobs_to_scores(
            logprobs_for_item, label_token_ids, apply_softmax
        )
        scores.append(score_list)

    return scores
```

---

## `apply_softmax` Semantics

> **IMPORTANT: There is a semantic difference between PyTorch PR and this RFC that must be resolved.**

### The Conflict

| `apply_softmax` | PyTorch PR Returns | Original RFC Proposed |
|-----------------|--------------------|-----------------------|
| `True` | Normalized probabilities | Normalized probabilities |
| `False` | **`exp(logprob)`** (probabilities) | **Raw logprobs** |

### PyTorch Behavior (from PR)

```python
# In tokenizer_manager_multiitem_mixin.py
if apply_softmax:
    # Softmax normalization
    scores = softmax(logprobs)
else:
    # Return exp(logprob) - still probabilities, just unnormalized
    scores = [math.exp(lp) for lp in logprobs]
```

### RFC/ADR-001 Expectation

ADR-001 and the original RFC expected:

```python
if apply_softmax:
    scores = softmax(logprobs)  # Probabilities
else:
    scores = logprobs  # Raw log probabilities
```

### Resolution Options

**Option A: Match PyTorch (recommended for parity)**
- `apply_softmax=False` returns `exp(logprob)`
- Pros: Exact PyTorch parity, no surprises when comparing
- Cons: Breaks expectation that "no softmax" means "raw logprobs"

**Option B: Keep RFC behavior (better semantics)**
- `apply_softmax=False` returns raw logprobs
- Pros: Clearer semantics, matches parameter name
- Cons: Different from PyTorch, harder to compare results

**Option C: Add new parameter**
- Add `return_logprobs: bool` separate from `apply_softmax`
- Pros: Explicit control
- Cons: API change, more complexity

### Recommendation

**Use Option A (match PyTorch)** for initial implementation to ensure parity. Document the behavior clearly:

```python
def _convert_logprobs_to_scores(self, logprobs, label_token_ids, apply_softmax):
    """Convert logprobs to scores.

    Args:
        apply_softmax: If True, return normalized probabilities (sum to 1).
                      If False, return unnormalized probabilities (exp of logprobs).

    Note: To get raw logprobs, access input_token_ids_logprobs directly.
    """
    score_list = [logprobs.get(tid, float("-inf")) for tid in label_token_ids]

    if apply_softmax:
        # Normalized probabilities
        max_lp = max(score_list)
        exp_scores = [math.exp(x - max_lp) if x != float("-inf") else 0.0
                      for x in score_list]
        total = sum(exp_scores)
        return [x / total if total > 0 else 0.0 for x in exp_scores]
    else:
        # Unnormalized probabilities (exp of logprobs) - matches PyTorch
        return [math.exp(x) if x != float("-inf") else 0.0 for x in score_list]
```

---

## Design Decisions

### Decision 1: Server-Level Configuration

**Choice:** Enable multi-item scoring via server arg, not per-request.

**Rationale:**
- Simpler API (no new request parameters)
- Consistent behavior across requests
- Delimiter token is model-specific, not request-specific
- Runtime constraints (radix cache, chunked prefill) are server-level

**Trade-off:** Can't mix single-item and multi-item in same server instance.

### Decision 2: Ignore `item_first` in Multi-Item Mode

**Choice:** Multi-item mode always uses `query<d>items...` format.

**Rationale:**
- `item_first=True` would require `item1<d>item2<d>...<d>query<d>` which breaks attention isolation
- Simpler implementation
- Matches PyTorch behavior

**Trade-off:** Users needing `item_first` must use single-item mode.

### Decision 3: Prefill-Only Activation

**Choice:** Multi-item scoring only activates for prefill-only requests (no token generation).

**Rationale:**
- Scoring doesn't need generation
- Simplifies implementation (no decode path needed)
- Allows `next_token_logits` to be `None`
- Matches PyTorch behavior

**Trade-off:** Cannot use multi-item for generate-then-score workflows.

### Decision 4: Delimiter-Only Logprob Computation

**Choice:** Compute logprobs only at delimiter positions, not all positions.

**Rationale:**
- More efficient (fewer logit computations)
- Smaller output tensors
- Simpler TokenizerManager logic (no index computation)
- Matches PyTorch implementation

**Trade-off:** Cannot inspect logprobs at non-delimiter positions.

### Decision 5: Final Delimiter Required

**Choice:** Sequence ends with delimiter: `query<d>item1<d>item2<d>`

**Rationale:**
- Logprob at final delimiter scores the last item
- Without it, last item has no scoring position
- Consistent format

**Trade-off:** One extra token per sequence.

### Decision 6: Disable Incompatible Features

**Choice:** Require disabling radix cache, chunked prefill, and sliding window.

**Rationale:**
- Radix cache: Cache keys don't account for item boundaries
- Chunked prefill: Could split sequences at item boundaries
- Sliding window: Could attend across item boundaries

**Trade-off:** Reduced functionality when multi-item is enabled.

---

## JAX-Specific Considerations

### Attention Mask Implementation

The PyTorch implementation uses FlashInfer's specialized parameters. For JAX/TPU:

**Option 1: Explicit Mask Tensor**
```python
# Create attention mask that blocks cross-item attention
# Shape: [seq_len, seq_len], may be memory-intensive for long sequences
mask = create_multi_item_attention_mask(
    query_len=query_len,
    item_lengths=item_lengths,
    delimiter_positions=delimiter_positions,
)
```

**Option 2: Pallas Custom Kernel**
```python
# Custom Pallas kernel with item-boundary-aware attention
# More efficient but requires kernel development
@jax.named_call
def multi_item_attention_kernel(...):
    # Implement custom attention with position-based masking
    pass
```

**Option 3: Segment IDs (if supported)**
```python
# Some attention implementations support segment-based masking
segment_ids = [0] * query_len + [1] * item1_len + [2] * item2_len + ...
# Each segment only attends to itself + segment 0 (query)
```

### Position Manipulation

PyTorch modifies positions in-place for attention computation:

```python
# In _process_multi_item_scoring
pos[first_delim:] = diff - 1
forward_batch.positions[seq_start:seq_end] = pos
```

JAX equivalent needs to handle immutability:

```python
# JAX: Create new position array (immutable)
new_positions = jnp.where(
    positions >= first_delim,
    compute_relative_positions(positions, delimiter_indices),
    positions
)
```

### Softmax Location

Per ADR-001, softmax must remain in TokenizerManager and be device-agnostic (not JAX):

```python
# CORRECT: Device-agnostic in TokenizerManager
def _convert_logprobs_to_scores(self, ...):
    import math  # Pure Python
    exp_scores = [math.exp(x) for x in logprobs]
    ...

# WRONG: JAX in TokenizerManager (would cause device conflicts)
def _convert_logprobs_to_scores(self, ...):
    import jax.numpy as jnp  # NO! Device conflict
    exp_scores = jnp.exp(logprobs)
```

> **Current JAX Implementation Note:**
>
> The JAX TokenizerManager currently uses `scipy.special.softmax` (see `tokenizer_manager.py:1294-1301`), not pure Python `math`. SciPy is device-agnostic (CPU-only, NumPy-based), so it satisfies ADR-001's intent of avoiding JAX device conflicts.
>
> **Decision needed:** Either:
> - Accept SciPy as compliant with ADR-001 (it's device-agnostic)
> - Update to pure Python for stricter compliance
> - Update ADR-001 to explicitly allow SciPy
>
> For multi-item scoring, either approach works as long as no JAX ops are used in TokenizerManager.

### TPU-Specific Constraints

| Aspect | Consideration |
|--------|---------------|
| Memory | Explicit mask tensor may exceed HBM for very long sequences |
| bf16 | Ensure attention mask computation handles bf16 precision |
| Multi-host | Position manipulation must be consistent across hosts |
| XLA compilation | Mask creation should be JIT-compatible |

---

## Implementation Plan

### Phase 0: Investigation

- [ ] Determine JAX attention mask approach (explicit tensor vs Pallas vs segment IDs)
- [ ] Prototype attention isolation on simple test case
- [ ] Verify JAX scheduler supports necessary logprob extraction
- [ ] Benchmark mask tensor memory usage for target sequence lengths

### Phase 1: Infrastructure

- [ ] Add `multi_item_scoring_delimiter` to server args
- [ ] **Add `multi_item_scoring_delimiter` to `GLOBAL_SERVER_ARGS_KEYS`** (required for LogitsProcessor/attention to see it)
- [ ] Add startup validation for required flags (radix cache, chunked prefill)
- [ ] Add startup validation for incompatible flags (speculative decoding)
- [ ] Add `multi_item_delimiter_text` derivation from token ID (with `skip_special_tokens=False`)
- [ ] Implement delimiter text validation (single-token re-encoding)
- [ ] Add runtime validation: delimiter token not in query/items (**JAX-only guard**, not in PyTorch)
- [ ] Add runtime validation: query must have at least one token (**JAX-only guard**, not in PyTorch)

> **Note on JAX-Only Guards:** PyTorch does not validate delimiter collision or empty query. These are safety improvements for JAX. If strict parity is required, make them opt-in via a flag or remove them.

### Phase 2: Attention Mechanism (Critical Path)

- [ ] Implement multi-item attention mask generation
- [ ] Integrate with JAX attention computation
- [ ] Verify item isolation (items don't attend to each other)
- [ ] Handle position manipulation for attention

### Phase 3: Logprob Extraction

- [ ] Implement `compute_logprobs_for_multi_item_scoring()` in LogitsProcessor
- [ ] Recompute `extend_logprob_pruned_lens_cpu` based on delimiter counts
- [ ] Reshape all logprob arrays (`input_token_logprobs`, `input_top_logprobs_*`, `input_token_ids_logprobs_*`) to delimiter-only positions
- [ ] Modify scheduler to emit delimiter-only logprobs
- [ ] Update TokenizerManager to consume pre-sliced logprobs
- [ ] Handle `next_token_logits=None` in workers

### Phase 4: Score Computation

- [ ] Implement `_convert_logprobs_to_scores()` with PyTorch-matching semantics
- [ ] Handle `apply_softmax=True/False` correctly
- [ ] Ensure pure Python (no JAX) in TokenizerManager

### Phase 5: Testing

- [ ] Unit tests for attention mask generation
- [ ] Integration test: multi-item vs single-item equivalence
- [ ] Performance benchmark: multi-item speedup
- [ ] Edge cases from PyTorch test suite (see below)

### Phase 6: Documentation

- [ ] Update RFC-000 with multi-item mode
- [ ] Add usage examples with required flags
- [ ] Document delimiter token selection guidance
- [ ] Update ADR-001 if `apply_softmax` semantics change

---

## Testing Strategy

### Correctness Tests

#### Multi-Item vs Single-Item Equivalence

```python
def test_multi_item_equals_single_item():
    """Multi-item scores should match single-item scores.

    This is the critical correctness test - if attention isolation
    is implemented correctly, results should match.
    """
    query = "The answer is"
    items = [" yes", " no", " maybe"]

    # Single-item mode (current implementation)
    single_scores = []
    for item in items:
        score = score_single_item(query, item, label_token_ids)
        single_scores.append(score)

    # Multi-item mode
    multi_scores = score_multi_item(query, items, label_token_ids)

    # Should match within tolerance
    for i, (single, multi) in enumerate(zip(single_scores, multi_scores)):
        assert_allclose(single, multi, rtol=1e-4,
            err_msg=f"Item {i} mismatch: single={single}, multi={multi}")
```

#### Attention Isolation Verification

```python
def test_attention_isolation():
    """Verify items don't attend to each other.

    If item2 could see item1, changing item1 would change item2's score.
    """
    query = "The capital is"

    # Score with items [A, B, C]
    scores_abc = score_multi_item(query, [" A", " B", " C"], label_ids)

    # Score with items [X, B, C] - only item1 changed
    scores_xbc = score_multi_item(query, [" X", " B", " C"], label_ids)

    # Items 2 and 3 should have IDENTICAL scores (they can't see item 1)
    assert scores_abc[1] == scores_xbc[1], "Item 2 score changed when item 1 changed!"
    assert scores_abc[2] == scores_xbc[2], "Item 3 score changed when item 1 changed!"

    # Item 1 should be different
    assert scores_abc[0] != scores_xbc[0], "Item 1 score should differ"
```

### Edge Cases (from PyTorch Test Suite)

```python
def test_empty_items():
    """Handle empty item strings."""
    scores = score_multi_item("Query", ["", " valid", ""], label_ids)
    assert len(scores) == 3

def test_single_item():
    """Single item should work (degenerate case)."""
    scores = score_multi_item("Query", [" item"], label_ids)
    assert len(scores) == 1

def test_many_items():
    """Test with large number of items."""
    items = [f" item{i}" for i in range(100)]
    scores = score_multi_item("Query", items, label_ids)
    assert len(scores) == 100

def test_unicode_items():
    """Handle unicode in items."""
    scores = score_multi_item("Query", [" 日本語", " emoji 🎉", " mixed"], label_ids)
    assert len(scores) == 3

def test_apply_softmax_false():
    """Verify apply_softmax=False returns exp(logprob), not raw logprobs."""
    scores = score_multi_item("Query", [" a", " b"], label_ids, apply_softmax=False)
    # All scores should be positive (exp of anything is positive)
    assert all(s >= 0 for score_list in scores for s in score_list)

def test_deterministic_consistency():
    """Multiple runs should give identical results."""
    query = "The answer is"
    items = [" yes", " no"]

    scores1 = score_multi_item(query, items, label_ids)
    scores2 = score_multi_item(query, items, label_ids)

    assert scores1 == scores2, "Results should be deterministic"

def test_empty_query_tokens():
    """Handle empty tokenized query (edge case).

    WARNING: If query tokenizes to empty list, the first delimiter
    is at position 0, and (delimiter_index - 1) = -1, causing
    incorrect slicing.

    Text inputs usually inject BOS token, masking this issue.
    Token inputs are vulnerable.
    """
    # This should either:
    # 1. Raise a clear error for empty query
    # 2. Handle gracefully by ensuring minimum query length
    with pytest.raises(ValueError, match="Query must not be empty"):
        score_multi_item_tokens(
            query_tokens=[],  # Empty!
            item_tokens=[[1, 2], [3, 4]],
            label_token_ids=[1, 2, 3],
        )

def test_delimiter_not_in_content():
    """Verify delimiter doesn't appear in query or items."""
    delimiter_token = 128009

    # This should fail validation or produce a warning
    query_with_delimiter = f"Query with delimiter token {delimiter_token}"
    # If delimiter appears in tokenized query, results will be wrong
```

### Edge Case: Empty Query with Token Inputs

> **WARNING:** In the PR, scoring positions are computed as `(delimiter_index - 1)`. If the query token list is empty, the first delimiter is at position 0, leading to index `-1` and incorrect slicing.

Text inputs usually inject BOS token, so this is less visible. Token inputs are vulnerable:

```python
# Problematic case
query_tokens = []  # Empty query
item_tokens = [[100, 101], [200, 201]]
delimiter = 50256

# Combined: [<d>, 100, 101, <d>, 200, 201, <d>]
# First delimiter at position 0
# Scoring position = 0 - 1 = -1  # WRONG!
```

**Recommendation:** Add validation that query must have at least one token (or BOS).

### Performance Tests

```python
def test_multi_item_speedup():
    """Multi-item should be faster for large N."""
    query = "Context " * 50  # ~300 tokens
    items = [f" item{i}" for i in range(10)]

    t_single = benchmark(score_single_item_batch, query, items)
    t_multi = benchmark(score_multi_item, query, items)

    # Expect significant speedup
    speedup = t_single / t_multi
    assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"

def test_memory_usage():
    """Verify memory doesn't explode with many items."""
    query = "Query"
    items = [f" item{i}" for i in range(1000)]

    initial_memory = get_memory_usage()
    scores = score_multi_item(query, items, label_ids)
    final_memory = get_memory_usage()

    # Memory increase should be bounded
    assert final_memory - initial_memory < 1_000_000_000  # 1GB
```

---

## Compatibility

### Backward Compatibility

- **API:** No changes - existing requests work unchanged
- **Default behavior:** Without `--multi-item-scoring-delimiter`, behavior is identical to current
- **Response format:** Unchanged

### PyTorch Parity Checklist

| Aspect | Status | Notes |
|--------|--------|-------|
| Server arg name | ✅ | `--multi-item-scoring-delimiter` |
| Sequence format | ✅ | `query<d>item1<d>...<d>` |
| Attention isolation | ⚠️ | Requires JAX implementation |
| Logprob extraction | ⚠️ | Delimiter-only, pre-sliced |
| `apply_softmax=False` | ⚠️ | Returns `exp(logprob)`, not raw |
| `item_first` handling | ✅ | Ignored in multi-item mode |
| Runtime constraints | ⚠️ | Validate at startup |
| Prefill-only gating | ⚠️ | Only for scoring requests |

---

## Open Questions

### Resolved

1. ~~**Delimiter token selection:** What's the best delimiter token?~~
   **Answer:** Model-specific. Llama-3 uses `128009` (<|eot_id|>). User must configure.

2. ~~**`logprob_start_len` support:** Does JAX scheduler support this?~~
   **Answer:** Still required. The PR sets `logprob_start_len=0` and uses it to count delimiters and shape logprob arrays correctly. While the final output only contains delimiter positions, the scheduler uses this parameter internally. JAX needs an equivalent mechanism for delimiter slicing.

3. ~~**Prefix caching interaction:** Does multi-item work with prefix caching?~~
   **Answer:** No - radix cache must be disabled.

### Open

1. **JAX attention mask approach:** Which implementation (explicit tensor, Pallas, segments) is best for TPU?

2. **Max items limit:** Should we limit items to prevent OOM? PyTorch doesn't limit.

3. **Memory budget:** What's the maximum sequence length / item count for TPU v6e?

4. **bf16 precision:** Does attention mask computation need special handling for bf16?

5. **Multi-host consistency:** How to ensure position manipulation is consistent across TPU hosts?

---

## Alternatives Considered

### Alternative 1: Batched Single-Item

**Description:** Keep separate sequences but batch them together.

```python
# Batch of 3 sequences
["query + item1", "query + item2", "query + item3"]
```

**Pros:**
- Simpler implementation (no attention masking needed)
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

---

## Success Metrics

- [ ] Multi-item scores match single-item within tolerance (1e-4)
- [ ] Attention isolation verified (changing item N doesn't affect item M)
- [ ] 10x+ speedup for 100 items
- [ ] All edge cases pass (empty, unicode, many items)
- [ ] Required flags validated at startup
- [ ] `apply_softmax` semantics match PyTorch
- [ ] Memory usage bounded for large item counts
- [ ] All existing single-item tests still pass

---

## References

- [RFC-000: Score API Design and Architecture](000-score-api-design.md)
- [ADR-001: Pure Python Softmax Decision](../decisions/001-pure-python-softmax.md)
- **PyTorch PR #10979:** [sgl-project/sglang#10979](https://github.com/sgl-project/sglang/pull/10979)
- PyTorch implementation files:
  - `sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  - `sglang/python/sglang/srt/layers/logits_processor.py`
  - `sglang/python/sglang/srt/managers/tokenizer_manager_multiitem_mixin.py`
  - `sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- [Investigation: Score API PyTorch vs JAX Comparison](../investigations/score-api-pytorch-vs-jax.md)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-01 | Initial draft |
| 2026-02-05 | Major update based on PR #10979 analysis: added attention mask section, runtime constraints, corrected logprob dataflow, resolved apply_softmax semantics, added JAX considerations, expanded testing |
| 2026-02-05 | Second review pass: added FlashInfer backend requirement, corrected logprob_start_len explanation, added extend_logprob_pruned_lens_cpu metadata adjustment, documented all logprob array shape changes, strengthened speculative decoding warning, added delimiter collision as hard requirement, added empty query edge case, scoped sliding window to FlashInfer |
| 2026-02-05 | Third review pass: added `skip_special_tokens=False` to delimiter decode, clarified delimiter validation must be token-ID based, acknowledged JAX scipy.softmax vs pure Python, added GLOBAL_SERVER_ARGS_KEYS propagation requirement, documented specific JAX changes for `next_token_logits=None`, marked JAX-only validations as divergence from PyTorch |
