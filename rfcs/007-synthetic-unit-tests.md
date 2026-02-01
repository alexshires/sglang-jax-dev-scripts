# RFC-007: Synthetic Unit Tests

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-01 |
| **Updated** | 2026-02-01 |
| **Related** | [RFC-001](001-score-api-comprehensive-tests.md), [RFC-003](003-score-api-comprehensive-test-suite.md), [RFC-004](004-score-api-performance-benchmarks.md) |

## Summary

Add a layer of fast, deterministic synthetic unit tests that validate Score API correctness using mock logits and controlled inputs. The core tests (shift, mask, boundary, numerical stability) run without model inference, while JAX compilation tests require runtime but focus on caching behavior rather than model accuracy. This catches shift/mask/boundary bugs early, before they reach heavier integration tests.

## Motivation

### Gap Analysis

RFC-003 provides comprehensive integration tests, but they all require real model inference. This creates gaps:

| Test Type | Current Coverage | Gap |
|-----------|------------------|-----|
| Shift correctness (position t-1 → token t) | Implicit via HF comparison | No explicit unit test with known logits |
| Mask correctness (pad = 0 contribution) | Not tested | Pad positions could contribute incorrectly |
| Continuation boundary math | Implicit | No explicit `full - prompt = continuation` validation |
| JAX compilation caching | Not tested | Same-shape requests could trigger recompilation |
| Fuzz/property testing | Not tested | No random input validation |

### Why Synthetic Tests Matter

1. **Speed**: Run in milliseconds (no model loading)
2. **Determinism**: No floating-point variance from model weights
3. **Debuggability**: When they fail, you know exactly what broke
4. **Early detection**: Catch off-by-one bugs before integration tests
5. **CI-friendly**: Can run on any hardware (no TPU required)

### Problems These Tests Catch

Based on common Score API bugs in similar systems:

1. **Off-by-one in shift**: Using logits at position t instead of t-1
2. **Incorrect masking**: Pad tokens contributing to scores
3. **Boundary errors**: Wrong token marked as continuation start
4. **Shape pollution**: JAX recompiling for same shapes
5. **Edge case crashes**: Empty inputs, extreme values

## Proposed Solution

### New Test File

**File:** `test/srt/test_score_api_synthetic.py`

**Characteristics:**
- No model loading (uses mock logits)
- Runs on CPU (no TPU required)
- Fast (<1 second for all tests)
- Included in default CI suite

### Test Categories

#### Category 1: Shift Correctness Tests

Validate that scoring uses logits at position t-1 to score token t.

**Note on API Contract:**
These tests assume a `compute_token_logprobs` function with the following signature:
```python
def compute_token_logprobs(
    logits: np.ndarray,  # Shape: (seq_len, vocab_size)
    token_ids: List[int],  # Actual token IDs in sequence
    label_token_ids: List[int],  # Vocabulary token IDs to extract logprobs for (arbitrary, not necessarily in sequence)
    start_position: int,  # Position to start scoring from
    attention_mask: Optional[List[int]] = None,  # 1 = real, 0 = padding
) -> List[float]:
    """
    Compute log probabilities for specified vocabulary tokens at each position.

    Args:
        label_token_ids: Arbitrary vocabulary token IDs to score. These are
            candidate tokens we want probabilities for, NOT tokens that must
            appear in the sequence. For example, scoring [" yes", " no"] tokens
            regardless of what tokens are actually in token_ids.

    Returns log probabilities for each token in label_token_ids, where
    the probability of token at position t is computed using logits[t-1].
    """
```

This is a proposed internal API for synthetic testing. If not yet implemented,
this should be extracted or created from the existing score logic.

```python
# test/srt/test_score_api_synthetic.py

import numpy as np
import pytest
from sgl_jax.srt.managers.tokenizer_manager import compute_token_logprobs

class TestShiftCorrectness:
    """
    Validate that Score API correctly uses logits[t-1] to score token[t].

    The Score API computes P(token[t] | context[0:t]) using the logits
    output at position t-1. This is because autoregressive models output
    next-token predictions at each position.

    These tests use synthetic logits where the correct answer is known,
    making it trivial to detect off-by-one errors.
    """

    def test_shift_basic(self):
        """
        Verify position t-1 logits score token t.

        Setup:
        - 4 tokens: [A, B, C, D] at positions [0, 1, 2, 3]
        - Logits shape: (4, vocab_size)
        - logits[0] predicts token at position 1 (B)
        - logits[1] predicts token at position 2 (C)
        - logits[2] predicts token at position 3 (D)

        We construct logits where:
        - logits[t-1] has high value only at token[t]
        - All other positions have uniform low values

        Expected: High scores for each token (using correct shift)
        If shift is wrong: Low/random scores (using wrong logits row)
        """
        vocab_size = 100
        tokens = [10, 20, 30, 40]  # A=10, B=20, C=30, D=40

        # Construct synthetic logits
        # logits[i] should have high value at tokens[i+1]
        logits = np.full((4, vocab_size), -10.0, dtype=np.float32)
        logits[0, tokens[1]] = 0.0  # Position 0 predicts B (token 20)
        logits[1, tokens[2]] = 0.0  # Position 1 predicts C (token 30)
        logits[2, tokens[3]] = 0.0  # Position 2 predicts D (token 40)
        # logits[3] doesn't matter (no token after D)

        # Score tokens 1, 2, 3 (B, C, D) - skipping first token
        label_token_ids = [tokens[1], tokens[2], tokens[3]]

        scores = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=label_token_ids,
            start_position=1,  # Start scoring from position 1
        )

        # With correct shift: scores should be high (close to 0 in logprob space)
        # Background logprob ≈ log(exp(-10) / (exp(0) + 99*exp(-10))) ≈ -10
        # Target logprob ≈ log(exp(0) / (exp(0) + 99*exp(-10))) ≈ 0
        for score in scores:
            assert score > -1.0, f"Score {score} too low - shift may be wrong"

    def test_shift_off_by_one_detection(self):
        """
        Explicitly verify that using wrong shift gives wrong answer.

        This test documents the expected failure mode if shift is
        implemented incorrectly.
        """
        vocab_size = 100
        tokens = [10, 20, 30]

        # logits[0] predicts token 20, logits[1] predicts token 30
        logits = np.full((3, vocab_size), -10.0, dtype=np.float32)
        logits[0, 20] = 0.0  # Correct: use this for token[1]
        logits[1, 30] = 0.0  # Correct: use this for token[2]

        # Compute with correct shift
        correct_score = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
        )

        # The correct score should be close to 0 (high probability)
        assert correct_score[0] > -1.0

        # If we used wrong shift (logits[1] instead of logits[0] for token[1]):
        # We'd get logprob of token 20 under logits[1], which is -10
        # This demonstrates what the bug would look like

    def test_shift_with_varying_sequence_lengths(self):
        """Test shift correctness with different sequence lengths."""
        vocab_size = 50

        for seq_len in [2, 5, 10, 20]:
            tokens = list(range(seq_len))  # [0, 1, 2, ..., seq_len-1]

            # Each position predicts next token with high confidence
            logits = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
            for i in range(seq_len - 1):
                logits[i, tokens[i + 1]] = 0.0

            # Score all tokens except first
            label_token_ids = tokens[1:]

            scores = compute_token_logprobs(
                logits=logits,
                token_ids=tokens,
                label_token_ids=label_token_ids,
                start_position=1,
            )

            # All scores should be high
            for i, score in enumerate(scores):
                assert score > -1.0, f"Seq len {seq_len}, pos {i}: score {score} too low"
```

#### Category 2: Mask Correctness Tests

Validate that padded positions contribute zero to the score.

```python
class TestMaskCorrectness:
    """
    Validate that padding tokens do not contribute to scores.

    When scoring batched sequences of different lengths, shorter sequences
    are padded. The padding positions must contribute 0 to the total score.

    Common bugs:
    - Padding contributes negative infinity (crashes or NaN)
    - Padding contributes non-zero value (incorrect scores)
    - Mask applied at wrong positions
    """

    def test_pad_contribution_is_zero(self):
        """
        Verify that padding positions contribute exactly 0 to score.

        Setup:
        - Actual tokens: [A, B, PAD, PAD]
        - Score should only include A and B contributions
        """
        vocab_size = 100
        pad_token_id = 0
        tokens = [10, 20, pad_token_id, pad_token_id]

        # Logits that would give non-zero score if pad tokens are scored
        logits = np.full((4, vocab_size), -5.0, dtype=np.float32)
        logits[0, 20] = 0.0  # High prob for token B at position 1
        logits[1, pad_token_id] = 0.0  # High prob for PAD at position 2
        logits[2, pad_token_id] = 0.0  # High prob for PAD at position 3

        # Score without padding (just [A, B])
        tokens_no_pad = [10, 20]
        logits_no_pad = logits[:2]

        score_no_pad = compute_token_logprobs(
            logits=logits_no_pad,
            token_ids=tokens_no_pad,
            label_token_ids=[20],
            start_position=1,
        )

        # Score with padding (should be identical)
        score_with_pad = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
            attention_mask=[1, 1, 0, 0],  # Mask out pad positions
        )

        assert abs(score_no_pad[0] - score_with_pad[0]) < 1e-6, \
            "Padding affected score - mask not working correctly"

    def test_all_pad_tail(self):
        """Sequence ending in all padding."""
        vocab_size = 50
        pad_token_id = 0

        # [REAL, REAL, PAD, PAD, PAD]
        tokens = [10, 20, pad_token_id, pad_token_id, pad_token_id]
        mask = [1, 1, 0, 0, 0]

        logits = np.random.randn(5, vocab_size).astype(np.float32)

        score = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
            attention_mask=mask,
        )

        # Should not crash and should be finite
        assert np.isfinite(score[0]), "Score is not finite with pad tail"

    def test_empty_continuation(self):
        """
        Edge case: continuation length is 0.

        This happens when prompt exactly fills the context.
        Score should be 0 (no tokens to score).
        """
        vocab_size = 50
        tokens = [10, 20, 30]  # All prompt, no continuation

        logits = np.random.randn(3, vocab_size).astype(np.float32)

        # No continuation tokens to score
        score = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[],
            start_position=3,  # Start after all tokens
        )

        assert len(score) == 0 or sum(score) == 0.0, \
            "Empty continuation should have zero score"

    def test_continuation_length_one(self):
        """Edge case: exactly one continuation token."""
        vocab_size = 50
        tokens = [10, 20]  # One prompt token, one continuation

        logits = np.full((2, vocab_size), -10.0, dtype=np.float32)
        logits[0, 20] = 0.0  # High prob for continuation token

        score = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
        )

        assert len(score) == 1
        assert score[0] > -1.0, "Single token continuation score incorrect"

    def test_mask_with_interior_padding(self):
        """
        Test that interior padding (if ever used) is handled.

        Note: This is unusual but tests robustness.
        [REAL, PAD, REAL, PAD]
        """
        vocab_size = 50
        pad_token_id = 0

        tokens = [10, pad_token_id, 30, pad_token_id]
        mask = [1, 0, 1, 0]

        logits = np.random.randn(4, vocab_size).astype(np.float32)

        score = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[30],
            start_position=1,
            attention_mask=mask,
        )

        assert np.isfinite(score[0]), "Interior padding caused non-finite score"
```

#### Category 3: Continuation Boundary Tests

Validate that continuation scoring math is correct.

```python
class TestContinuationBoundary:
    """
    Validate continuation-only scoring correctness.

    In rerank mode, we score only the continuation (not the prompt).
    This requires correct boundary detection:

    score(continuation | prompt) = score(prompt + continuation) - score(prompt)

    Or equivalently: only sum logprobs from continuation start position.

    Common bugs:
    - Off-by-one in continuation start position
    - Including prompt tokens in continuation score
    - Excluding first continuation token
    """

    def test_continuation_equals_full_minus_prompt(self):
        """
        Verify: score(continuation) ≈ score(full) - score(prompt_portion)

        This is the fundamental property of continuation scoring.
        """
        vocab_size = 100
        prompt_tokens = [10, 20, 30]
        continuation_tokens = [40, 50]
        full_tokens = prompt_tokens + continuation_tokens

        # Construct logits where each position predicts next token
        logits = np.full((5, vocab_size), -10.0, dtype=np.float32)
        for i in range(4):
            logits[i, full_tokens[i + 1]] = 0.0

        # Score full sequence (positions 1-4, tokens 20, 30, 40, 50)
        full_score = compute_token_logprobs(
            logits=logits,
            token_ids=full_tokens,
            label_token_ids=full_tokens[1:],
            start_position=1,
        )

        # Score just continuation (positions 3-4, tokens 40, 50)
        continuation_score = compute_token_logprobs(
            logits=logits,
            token_ids=full_tokens,
            label_token_ids=continuation_tokens,
            start_position=3,  # Start at continuation
        )

        # Score just prompt portion (positions 1-2, tokens 20, 30)
        prompt_portion_score = compute_token_logprobs(
            logits=logits,
            token_ids=full_tokens,
            label_token_ids=prompt_tokens[1:],  # [20, 30]
            start_position=1,
        )

        # Verify: full ≈ prompt_portion + continuation
        full_sum = sum(full_score)
        expected_sum = sum(prompt_portion_score) + sum(continuation_score)

        assert abs(full_sum - expected_sum) < 1e-5, \
            f"Continuation math wrong: full={full_sum}, expected={expected_sum}"

    def test_continuation_start_position_exact(self):
        """
        Verify the exact token where continuation scoring starts.

        For prompt=[A, B, C] and continuation=[D, E]:
        - Position 0: token A (not scored)
        - Position 1: token B (scored if full, not if continuation-only)
        - Position 2: token C (scored if full, not if continuation-only)
        - Position 3: token D (first continuation token - MUST be scored)
        - Position 4: token E (continuation token - MUST be scored)

        The continuation score should include D and E only.
        """
        vocab_size = 100
        prompt = [10, 20, 30]  # A, B, C
        continuation = [40, 50]  # D, E
        full = prompt + continuation

        # Make each token easily identifiable by its score
        # Token 40 gets logprob ~= -1, Token 50 gets logprob ~= -2
        logits = np.full((5, vocab_size), -10.0, dtype=np.float32)
        logits[2, 40] = -1.0  # Position 2 predicts D with logprob ~-1
        logits[3, 50] = -2.0  # Position 3 predicts E with logprob ~-2

        # Make prompt tokens have different scores
        logits[0, 20] = -3.0  # B
        logits[1, 30] = -4.0  # C

        # Score continuation only
        continuation_score = compute_token_logprobs(
            logits=logits,
            token_ids=full,
            label_token_ids=continuation,
            start_position=len(prompt),  # Start at position 3
        )

        # Should have exactly 2 scores (for D and E)
        assert len(continuation_score) == 2, \
            f"Expected 2 continuation scores, got {len(continuation_score)}"

        # Scores should be close to -1 and -2 (after softmax adjustment)
        # The key is that they should NOT be -3 or -4 (prompt token scores)

    def test_off_by_one_in_boundary(self):
        """
        Explicitly test that common off-by-one errors are caught.

        If start_position is off by 1:
        - Too early (2 instead of 3): includes last prompt token
        - Too late (4 instead of 3): excludes first continuation token

        Both are bugs that this test should catch.
        """
        vocab_size = 50
        prompt = [10, 20, 30]
        continuation = [40]
        full = prompt + continuation

        # Give each position a unique, identifiable score
        logits = np.full((4, vocab_size), -100.0, dtype=np.float32)
        logits[0, 20] = -1.0  # Score for token at pos 1 (B)
        logits[1, 30] = -2.0  # Score for token at pos 2 (C)
        logits[2, 40] = -3.0  # Score for token at pos 3 (D) <- continuation

        # Correct: start at position 3
        correct_score = compute_token_logprobs(
            logits=logits,
            token_ids=full,
            label_token_ids=continuation,
            start_position=3,
        )

        # The score should correspond to position 2's prediction of token 40
        # which has logit -3.0
        assert len(correct_score) == 1
        # Score should be based on logits[2, 40] = -3.0, not logits[1, 30] = -2.0
```

#### Category 4: JAX Compilation Tests

**Note:** These tests require JAX runtime and model inference. They are NOT
purely synthetic unit tests and should be considered integration tests.
They are included here for completeness but may be better suited for RFC-003's
comprehensive test suite.

Validate that same-shape requests don't trigger recompilation.

```python
class TestJAXCompilation:
    """
    Validate JAX compilation behavior.

    WARNING: These tests require JAX runtime and are NOT synthetic unit tests.
    They call score_request() which performs model inference on TPU/GPU.
    Consider moving to RFC-003's integration test suite.

    JAX compiles (traces) functions on first call with a new shape.
    Subsequent calls with the same shape should reuse the compiled version.

    Excessive recompilation causes:
    - High latency on "same" requests
    - Memory pressure from cached compilations
    - Unpredictable performance

    These tests validate compilation caching works correctly.
    """

    def test_no_recompile_same_shape(self):
        """
        Three requests with identical shapes should trigger only one compilation.

        This requires access to JAX compilation counters or timing analysis.

        NOTE: This test requires make_score_request() helper. Either implement or use
        the actual Score API request building logic.
        """
        import jax
        from sgl_jax.srt.managers.tokenizer_manager import score_request

        # TBD: Define make_score_request() helper
        # def make_score_request(batch_size, seq_len):
        #     """Create a Score API request with specified shape."""
        #     return {
        #         "query": "a" * seq_len,
        #         "items": ["x"] * batch_size,
        #         "label_token_ids": [1, 2, 3],
        #     }

        # Get initial compilation count (if available)
        # Note: This may require JAX internal APIs or custom instrumentation

        # Request 1: First call - triggers compilation
        # request1 = make_score_request(batch_size=4, seq_len=128)
        # result1 = score_request(request1)

        # Request 2: Same shape - should NOT recompile
        # request2 = make_score_request(batch_size=4, seq_len=128)
        # result2 = score_request(request2)

        # Request 3: Same shape - should NOT recompile
        # request3 = make_score_request(batch_size=4, seq_len=128)
        # result3 = score_request(request3)

        # Verification options:
        # 1. Check JAX compilation counter (if instrumented)
        # 2. Verify request 2 and 3 are faster than request 1
        # 3. Check that no new compilation logs appear

        # Timing-based verification (less reliable but portable)
        # import time

        # t1_start = time.perf_counter()
        # score_request(make_score_request(batch_size=4, seq_len=128))
        # t1 = time.perf_counter() - t1_start

        # t2_start = time.perf_counter()
        # score_request(make_score_request(batch_size=4, seq_len=128))
        # t2 = time.perf_counter() - t2_start

        # Second call should be at least 2x faster (no compilation)
        # This is a heuristic - compilation typically takes 10-100x longer
        # assert t2 < t1 * 0.5 or t1 < 0.1, \
        #     f"Possible recompilation: t1={t1:.3f}s, t2={t2:.3f}s"

        pytest.skip("TBD: make_score_request() not yet implemented")

    def test_recompile_on_different_shape(self):
        """
        Different shapes SHOULD trigger recompilation (expected behavior).

        This documents that shape changes cause compilation, which is
        expected JAX behavior but should be minimized in production.
        """
        # Shape 1
        # request1 = make_score_request(batch_size=4, seq_len=128)
        # result1 = score_request(request1)

        # Shape 2 (different) - SHOULD recompile
        # request2 = make_score_request(batch_size=8, seq_len=256)
        # result2 = score_request(request2)

        # Both should succeed
        # assert result1 is not None
        # assert result2 is not None

        pytest.skip("TBD: make_score_request() not yet implemented")

    def test_bucketed_shapes_reduce_compilations(self):
        """
        Verify that input bucketing reduces compilation count.

        If the system buckets sequence lengths to fixed sizes (e.g., 128, 256, 512),
        then sequences of length 100 and 120 should both bucket to 128 and
        share the same compiled function.
        """
        # Sequence length 100 - should bucket to 128
        # request1 = make_score_request(batch_size=1, seq_len=100)

        # Sequence length 120 - should also bucket to 128
        # request2 = make_score_request(batch_size=1, seq_len=120)

        # import time

        # First request with seq_len=100
        # t1_start = time.perf_counter()
        # score_request(request1)
        # t1 = time.perf_counter() - t1_start

        # Second request with seq_len=120 (same bucket)
        # t2_start = time.perf_counter()
        # score_request(request2)
        # t2 = time.perf_counter() - t2_start

        # If bucketing works, t2 should be fast (no recompilation)
        # Skip this assertion if bucketing isn't implemented
        # if hasattr(score_request, 'uses_bucketing') and score_request.uses_bucketing:
        #     assert t2 < t1 * 0.5, \
        #         "Bucketing may not be working - seq_len 100 and 120 should share compilation"

        pytest.skip("TBD: make_score_request() not yet implemented")
```

#### Category 5: Fuzz and Property Tests

Random input testing with invariant assertions.

```python
import hypothesis
from hypothesis import given, strategies as st

class TestFuzzAndProperties:
    """
    Property-based tests using random inputs.

    These tests generate random (but valid) inputs and verify that
    certain properties always hold:
    - Scores are finite (no NaN/Inf)
    - Batch invariance (same input = same output regardless of batch)
    - Monotonic masking (more padding = same or lower score contribution)

    Catches edge cases that structured tests miss.
    """

    @given(
        vocab_size=st.integers(min_value=100, max_value=1000),
        seq_len=st.integers(min_value=2, max_value=100),
        num_labels=st.integers(min_value=1, max_value=10),
    )
    def test_scores_always_finite(self, vocab_size, seq_len, num_labels):
        """Property: Scores must always be finite (no NaN or Inf)."""
        # Generate random tokens within vocab
        tokens = np.random.randint(0, vocab_size, size=seq_len).tolist()

        # Generate random logits
        logits = np.random.randn(seq_len, vocab_size).astype(np.float32)

        # Pick random label tokens (arbitrary vocab IDs, not necessarily in sequence)
        # This is correct: label_token_ids are candidate tokens to score, not tokens
        # that must appear in the sequence. We extract their logprobs from the model's
        # vocabulary distribution at each position.
        label_token_ids = np.random.randint(0, vocab_size, size=num_labels).tolist()

        scores = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=label_token_ids,
            start_position=1,
        )

        for score in scores:
            assert np.isfinite(score), \
                f"Non-finite score: {score} with seq_len={seq_len}, vocab={vocab_size}"

    @given(
        seq_len=st.integers(min_value=5, max_value=50),
    )
    def test_batch_invariance(self, seq_len):
        """
        Property: Same input scored alone vs in batch gives same result.

        score([A]) == score([A, B, C])[0]  # A's score unchanged by B, C
        """
        vocab_size = 100

        # Generate a test sequence
        tokens_a = np.random.randint(0, vocab_size, size=seq_len).tolist()
        tokens_b = np.random.randint(0, vocab_size, size=seq_len).tolist()

        logits_a = np.random.randn(seq_len, vocab_size).astype(np.float32)
        logits_b = np.random.randn(seq_len, vocab_size).astype(np.float32)

        label_token_ids = [tokens_a[1]]  # Score first token after start

        # Score A alone
        score_alone = compute_token_logprobs(
            logits=logits_a,
            token_ids=tokens_a,
            label_token_ids=label_token_ids,
            start_position=1,
        )

        # Score A in batch with B (if batching is supported at this level)
        # Note: This may need adjustment based on actual API
        score_batched = compute_token_logprobs(
            logits=logits_a,  # Same logits for A
            token_ids=tokens_a,
            label_token_ids=label_token_ids,
            start_position=1,
        )

        assert abs(score_alone[0] - score_batched[0]) < 1e-6, \
            "Batch affected score - batch invariance violated"

    @given(
        seq_len=st.integers(min_value=3, max_value=20),
    )
    def test_ordering_invariance(self, seq_len):
        """
        Property: Order of items in batch doesn't affect their individual scores.

        For batch [A, B, C]: score(A) same whether batch is [A,B,C] or [C,A,B]

        NOTE: This test requires batch_score() and make_score_request() to be defined.
        These are TBD - either implement or mark this test as @pytest.mark.skip(reason="TBD: batch_score not yet implemented")
        """
        # This tests the high-level batch scoring API
        vocab_size = 100

        items = [
            np.random.randint(0, vocab_size, size=seq_len).tolist()
            for _ in range(3)
        ]

        # TBD: Define batch_score() helper function
        # def batch_score(items, order):
        #     """Score items in specified order."""
        #     pass

        # Score in original order
        # scores_original = batch_score(items, order=[0, 1, 2])

        # Score in different order
        # scores_reordered = batch_score(items, order=[2, 0, 1])

        # Item 0's score should be the same in both
        # assert abs(scores_original[0] - scores_reordered[1]) < 1e-6, \
        #     "Batch order affected scores"

        pytest.skip("TBD: batch_score() not yet implemented")

    @given(
        real_len=st.integers(min_value=2, max_value=10),
        pad_len=st.integers(min_value=0, max_value=10),
    )
    def test_padding_monotonic(self, real_len, pad_len):
        """
        Property: Adding padding tokens should not change the score.

        score([A, B]) == score([A, B, PAD, PAD])

        (when properly masked)
        """
        vocab_size = 100
        pad_token_id = 0

        # Real tokens
        real_tokens = np.random.randint(1, vocab_size, size=real_len).tolist()

        # Padded version
        padded_tokens = real_tokens + [pad_token_id] * pad_len
        mask = [1] * real_len + [0] * pad_len

        # Logits for both (extend with random for padded positions)
        logits_real = np.random.randn(real_len, vocab_size).astype(np.float32)
        logits_padded = np.vstack([
            logits_real,
            np.random.randn(pad_len, vocab_size).astype(np.float32)
        ])

        label_token_ids = [real_tokens[1]]

        score_real = compute_token_logprobs(
            logits=logits_real,
            token_ids=real_tokens,
            label_token_ids=label_token_ids,
            start_position=1,
        )

        score_padded = compute_token_logprobs(
            logits=logits_padded,
            token_ids=padded_tokens,
            label_token_ids=label_token_ids,
            start_position=1,
            attention_mask=mask,
        )

        assert abs(score_real[0] - score_padded[0]) < 1e-6, \
            f"Padding changed score: {score_real[0]} vs {score_padded[0]}"
```

#### Category 6: Numerical Stability Tests

Test edge cases in numerical computation.

```python
class TestNumericalStability:
    """
    Validate numerical stability under extreme conditions.

    Scoring involves log_softmax which can be numerically unstable:
    - Very large logits → overflow before softmax
    - Very small logits → underflow in exp()
    - Large vocab → sum of many small numbers

    These tests verify the implementation handles edge cases.
    """

    def test_extreme_logits_no_overflow(self):
        """Large logits should not cause overflow."""
        vocab_size = 100
        tokens = [10, 20, 30]

        # Extreme positive logits
        logits = np.full((3, vocab_size), 1000.0, dtype=np.float32)
        logits[0, 20] = 1001.0  # Slightly higher for target

        scores = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
        )

        assert np.isfinite(scores[0]), "Overflow with large logits"

    def test_extreme_negative_logits_no_underflow(self):
        """Very negative logits should not cause underflow issues."""
        vocab_size = 100
        tokens = [10, 20, 30]

        # Extreme negative logits
        logits = np.full((3, vocab_size), -1000.0, dtype=np.float32)
        logits[0, 20] = -999.0  # Slightly higher for target

        scores = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[20],
            start_position=1,
        )

        assert np.isfinite(scores[0]), "Underflow with negative logits"

    def test_large_vocab_numerical_stability(self):
        """Large vocabulary should not cause precision issues."""
        vocab_size = 128000  # Typical LLM vocab size
        seq_len = 10

        tokens = np.random.randint(0, vocab_size, size=seq_len).tolist()
        logits = np.random.randn(seq_len, vocab_size).astype(np.float32)

        scores = compute_token_logprobs(
            logits=logits,
            token_ids=tokens,
            label_token_ids=[tokens[1]],
            start_position=1,
        )

        assert np.isfinite(scores[0]), "Precision issue with large vocab"

    def test_bf16_stability(self):
        """
        BF16 computation should not produce NaN/Inf.

        BF16 has less precision than FP32, which can cause issues
        in log_softmax computation if not handled carefully.
        """
        vocab_size = 1000
        seq_len = 100

        tokens = np.random.randint(0, vocab_size, size=seq_len).tolist()
        logits = np.random.randn(seq_len, vocab_size).astype(np.float32)

        # Simulate bf16 by reducing precision
        logits_bf16 = logits.astype(np.float32)  # Would be bfloat16 in JAX

        scores = compute_token_logprobs(
            logits=logits_bf16,
            token_ids=tokens,
            label_token_ids=[tokens[1]],
            start_position=1,
            dtype='bfloat16',  # Flag to use bf16 path
        )

        assert np.isfinite(scores[0]), "BF16 caused NaN/Inf"

    def test_log_softmax_not_prob_then_log(self):
        """
        Verify log_softmax is used, not softmax-then-log.

        softmax → log is numerically unstable for small probabilities:
        - softmax gives 0.0 for very unlikely tokens
        - log(0.0) = -inf

        log_softmax handles this correctly via log-sum-exp trick.

        This test verifies the implementation by checking that very
        unlikely tokens get finite (very negative) log probabilities,
        not -inf.
        """
        vocab_size = 100
        tokens = [10, 20, 30]

        # Make one token extremely unlikely
        logits = np.zeros((3, vocab_size), dtype=np.float32)
        logits[0, :] = 0.0  # Uniform
        logits[0, 99] = -100.0  # Token 99 extremely unlikely

        # Score the unlikely token
        scores = compute_token_logprobs(
            logits=logits,
            token_ids=[10, 99, 30],  # Token 99 at position 1
            label_token_ids=[99],
            start_position=1,
        )

        # Should be very negative but finite (not -inf)
        assert np.isfinite(scores[0]), \
            "Got -inf for unlikely token - using prob-then-log instead of log_softmax"
        assert scores[0] < -10.0, \
            "Unlikely token should have very negative log probability"
```

### Implementation Plan

#### Phase 1: Core Infrastructure (1 day)

- [ ] Create `test/srt/test_score_api_synthetic.py`
- [ ] Implement mock `compute_token_logprobs()` function signature
- [ ] Add pytest fixtures for synthetic logit generation
- [ ] Verify tests can run without model loading

#### Phase 2: Shift and Mask Tests (1 day)

- [ ] Implement `TestShiftCorrectness` (3 tests)
- [ ] Implement `TestMaskCorrectness` (5 tests)
- [ ] Verify all tests pass with current implementation
- [ ] Document any bugs found

#### Phase 3: Continuation and Compilation Tests (1 day)

- [ ] Implement `TestContinuationBoundary` (3 tests)
- [ ] Implement `TestJAXCompilation` (3 tests)
- [ ] Add timing-based compilation detection
- [ ] Document compilation behavior

#### Phase 4: Fuzz and Stability Tests (1 day)

- [ ] Add `hypothesis` to dev dependencies
- [ ] Implement `TestFuzzAndProperties` (4 tests)
- [ ] Implement `TestNumericalStability` (5 tests)
- [ ] Run fuzz tests with high iteration count locally

#### Phase 5: CI Integration (0.5 day)

- [ ] Add to default CI suite (fast, no TPU required)
- [ ] Configure hypothesis profiles for CI vs local
- [ ] Update test documentation
- [ ] Add to INDEX.md

### File Structure

```
test/srt/
├── test_score_api.py              # Existing: Integration tests (require model)
├── test_score_api_synthetic.py    # NEW: Synthetic unit tests (no model)
├── test_score_api_edge_cases.py   # From RFC-003: Edge case tests
├── test_score_api_jax_features.py # From RFC-003: JAX-specific tests
└── bench_score.py                 # From RFC-004: Performance benchmarks
```

### CI Configuration

```yaml
# Synthetic tests: Run on every PR (fast, CPU-only)
synthetic-tests:
  runs-on: ubuntu-latest
  steps:
    - run: pip install pytest hypothesis numpy
    - run: pytest test/srt/test_score_api_synthetic.py -v

# Full tests: Run on TPU (slower, after synthetic pass)
integration-tests:
  needs: synthetic-tests
  runs-on: tpu-v6e-1
  steps:
    - run: pytest test/srt/test_score_api.py -v
```

### Dependencies

**Required:**
- `pytest` (already present)
- `numpy` (already present)

**Optional (for fuzz tests):**
- `hypothesis` - Property-based testing library

```bash
pip install hypothesis
```

### Relationship to Existing Tests

| Test Layer | File | Model Required | TPU Required | CI Tier |
|------------|------|----------------|--------------|---------|
| **Synthetic** | `test_score_api_synthetic.py` | No | No | Default |
| Integration | `test_score_api.py` | Yes | Yes | Default |
| Edge Cases | `test_score_api_edge_cases.py` | Yes | Yes | Default |
| JAX Features | `test_score_api_jax_features.py` | Yes | Multi-TPU | Nightly |
| Performance | `bench_score.py` | Yes | Yes | Manual |

### Success Metrics

- [ ] 20+ synthetic tests passing
- [ ] All tests run in <5 seconds (no model loading)
- [ ] Tests run on CPU (no TPU required)
- [ ] At least one bug found by fuzz testing
- [ ] Compilation test catches real recompilation issues

### Open Questions

1. **Mock vs Real**: Should synthetic tests mock `compute_token_logprobs()` or test the actual implementation with synthetic inputs?
   - Recommendation: Test actual implementation to catch real bugs

2. **Hypothesis Configuration**: What's the right balance of iteration count for CI vs local?
   - Recommendation: CI = 100 iterations, Local = 1000 iterations

3. **Compilation Tracking**: Does JAX expose compilation counters we can use?
   - Fallback: Use timing-based heuristics

4. **Integration with RFC-003**: Should these tests be merged into existing test files or kept separate?
   - Recommendation: Keep separate for clear layering (synthetic vs integration)

## Alternatives Considered

### Alternative 1: Add Synthetic Tests to Existing Files

**Description:** Add synthetic tests to `test_score_api.py`

**Pros:**
- Fewer files to maintain
- Collocated with integration tests

**Cons:**
- Mixes fast (no model) and slow (model) tests
- Can't run synthetic tests in isolation on CPU
- Harder to gate in CI

**Why rejected:** Clear separation enables better CI gating and faster feedback loops.

### Alternative 2: Skip Fuzz Testing

**Description:** Only use structured tests, skip hypothesis

**Pros:**
- No new dependency
- Deterministic tests only

**Cons:**
- Misses edge cases that structured tests don't cover
- Fuzz tests are cheap and high-value

**Why rejected:** Hypothesis is well-established and catches real bugs. Worth the dependency.

### Alternative 3: Full Mocking

**Description:** Mock all dependencies, test in complete isolation

**Pros:**
- True unit tests
- No implementation coupling

**Cons:**
- Mocks can drift from real implementation
- Bugs in integration between components missed

**Why rejected:** We want to catch real bugs in the actual implementation, not just verify our understanding of what it should do.

## References

- [RFC-001: Score API Comprehensive Tests](001-score-api-comprehensive-tests.md)
- [RFC-003: Comprehensive Score API Test Suite](003-score-api-comprehensive-test-suite.md)
- [RFC-004: Score API Performance Benchmarks](004-score-api-performance-benchmarks.md)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [JAX Compilation Debugging](https://jax.readthedocs.io/en/latest/debugging/index.html)
- ChatGPT analysis of Score API test gaps (2026-02-01)
