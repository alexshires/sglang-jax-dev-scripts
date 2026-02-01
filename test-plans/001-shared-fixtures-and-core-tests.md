# Test Plan 001: Shared Fixtures and Core Engine Tests

**Related RFC:** [RFC-003](../rfcs/003-score-api-comprehensive-test-suite.md)
**Phase:** 1-2 (Weeks 1-2)
**Priority:** P0 (Foundation)

## Objective

Create shared test fixtures module and implement core engine-level tests for Score API, expanding coverage from 3 tests to 12+ tests.

## Deliverables

1. **New file:** `python/sgl_jax/test/score_test_utils.py` (~300 lines)
2. **Expanded:** `test/srt/test_score_api.py` (~900 lines, was ~600)
3. **Modified:** `python/sgl_jax/srt/managers/tokenizer_manager.py` (add input validation)

## Shared Fixtures Module Specification

### File: `python/sgl_jax/test/score_test_utils.py`

```python
"""
Shared fixtures and utilities for Score API tests.

Usage:
    from sgl_jax.test.score_test_utils import (
        ScoreTestConfig,
        build_engine,
        assert_scores_shape,
        assert_scores_probs
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import unittest


@dataclass
class ScoreTestConfig:
    """
    Configuration for Score API test environment.

    Attributes:
        model: HuggingFace model identifier
        device: Target device (tpu, gpu, cpu)
        dtype: Model dtype (bfloat16, float32)
        download_dir: Model cache directory
        seed: Random seed for reproducibility
        tp_size: Tensor parallel size
        context_length: Max context length
    """
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "tpu"
    dtype: str = "bfloat16"
    download_dir: str = "/dev/shm"
    seed: int = 3
    tp_size: int = 1
    context_length: int = 4096


def build_engine(config: Optional[ScoreTestConfig] = None):
    """
    Build test engine with specified configuration.

    Args:
        config: Test configuration, defaults to ScoreTestConfig()

    Returns:
        Engine instance ready for scoring tests

    Example:
        >>> config = ScoreTestConfig(model="meta-llama/Llama-3.2-1B-Instruct")
        >>> engine = build_engine(config)
    """
    if config is None:
        config = ScoreTestConfig()

    # Implementation mirrors test/srt/test_score_api.py setup
    from sgl_jax.test.test_utils import run_mmlu_test

    runner = run_mmlu_test(
        model=config.model,
        tp_size=config.tp_size,
        attn_backend="all",
        context_length=config.context_length,
        enable_mla=False,
        append_len=256,
        download_dir=config.download_dir,
        base_url="http://localhost:30000",
    )

    return runner


def get_tokenizer(model: str):
    """
    Get tokenizer for specified model.

    Args:
        model: HuggingFace model identifier

    Returns:
        PreTrainedTokenizer instance
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def get_label_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    """
    Convert label tokens to token IDs.

    Args:
        tokenizer: Tokenizer instance
        tokens: List of token strings (e.g., [" to", " of"])

    Returns:
        List of token IDs

    Example:
        >>> tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        >>> ids = get_label_token_ids(tokenizer, [" to", " of"])
        >>> print(ids)  # [311, 315]
    """
    token_ids = []
    for token in tokens:
        # Use encode with add_special_tokens=False to get raw token ID
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Token '{token}' does not map to exactly one token ID. "
                f"Got {len(ids)} IDs: {ids}. Use single-token strings."
            )
        token_ids.append(ids[0])
    return token_ids


def default_query_items() -> Dict[str, Any]:
    """
    Canonical query/items for reuse across tests.

    Returns:
        Dict with query, items, and label_tokens

    Example:
        >>> data = default_query_items()
        >>> print(data["query"])
        'I pledge allegiance'
    """
    return {
        "query": "I pledge allegiance",
        "items": [" to", " of", " for"],
        "label_tokens": [" to", " of", " for"],
    }


def assert_scores_shape(scores: List[List[float]], num_items: int, num_labels: int):
    """
    Validate score output dimensions.

    Args:
        scores: Output from score API
        num_items: Expected number of items
        num_labels: Expected number of labels per item

    Raises:
        AssertionError: If dimensions don't match

    Example:
        >>> scores = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
        >>> assert_scores_shape(scores, num_items=2, num_labels=3)
    """
    assert isinstance(scores, list), f"Expected list, got {type(scores)}"
    assert len(scores) == num_items, (
        f"Expected {num_items} items, got {len(scores)}"
    )

    for i, score_list in enumerate(scores):
        assert isinstance(score_list, list), (
            f"Item {i}: expected list, got {type(score_list)}"
        )
        assert len(score_list) == num_labels, (
            f"Item {i}: expected {num_labels} labels, got {len(score_list)}"
        )


def assert_scores_probs(scores: List[List[float]], apply_softmax: bool):
    """
    Validate score probability constraints.

    Args:
        scores: Output from score API
        apply_softmax: Whether softmax was applied

    Raises:
        AssertionError: If probability constraints violated

    Validation:
        - All values must be numeric
        - All values in [0, 1] range (for apply_softmax=True)
        - Sum ≈ 1.0 if apply_softmax=True (within 1e-6)
        - No sum constraint if apply_softmax=False

    Example:
        >>> scores = [[0.8, 0.15, 0.05]]
        >>> assert_scores_probs(scores, apply_softmax=True)  # OK: sums to 1.0
    """
    for i, score_list in enumerate(scores):
        # All values must be numeric
        for j, score in enumerate(score_list):
            assert isinstance(score, (int, float)), (
                f"Item {i}, label {j}: expected numeric, got {type(score)}"
            )

        if apply_softmax:
            # Softmax: all values in [0, 1]
            for j, score in enumerate(score_list):
                assert 0 <= score <= 1, (
                    f"Item {i}, label {j}: expected [0, 1], got {score}"
                )

            # Softmax: sum ≈ 1.0
            total = sum(score_list)
            assert abs(total - 1.0) < 1e-6, (
                f"Item {i}: expected sum ≈ 1.0, got {total}"
            )
        else:
            # Logprob: no sum constraint, but check reasonable range
            # Logprobs should be in (-inf, 0] when converted to prob space
            pass


def compute_hf_reference_scores(
    model: str,
    query: str,
    items: List[str],
    label_token_ids: List[int],
    item_first: bool = False,
    apply_softmax: bool = False,
) -> Optional[List[List[float]]]:
    """
    Compute reference scores using HuggingFace.

    Gated by SGLANG_JAX_RUN_HF_REFERENCE env var.
    Requires torch + transformers installed.

    Args:
        model: HuggingFace model identifier
        query: Query text
        items: Item texts
        label_token_ids: Label token IDs to score
        item_first: If True, concatenate as item+query
        apply_softmax: If True, apply softmax to logprobs

    Returns:
        Reference scores, or None if gating disabled

    Example:
        >>> # Set env: export SGLANG_JAX_RUN_HF_REFERENCE=1
        >>> scores = compute_hf_reference_scores(
        ...     model="meta-llama/Llama-3.2-1B-Instruct",
        ...     query="I pledge allegiance",
        ...     items=[" to", " of"],
        ...     label_token_ids=[311, 315]
        ... )
    """
    if not should_run_hf_reference():
        return None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "HuggingFace reference requires torch and transformers. "
            "Install with: pip install torch transformers"
        )

    # Implementation from existing test_score_consistency
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    all_scores = []
    for item in items:
        # Concatenate based on item_first
        if item_first:
            text = item + query
        else:
            text = query + item

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(hf_model.device)

        # Forward pass
        with torch.no_grad():
            outputs = hf_model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Extract logprobs for label tokens
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        scores = []
        for token_id in label_token_ids:
            logprob = logprobs[token_id].item()
            scores.append(logprob)

        # Apply softmax if requested
        if apply_softmax:
            import math
            max_logprob = max(scores)
            exp_scores = [
                math.exp(s - max_logprob) if s != float("-inf") else 0.0
                for s in scores
            ]
            sum_exp = sum(exp_scores)
            scores = [s / sum_exp if sum_exp > 0 else 0.0 for s in exp_scores]

        all_scores.append(scores)

    return all_scores


def should_run_hf_reference() -> bool:
    """
    Check if HuggingFace reference tests should run.

    Returns:
        True if SGLANG_JAX_RUN_HF_REFERENCE=1
    """
    return os.getenv("SGLANG_JAX_RUN_HF_REFERENCE") == "1"


def skip_if_no_hf_reference():
    """
    Skip test decorator if HF reference not enabled.

    Usage:
        @skip_if_no_hf_reference()
        def test_with_hf_reference(self):
            ...
    """
    import pytest
    return pytest.mark.skipif(
        not should_run_hf_reference(),
        reason="Set SGLANG_JAX_RUN_HF_REFERENCE=1 to run HF reference tests"
    )


def skip_if_no_multidevice():
    """
    Skip test decorator if < 2 JAX devices available.

    Usage:
        @skip_if_no_multidevice()
        def test_multidevice_sharding(self):
            ...
    """
    import pytest
    import jax
    return pytest.mark.skipif(
        len(jax.devices()) < 2,
        reason="Requires multi-device setup (found {len(jax.devices())} devices)"
    )
```

## Core Engine Tests Specification

### File: `test/srt/test_score_api.py`

Expand with these new tests:

```python
class TestScoreAPI(CustomTestCase):
    """Core engine-level tests for Score API"""

    # ... existing tests (test_score_consistency, test_score_batch_handling,
    #     test_score_request_construction) ...

    def test_score_text_input(self):
        """
        Test scoring with text inputs (query and items as strings).

        Validates:
        - Tokenization path works correctly
        - Results have correct shape
        - Probabilities are valid
        """
        from sgl_jax.test.score_test_utils import (
            default_query_items,
            get_tokenizer,
            get_label_token_ids,
            assert_scores_shape,
            assert_scores_probs
        )

        data = default_query_items()
        tokenizer = get_tokenizer(self.model)
        label_token_ids = get_label_token_ids(tokenizer, data["label_tokens"])

        result = self.runner.score(
            query=data["query"],
            items=data["items"],
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        assert_scores_shape(result, num_items=3, num_labels=3)
        assert_scores_probs(result, apply_softmax=True)

    def test_score_token_input(self):
        """
        Test scoring with token inputs (query and items as token IDs).

        Validates:
        - Direct token path works correctly
        - No tokenization overhead
        - Results match text input (when tokens derived from text)
        """
        from sgl_jax.test.score_test_utils import (
            default_query_items,
            get_tokenizer,
            get_label_token_ids
        )

        data = default_query_items()
        tokenizer = get_tokenizer(self.model)

        # Tokenize query and items
        query_ids = tokenizer.encode(data["query"], add_special_tokens=False)
        items_ids = [
            tokenizer.encode(item, add_special_tokens=False)
            for item in data["items"]
        ]
        label_token_ids = get_label_token_ids(tokenizer, data["label_tokens"])

        result = self.runner.score(
            query=query_ids,
            items=items_ids,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        assert len(result) == 3
        for scores in result:
            assert len(scores) == 3
            assert abs(sum(scores) - 1.0) < 1e-6

    def test_score_apply_softmax_true(self):
        """
        Test apply_softmax=True.

        Validates:
        - Probabilities sum to 1.0 (within 1e-6)
        - All values in [0, 1]
        """
        # ... implementation ...

    def test_score_apply_softmax_false(self):
        """
        Test apply_softmax=False.

        Validates:
        - Returns logprobs (not probabilities)
        - No sum constraint
        - Values in reasonable range
        """
        # ... implementation ...

    def test_score_item_first_false(self):
        """
        Test item_first=False (default).

        Validates:
        - Concatenation order is query + item
        - Sanity check on results
        """
        # ... implementation ...

    def test_score_item_first_true(self):
        """
        Test item_first=True.

        Validates:
        - Concatenation order is item + query
        - Results differ from item_first=False
        """
        # ... implementation ...

    def test_score_different_label_tokens(self):
        """
        Test with varying numbers of label tokens.

        Validates:
        - Label counts: 1, 2, 4, 8, 16
        - Output dimensions scale correctly
        """
        # ... implementation ...

    def test_score_single_item(self):
        """
        Test with single item (batch size 1).

        Validates:
        - Edge case for batching logic
        - No batching bugs
        """
        # ... implementation ...

    def test_score_determinism(self):
        """
        Test that same input yields identical scores.

        Validates:
        - Multiple runs with same seed
        - Bit-exact reproducibility
        """
        # ... implementation ...

    def test_score_default_params(self):
        """
        Test default parameter values.

        Validates:
        - apply_softmax defaults to False
        - item_first defaults to False
        """
        # ... implementation ...
```

## Input Validation in tokenizer_manager.py

### File: `python/sgl_jax/srt/managers/tokenizer_manager.py`

Add validation at start of `score_request()` method:

```python
def score_request(
    self,
    query: Union[str, List[int]],
    items: Union[List[str], List[List[int]]],
    label_token_ids: List[int],
    item_first: bool = False,
    apply_softmax: bool = False,
    return_logprob: bool = False,
) -> List[List[float]]:
    """
    Score items against query using specified label tokens.

    ... existing docstring ...
    """

    # VALIDATION: Empty label_token_ids
    if not label_token_ids or len(label_token_ids) == 0:
        raise ValueError(
            "label_token_ids cannot be empty. "
            "At least one label token ID is required for scoring."
        )

    # VALIDATION: Negative token IDs
    if any(tid < 0 for tid in label_token_ids):
        negative_ids = [tid for tid in label_token_ids if tid < 0]
        raise ValueError(
            f"label_token_ids cannot contain negative values. "
            f"Got negative IDs: {negative_ids}"
        )

    # VALIDATION: Mixed input types
    query_is_text = isinstance(query, str)
    items_is_text = all(isinstance(item, str) for item in items)
    items_is_tokens = all(isinstance(item, list) for item in items)

    if not (items_is_text or items_is_tokens):
        raise ValueError(
            "items must be either all strings or all token ID lists. "
            "Mixed types are not allowed."
        )

    if query_is_text != items_is_text:
        raise ValueError(
            f"query and items must both be text (str) or both be tokens (list[int]). "
            f"Got query type: {type(query).__name__}, "
            f"items type: {'text (str)' if items_is_text else 'tokens (list[int])'}"
        )

    # ... existing implementation ...
```

## Test Execution Plan

### Step 1: Create Fixtures Module

```bash
cd /Users/kanna/Sandbox/sglang-jax
touch python/sgl_jax/test/score_test_utils.py
# Implement module as specified above
```

### Step 2: Validate Fixtures Don't Break Existing Tests

```bash
# Refactor existing tests to use fixtures
# Run to ensure no regression
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# Should show 3 tests passing (existing tests)
```

### Step 3: Add Input Validation

```bash
# Edit python/sgl_jax/srt/managers/tokenizer_manager.py
# Add validation as specified above
```

### Step 4: Implement New Tests One by One

```bash
# Add test_score_text_input
python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_text_input -v

# Add test_score_token_input
python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_token_input -v

# ... continue for all 12 tests
```

### Step 5: Run Full Suite on TPU

```bash
# All core tests together
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# Expected: 12 tests pass, < 5 min runtime
```

### Step 6: Measure Coverage

```bash
pytest --cov=python/sgl_jax/srt/managers/tokenizer_manager \
       --cov-report=term-missing \
       test/srt/test_score_api.py

# Target: 95%+ line coverage on score_request()
```

## Success Criteria

- [ ] `score_test_utils.py` created with all fixtures
- [ ] Existing 3 tests still pass after refactoring
- [ ] 9 new core tests implemented and passing
- [ ] Input validation added to tokenizer_manager.py
- [ ] All 12 tests pass on TPU in < 5 minutes
- [ ] Line coverage on score_request() >= 95%
- [ ] No test code duplication (all use shared fixtures)

## Dependencies

None (foundation for all other test plans)

## Risks

1. **HF reference flakiness** - Network timeouts, model download failures
   - Mitigation: Gate with env var, only run in nightly CI

2. **TPU availability** - May not have TPU for testing
   - Mitigation: Tests should skip gracefully on CPU with warning

3. **Performance regression** - More tests = slower CI
   - Mitigation: Keep tests focused, remove sleeps, use smaller batches

## Follow-Up

After this phase completes:
- Test Plan 002: Edge Cases (build on these fixtures)
- Test Plan 003: HTTP + Protocol (reuse fixtures)
- Test Plan 004: JAX Features + Perf (advanced usage)
