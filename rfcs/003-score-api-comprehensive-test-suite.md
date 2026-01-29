# RFC-003: Comprehensive Score API Test Suite

**Status:** Draft
**Author:** Engineering Team
**Created:** 2026-01-29
**Updated:** 2026-01-29
**Related RFC:** RFC-001

## Summary

Expand Score API test coverage from 4 Tier 1 tests to a comprehensive suite (~30+ tests) with clear CI vs local/perf gating, shared test fixtures, and validation of edge cases and JAX-specific features.

## Motivation

### Current State (Post RFC-001)
- **Tests implemented:** 4 (engine: 3, HTTP: 1)
- **Coverage:** Basic correctness, batching, optimization verification
- **Gaps identified by PyTorch comparison:**
  - No edge case validation (empty inputs, invalid types, unicode)
  - No error handling tests
  - No JAX-specific feature tests (bf16/fp32 stability, sharding)
  - No performance benchmarks
  - No protocol validation tests
  - Test code duplication (no shared fixtures)

### Problems
1. **Incomplete edge case coverage** - Missing validation for empty labels, negative IDs, mixed types
2. **No performance tracking** - Can't detect regressions in throughput/latency
3. **Test duplication** - Each test reimplements setup (engine init, tokenizer, assertions)
4. **No CI gating strategy** - All tests mixed together (fast deterministic + slow network-dependent)
5. **JAX-specific features untested** - bf16 stability, multi-device sharding not validated
6. **Protocol layer untested** - HTTP schema validation, serialization not covered

### Goals
1. Achieve **95%+ line coverage** on Score API code paths
2. **Separate CI tests** (fast, deterministic) from **perf tests** (slow, analysis)
3. **Shared fixtures** to eliminate duplication and ensure consistency
4. **Document behavior** for edge cases (empty inputs, invalid types)
5. **JAX-specific validation** (dtype stability, sharding correctness)
6. **Performance baseline** for regression detection

## Proposed Solution

### File Structure

```
python/sgl_jax/test/
├── score_test_utils.py          # NEW: Shared fixtures & helpers

test/srt/
├── test_score_api.py            # EXPAND: Core engine tests
├── test_score_api_edge_cases.py # NEW: Validation & corner cases
├── test_score_api_jax_features.py # NEW: JAX-specific (bf16, sharding)
├── bench_score.py               # NEW: Performance benchmarks
└── openai_server/basic/
    ├── test_openai_server.py    # EXPAND: HTTP endpoint tests
    └── test_protocol.py         # EXPAND: Protocol validation
```

**Gating strategy:**
- **CI (default):** `test_score_api.py` + `test_score_api_edge_cases.py` + HTTP tests
- **CI (nightly):** Above + HF reference tests (requires network)
- **Local/TPU-multi:** `test_score_api_jax_features.py` (multi-device)
- **Perf only:** `bench_score.py` (not in CI by default)

### Shared Fixtures Module

**New file:** `python/sgl_jax/test/score_test_utils.py`

Design principles:
- Importable as `from sgl_jax.test.score_test_utils import ...`
- Matches existing pattern in `sgl_jax.test.test_utils`
- Eliminates duplication across test files
- Provides canonical test data and assertions

**API surface:**

```python
# python/sgl_jax/test/score_test_utils.py
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class ScoreTestConfig:
    """Configuration for score API tests"""
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "tpu"
    dtype: str = "bfloat16"
    download_dir: str = "/dev/shm"
    seed: int = 3

def build_engine(config: ScoreTestConfig):
    """Initialize engine with test configuration"""
    # Aligned with test/srt/test_score_api.py defaults
    ...

def get_tokenizer(model: str):
    """Get tokenizer for model"""
    ...

def get_label_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    """Convert label tokens to IDs"""
    ...

def default_query_items():
    """Canonical query/items for reuse across tests"""
    return {
        "query": "I pledge allegiance",
        "items": [" to", " of", " for"],
        "label_tokens": [" to", " of", " for"]
    }

def assert_scores_shape(scores, num_items: int, num_labels: int):
    """Validate score output dimensions"""
    assert len(scores) == num_items
    for score_list in scores:
        assert len(score_list) == num_labels

def assert_scores_probs(scores, apply_softmax: bool):
    """Validate score probability constraints"""
    for score_list in scores:
        # All values in valid range
        assert all(0 <= s <= 1 for s in score_list)

        if apply_softmax:
            # Softmax: sum ≈ 1
            assert abs(sum(score_list) - 1.0) < 1e-6
        else:
            # Logprob: no sum constraint
            pass

def compute_hf_reference_scores(
    model: str,
    query: str,
    items: List[str],
    label_token_ids: List[int],
    item_first: bool = False,
    apply_softmax: bool = False
) -> List[List[float]]:
    """
    Compute reference scores using HuggingFace.

    Gated by SGLANG_JAX_RUN_HF_REFERENCE env var.
    Requires torch + transformers installed.
    """
    if not should_run_hf_reference():
        return None
    ...

def should_run_hf_reference() -> bool:
    """Check if HF reference tests should run"""
    return os.getenv("SGLANG_JAX_RUN_HF_REFERENCE") == "1"

def skip_if_no_multidevice():
    """Skip test if < 2 JAX devices available"""
    import jax
    if len(jax.devices()) < 2:
        pytest.skip("Requires multi-device setup")

def skip_if_no_hf_reference():
    """Skip test if HF reference not enabled"""
    if not should_run_hf_reference():
        pytest.skip("Set SGLANG_JAX_RUN_HF_REFERENCE=1 to run")
```

**Benefits:**
- DRY: No duplication of engine setup, tokenizer loading, assertions
- Consistency: All tests use same canonical data and validation logic
- Maintainability: Change fixture once, all tests benefit
- Documentation: Fixtures serve as examples of proper usage

## Test Matrix

### CI Tests (Fast, Deterministic)

#### Core Engine Tests (`test/srt/test_score_api.py`)

Expand existing file with:

1. **test_score_batch_handling** (existing, expand)
   - Batch sizes: 1, 2, 4, 8, 16
   - Validate shape + probability constraints

2. **test_score_text_input** (new)
   - Query + items as strings
   - Validate tokenization path

3. **test_score_token_input** (new)
   - Query as `list[int]`, items as `list[list[int]]`
   - Validate direct token path

4. **test_score_apply_softmax_true** (new)
   - Verify probabilities sum to 1.0
   - Check all values in [0, 1]

5. **test_score_apply_softmax_false** (new)
   - Verify no sum constraint
   - Check logprob range

6. **test_score_item_first_false** (new)
   - Default: `query + item` concatenation
   - Sanity check on order

7. **test_score_item_first_true** (new)
   - Reversed: `item + query` concatenation
   - Verify different results than item_first=False

8. **test_score_different_label_tokens** (new)
   - Label token counts: 1, 2, 4, 8, 16
   - Validate output dimensions scale correctly

9. **test_score_single_item** (new)
   - Single item in batch
   - Edge case for batching logic

10. **test_score_request_construction** (existing)
    - Verify max_new_tokens=0 (prefill-only)
    - Verify token_ids_logprob set
    - Verify return_logprob=True
    - Verify stream=False

11. **test_score_determinism** (new)
    - Same input → identical scores
    - Multiple runs with same seed

12. **test_score_default_params** (new)
    - Verify apply_softmax defaults to False
    - Verify item_first defaults to False

#### Edge Case Tests (`test/srt/test_score_api_edge_cases.py`)

New file for validation corner cases:

1. **test_score_empty_items** (new)
   - `items=[]` should raise `ValueError`
   - Clear error message

2. **test_score_empty_label_token_ids** (new)
   - `label_token_ids=[]` should raise `ValueError`
   - Can't compute scores over zero labels

3. **test_score_negative_token_ids** (new)
   - Negative IDs in label_token_ids should raise `ValueError`
   - Never valid in vocabulary

4. **test_score_token_ids_exceeds_vocab** (existing behavior)
   - Token ID >= vocab_size should raise error
   - Already implemented, add explicit test

5. **test_score_mixed_input_types_raises** (new)
   - Text query + token items → ValueError
   - Token query + text items → ValueError
   - Document this is intentionally rejected

6. **test_score_duplicate_label_tokens** (new)
   - Duplicates in label_token_ids
   - Should work (return scores for each occurrence)

7. **test_score_unicode_handling** (new)
   - Unicode in query/items
   - Emoji, non-ASCII characters

8. **test_score_whitespace_handling** (new)
   - Leading/trailing whitespace
   - Multiple spaces

9. **test_score_ordering_preserved** (new)
   - Output order matches input items order
   - Validate indices align

10. **test_score_invalid_types** (new)
    - items not list → TypeError
    - label_token_ids not list → TypeError
    - Mixed valid/invalid in list → TypeError

#### HTTP Endpoint Tests (`test/srt/openai_server/basic/test_openai_server.py`)

Expand `TestOpenAIV1Score` class:

1. **test_score_text_input** (existing)
   - POST /v1/score with text inputs
   - Validate response format

2. **test_score_token_input** (new)
   - Query/items as token IDs
   - Validate API accepts both

3. **test_score_usage_info** (new)
   - Response includes usage field
   - prompt_tokens > 0
   - completion_tokens == 0 (scoring has no completion)

4. **test_score_error_handling** (new)
   - Invalid label_token_ids → 400 error
   - Missing required fields → 400 error
   - Wrong types → 400 error
   - Validate error schema

5. **test_score_default_fields** (new)
   - Response includes "object": "scoring"
   - Response includes "model" field
   - Created timestamp present

6. **test_score_ordering** (new)
   - Scores array order matches items order
   - Validate indices

#### Protocol Tests (`test/srt/openai_server/basic/test_protocol.py`)

Add `TestScoringProtocol` class:

1. **test_scoring_request_validation** (new)
   - Required fields: query, items, label_token_ids
   - Missing field → validation error

2. **test_scoring_request_defaults** (new)
   - apply_softmax defaults to False
   - item_first defaults to False

3. **test_scoring_request_accepts_token_and_text** (new)
   - Union types for query/items
   - Both str and list[int] accepted

4. **test_scoring_response_serialization** (new)
   - exclude_none=True applied
   - No null fields in JSON

### CI Tests (Nightly, Requires Network)

#### HF Reference Tests (`test/srt/test_score_api.py`)

1. **test_score_consistency_with_hf** (existing)
   - Compare SGLang vs HuggingFace
   - < 1% difference tolerance
   - Gated by `SGLANG_JAX_RUN_HF_REFERENCE=1`

### Local/TPU-Multi Tests

#### JAX-Specific Tests (`test/srt/test_score_api_jax_features.py`)

New file for JAX-specific validation:

1. **test_score_numerical_stability** (new)
   - bf16 vs fp32 comparison
   - Acceptable tolerance for bf16 precision loss
   - Skip if single dtype environment

2. **test_score_sharding_correctness** (new)
   - Multi-device sharding
   - Results match single-device
   - Gated by `skip_if_no_multidevice()`

3. **test_score_with_prefix_caching** (new)
   - Verify cache hit/miss metrics
   - Requires access to meta_info
   - May need patching or testing with modified engine

### Performance Tests (Not in CI)

#### Benchmark Tool (`test/srt/bench_score.py`)

New standalone tool for performance analysis:

```python
# test/srt/bench_score.py
"""
Score API performance benchmark.

Usage:
    python test/srt/bench_score.py --batch-sizes 1,2,4,8,16,32 --num-labels 2,4,8
"""

Features:
- Throughput measurement (scores/sec)
- Latency percentiles (p50, p95, p99)
- Batch size scaling analysis
- Label count scaling analysis
- CSV output for tracking over time
- Comparison with baseline (detect regressions)
```

**Metrics tracked:**
- Throughput (items scored / second)
- Latency (ms per request)
- Batch scaling efficiency
- Label count scaling
- Memory usage

**Not in CI because:**
- Takes 5-10 minutes to run full suite
- Results vary based on hardware
- Requires manual analysis/visualization
- Used for investigation, not regression gating

## Behavior Validation Decisions

### Decision 1: Empty `label_token_ids` Should Raise

**Current behavior:** Will error during softmax with cryptic message

**Proposed:**
```python
if not label_token_ids or len(label_token_ids) == 0:
    raise ValueError(
        "label_token_ids cannot be empty. "
        "At least one label token ID is required for scoring."
    )
```

**Rationale:**
- Logical error: can't compute scores over zero labels
- Fail fast with clear message
- Matches input validation best practices

**Test:** `test_score_empty_label_token_ids`

### Decision 2: Negative Token IDs Should Raise

**Current behavior:** Only checks `>= vocab_size`, not `< 0`

**Proposed:**
```python
if any(tid < 0 for tid in label_token_ids):
    raise ValueError(
        f"label_token_ids cannot contain negative values. "
        f"Got: {[tid for tid in label_token_ids if tid < 0]}"
    )
```

**Rationale:**
- Negative IDs are never valid in vocabulary
- Likely indicates a bug in caller's code
- Fail fast to aid debugging

**Test:** `test_score_negative_token_ids`

### Decision 3: Mixed Input Types Should Raise

**Current behavior:** Implicitly rejects (crashes or undefined)

**Proposed:**
```python
query_is_text = isinstance(query, str)
items_is_text = isinstance(items[0], str)

if query_is_text != items_is_text:
    raise ValueError(
        "query and items must both be text (str) or both be tokens (list[int]). "
        f"Got query type: {type(query).__name__}, items[0] type: {type(items[0]).__name__}"
    )
```

**Rationale:**
- Mixed types create ambiguity in tokenization
- Clearer API contract
- Better error message

**Test:** `test_score_mixed_input_types_raises`

## Implementation Plan

### Phase 1: Shared Fixtures (Week 1)
- [ ] Create `python/sgl_jax/test/score_test_utils.py`
- [ ] Implement `ScoreTestConfig` dataclass
- [ ] Implement `build_engine()`, `get_tokenizer()`, `get_label_token_ids()`
- [ ] Implement assertion helpers (`assert_scores_shape`, `assert_scores_probs`)
- [ ] Implement HF reference helper (gated)
- [ ] Implement skip decorators (`skip_if_no_multidevice`, etc.)
- [ ] Write docstrings and usage examples
- [ ] Refactor existing tests to use fixtures (validate no regression)

### Phase 2: Core Engine Tests (Week 2)
- [ ] Expand `test/srt/test_score_api.py` with 12 core tests
- [ ] Use shared fixtures throughout
- [ ] Add behavior validation to `tokenizer_manager.py`:
  - [ ] Empty label_token_ids check
  - [ ] Negative token ID check
  - [ ] Mixed input type check
- [ ] Run full suite on TPU
- [ ] Validate coverage increase (expect 80% → 95%+)

### Phase 3: Edge Cases (Week 3)
- [ ] Create `test/srt/test_score_api_edge_cases.py`
- [ ] Implement 10 edge case tests
- [ ] Add to CI suite in `test/srt/run_suite.py`
- [ ] Document edge case behavior in code comments
- [ ] Update API documentation with validation rules

### Phase 4: HTTP + Protocol (Week 3)
- [ ] Expand `TestOpenAIV1Score` with 6 HTTP tests
- [ ] Create `TestScoringProtocol` with 4 protocol tests
- [ ] Add to e2e test suite
- [ ] Validate error schemas match OpenAI API conventions

### Phase 5: JAX-Specific (Week 4)
- [ ] Create `test/srt/test_score_api_jax_features.py`
- [ ] Implement 3 JAX-specific tests
- [ ] Add to local/nightly suite (not default CI)
- [ ] Document multi-device testing setup

### Phase 6: Performance (Week 4)
- [ ] Create `test/srt/bench_score.py`
- [ ] Implement throughput measurement
- [ ] Implement latency percentiles
- [ ] Add batch/label scaling analysis
- [ ] Create baseline results CSV
- [ ] Document usage in runbook

### Phase 7: CI Integration (Week 5)
- [ ] Update `test/srt/run_suite.py`:
  - [ ] Add core + edge case tests to default suite
  - [ ] Create nightly suite with HF reference
  - [ ] Exclude JAX-specific tests from default
  - [ ] Exclude perf tests from all suites
- [ ] Update GitHub Actions workflow (RFC-002)
- [ ] Add HF reference env var to nightly runs
- [ ] Validate CI runtime < 10 minutes

## Testing Strategy

### Validation

**Phase 1:** Fixtures must not break existing tests
```bash
# Before adding new tests, validate fixtures work
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
# All 3 existing tests must pass
```

**Phase 2:** Core tests must pass on TPU
```bash
# Full core suite
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
# Expect 12 tests, 100% pass rate, < 5 min runtime
```

**Phase 3:** Edge cases must validate error handling
```bash
python3 -m unittest test.srt.test_score_api_edge_cases -v
# All 10 tests pass, each validates specific error case
```

**Phase 4:** HTTP tests must validate full stack
```bash
# Start server + run HTTP tests
python3 -m unittest test.srt.openai_server.basic.test_openai_server.TestOpenAIV1Score -v
# Validate end-to-end flow
```

**Phase 5:** JAX tests only on multi-device
```bash
# Only run if len(jax.devices()) > 1
python3 -m unittest test.srt.test_score_api_jax_features -v
# Tests skip gracefully on single device
```

**Phase 6:** Perf baseline
```bash
python test/srt/bench_score.py --batch-sizes 1,2,4,8 --num-labels 2,4,8 --output baseline.csv
# Establish baseline for future comparison
```

### Monitoring

**Coverage tracking:**
```bash
pytest --cov=python/sgl_jax/srt/managers/tokenizer_manager \
       --cov-report=term-missing \
       test/srt/test_score_api*.py
# Target: 95%+ line coverage on score_request()
```

**CI metrics:**
- Test count: 4 → ~30
- Coverage: ~80% → 95%+
- CI runtime: < 10 minutes (without perf tests)
- Nightly runtime: < 15 minutes (with HF reference)

## Cost Analysis

**Development Cost:**
- Phase 1 (fixtures): 4 hours
- Phase 2 (core tests): 6 hours
- Phase 3 (edge cases): 4 hours
- Phase 4 (HTTP/protocol): 3 hours
- Phase 5 (JAX-specific): 3 hours
- Phase 6 (perf): 4 hours
- Phase 7 (CI integration): 2 hours
- **Total:** ~26 hours over 5 weeks

**Ongoing Cost:**
- CI runtime: +5 min/run (30 new tests)
- TPU cost: 5 min × 30 runs/month × $0.64/hr = $1.60/month
- Maintenance: ~1 hour/month (test updates)

**ROI:**
- Prevented bugs: ~2-3 per month (based on RFC-001 results)
- Debug time saved: 3 bugs × 2 hours = 6 hours/month
- **Monthly ROI:** 600% (6 hours saved vs 1 hour maintenance)

## Open Questions

- [ ] Should we add load testing for high-throughput scenarios (>100 items/batch)?
- [ ] Need tests for concurrent requests (thread safety)?
- [ ] Should bench_score.py integrate with CI performance tracking tools?
- [ ] Add memory profiling to perf tests?
- [ ] Should we test with multiple model sizes (1B, 7B, 70B)?
- [ ] Add tests for streaming (even though scoring doesn't stream)?

## Alternatives Considered

### Alternative 1: Monolithic Test File

**Description:** Put all 30+ tests in single `test_score_api.py`

**Pros:**
- Single file to maintain
- No confusion about where tests go

**Cons:**
- 2000+ line file (hard to navigate)
- Can't selectively run CI vs perf tests
- Slow test discovery
- Merge conflicts

**Why rejected:** File separation by purpose improves maintainability and enables selective test execution.

### Alternative 2: No Shared Fixtures

**Description:** Each test file implements its own setup

**Pros:**
- No shared dependency
- Tests fully independent

**Cons:**
- Massive duplication (each test reimplements engine setup)
- Inconsistent test data across files
- Hard to change defaults (must update 30+ tests)

**Why rejected:** DRY principle. Fixtures eliminate duplication and ensure consistency.

### Alternative 3: All Tests in CI

**Description:** Run all tests (including perf) in every CI run

**Pros:**
- Complete validation every time
- Catch all regressions

**Cons:**
- 20+ minute CI runs (too slow)
- Flaky due to network-dependent HF reference
- Perf results meaningless without dedicated hardware

**Why rejected:** Need fast CI feedback loop. Separate nightly/perf tests from default CI.

### Alternative 4: Skip Behavior Validation

**Description:** Don't validate edge cases (empty inputs, negative IDs)

**Pros:**
- Less validation code
- Slightly faster happy path

**Cons:**
- Cryptic errors for invalid input
- Users waste time debugging
- API contract unclear

**Why rejected:** Input validation is essential for good API design. Clear error messages save user time.

## Success Metrics

**Phase 1-3 Complete:**
- ✅ 25+ tests passing on TPU
- ✅ 95%+ line coverage on score_request()
- ✅ All edge cases validated
- ✅ CI runtime < 10 minutes

**Phase 4-5 Complete:**
- ✅ HTTP/protocol tests passing
- ✅ JAX-specific tests documented
- ✅ Multi-device validation available

**Phase 6-7 Complete:**
- ✅ Performance baseline established
- ✅ CI integration complete
- ✅ Nightly suite with HF reference

## Timeline

- **Week 1:** Shared fixtures + refactor existing tests
- **Week 2:** Core engine tests (12 new)
- **Week 3:** Edge cases (10 new) + HTTP/protocol (10 new)
- **Week 4:** JAX-specific (3 new) + perf tool
- **Week 5:** CI integration + documentation

**Total:** 5 weeks part-time or 2-3 weeks full-time

## References

- RFC-001: Score API Comprehensive Tests (baseline)
- RFC-002: CI/CD for TPU Testing (infrastructure)
- ADR-001: Pure Python Softmax Decision
- PyTorch reference: `sglang/test/registered/core/test_score_api.py`
- Investigation: Score API PyTorch vs JAX comparison
- Runbook: Debugging TPU test failures
