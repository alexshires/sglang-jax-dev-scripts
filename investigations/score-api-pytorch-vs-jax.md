# Investigation: Score API - PyTorch vs JAX Comparison

**Date:** 2026-01-29
**Status:** Complete
**Related:** RFC-001, ADR-001

## Summary

Comprehensive comparison of `/v1/score` API implementation between PyTorch (sglang) and JAX (sglang-jax) versions to identify missing features, bugs, and optimization opportunities.

## Methodology

**Files Compared:**
- **PyTorch:** `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- **JAX:** `python/sgl_jax/srt/managers/tokenizer_manager.py`

**Approach:**
- Side-by-side code review
- Test coverage comparison
- API parameter validation
- Performance optimization analysis

## High-Level Comparison

| Aspect | PyTorch | JAX | Status |
|--------|---------|-----|--------|
| **Core Functionality** | ✅ Working | ✅ Working | Match |
| **Test Coverage** | 17 tests | 0 tests → 4 tests | Fixed |
| **HTTP Integration** | ✅ Tested | ❌ → ✅ | Fixed |
| **Batching** | ✅ Validated | ❌ → ✅ | Fixed |
| **Softmax Impl** | Pure Python | JAX → Python | Fixed |
| **Prefill-only** | max_new_tokens=0 | 1 → 0 | Fixed |
| **Error Handling** | ✅ Robust | ❌ → ✅ | Fixed |

## Detailed Findings

### 1. Softmax Implementation

**PyTorch Version:**
```python
# Pure Python implementation
if apply_softmax:
    max_logprob = max(score_list)
    exp_scores = [math.exp(x - max_logprob) if x != float("-inf") else 0.0
                  for x in score_list]
    sum_exp = sum(exp_scores)
    score_list = [x / sum_exp if sum_exp > 0 else 0.0 for x in exp_scores]
```

**JAX Version (Original):**
```python
# Used JAX (caused device conflicts)
if apply_softmax:
    score_list = jax.nn.softmax(jnp.asarray(score_list), axis=0).tolist()
```

**Issue:**
- JAX initialization conflicts with TPU in Scheduler subprocess
- Even CPU device causes conflict

**Fix:**
- Adopted PyTorch's pure Python approach
- See ADR-001 for rationale

**Status:** ✅ Fixed

### 2. Prefill-Only Optimization

**PyTorch Version:**
```python
# Line ~1205, ~1223
sampling_params={
    "max_new_tokens": 0,  # Prefill only
    "temperature": 0,
    # ...
}
```

**JAX Version (Original):**
```python
# Line 1242, 1260
sampling_params={
    "max_new_tokens": 1,  # WRONG: Runs decode phase
    "temperature": 0,
    # ...
}
```

**Issue:**
- `max_new_tokens=1` enables decode phase
- Decode phase unnecessary for scoring (only need prefill logprobs)
- Wastes compute and time

**Impact:**
```
Before: 159 seconds (prefill + decode)
After:  105 seconds (prefill only)
Improvement: 34% faster
```

**Fix:**
```python
sampling_params={"max_new_tokens": 0}
```

**Status:** ✅ Fixed

### 3. Error Handling

**PyTorch Version:**
```python
# Line ~1235
output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])
if not output_logprobs:
    raise RuntimeError(
        f"output_token_ids_logprobs is empty for request {result['meta_info'].get('id')}. "
        "This indicates token_ids_logprobs were not computed properly."
    )

for logprob, token_id, _ in output_logprobs[0]:
    # Safe: validated output_logprobs is not empty
```

**JAX Version (Original):**
```python
# Line ~1271
for logprob, token_id, _ in result["meta_info"].get("output_token_ids_logprobs", [])[0]:
    # UNSAFE: Direct [0] access can raise IndexError
```

**Issue:**
- No validation before accessing `[0]`
- Generic IndexError instead of informative error
- Harder to debug root cause

**Fix:**
```python
output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])
if not output_logprobs or len(output_logprobs) == 0:
    raise RuntimeError(
        f"output_token_ids_logprobs is empty for request {result['meta_info'].get('id', '<unknown>')}. "
        "This indicates token_ids_logprobs were not computed properly."
    )

for logprob, token_id, _ in output_logprobs[0]:
    # Now safe
```

**Status:** ✅ Fixed

### 4. Test Coverage

**PyTorch Tests:**
```
test/registered/core/test_score_api.py:
- test_score_simple()
- test_score_consistency()
- test_score_multi_prefix()
- test_score_batch()
- test_score_batch_multi_prefix()
- test_score_item_first_false()
- test_score_item_first_true()
- test_score_with_empty_label()
- test_score_with_invalid_tokens()
- test_score_return_format()
- test_score_numerical_stability()
- test_score_large_batch()
- test_score_edge_cases()
- test_score_parallel_requests()

test/registered/openai_server/basic/test_openai_server.py:
- test_score_text_input()
- test_score_token_input()
- test_score_batch_request()

Total: 17 tests
```

**JAX Tests (Original):**
```
test/srt/test_score_api.py: (didn't exist)

Total: 0 tests
```

**JAX Tests (After RFC-001):**
```
test/srt/test_score_api.py:
- test_score_consistency()        # Validates against HuggingFace
- test_score_batch_handling()     # Tests batch sizes 1,2,4,8
- test_score_request_construction() # Validates optimizations

test/srt/openai_server/basic/test_openai_server.py:
- test_score_text_input()         # HTTP endpoint test

Total: 4 tests (Tier 1 Critical)
```

**Status:** ✅ Fixed (initial coverage achieved, more tests planned)

## Similarities (Good Design)

### 1. API Surface

Both versions support identical parameters:
```python
{
    "query": str | list[int],
    "items": list[str] | list[list[int]],
    "label_token_ids": list[int],
    "item_first": bool,
    "return_logprob": bool
}
```

### 2. Request Construction

Both create GenerateReqInput with:
- `token_ids_logprob` for selective logprob extraction
- `return_logprob=True`
- `stream=False`
- `temperature=0`

### 3. Batch Handling

Both handle batching identically:
- Iterate over items
- Create one request per item
- Collect and return all results

### 4. Architecture

Both isolate model execution in subprocess:
- TokenizerManager in main process
- Scheduler in subprocess with device access
- IPC for logprob communication

## Performance Comparison

### Test Runtime (TPU v6e-1)

**JAX Version:**
- Before fixes: Failed (device conflict)
- After fixes: 104.9 seconds for 3 tests

**PyTorch Version:**
- Not tested on TPU (GPU-focused)
- CPU reference: ~30 seconds for similar tests

**Conclusion:** JAX version competitive once bugs fixed

### Optimization Effectiveness

**max_new_tokens=0:**
- Both versions use this optimization
- Avoids decode loop entirely
- ~30-50% speedup vs max_new_tokens=1

**token_ids_logprob:**
- Both versions use selective logprob extraction
- Only compute logprobs for label tokens
- Saves memory and computation

## Lessons Learned

### 1. Reference Implementation is Valuable

**Discovery:** Many bugs found by comparing with PyTorch
- Softmax: Pure Python matches PyTorch
- max_new_tokens: PyTorch had correct value
- Error handling: PyTorch more robust

**Takeaway:** Always check reference implementation when porting

### 2. Testing Matters

**Discovery:** PyTorch has 17 tests, JAX had 0
- Tests caught all 3 bugs immediately
- Would have caught bugs before production

**Takeaway:** Test coverage is not optional

### 3. Architecture Drives Implementation

**Discovery:** Multi-process architecture constrains design
- TokenizerManager must be device-agnostic
- Can't use JAX in main process
- Pure Python is the right choice

**Takeaway:** Understand architecture before coding

## Remaining Gaps

### Features in PyTorch Not Yet in JAX

1. **Extended test coverage:**
   - Edge cases (empty labels, invalid tokens)
   - Numerical stability tests
   - Large batch tests (>100 items)
   - Parallel request tests

2. **Error handling:**
   - More detailed error messages
   - Input validation
   - Edge case handling

3. **Documentation:**
   - Docstrings for score_request()
   - Usage examples
   - API reference

### Planned Improvements

- [ ] Achieve 95%+ test coverage (RFC-001)
- [ ] Add input validation
- [ ] Add performance benchmarks vs PyTorch
- [ ] Document scoring API in README

## References

- PyTorch source: `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- JAX source: `python/sgl_jax/srt/managers/tokenizer_manager.py`
- PyTorch tests: `sglang/test/registered/core/test_score_api.py`
- JAX tests: `test/srt/test_score_api.py`
- RFC-001: Score API Comprehensive Tests
- ADR-001: Pure Python Softmax Decision

## Conclusion

**Summary:**
- JAX version had 3 critical bugs
- All bugs were fixed by following PyTorch patterns
- Test coverage improved from 0 to 4 tests
- Runtime improved 34% with max_new_tokens fix
- API surface is feature-complete

**Next Steps:**
1. Expand test coverage to match PyTorch (17 tests)
2. Add CI/CD for regression prevention (RFC-002)
3. Benchmark performance vs PyTorch
4. Document scoring API usage
