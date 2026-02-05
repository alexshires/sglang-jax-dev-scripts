# ADR-001: SciPy Softmax in TokenizerManager

| | |
|------------|------|
| **Date** | 2026-01-29 |
| **Status** | Implemented |
| **Deciders** | Engineering Team |
| **Related** | [RFC-001](../rfcs/001-score-api-comprehensive-tests.md) |

## Context

The `/v1/score` API in `tokenizer_manager.py` needs to apply softmax to convert log probabilities to normalized probabilities. The initial implementation used `jax.nn.softmax()`, which caused runtime failures on TPU.

### Technical Constraints

**Architecture:**
- TokenizerManager runs in the **main process** (device-agnostic)
- Scheduler runs in a **subprocess** with exclusive TPU access via `mp.Process`
- JAX can only initialize devices once per process
- Scheduler subprocess claims TPU exclusively for model execution

**The Problem:**
```python
# In tokenizer_manager.py (main process)
if apply_softmax:
    score_list = jax.nn.softmax(jnp.asarray(score_list), axis=0).tolist()
```

**Error:**
```
RuntimeError: TPU is already in use by process with pid 12345
```

**Why it fails:**
- Even `jax.devices('cpu')` triggers JAX initialization
- JAX initialization detects TPU in use by subprocess
- Conflict occurs regardless of target device

## Decision

Use **SciPy's softmax** implementation in `tokenizer_manager.py`:

```python
from scipy.special import softmax

if apply_softmax:
    score_list = softmax(score_list).tolist()
```

**Key aspects:**
- No JAX dependencies in TokenizerManager
- Uses well-tested, numerically stable library function
- SciPy is already a project dependency
- Concise and readable

## Consequences

### Positive
- **Eliminates device conflicts:** TokenizerManager stays device-agnostic
- **Numerically stable:** SciPy handles overflow/underflow correctly
- **Battle-tested:** Uses established library rather than custom implementation
- **Concise:** Single function call vs manual implementation
- **No performance impact:** Softmax over small arrays (typically <10 items)

### Negative
- **Not using JAX:** Mixing paradigms (JAX for model, SciPy for utils)

### Neutral
- **Performance:** Negligible difference for small arrays (<100 elements)
- **Maintenance:** Uses standard library, unlikely to need changes

## Alternatives Considered

### Alternative 1: Pure Python
**Description:**
```python
import math
max_logprob = max(score_list)
exp_scores = [math.exp(x - max_logprob) if x != float("-inf") else 0.0
              for x in score_list]
sum_exp = sum(exp_scores)
score_list = [x / sum_exp if sum_exp > 0 else 0.0 for x in exp_scores]
```

**Pros:**
- No external dependencies
- Full control over implementation

**Cons:**
- Reimplements standard operation
- More code to maintain
- Need to handle edge cases manually

**Why rejected:** SciPy provides a well-tested implementation with proper edge case handling. No need to reinvent the wheel.

### Alternative 2: NumPy
**Description:**
```python
import numpy as np
score_list = np.exp(score_list - np.max(score_list))
score_list = (score_list / score_list.sum()).tolist()
```

**Pros:**
- Concise and readable
- Numerically stable
- No device conflicts

**Cons:**
- Still requires manual implementation of softmax formula
- SciPy provides a dedicated function

**Why rejected:** SciPy's `softmax` is more direct and handles edge cases automatically.

### Alternative 3: JAX with Explicit CPU Device
**Description:**
```python
import jax
with jax.default_device(jax.devices('cpu')[0]):
    score_list = jax.nn.softmax(jnp.asarray(score_list), axis=0).tolist()
```

**Pros:**
- Uses JAX consistently
- Forces CPU execution

**Cons:**
- Still triggers JAX initialization
- Still detects TPU in use by subprocess
- **Tested and failed with same error**

**Why rejected:** Empirically failed during testing. JAX initialization happens before device selection.

### Alternative 4: Move Softmax to Scheduler Subprocess
**Description:**
Apply softmax inside the Scheduler subprocess where JAX/TPU is available.

**Pros:**
- Can use JAX natively
- No device conflicts

**Cons:**
- Violates architecture (TokenizerManager should stay device-agnostic)
- Requires IPC for simple operation
- Adds latency and complexity
- Breaks separation of concerns

**Why rejected:** Architectural violation. TokenizerManager should not depend on Scheduler's device access.

## Implementation Notes

**Location:** `python/sgl_jax/srt/managers/tokenizer_manager.py`

**Import added:**
```python
from scipy.special import softmax
```

**Removed imports:**
```python
# Removed (no longer needed):
import jax
import jax.numpy as jnp
```

## Validation

**Test coverage:**
- `test_score_consistency`: Validates softmax output matches HuggingFace (< 1% diff)
- `test_score_batch_handling`: Confirms probabilities sum to 1.0 (6 decimal places)

**Numerical validation:**
```python
# Example from test run
HuggingFace reference: [0.8234, 0.1234, 0.0532]
SGLang JAX output:     [0.8231, 0.1236, 0.0533]
Difference:            [0.036%, 0.162%, 0.188%]  # All < 1%
```

**Runtime impact:**
- Before fix: Tests failed with device conflict
- After fix: All tests pass
- Softmax execution time: Negligible (<1ms for typical array sizes)

## References

- [RFC-001: Score API Comprehensive Tests](../rfcs/001-score-api-comprehensive-tests.md)
- PyTorch implementation: `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- [Investigation: TokenizerManager Architecture](../investigations/tokenizer-manager-architecture.md)
- Test file: `test/srt/test_score_api.py`
- SciPy softmax: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
