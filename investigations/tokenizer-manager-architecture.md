# Investigation: TokenizerManager Architecture

| | |
|------------|------|
| **Date** | 2026-01-29 |
| **Status** | Complete |
| **Related** | [ADR-001](../decisions/001-pure-python-softmax.md), [RFC-001](../rfcs/001-score-api-comprehensive-tests.md) |

## Summary

Investigation into the multi-process architecture of sglang-jax to understand why JAX operations in TokenizerManager cause device conflicts.

## Architecture Overview

### Process Layout

```
Main Process
├── TokenizerManager (device-agnostic)
│   ├── Handles tokenization
│   ├── Processes scoring requests
│   └── Returns results to users
└── Scheduler (subprocess via mp.Process)
    ├── Claims TPU exclusively
    ├── Runs JAX model inference
    └── Returns logprobs to TokenizerManager
```

## Key Findings

### 1. TokenizerManager Runs in Main Process

**Evidence:**
```python
# In launch_server.py or similar
tokenizer_manager = TokenizerManager(...)

# Scheduler starts in subprocess
scheduler_process = mp.Process(target=scheduler.run)
scheduler_process.start()
```

**Implications:**
- TokenizerManager has no direct TPU access
- Must remain device-agnostic
- Cannot use JAX operations that trigger device initialization

### 2. Scheduler Has Exclusive TPU Access

**Evidence:**
```python
# In scheduler subprocess
jax.distributed.initialize()  # Claims TPU
model = load_model()           # Loads on TPU
```

**Implications:**
- TPU is locked to Scheduler subprocess
- Main process cannot access TPU
- Even JAX CPU operations in main process cause conflicts

### 3. JAX Initialization is Global

**Key Discovery:**
```python
# Even this fails in main process:
with jax.default_device(jax.devices('cpu')[0]):
    result = jax.nn.softmax(array)

# Error: RuntimeError: TPU is already in use by process with pid XXXXX
```

**Why:**
1. `jax.devices('cpu')` triggers JAX initialization
2. JAX initialization scans all available devices
3. Sees TPU is locked by subprocess
4. Raises conflict error before device selection happens

### 4. Communication Flow

```
User Request
    ↓
TokenizerManager.score_request()
    ↓
Creates GenerateReqInput with token_ids_logprob
    ↓
Sends to Scheduler (via IPC)
    ↓
Scheduler runs JAX inference on TPU
    ↓
Returns logprobs to TokenizerManager
    ↓
TokenizerManager applies softmax (MUST BE DEVICE-AGNOSTIC)
    ↓
Returns normalized scores to user
```

## Design Rationale

### Why This Architecture?

**Benefits:**
1. **Isolation:** Model inference isolated in subprocess
2. **Clean shutdown:** Can kill subprocess without affecting server
3. **Resource control:** Scheduler exclusively manages TPU
4. **Fault tolerance:** Scheduler crash doesn't take down entire server

**Trade-offs:**
1. **IPC overhead:** Communication between processes
2. **Device constraints:** Main process can't use JAX/TPU
3. **Debugging complexity:** Multi-process debugging harder

## Comparison with PyTorch Version

### PyTorch Architecture

```
Main Process
├── TokenizerManager
│   ├── Uses Python/NumPy for utils
│   └── No torch.nn in tokenization logic
└── Scheduler (subprocess)
    └── Uses torch.nn for model inference
```

**Key insight:** PyTorch version also keeps TokenizerManager device-agnostic. Uses Python/NumPy for utilities like softmax.

### Architectural Similarity

Both versions:
- Isolate model execution in subprocess
- Keep TokenizerManager device-agnostic
- Use IPC for logprob communication
- Apply probability normalization in main process

## Lessons Learned

### 1. Respect Process Boundaries

**Problem:** Tried to use JAX in main process
**Solution:** Use pure Python for device-agnostic operations

### 2. Framework Initialization is Global

**Problem:** Assumed we could force CPU device
**Reality:** JAX initialization scans all devices first

### 3. Follow Reference Implementation

**Problem:** JAX version used `jax.nn.softmax()`
**Solution:** PyTorch version uses Python - we should too

## Testing Implications

### Tests Must Respect Architecture

**Correct:**
```python
# Test runs in main process
# Launches server (which starts Scheduler subprocess)
# Makes requests to TokenizerManager
# TokenizerManager uses pure Python utils
```

**Incorrect:**
```python
# Test tries to use JAX directly
# Conflicts with Scheduler's TPU access
```

### Why Test Isolation Matters

```python
@classmethod
def tearDownClass(cls):
    cls.runner.shutdown()
    # DON'T: jax.clear_backends()  # Would conflict with Scheduler
    # DO: Let subprocess cleanup handle JAX
```

## References

- Source: `python/sgl_jax/srt/managers/tokenizer_manager.py`
- Source: `python/sgl_jax/srt/managers/scheduler.py`
- PyTorch comparison: `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- JAX docs: https://jax.readthedocs.io/en/latest/multi_process.html

## Open Questions

- [ ] Could we use a shared memory approach instead of IPC?
- [ ] Would thread-based architecture work better than multiprocessing?
- [ ] Can we benchmark IPC overhead vs local computation?

## Recommendations

1. **Document architecture clearly** in main repo README
2. **Add architecture diagram** to docs
3. **Lint rule:** Prevent JAX imports in tokenizer_manager.py
4. **Code comments:** Explain why pure Python is used
