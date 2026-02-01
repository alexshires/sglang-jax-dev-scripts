# Runbook: Debugging TPU Test Failures

**Last Updated:** 2026-01-29
**Maintainer:** Engineering Team
**Related:** [RFC-002](../rfcs/002-cicd-tpu-testing.md) (CI/CD)

## Overview

Step-by-step guide for debugging test failures on TPU VMs, whether running locally or in CI/CD.

## Prerequisites

- gcloud CLI installed and authenticated
- Access to GCP project with TPU quota
- SSH access to TPU VMs

## Quick Diagnostics

### 1. Check TPU VM Status

```bash
# List all TPU VMs
gcloud compute tpus tpu-vm list --zone=us-central1-b

# Check specific VM
gcloud compute tpus tpu-vm describe sglang-test-vm --zone=us-central1-b
```

**Look for:**
- State: `READY` (good) vs `CREATING`, `FAILED`, `TERMINATED`
- Health: `HEALTHY` vs `UNHEALTHY`
- API version: Should match your code (e.g., `v2-alpha-tpuv6e`)

### 2. Check Running Processes

```bash
# SSH and check processes
gcloud compute tpus tpu-vm ssh sglang-test-vm --zone=us-central1-b

# On TPU VM:
ps aux | grep python    # Look for hanging processes
nvidia-smi              # Check GPU usage (if applicable)
top                     # Check CPU/memory usage
```

### 3. Quick Test Run

```bash
# Run single test to isolate issue
gcloud compute tpus tpu-vm ssh sglang-test-vm --zone=us-central1-b \
  --command="cd ~/sglang-jax && source .venv/bin/activate && \
             python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency -v"
```

## Common Failure Scenarios

### Scenario 1: Device Conflict Error

**Symptom:**
```
RuntimeError: TPU is already in use by process with pid 12345
```

**Root Cause:**
- JAX operation in main process
- Scheduler subprocess has exclusive TPU access
- JAX initialization triggers device scan

**Diagnosis:**
```bash
# Check if JAX is being used in tokenizer_manager
cd ~/sglang-jax
grep -n "import jax" python/sgl_jax/srt/managers/tokenizer_manager.py

# Should return no results
```

**Fix:**
1. Remove JAX imports from tokenizer_manager.py
2. Use pure Python alternatives (e.g., softmax)
3. See ADR-001 for rationale

**Validation:**
```bash
# Test should pass after fix
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
```

### Scenario 2: max_new_tokens Performance Issue

**Symptom:**
- Tests pass but take 150+ seconds
- Expected: ~105 seconds

**Root Cause:**
- `max_new_tokens=1` instead of `0`
- Runs unnecessary decode phase

**Diagnosis:**
```bash
# Check tokenizer_manager for max_new_tokens
cd ~/sglang-jax
grep -n "max_new_tokens" python/sgl_jax/srt/managers/tokenizer_manager.py

# Look for score_request method (around line 1242, 1260)
```

**Fix:**
```python
# Change from:
sampling_params={"max_new_tokens": 1}

# To:
sampling_params={"max_new_tokens": 0}
```

**Validation:**
```bash
# Time the test
time python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# Should complete in ~105 seconds
```

### Scenario 3: Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Diagnosis:**
```bash
# Check installed packages
source .venv/bin/activate
pip list | grep transformers
```

**Fix:**
```bash
# Install missing dependencies
cd ~/sglang-jax
source .venv/bin/activate
pip install -e .

# Or install specific package
pip install transformers
```

**Validation:**
```bash
python3 -c "import transformers; print(transformers.__version__)"
```

### Scenario 4: Model Download Timeout

**Symptom:**
```
HTTPSConnectionPool: Read timed out
```

**Root Cause:**
- HuggingFace model download slow/failed
- Network issues
- Large model files

**Diagnosis:**
```bash
# Check HuggingFace cache
ls -lh ~/.cache/huggingface/hub/

# Test network to HuggingFace
curl -I https://huggingface.co
```

**Fix:**
```bash
# Pre-download model
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Llama-3.2-1B-Instruct'
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
"

# Increase timeout in test
# (Edit test file to add timeout parameter)
```

**Validation:**
```bash
# Model should be cached now
ls -lh ~/.cache/huggingface/hub/ | grep Llama-3.2-1B
```

### Scenario 5: Insufficient Memory

**Symptom:**
```
RuntimeError: Out of memory
SIGKILL (killed by OS)
```

**Diagnosis:**
```bash
# Check memory during test run
watch -n 1 'free -h'

# Check TPU memory
# (TPU-specific commands)
```

**Fix:**
1. Use smaller batch sizes in test
2. Clear cache between tests:
```python
@classmethod
def tearDownClass(cls):
    cls.runner.shutdown()
    jax.clear_caches()  # Clear JAX cache
```
3. Use smaller model for testing

**Validation:**
```bash
# Monitor memory during test
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v &
watch -n 1 'free -h'
```

### Scenario 6: HTTP Server Connection Failed

**Symptom:**
```
ConnectionError: Connection refused
http.client.RemoteDisconnected: Remote end closed connection
```

**Root Cause:**
- Server not started
- Server crashed during startup
- Port conflict

**Diagnosis:**
```bash
# Check if server is running
ps aux | grep "launch_server"

# Check port
netstat -tuln | grep 30000

# Check logs
tail -f /tmp/sglang_server.log
```

**Fix:**
```bash
# Kill any existing servers
pkill -f "launch_server"

# Check logs for startup errors
cat /tmp/sglang_server.log

# Ensure proper cleanup in tests
# (Add proper tearDown/tearDownClass methods)
```

**Validation:**
```bash
# Test server startup manually
python3 -m sglang_jax.launch_server --model meta-llama/Llama-3.2-1B-Instruct &

# Wait for startup
sleep 10

# Test endpoint
curl http://localhost:30000/v1/models

# Cleanup
pkill -f "launch_server"
```

## CI/CD Specific Issues

### GitHub Actions Timeout

**Symptom:**
```
Error: The operation was canceled
```

**Diagnosis:**
- Check workflow timeout setting (should be 20 min)
- Check test runtime (should be <10 min)

**Fix:**
```yaml
jobs:
  tpu-tests:
    timeout-minutes: 20  # Increase if needed
```

### GCP Authentication Failed

**Symptom:**
```
ERROR: (gcloud.auth.activate-service-account) Invalid token
```

**Diagnosis:**
```bash
# Check if secret is set in GitHub
gh secret list

# Should show: GCP_SERVICE_ACCOUNT_KEY
```

**Fix:**
1. Create service account key in GCP
2. Add to GitHub secrets:
```bash
gh secret set GCP_SERVICE_ACCOUNT_KEY < key.json
```

### TPU Cleanup Failed

**Symptom:**
- TPU VMs not deleted after test
- Increasing costs

**Diagnosis:**
```bash
# List orphaned VMs
gcloud compute tpus tpu-vm list --filter="name~ci-tpu-*"
```

**Fix:**
```bash
# Add cleanup step with || true
- name: Cleanup TPU
  if: always()
  run: |
    gcloud compute tpus tpu-vm delete $TPU_NAME \
      --zone=$TPU_ZONE \
      --quiet || true
```

## Debugging Workflow

### Step 1: Identify Failure Type

1. **Immediate failure (< 10s):** Likely dependency/setup issue
2. **During test (30-120s):** Logic error, device conflict
3. **Timeout (> 10 min):** Performance issue, hanging process

### Step 2: Reproduce Locally

```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh sglang-test-vm --zone=us-central1-b

# Navigate to repo
cd ~/sglang-jax
git pull origin main
source .venv/bin/activate

# Run failing test
python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_<failing_test> -v
```

### Step 3: Add Debug Logging

```python
# In test file
import logging
logging.basicConfig(level=logging.DEBUG)

def test_score_consistency(self):
    logging.debug("Starting test_score_consistency")
    # ... test code ...
    logging.debug("Completed HF reference")
    # ... more code ...
```

### Step 4: Isolate Issue

```python
# Create minimal reproduction
def test_minimal_repro(self):
    """Minimal test to isolate issue"""
    self.runner.launch_server(...)

    # Simplest possible test
    result = self.runner.score(
        query="Test",
        items=["A"],
        label_token_ids=[123]
    )

    self.assertIsNotNone(result)
```

### Step 5: Check Recent Changes

```bash
# What changed recently?
git log --oneline -10

# Diff against working commit
git diff <working-commit> <failing-commit> -- python/sgl_jax/srt/managers/tokenizer_manager.py
```

## Useful Commands

### Log Collection

```bash
# Collect all logs
gcloud compute tpus tpu-vm ssh sglang-test-vm --zone=us-central1-b \
  --command="tar czf /tmp/logs.tar.gz /tmp/*.log ~/sglang-jax/test_*.log 2>/dev/null; echo 'Logs archived'"

# Download logs
gcloud compute tpus tpu-vm scp sglang-test-vm:/tmp/logs.tar.gz ./logs.tar.gz --zone=us-central1-b

# Extract and view
tar xzf logs.tar.gz
```

### Process Management

```bash
# Kill all Python processes
pkill -f python

# Kill specific test
pkill -f "test_score_api"

# Kill server
pkill -f "launch_server"
```

### Environment Check

```bash
# Full environment diagnostic
python3 -c "
import sys
import jax
import transformers
print(f'Python: {sys.version}')
print(f'JAX: {jax.__version__}')
print(f'Devices: {jax.devices()}')
print(f'Transformers: {transformers.__version__}')
"
```

## Prevention

### 1. Pre-commit Checks

```bash
# Run tests locally before pushing
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# Check for JAX in wrong places
grep -r "import jax" python/sgl_jax/srt/managers/tokenizer_manager.py && echo "ERROR: JAX found in tokenizer_manager" || echo "OK"
```

### 2. CI/CD Best Practices

- Always use `if: always()` for cleanup steps
- Set reasonable timeouts (20 min max)
- Use preemptible TPUs for cost savings
- Archive test results for debugging

### 3. Monitoring

- Track test duration over time
- Alert on failures
- Monitor TPU costs

## References

- [RFC-001: Score API Comprehensive Tests](../rfcs/001-score-api-comprehensive-tests.md)
- [RFC-002: CI/CD for TPU Testing](../rfcs/002-cicd-tpu-testing.md)
- [ADR-001: Pure Python Softmax Decision](../decisions/001-pure-python-softmax.md)
- [Investigation: TokenizerManager Architecture](../investigations/tokenizer-manager-architecture.md)

## Escalation

If issue persists after following this runbook:

1. **Check GitHub Issues:** https://github.com/sgl-project/sglang/issues
2. **Post in Discord:** #tpu-testing channel
3. **Create detailed bug report** with:
   - Full error traceback
   - Test output
   - Environment info
   - Reproduction steps
