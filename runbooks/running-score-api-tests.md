# Runbook: Running Score API Tests

| | |
|------------|------|
| **Last Updated** | 2026-01-29 |
| **Maintainer** | Engineering Team |
| **Related** | [RFC-001](../rfcs/001-score-api-comprehensive-tests.md), [RFC-003](../rfcs/003-score-api-comprehensive-test-suite.md) |

## Overview

Guide for running Score API tests in different environments: CI (automatic), local TPU (manual), and local CPU (quick validation).

## Test Files

- **`test/srt/test_score_api.py`** - Core engine-level tests (3 tests, ~2 min)
- **`test/srt/openai_server/basic/test_openai_server.py`** - HTTP endpoint tests (includes `TestOpenAIV1Score`)

## Running in CI (Automatic)

### On Pull Requests

Tests run automatically on every PR via GitHub Actions:

```yaml
# .github/workflows/pr-test.yml
# Runs on self-hosted TPU runners: arc-runner-v6e-1
```

**What runs:**
- Full `e2e-test-tpu-v6e-1` suite (includes Score API tests)
- Triggered on: PRs to `main` or `epic/*` branches
- Triggered when: Changes to `python/**`, `test/**`, or workflows

**How to check results:**
1. Go to PR on GitHub
2. Check "Checks" tab
3. Look for "PR Test / e2e-test-tpu-v6e-1" job
4. View logs for test results

**To run tests on your PR:**
- Just push changes to `python/**` or `test/**`
- Tests run automatically (no label needed)
- Draft PRs are skipped

### Nightly Runs

Full comprehensive tests run nightly:

```yaml
# .github/workflows/nightly-test.yml
# Runs complete test matrix including e2e tests
```

**What runs:**
- All test suites (unit, e2e, performance, accuracy)
- Includes Score API tests in e2e suite
- Publishes results to GitHub

## Running Locally on TPU

### Prerequisites

- gcloud CLI installed and authenticated
- GCP project with TPU quota
- Billing enabled

### Method 1: Quick Test Run (Recommended)

Use the wrapper script:

```bash
cd /Users/kanna/Sandbox/sglang-jax
./scripts/run_score_tests_on_tpu.sh
```

**What it does:**
1. Creates preemptible TPU VM (v6e-1)
2. Sets up environment (git clone, venv, pip install)
3. Runs Score API tests
4. Shows results
5. Cleans up TPU VM automatically

**Cost:** ~$0.03 per run (5 min × $0.64/hr preemptible)

**Output:**
```
🚀 Creating TPU VM: score-test-1738185234
⏳ Waiting for TPU to be ready...
📦 Setting up environment...
🧪 Running tests...

test_score_batch_handling (test.srt.test_score_api.TestScoreAPI) ... ok
test_score_consistency (test.srt.test_score_api.TestScoreAPI) ... ok
test_score_request_construction (test.srt.test_score_api.TestScoreAPI) ... ok

----------------------------------------------------------------------
Ran 3 tests in 104.892s

OK
🧹 Cleaning up TPU VM...
✅ Done!
```

### Method 2: Manual gcloud Commands

For more control:

**Step 1: Create TPU VM**
```bash
TPU_NAME="score-test-$(date +%s)"
ZONE="us-east5-b"

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e \
  --preemptible \
  --quiet

# Wait for ready
sleep 30
```

**Step 2: Setup Environment**
```bash
# Clone and setup
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="
    git clone https://github.com/sgl-project/sglang-jax.git ~/sglang-jax &&
    cd ~/sglang-jax &&
    python3 -m venv .venv &&
    source .venv/bin/activate &&
    pip install -e .
  "
```

**Step 3: Run Tests**
```bash
# Run Score API tests
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="
    cd ~/sglang-jax &&
    source .venv/bin/activate &&
    python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
  "
```

**Step 4: Cleanup**
```bash
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet
```

### Method 3: Persistent TPU (Development)

For active development, keep TPU running:

```bash
# Create once
gcloud compute tpus tpu-vm create dev-tpu \
  --zone=us-east5-b \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e \
  --preemptible

# SSH in and work
gcloud compute tpus tpu-vm ssh dev-tpu --zone=us-east5-b

# On TPU:
cd ~/sglang-jax
git pull
source .venv/bin/activate
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# When done (delete to avoid charges)
gcloud compute tpus tpu-vm delete dev-tpu --zone=us-east5-b
```

**Cost:** $0.64/hr preemptible, $1.90/hr on-demand

## Running Locally on CPU

For quick validation (not full correctness):

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Activate venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run tests (will use CPU, slower)
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
```

**Notes:**
- CPU tests take longer (~10-15 min vs 2 min on TPU)
- Some JAX operations may behave differently on CPU
- Good for syntax/import checks, not accuracy validation

## Running Specific Tests

### Single Test Method

```bash
# On TPU
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="cd ~/sglang-jax && source .venv/bin/activate && \
    python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency -v"

# Or locally
python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency -v
```

### HTTP Endpoint Tests Only

```bash
python3 -m unittest test.srt.openai_server.basic.test_openai_server.TestOpenAIV1Score -v
```

### Via Test Suite Runner

```bash
# Run entire e2e suite (includes Score API tests)
python test/srt/run_suite.py --suite e2e-test-tpu-v6e-1 --timeout-per-file 600
```

## Test Suite Integration

Score API tests are part of the `e2e-test-tpu-v6e-1` suite:

```python
# In test/srt/run_suite.py
"e2e-test-tpu-v6e-1": [
    # ... other tests ...
    TestFile("test/srt/test_score_api.py", 2),  # 2 min estimated
    TestFile("test/srt/openai_server/basic/test_openai_server.py", 1),  # includes HTTP tests
]
```

This means:
- ✅ Runs automatically on every PR
- ✅ Included in nightly runs
- ✅ Part of release testing
- ✅ Uses test suite timeout/retry logic

## Debugging Failed Tests

### Check Logs

**In CI:**
1. Go to GitHub Actions run
2. Click on failed job
3. Expand test step
4. Search for "FAILED" or "ERROR"

**On TPU:**
```bash
# Get detailed output
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="cd ~/sglang-jax && source .venv/bin/activate && \
    python3 -m unittest test.srt.test_score_api.TestScoreAPI -v 2>&1" | tee test_output.txt
```

### Common Issues

See [debugging-tpu-test-failures.md](debugging-tpu-test-failures.md) for full troubleshooting guide.

**Quick fixes:**

1. **Device conflict error**
   - Check: JAX not imported in tokenizer_manager.py
   - Fix: Use SciPy softmax (see [ADR-001](../decisions/001-pure-python-softmax.md))

2. **Slow tests (>150s)**
   - Check: `max_new_tokens=1` instead of `0`
   - Fix: Set `max_new_tokens=0` for prefill-only

3. **Model download timeout**
   - Run: Pre-download model to `/dev/shm`
   - Or: Use cached model from previous run

## Cost Management

### Estimated Costs

| Test Type | Duration | Cost/Run | Runs/Month | Monthly Cost |
|-----------|----------|----------|------------|--------------|
| Quick test (script) | 5 min | $0.03 | 20 | $0.60 |
| Manual run | 5 min | $0.03 | 10 | $0.30 |
| CI (per PR) | 2 min | $0.02 | 60 | $1.20 |
| Nightly | 2 min | $0.02 | 30 | $0.60 |
| **TOTAL** | | | | **~$2.70/month** |

### Cost Optimization

1. **Use preemptible TPUs** - 70% cheaper
2. **Delete after use** - Don't leave running
3. **Use script cleanup** - Automatic deletion
4. **Share TPU for dev** - Keep one running, delete when done

## Environment Variables

Useful for controlling test behavior:

```bash
# Enable HuggingFace reference tests (slow, requires network)
export SGLANG_JAX_RUN_HF_REFERENCE=1

# CI mode (used in GitHub Actions)
export SGLANG_JAX_IS_IN_CI=true

# Custom model cache directory
export HF_HOME=/dev/shm/huggingface
```

## Quick Reference

```bash
# ============ QUICK START ============

# 1. Run tests in CI (automatic)
# → Just push to PR, tests run automatically

# 2. Run tests on TPU (manual, one command)
./scripts/run_score_tests_on_tpu.sh

# 3. Run tests locally on CPU (quick check)
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v

# ============ ADVANCED ============

# Run specific test
python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency -v

# Run with HF reference (slow)
SGLANG_JAX_RUN_HF_REFERENCE=1 python3 -m unittest test.srt.test_score_api.TestScoreAPI.test_score_consistency -v

# Run entire e2e suite
python test/srt/run_suite.py --suite e2e-test-tpu-v6e-1
```

## Related Documentation

- **[RFC-001: Score API Comprehensive Tests](../rfcs/001-score-api-comprehensive-tests.md)** - Test design and implementation
- **[RFC-003: Comprehensive Test Suite](../rfcs/003-score-api-comprehensive-test-suite.md)** - Future expansion plan
- **[ADR-001: SciPy Softmax](../decisions/001-pure-python-softmax.md)** - Why we avoid JAX in tokenizer_manager
- **[Runbook: Debugging TPU Test Failures](debugging-tpu-test-failures.md)** - Troubleshooting guide

## Support

**If tests fail:**
1. Check [debugging-tpu-test-failures.md](debugging-tpu-test-failures.md)
2. Look for similar issues in GitHub
3. Check recent commits for related changes
4. Ask in team chat with test output

**For cost concerns:**
- Review monthly TPU usage in GCP console
- Check for orphaned TPU VMs: `gcloud compute tpus tpu-vm list`
- Set up billing alerts in GCP

## Appendix: Test Suite Structure

```
test/srt/
├── test_score_api.py                    # Engine tests (3 tests)
├── openai_server/basic/
│   └── test_openai_server.py            # HTTP tests (includes TestOpenAIV1Score)
└── run_suite.py                         # Test orchestration
    └── suites["e2e-test-tpu-v6e-1"]     # Score API tests run here
```
