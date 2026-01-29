# Implementation Checklist: Score API Test Integration

**Task:** Integrate Score API tests into existing CI/CD and create gcloud-based test runner
**Repository:** sglang-jax
**Documentation:** This repo (sglang-jax-dev-scripts)

## Changes Needed in sglang-jax Repository

### 1. Add Score API Tests to Test Suite

**File:** `test/srt/run_suite.py`

**Change:** Add Score API tests to `e2e-test-tpu-v6e-1` suite

```python
# Around line 455-470
"e2e-test-tpu-v6e-1": [
    # ... existing tests ...
    TestFile("test/srt/openai_server/basic/test_protocol.py", 0.1),
    TestFile("test/srt/openai_server/basic/test_serving_chat.py", 0.1),
    TestFile("test/srt/openai_server/basic/test_serving_completions.py", 0.1),
    TestFile("test/srt/openai_server/basic/test_openai_server.py", 1),
    TestFile("test/srt/test_score_api.py", 2),  # ← ADD THIS LINE
    TestFile("test/srt/openai_server/features/test_ebnf.py", 2),
    # ... rest of tests ...
],
```

**Why:** This makes Score API tests run automatically:
- On every PR (via `.github/workflows/pr-test.yml`)
- In nightly runs (via `.github/workflows/nightly-test.yml`)
- When running test suite manually

**Impact:**
- CI runs will include Score API tests (+2 min runtime)
- Tests will catch regressions automatically
- No workflow changes needed (existing CI picks it up)

---

### 2. Create gcloud-based Test Runner Script

**File:** `scripts/run_score_tests_on_tpu.sh` (NEW)

**Content:**

```bash
#!/bin/bash
# Run Score API tests on TPU using gcloud (no SkyPilot)
# Usage: ./scripts/run_score_tests_on_tpu.sh

set -e

echo "=================================================="
echo "Score API TPU Test Runner"
echo "=================================================="
echo ""

# Configuration
TPU_NAME="score-test-$(date +%s)"
ZONE="us-east5-b"
ACCELERATOR="v6e-1"
VERSION="v2-alpha-tpuv6e"

# Cleanup on exit (success or failure)
cleanup() {
    echo ""
    echo "🧹 Cleaning up TPU VM: $TPU_NAME"
    gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet 2>/dev/null || true
    echo "✅ Cleanup complete"
}
trap cleanup EXIT

# Create TPU VM
echo "🚀 Creating TPU VM: $TPU_NAME"
echo "   Zone: $ZONE"
echo "   Type: $ACCELERATOR"
echo "   Mode: preemptible"
echo ""

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR \
  --version=$VERSION \
  --preemptible \
  --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to create TPU VM"
    exit 1
fi

# Wait for TPU to be ready
echo "⏳ Waiting for TPU to be ready..."
sleep 30

# Setup and run tests
echo "📦 Setting up environment and running tests..."
echo ""

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command="
set -e

echo '📥 Cloning repository...'
git clone https://github.com/sgl-project/sglang-jax.git ~/sglang-jax

echo '🔧 Setting up Python environment...'
cd ~/sglang-jax
python3 -m venv .venv
source .venv/bin/activate

echo '📦 Installing dependencies...'
pip install -q -e .

echo ''
echo '🧪 Running Score API tests...'
echo '=================================================='
python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
TEST_EXIT=\$?
echo '=================================================='
echo ''

if [ \$TEST_EXIT -eq 0 ]; then
    echo '✅ All tests passed!'
else
    echo '❌ Tests failed with exit code:' \$TEST_EXIT
fi

exit \$TEST_EXIT
"

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Score API tests completed successfully"
else
    echo "❌ Score API tests failed (exit code: $TEST_RESULT)"
fi

echo ""
echo "=================================================="
echo "Test run complete"
echo "=================================================="

exit $TEST_RESULT
```

**Make executable:**
```bash
chmod +x scripts/run_score_tests_on_tpu.sh
```

**Why:**
- Simple one-command testing for developers
- No SkyPilot dependency (per ADR-002)
- Automatic cleanup (even on failure)
- Clear output and error handling

**Usage:**
```bash
cd /path/to/sglang-jax
./scripts/run_score_tests_on_tpu.sh
```

---

### 3. Optional: Update README (Recommended)

**File:** `README.md`

**Add section:**

```markdown
## Running Tests

### CI Tests (Automatic)

Tests run automatically on every PR via GitHub Actions.

### Manual Testing on TPU

Run Score API tests on TPU with one command:

```bash
./scripts/run_score_tests_on_tpu.sh
```

Cost: ~$0.03 per run (5 minutes on preemptible v6e-1)

See [test/srt/run_suite.py](test/srt/run_suite.py) for available test suites.
```

**Why:**
- Documents the new test runner
- Makes it discoverable for new contributors
- Shows cost transparency

---

## No Changes Needed

These parts of the repo **DO NOT** need changes:

### ✅ GitHub Actions Workflows
- `.github/workflows/pr-test.yml` - Already runs e2e tests
- `.github/workflows/nightly-test.yml` - Already runs e2e tests
- Tests will automatically include Score API once added to suite

### ✅ Test Files
- `test/srt/test_score_api.py` - Already exists (from RFC-001)
- `test/srt/openai_server/basic/test_openai_server.py` - Already has TestOpenAIV1Score

### ✅ SkyPilot Config (Optional to Keep)
- `sglang-jax-tests.yaml` - Can keep for those who want to use SkyPilot
- Not required, but doesn't hurt to leave it
- ADR-002 just means we don't require SkyPilot

---

## Testing the Changes

### Step 1: Test Suite Integration

```bash
# After editing run_suite.py
cd /path/to/sglang-jax

# Run just Score API tests from suite
python test/srt/run_suite.py --suite e2e-test-tpu-v6e-1 \
  --range-begin 10 --range-end 11 \
  --timeout-per-file 300

# If test_score_api.py is 11th in list (0-indexed = 10)
# Adjust range based on actual position
```

### Step 2: Test gcloud Script

```bash
# After creating the script
cd /path/to/sglang-jax
chmod +x scripts/run_score_tests_on_tpu.sh

# Dry run (check script syntax)
bash -n scripts/run_score_tests_on_tpu.sh

# Full run (creates TPU, runs tests, cleans up)
./scripts/run_score_tests_on_tpu.sh
```

Expected output:
```
==================================================
Score API TPU Test Runner
==================================================

🚀 Creating TPU VM: score-test-1738185234
   Zone: us-east5-b
   Type: v6e-1
   Mode: preemptible

⏳ Waiting for TPU to be ready...
📦 Setting up environment and running tests...

📥 Cloning repository...
🔧 Setting up Python environment...
📦 Installing dependencies...

🧪 Running Score API tests...
==================================================
test_score_batch_handling ... ok
test_score_consistency ... ok
test_score_request_construction ... ok

----------------------------------------------------------------------
Ran 3 tests in 104.892s

OK
==================================================

✅ All tests passed!

🧹 Cleaning up TPU VM: score-test-1738185234
✅ Cleanup complete

==================================================
Test run complete
==================================================
```

### Step 3: Verify CI Integration

After merging changes:

1. Create a test PR
2. Check that "PR Test / e2e-test-tpu-v6e-1" job runs
3. Verify Score API tests appear in logs
4. Confirm tests pass

---

## Rollback Plan

If issues arise:

### Rollback Step 1: Remove from Test Suite

```python
# In test/srt/run_suite.py
# Comment out the line:
# TestFile("test/srt/test_score_api.py", 2),
```

### Rollback Step 2: Delete Script

```bash
rm scripts/run_score_tests_on_tpu.sh
```

CI will continue working without Score API tests.

---

## Cost Impact

### Before Changes
- PR tests: ~15 min/run × $0.64/hr = $0.16
- 60 PRs/month = $9.60/month

### After Changes
- PR tests: ~17 min/run × $0.64/hr = $0.18 (+$0.02)
- 60 PRs/month = $10.80/month (+$1.20)

**Additional:**
- Manual test runs: ~20/month × $0.03 = $0.60/month

**Total increase:** ~$1.80/month

---

## Success Criteria

- [x] Documentation created (runbook + ADR)
- [ ] Test suite updated (1 line change)
- [ ] gcloud script created (~70 lines)
- [ ] README updated (optional)
- [ ] Changes tested locally
- [ ] PR created and merged
- [ ] CI validates changes work

---

## Timeline

- **Documentation:** Complete ✅
- **Implementation:** 15-20 minutes
- **Testing:** 10 minutes (one test run)
- **PR review:** Team dependent
- **Total:** ~30 minutes hands-on work

---

## Next Steps

1. Review this checklist
2. Make changes in sglang-jax repo
3. Test changes locally
4. Create PR with changes
5. Monitor first CI run to ensure tests work

---

## Support

Questions? Check:
- **[Runbook: Running Score API Tests](runbooks/running-score-api-tests.md)** - Usage guide
- **[ADR-002: No SkyPilot](decisions/002-no-skypilot-for-unit-tests.md)** - Design rationale
- **[Runbook: Debugging TPU Tests](runbooks/debugging-tpu-test-failures.md)** - Troubleshooting
