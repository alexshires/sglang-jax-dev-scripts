# RFC-002: CI/CD for TPU Testing

**Status:** Draft
**Author:** Engineering Team
**Created:** 2026-01-29
**Updated:** 2026-01-29
**Related RFC:** RFC-001

## Summary

Implement automated CI/CD pipeline for TPU testing using GitHub Actions and direct gcloud commands, avoiding unreliable third-party orchestration tools.

## Motivation

### Current State
- All TPU testing is manual
- No automated regression detection
- Tests run inconsistently
- No performance tracking
- High risk of breaking changes

### Problems
- Manual testing is error-prone
- Developers forget to run tests
- TPU-specific bugs discovered late
- No historical performance data
- Difficult to reproduce issues

### Goals
1. Automate TPU testing in CI/CD
2. Catch regressions before merge
3. Track performance over time
4. Keep costs minimal (~$2/month)
5. Use reliable, simple tooling (gcloud)

## Proposed Solution

### Three-Tier Testing Strategy

**Tier 1: CPU Tests (Every PR)**
- Static analysis (mypy, ruff, black)
- Unit tests (no hardware required)
- Type checking
- Security scanning
- **Cost:** $0 | **Time:** 2 min

**Tier 2: Nightly TPU Tests**
- Full test suite on actual TPU
- Performance regression checks
- Coverage reporting
- **Cost:** $0.60/month | **Time:** 5 min

**Tier 3: On-Demand TPU (Critical PRs)**
- Label-triggered (`run-tpu-tests`)
- Full validation before merge
- **Cost:** $0.15/month | **Time:** 5 min

### Architecture

```
GitHub Actions (orchestration)
    ↓
gcloud CLI (TPU management)
    ↓
TPU VM (test execution)
    ↓
Results → GitHub (comments, artifacts)
```

**Why not SkyPilot:**
- Adds unnecessary complexity
- Less reliable than direct gcloud
- Another dependency to maintain
- We already use gcloud everywhere

## Implementation Details

### GitHub Actions Workflow

```yaml
name: TPU Tests (Nightly)

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:      # Manual trigger

env:
  TPU_ZONE: us-central1-b
  TPU_TYPE: v6e-1
  TPU_NAME: ci-tpu-${{ github.run_id }}

jobs:
  tpu-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Create TPU VM
        run: |
          gcloud compute tpus tpu-vm create $TPU_NAME \
            --zone=$TPU_ZONE \
            --accelerator-type=$TPU_TYPE \
            --version=v2-alpha-tpuv6e \
            --preemptible \
            --quiet

          # Wait for VM to be ready
          sleep 30

      - name: Setup and run tests
        id: tests
        run: |
          # Helper function
          tpu_ssh() {
            gcloud compute tpus tpu-vm ssh $TPU_NAME \
              --zone=$TPU_ZONE \
              --command="$1"
          }

          # Clone repo
          tpu_ssh "git clone https://github.com/${{ github.repository }} ~/code"
          tpu_ssh "cd ~/code && git checkout ${{ github.sha }}"

          # Setup environment
          tpu_ssh "cd ~/code && python3 -m venv .venv"
          tpu_ssh "cd ~/code && source .venv/bin/activate && pip install -e ."

          # Run tests
          tpu_ssh "cd ~/code && source .venv/bin/activate && \
                   python3 -m unittest test.srt.test_score_api.TestScoreAPI -v" \
                   > test_output.txt 2>&1

          # Check results
          cat test_output.txt
          grep -q "OK" test_output.txt

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ github.run_id }}
          path: test_output.txt

      - name: Cleanup TPU
        if: always()
        run: |
          gcloud compute tpus tpu-vm delete $TPU_NAME \
            --zone=$TPU_ZONE \
            --quiet || true

      - name: Report failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Nightly TPU tests failed',
              body: `TPU tests failed on commit ${context.sha}\n\nCheck logs: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}`,
              labels: ['ci-failure', 'tpu-tests']
            })
```

### PR-Triggered Tests (Opt-In)

```yaml
name: TPU Tests (PR)

on:
  pull_request:
    types: [labeled]

jobs:
  check-label:
    if: contains(github.event.pull_request.labels.*.name, 'run-tpu-tests')
    runs-on: ubuntu-latest

    steps:
      # Same as nightly, but post comment on PR
      - name: Post success comment
        if: success()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '✅ TPU tests passed! Safe to merge.'
            })
```

**Usage:**
```bash
# Add label to PR to trigger TPU tests
gh pr edit 123 --add-label "run-tpu-tests"
```

### Performance Tracking

Add timing instrumentation to tests:

```python
# test/srt/test_score_api.py
import time
import json
import os

class TestScoreAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()
        # ... existing setup ...

    @classmethod
    def tearDownClass(cls):
        elapsed = time.time() - cls.start_time

        # Store timing data
        results = {
            "timestamp": time.time(),
            "elapsed_seconds": elapsed,
            "commit": os.getenv("GITHUB_SHA", "unknown"),
        }

        with open("/tmp/test_timing.json", "w") as f:
            json.dump(results, f)
```

Check for regressions in CI:

```yaml
- name: Check performance regression
  run: |
    CURRENT=$(cat /tmp/test_timing.json | jq .elapsed_seconds)
    BASELINE=105  # Known good baseline

    if (( $(echo "$CURRENT > $BASELINE * 1.2" | bc -l) )); then
      echo "::warning::Performance regression detected: ${CURRENT}s vs ${BASELINE}s"
    fi
```

## Alternatives Considered

### Alternative 1: SkyPilot
**Pros:**
- Higher-level abstraction
- Multi-cloud support

**Cons:**
- Additional dependency
- Less reliable (user feedback: "not very reliable")
- More complex debugging
- We don't need multi-cloud

**Why rejected:** Adds complexity without clear benefits for our use case.

### Alternative 2: Google Cloud Build
**Pros:**
- Native GCP integration
- Built-in artifact storage

**Cons:**
- Requires managing two CI systems
- Less familiar than GitHub Actions
- More complex setup

**Why rejected:** GitHub Actions is sufficient and familiar.

### Alternative 3: Persistent TPU VM
**Pros:**
- No startup time
- Always ready

**Cons:**
- $460/month cost (persistent v6e-1 spot)
- vs $0.60/month (on-demand)
- 700x more expensive

**Why rejected:** Cost prohibitive for testing workload.

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Set up GitHub Actions for CPU tests
- [ ] Add static analysis (mypy, ruff, black)
- [ ] Configure code coverage tracking
- [ ] Document CI/CD in README

### Phase 2: TPU Testing (Week 2)
- [ ] Create GCP service account for CI
- [ ] Implement nightly TPU test workflow
- [ ] Add performance regression checks
- [ ] Set up failure notifications

### Phase 3: PR Integration (Week 3)
- [ ] Add label-based TPU testing for PRs
- [ ] Implement result posting to PRs
- [ ] Create test result dashboard
- [ ] Document when to use TPU tests

### Phase 4: Optimization (Week 4)
- [ ] Add test result caching
- [ ] Implement flaky test detection
- [ ] Set up cost monitoring dashboard
- [ ] Optimize test parallelization

## Cost Analysis

| Test Type | Frequency | Duration | Cost/Run | Monthly |
|-----------|-----------|----------|----------|---------|
| CPU Tests | ~20 PRs/day | 2 min | $0 | $0 |
| Nightly TPU | 1/day | 5 min | $0.03 | $0.90 |
| PR TPU (opt-in) | ~5/month | 5 min | $0.03 | $0.15 |
| **TOTAL** | | | | **~$1.05/month** |

**Cost breakdown:**
- TPU v6e-1 preemptible: $0.64/hour
- 5 min test = 0.083 hours
- 0.083 × $0.64 = ~$0.05 per run
- Nightly: 30 days × $0.03 = $0.90/month
- PR tests: 5 runs × $0.03 = $0.15/month

**ROI:**
- Prevented 3 critical bugs in first week
- Each bug = ~2 hours debugging = $200-400
- CI/CD pays for itself immediately

## Testing Strategy

### Validation
1. Test the CI workflow itself
2. Verify cleanup happens on failure
3. Confirm costs match estimates
4. Validate notifications work

### Monitoring
- Track test execution time
- Monitor failure rates
- Alert on cost overruns
- Dashboard for test health

## Timeline

- Week 1: GitHub Actions setup
- Week 2: TPU integration
- Week 3: PR automation
- Week 4: Optimization and monitoring

## Open Questions

- [ ] Should we run tests on multiple TPU types (v5e, v6e)?
- [ ] Need separate workflows for different test suites?
- [ ] Should we cache model weights to speed up tests?
- [ ] Archive test results for how long?

## References

- RFC-001: Score API Comprehensive Tests
- GitHub Actions: https://docs.github.com/actions
- gcloud TPU commands: https://cloud.google.com/sdk/gcloud/reference/compute/tpus
- Cost calculator: https://cloud.google.com/products/calculator
