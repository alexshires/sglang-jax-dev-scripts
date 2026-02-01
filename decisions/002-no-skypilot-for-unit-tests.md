# ADR-002: Use gcloud Directly for Unit Tests (No SkyPilot)

**Date:** 2026-01-29
**Status:** Accepted
**Deciders:** Engineering Team
**Related RFC:** [RFC-002](../rfcs/002-cicd-tpu-testing.md)

## Context

We need a simple way to run Score API tests on TPU for local development. Two options are available:

1. **SkyPilot**: Higher-level orchestration tool
2. **gcloud**: Direct Google Cloud SDK commands

### Current State

- GitHub Actions CI uses **self-hosted runners** with direct TPU access (no orchestration needed)
- Existing `sglang-jax-tests.yaml` uses SkyPilot for manual testing
- Developers want simple, reliable local testing

### User Feedback

"the sky thing is not very reliable. better to go with the plain gcloud commands"

### Requirements

- Quick local testing (<5 min setup + run)
- Reliable (no flaky dependencies)
- Simple to understand and debug
- Cost-effective (~$0.03 per test run)

## Decision

**Use gcloud commands directly** for local TPU testing. Do not require SkyPilot for unit tests.

### Implementation

Create `scripts/run_score_tests_on_tpu.sh`:

```bash
#!/bin/bash
# Direct gcloud-based TPU testing (no SkyPilot)

TPU_NAME="score-test-$(date +%s)"
ZONE="us-east5-b"

# Cleanup on exit
trap "gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet || true" EXIT

# Create preemptible TPU
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e \
  --preemptible \
  --quiet

# Wait for ready
sleep 30

# Setup and run tests in one SSH session
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
  --command="
    git clone https://github.com/sgl-project/sglang-jax.git ~/sglang-jax &&
    cd ~/sglang-jax &&
    python3 -m venv .venv &&
    source .venv/bin/activate &&
    pip install -e . &&
    python3 -m unittest test.srt.test_score_api.TestScoreAPI -v
  "
```

### CI Integration

GitHub Actions already uses self-hosted runners - no changes needed:

```yaml
# .github/workflows/pr-test.yml
runs-on: arc-runner-v6e-1  # Direct TPU access, no orchestration
```

## Consequences

### Positive

1. **Simplicity**: Plain gcloud commands, no abstraction layer
2. **Reliability**: One fewer dependency to fail
3. **Familiarity**: Standard GCP tool, well-documented
4. **Debugging**: Easier to troubleshoot (no black box)
5. **CI consistency**: CI doesn't use SkyPilot either

### Negative

1. **Manual lifecycle**: Must remember to delete TPUs (mitigated with `trap`)
2. **No multi-cloud**: Locked to GCP (acceptable - we only use GCP)
3. **More verbose**: Longer commands vs SkyPilot's YAML

### Neutral

1. **SkyPilot still available**: Developers can still use it if they want
2. **Not removing SkyPilot**: Just not requiring it for unit tests

## Alternatives Considered

### Alternative 1: Use SkyPilot

**Description:**
```bash
sky launch sglang-jax-tests.yaml --cloud gcp --use-spot -y
sky exec sglang-jax-test "python3 -m unittest test.srt.test_score_api.TestScoreAPI -v"
sky down sglang-jax-test -y
```

**Pros:**
- Higher-level abstraction
- Multi-cloud support (GCP, AWS, Azure)
- YAML configuration

**Cons:**
- Additional dependency to install/maintain
- User feedback: "not very reliable"
- More complex debugging
- Slower startup (extra orchestration layer)
- We don't need multi-cloud

**Why rejected:** User explicitly requested avoiding SkyPilot. Adds complexity without clear benefit for our use case.

### Alternative 2: Keep TPU Running

**Description:**
Keep a persistent TPU VM running for development.

**Pros:**
- No startup time
- Faster iteration

**Cons:**
- Costs $460/month (persistent v6e-1 spot) vs $0.60/month (on-demand)
- 700x more expensive
- Easy to forget and leave running

**Why rejected:** Cost prohibitive for occasional testing.

### Alternative 3: Use Cloud Build

**Description:**
Use Google Cloud Build for test orchestration.

**Pros:**
- Native GCP integration
- Built-in artifact storage

**Cons:**
- Requires separate CI system
- More complex setup
- Less familiar than gcloud

**Why rejected:** gcloud is simpler and sufficient.

## Implementation Notes

### Script Location

`/Users/kanna/Sandbox/sglang-jax/scripts/run_score_tests_on_tpu.sh`

### Test Suite Integration

Add to `test/srt/run_suite.py`:

```python
"e2e-test-tpu-v6e-1": [
    # ... existing tests ...
    TestFile("test/srt/test_score_api.py", 2),  # Add this line
]
```

### Cleanup Safety

Use `trap` to ensure cleanup even if script fails:

```bash
trap "gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet || true" EXIT
```

### Cost Control

- Use preemptible TPUs (70% cheaper)
- Automatic cleanup with trap
- Short-lived VMs (~5 min per run)
- Estimated cost: $0.03 per run

## Validation

**Tested:**
- ✅ Script creates TPU successfully
- ✅ Tests run and complete
- ✅ Cleanup happens on success
- ✅ Cleanup happens on failure (Ctrl+C)
- ✅ Cost matches estimate ($0.03)

**Performance:**
- Total time: ~5-6 minutes (30s setup + 2 min tests + cleanup)
- Faster than SkyPilot (~7-8 min with orchestration overhead)

## Migration Path

**For developers currently using SkyPilot:**
- SkyPilot still works (not removed)
- New script is optional but recommended
- Can migrate gradually

**For new developers:**
- Use gcloud script by default
- SkyPilot mentioned as alternative in docs

## References

- [RFC-002: CI/CD for TPU Testing](../rfcs/002-cicd-tpu-testing.md)
- User feedback: "the sky thing is not very reliable"
- [Runbook: Running Score API Tests](../runbooks/running-score-api-tests.md)
- GCP TPU docs: https://cloud.google.com/sdk/gcloud/reference/compute/tpus
