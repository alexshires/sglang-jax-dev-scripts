# RFC-002: CI/CD Strategy for Score API Testing

| | |
|------------|------|
| **Status** | Implementing |
| **Author** | Engineering Team |
| **Created** | 2026-01-29 |
| **Updated** | 2026-01-30 |
| **Related** | [RFC-001](001-score-api-comprehensive-tests.md), [RFC-004](004-score-api-performance-benchmarks.md) |

## Summary

Define a CI/CD strategy for testing the `/v1/score` API in a fork of sglang-jax, covering local development testing with on-demand TPUs, the missing Score API performance benchmark, and contribution workflow to upstream.

## Context: Fork vs Upstream

This RFC is written for development on a **fork** of sglang-jax. Key differences:

| Aspect | Upstream sglang-jax | Your Fork |
|--------|---------------------|-----------|
| Self-hosted runners | `arc-runner-v6e-1/4` available | **Not available** |
| PR CI | Full TPU tests on every PR | No automatic TPU access |
| Cost | Covered by upstream maintainers | Your responsibility |

**Important:** When you submit PRs to upstream, their CI will run using their self-hosted runners. This RFC focuses on local development and testing before submitting upstream.

## Current State Analysis

### What Upstream Already Has (PR CI)

The upstream sglang-jax runs these on **every PR** via self-hosted TPU runners:

```yaml
# From .github/workflows/pr-test.yml
Jobs that run on every PR:
├── unit-test-1-tpu      (arc-runner-v6e-1)
├── unit-test-4-tpu      (arc-runner-v6e-4)
├── e2e-test-1-tpu       (arc-runner-v6e-1)  ← Includes Score API tests
├── e2e-test-4-tpu       (arc-runner-v6e-4)
├── accuracy-test-1-tpu  (arc-runner-v6e-1)
├── accuracy-test-4-tpu  (arc-runner-v6e-4)
├── performance-test-1-tpu (arc-runner-v6e-1)
└── performance-test-4-tpu (arc-runner-v6e-4)
```

### Score API Coverage in Upstream

| Test | Suite | Runs On |
|------|-------|---------|
| `test/srt/test_score_api.py` | `e2e-test-tpu-v6e-1` | Every PR |
| `test/srt/openai_server/basic/test_openai_server.py` | `e2e-test-tpu-v6e-1` | Every PR |

### The Gap: No Score API Performance Benchmark

**PyTorch SGLang has:** `run_score_benchmark` tests with latency/throughput thresholds in per-commit CI.

**JAX SGLang has:** General serving benchmarks (`test_bench_serving_dense.py`) but **no Score API specific performance benchmark**.

This is the primary gap this RFC addresses.

## Goals

1. **Enable local TPU testing** for fork development (gcloud on-demand)
2. **Add Score API performance benchmark** with regression thresholds
3. **Document contribution workflow** to upstream
4. **Maintain compatibility** with upstream CI when PRs are submitted

## Proposed Solution

### Part 1: Local Development Testing (Fork)

#### Option A: On-Demand TPU via gcloud (Recommended)

For occasional testing during development:

```bash
#!/bin/bash
# scripts/run_score_tests_tpu.sh

set -e

TPU_NAME="score-api-test-$(date +%s)"
TPU_ZONE="us-central1-b"
TPU_TYPE="v6e-1"
PROJECT_ID="your-project-id"

# Create TPU
echo "Creating TPU VM..."
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$TPU_ZONE \
  --accelerator-type=$TPU_TYPE \
  --version=v2-alpha-tpuv6e \
  --preemptible

# Setup and run tests
echo "Running Score API tests..."
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --command="
  git clone https://github.com/YOUR_USERNAME/sglang-jax.git ~/sglang-jax
  cd ~/sglang-jax
  git checkout YOUR_BRANCH
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install uv
  uv pip install -e 'python[all]'

  export SGLANG_JAX_IS_IN_CI=true

  # Run Score API tests
  python test/srt/run_suite.py --suite e2e-test-tpu-v6e-1 --range-begin 4 --range-end 5
"

# Cleanup
echo "Cleaning up..."
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$TPU_ZONE --quiet

echo "Done!"
```

**Cost:** ~$0.05 per run (5 min × $0.64/hr preemptible)

#### Option B: CPU Testing (Quick Iteration)

For fast iteration without TPU:

```bash
# Limited but fast - good for syntax/logic errors
cd sglang-jax
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "python[all]"

# Run protocol/unit tests (no TPU needed)
python -m pytest test/srt/openai_server/basic/test_protocol.py -v

# Note: Full Score API tests require TPU for model inference
```

### Part 2: Score API Performance Benchmark (The Real Gap)

#### New File: `test/srt/test_bench_score.py`

```python
"""
Score API performance benchmark with regression detection.

Runs as part of performance-test-tpu-v6e-1 suite.
Validates latency and throughput thresholds.

Usage:
    python test/srt/test_bench_score.py
"""

import time
import statistics
import unittest
from dataclasses import dataclass
from typing import List

from sgl_jax.test.test_utils import CustomTestCase
from sglang import Engine


# Thresholds (adjust based on baseline measurements)
SCORE_LATENCY_P50_THRESHOLD_MS = 50.0   # p50 latency must be under 50ms
SCORE_LATENCY_P99_THRESHOLD_MS = 150.0  # p99 latency must be under 150ms
SCORE_THROUGHPUT_THRESHOLD_IPS = 100.0  # Must achieve at least 100 items/sec


@dataclass
class BenchmarkResult:
    throughput_ips: float
    latency_p50_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    num_requests: int
    total_items: int


class TestScoreAPIPerformance(CustomTestCase):
    """
    Score API performance benchmarks with regression thresholds.

    These tests validate that Score API performance meets minimum
    requirements and catches regressions.
    """

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    engine = None

    @classmethod
    def setUpClass(cls):
        print(f"[Benchmark] Loading model: {cls.model_name}")
        cls.engine = Engine(model_path=cls.model_name)
        cls.tokenizer = cls.engine.tokenizer

        # Get label token IDs
        cls.label_tokens = [" yes", " no", " maybe"]
        cls.label_token_ids = [
            cls.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in cls.label_tokens
        ]
        print(f"[Benchmark] Model loaded, label_token_ids: {cls.label_token_ids}")

    @classmethod
    def tearDownClass(cls):
        if cls.engine:
            cls.engine.shutdown()

    def run_score_benchmark(
        self,
        batch_size: int,
        num_requests: int,
        warmup_requests: int = 5
    ) -> BenchmarkResult:
        """Run benchmark and return results."""

        query = "Is this statement true or false? The answer is"
        items = [f" Statement number {i} is being evaluated." for i in range(batch_size)]

        # Warmup
        for _ in range(warmup_requests):
            self.engine.score(
                query=query,
                items=items,
                label_token_ids=self.label_token_ids,
                apply_softmax=True
            )

        # Benchmark
        latencies_ms = []
        for _ in range(num_requests):
            start = time.perf_counter()
            self.engine.score(
                query=query,
                items=items,
                label_token_ids=self.label_token_ids,
                apply_softmax=True
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        # Compute metrics
        latencies_ms.sort()
        total_items = batch_size * num_requests
        total_time_sec = sum(latencies_ms) / 1000

        p50_idx = int(len(latencies_ms) * 0.50)
        p99_idx = min(int(len(latencies_ms) * 0.99), len(latencies_ms) - 1)

        return BenchmarkResult(
            throughput_ips=total_items / total_time_sec,
            latency_p50_ms=latencies_ms[p50_idx],
            latency_p99_ms=latencies_ms[p99_idx],
            latency_mean_ms=statistics.mean(latencies_ms),
            num_requests=num_requests,
            total_items=total_items
        )

    def test_score_latency_single_item(self):
        """
        Test Score API latency with single item.

        Validates p50 and p99 latency thresholds.
        """
        result = self.run_score_benchmark(batch_size=1, num_requests=50)

        print(f"[Benchmark] Single item latency: "
              f"p50={result.latency_p50_ms:.1f}ms, "
              f"p99={result.latency_p99_ms:.1f}ms")

        self.assertLess(
            result.latency_p50_ms,
            SCORE_LATENCY_P50_THRESHOLD_MS,
            f"p50 latency {result.latency_p50_ms:.1f}ms exceeds threshold {SCORE_LATENCY_P50_THRESHOLD_MS}ms"
        )

        self.assertLess(
            result.latency_p99_ms,
            SCORE_LATENCY_P99_THRESHOLD_MS,
            f"p99 latency {result.latency_p99_ms:.1f}ms exceeds threshold {SCORE_LATENCY_P99_THRESHOLD_MS}ms"
        )

    def test_score_throughput_batch(self):
        """
        Test Score API throughput with batched items.

        Validates minimum throughput threshold.
        """
        result = self.run_score_benchmark(batch_size=8, num_requests=30)

        print(f"[Benchmark] Batch throughput: "
              f"{result.throughput_ips:.1f} items/sec, "
              f"latency p50={result.latency_p50_ms:.1f}ms")

        self.assertGreater(
            result.throughput_ips,
            SCORE_THROUGHPUT_THRESHOLD_IPS,
            f"Throughput {result.throughput_ips:.1f} IPS below threshold {SCORE_THROUGHPUT_THRESHOLD_IPS} IPS"
        )

    def test_score_latency_large_batch(self):
        """
        Test Score API latency with large batch (20 items).

        Ensures large batches don't cause excessive latency.
        """
        result = self.run_score_benchmark(batch_size=20, num_requests=20)

        print(f"[Benchmark] Large batch (20 items): "
              f"p50={result.latency_p50_ms:.1f}ms, "
              f"throughput={result.throughput_ips:.1f} IPS")

        # Large batch allowed higher latency, but should still be reasonable
        large_batch_latency_threshold = SCORE_LATENCY_P99_THRESHOLD_MS * 2

        self.assertLess(
            result.latency_p99_ms,
            large_batch_latency_threshold,
            f"Large batch p99 latency {result.latency_p99_ms:.1f}ms exceeds threshold {large_batch_latency_threshold}ms"
        )


if __name__ == "__main__":
    unittest.main()
```

#### Add to Suite

Update `test/srt/run_suite.py`:

```python
# Add to performance-test-tpu-v6e-1 suite
"performance-test-tpu-v6e-1": [
    TestFile("test/srt/test_bench_serving_dense.py", 7),
    TestFile("test/srt/test_bench_score.py", 3),  # NEW: Score API benchmark
],
```

### Part 3: Contribution Workflow

#### Development Model: Fork-First

All Score API development happens on the fork first. This allows:
- Rapid iteration without upstream review cycles
- Building comprehensive test coverage
- Validating thresholds with real TPU runs
- Bundling related changes for cleaner upstream PRs

**Timeline:** ~2-3 months of fork development, then batch contribution to upstream.

```
Feature Branch → PR → Fork Main → (later) Batch PR → Upstream Main
```

#### PRs to Fork (Current Phase)

```bash
# 1. Create feature branch
git checkout -b feature/score-api-perf-benchmark

# 2. Make changes and test locally
python3 -m unittest test.srt.test_bench_score -v

# 3. Create PR to fork's main branch
gh pr create --title "Add Score API performance benchmark"

# 4. Merge to fork main after review
```

#### PRs to Upstream (Future Phase)

When fork development is mature:

```bash
# 1. Run local tests (CPU - quick validation)
python -m pytest test/srt/openai_server/basic/test_protocol.py -v

# 2. Run Score API tests on TPU (using script from Part 1)
./scripts/run_score_tests_tpu.sh

# 3. Run Score API performance benchmark
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --command="
  cd ~/sglang-jax
  source .venv/bin/activate
  python test/srt/test_bench_score.py -v
"

# 4. Submit PR to upstream
# Upstream CI will run full test suite on their self-hosted runners
gh pr create --repo sgl-project/sglang-jax --title "Add Score API performance benchmarks"
```

#### What Upstream CI Will Run

When you submit a PR, upstream's `arc-runner-v6e-*` runners will execute:

1. **Unit tests** - `unit-test-tpu-v6e-1/4`
2. **E2E tests** - `e2e-test-tpu-v6e-1/4` (includes Score API)
3. **Accuracy tests** - `accuracy-test-tpu-v6e-1/4`
4. **Performance tests** - `performance-test-tpu-v6e-1/4`

You don't need to replicate all of this locally - just validate your specific changes work.

### Part 4: Environment Variables

The codebase already uses these environment variables:

| Variable | Purpose | Set By |
|----------|---------|--------|
| `SGLANG_JAX_IS_IN_CI` | Indicates CI environment | Upstream CI, your scripts |
| `HF_TOKEN` | HuggingFace authentication | GitHub secrets, local env |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Model download timeout | CI workflow |

Use in your tests:

```python
import os

def is_in_ci() -> bool:
    return os.getenv("SGLANG_JAX_IS_IN_CI") == "true"
```

## Implementation Plan

### Phase 1: Score API Performance Benchmark

- [ ] Create `test/srt/test_bench_score.py` with threshold-based tests
- [ ] Establish baseline thresholds on TPU v6e
- [ ] Add to `performance-test-tpu-v6e-1` suite in `run_suite.py`
- [ ] Test locally on TPU

### Phase 2: Local Development Scripts

- [ ] Create `scripts/run_score_tests_tpu.sh` for on-demand TPU testing
- [ ] Document cost estimates
- [ ] Test the workflow end-to-end

### Phase 3: Documentation

- [ ] Update runbooks with fork-specific instructions
- [ ] Document contribution workflow
- [ ] Add troubleshooting guide

## Cost Analysis

### Local Development (Your Fork)

| Activity | Frequency | Duration | Cost/Run | Monthly |
|----------|-----------|----------|----------|---------|
| Score API tests | 10/month | 5 min | $0.05 | $0.50 |
| Performance benchmark | 5/month | 10 min | $0.11 | $0.55 |
| Full e2e suite | 3/month | 30 min | $0.32 | $0.96 |
| **Total** | | | | **~$2/month** |

*Using preemptible TPU v6e at $0.64/hr*

### Upstream CI (Free to You)

When you submit PRs, upstream's CI runs on their self-hosted runners at no cost to you.

## Alternatives Considered

### Alternative 1: Set Up Your Own Self-Hosted Runners

**Description:** Deploy persistent TPU VMs as GitHub Actions runners for your fork.

**Pros:**
- Matches upstream experience exactly
- No spin-up time for tests

**Cons:**
- Cost: ~$460/month for persistent TPU v6e
- Setup complexity (ARC runner deployment)
- Overkill for occasional development

**Why rejected:** Cost prohibitive for fork development. Use upstream CI for full validation.

### Alternative 2: Skip Local TPU Testing

**Description:** Only test on CPU locally, rely entirely on upstream CI.

**Pros:**
- Zero cost
- Simpler workflow

**Cons:**
- TPU-specific bugs not caught until PR
- Slower iteration cycle
- More failed PRs

**Why rejected:** Some local TPU validation is valuable before submitting PRs.

### Alternative 3: Use Cloud Build Instead of gcloud

**Description:** Use Google Cloud Build for automated TPU testing.

**Pros:**
- More automated
- Better logging/artifacts

**Cons:**
- More complex setup
- Another system to maintain
- gcloud scripts are simpler for occasional use

**Why rejected:** gcloud scripts are sufficient for fork development needs.

## Success Metrics

- [ ] Score API performance benchmark implemented with thresholds
- [ ] Local TPU testing script works reliably
- [ ] Contribution workflow documented
- [ ] At least one PR submitted to upstream using this workflow
- [ ] Performance regression caught before reaching upstream

## Open Questions

- [ ] What should the exact latency/throughput thresholds be? (Need baseline measurement)
- [ ] Should we add Score API benchmark to upstream's suite? (Propose in upstream PR)
- [ ] Need `HF_TOKEN` for model downloads - document how to set up?

## References

- Upstream PR workflow: `sglang-jax/.github/workflows/pr-test.yml`
- Test suite definition: `sglang-jax/test/srt/run_suite.py`
- PyTorch Score API benchmark: `sglang/test/registered/perf/test_bench_serving_1gpu_part2.py`
- [RFC-001: Score API Comprehensive Tests](001-score-api-comprehensive-tests.md)
- [RFC-004: Score API Performance Benchmarks and Stress Tests](004-score-api-performance-benchmarks.md)
