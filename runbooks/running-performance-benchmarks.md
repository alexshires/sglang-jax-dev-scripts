# Runbook: Running Performance Benchmarks

**Last Updated:** 2026-01-29
**Maintainer:** Engineering Team
**Related Docs:** RFC-004, Test Plan 004, Test Plan 003

## Overview

This runbook covers how to run performance benchmarks and stress tests for the `/v1/score` Scoring API. It includes instructions for local execution, cluster execution, and CI/CD integration.

## Quick Reference

```bash
# Quick smoke test (2-3 minutes)
python test/srt/bench_score.py --profile smoke

# Standard benchmark with baseline comparison (10-15 minutes)
python test/srt/bench_score.py --profile standard \
  --baseline test/srt/baselines/tpu-v6e-baseline.csv

# Full comprehensive benchmark (45-60 minutes)
python test/srt/bench_score.py --profile full --output results.csv

# Run stress tests
python test/srt/bench_score_stress.py --scenario all

# K8s cluster benchmark
./v1/benchmark/run_benchmark.sh --accelerator tpu-v6e-slice --topology 1x1
```

## Benchmark Profiles

| Profile | Runtime | Use Case |
|---------|---------|----------|
| `smoke` | 2-3 min | Quick validation after changes |
| `standard` | 10-15 min | Regular testing, regression detection |
| `full` | 45-60 min | Releases, deep investigation |

## Running Benchmarks Locally

### Prerequisites

1. Python 3.10+ with sglang-jax installed
2. Access to TPU or GPU (CPU works but is slower)
3. Model downloaded to `/dev/shm` or accessible path

### Method 1: Quick Smoke Test

Use this after making changes to quickly verify nothing broke:

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Run smoke profile
python test/srt/bench_score.py --profile smoke

# Expected output:
# ================================================================================
# BENCHMARK RESULTS - Profile: smoke
# ================================================================================
# Batch | Labels |          IPS | p50 (ms) | p95 (ms) | p99 (ms) |     Status
# ------|--------|--------------|----------|----------|----------|------------
#     1 |      2 |        85.3  |    11.7  |    13.2  |    14.1  |         OK
#     4 |      2 |       312.5  |    12.8  |    14.5  |    15.8  |         OK
# ...
```

**Time:** ~2-3 minutes
**Cost:** Minimal (uses existing hardware)

### Method 2: Standard Benchmark with Regression Detection

Use this for regular testing and detecting performance regressions:

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Run standard profile with baseline comparison
python test/srt/bench_score.py \
  --profile standard \
  --baseline test/srt/baselines/tpu-v6e-baseline.csv \
  --output results/current-run.csv \
  --regression-threshold 10

# Exit code: 0 = passed, 1 = regression detected
echo "Exit code: $?"
```

**Time:** ~10-15 minutes
**Cost:** Moderate (extended TPU usage)

### Method 3: Full Benchmark for Releases

Use this before releases or when investigating performance deeply:

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Run full profile with all outputs
python test/srt/bench_score.py \
  --profile full \
  --output results/full-benchmark.csv \
  --json-output results/full-benchmark.json \
  --metadata-output results/full-metadata.json

# Review results
cat results/full-benchmark.csv
cat results/full-metadata.json
```

**Time:** ~45-60 minutes
**Cost:** Significant (extended TPU usage, multiple models)

### Method 4: Custom Configuration

For specific investigations:

```bash
# Test specific batch sizes
python test/srt/bench_score.py \
  --batch-sizes 1,8,32,64 \
  --num-labels 2,4 \
  --num-runs 50

# Test specific model
python test/srt/bench_score.py \
  --profile smoke \
  --model meta-llama/Llama-3.2-3B-Instruct

# Quiet mode (no progress output)
python test/srt/bench_score.py --profile smoke --quiet
```

## Running Stress Tests

### Large Batch Stress Test

Tests behavior with very large batch sizes (50, 100, 200 items):

```bash
python test/srt/bench_score_stress.py --scenario large-batch

# Custom batch sizes
python test/srt/bench_score_stress.py \
  --scenario large-batch \
  --batch-sizes 50,100,150,200
```

**What it validates:**
- No OOM (out of memory) errors
- Completion within timeout
- Correct result shape

### Concurrent Request Stress Test

Tests thread safety and resource contention:

```bash
python test/srt/bench_score_stress.py --scenario concurrent

# Custom concurrency levels
python test/srt/bench_score_stress.py \
  --scenario concurrent \
  --concurrency 2,4,8,16
```

**What it validates:**
- Thread safety
- No race conditions
- Consistent results under load

### Sustained Load Stress Test

Tests stability over extended period:

```bash
python test/srt/bench_score_stress.py --scenario sustained
```

**What it validates:**
- No memory leaks
- Stable latency over time
- No degradation under sustained load

**Time:** ~5 minutes for sustained, ~2-3 minutes each for others

### Run All Stress Tests

```bash
python test/srt/bench_score_stress.py --scenario all
```

**Time:** ~10-15 minutes total

## Running on Kubernetes Cluster

### Prerequisites

1. `kubectl` configured with cluster access
2. Benchmark Docker image available
3. TPU or GPU node pool available

### Method: K8s Job

```bash
cd /Users/kanna/Sandbox/sglang-jax/v1/benchmark

# TPU v6e benchmark
./run_benchmark.sh --accelerator tpu-v6e-slice --topology 1x1

# GPU benchmark (if available)
./run_benchmark.sh --accelerator nvidia-tesla-l4 --count 1 --image <gpu-image>

# View logs
kubectl logs -f job/sglang-benchmark-xxxxx -n eval-serving
```

### Retrieving Results

```bash
# Get pod name
POD=$(kubectl get pods -n eval-serving -l job-name=sglang-benchmark-xxxxx -o jsonpath='{.items[0].metadata.name}')

# Copy results locally
kubectl cp eval-serving/$POD:/results/benchmark.csv ./cluster-results.csv
```

## CI/CD Integration

### Nightly Runs (Automatic)

Benchmarks run automatically at 2 AM UTC daily via GitHub Actions:

1. Go to **Actions** tab in GitHub
2. Select **Nightly Performance Benchmarks**
3. View latest run results
4. Download artifacts for detailed CSV

### On-Demand Runs

Trigger a benchmark manually:

1. Go to **Actions** > **Nightly Performance Benchmarks**
2. Click **Run workflow**
3. Select profile: `smoke`, `standard`, or `full`
4. Optionally enable stress tests
5. Click **Run workflow**

### Checking Results

```bash
# Via GitHub CLI
gh run list --workflow=nightly-perf.yaml
gh run view <run-id>
gh run download <run-id> -n perf-results-<run-number>
```

## Baseline Management

### Viewing Current Baseline

```bash
cat test/srt/baselines/tpu-v6e-baseline.csv
cat test/srt/baselines/tpu-v6e-metadata.json
```

### Updating Baseline

Update the baseline when:
- Intentional performance improvement
- Hardware/infrastructure change
- Monthly refresh (if no regressions)

```bash
# 1. Run comprehensive benchmark
python test/srt/bench_score.py \
  --profile standard \
  --output test/srt/baselines/tpu-v6e-baseline.csv \
  --metadata-output test/srt/baselines/tpu-v6e-metadata.json

# 2. Review changes
git diff test/srt/baselines/

# 3. Commit with explanation
git add test/srt/baselines/
git commit -m "Update performance baseline

Reason: [intentional optimization | infrastructure change | monthly refresh]
Key changes:
- Throughput improved X% for batch_size=16
- Latency p95 reduced by Yms
"
```

### Baseline Files

| File | Purpose |
|------|---------|
| `tpu-v6e-baseline.csv` | Throughput and latency baseline for TPU v6e |
| `tpu-v6e-metadata.json` | Environment metadata (commit, timestamp, hardware) |

## Interpreting Results

### Throughput (IPS)

- **Items per second** - Higher is better
- Expect linear scaling with batch size up to a point
- Compare against baseline for same configuration

### Latency

| Percentile | Meaning |
|------------|---------|
| p50 | Median latency (typical experience) |
| p95 | 95th percentile (most users see this or better) |
| p99 | 99th percentile (tail latency, worst case) |

### Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `OK` | Within threshold | No action needed |
| `WARN` | 5-10% degradation | Monitor, may need investigation |
| `REGRESSION` | >10% degradation | Investigate immediately |

### Example Analysis

```
Batch | Labels |          IPS |     Status
------|--------|--------------|------------
    8 |      4 |       450.2  |         OK     # Good
   16 |      4 |       380.1  |       WARN     # Slightly slower than baseline
   32 |      4 |       290.5  | REGRESSION     # Significant regression
```

**Investigation steps for regression:**
1. Check recent commits for performance-impacting changes
2. Compare memory usage during run
3. Check if baseline was taken on same hardware
4. Run multiple times to rule out flakiness

## Troubleshooting

### Benchmark Fails with OOM

```
Error: RESOURCE_EXHAUSTED: Out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-sizes 1,2,4,8`
2. Use smaller model: `--model meta-llama/Llama-3.2-1B-Instruct`
3. Ensure no other processes using device memory

### High Variance in Results

**Symptoms:** Large standard deviation, inconsistent runs

**Solutions:**
1. Increase warmup runs: `--warmup-runs 10`
2. Increase measurement runs: `--num-runs 50`
3. Ensure no other workloads on hardware
4. Check for thermal throttling

### Baseline Comparison Fails

```
FileNotFoundError: baselines/tpu-v6e-baseline.csv
```

**Solutions:**
1. Run without baseline first: remove `--baseline` flag
2. Create baseline: run with `--output baselines/tpu-v6e-baseline.csv`
3. Check file path is correct

### CI Workflow Not Running

**Check:**
1. Workflow file exists: `.github/workflows/nightly-perf.yaml`
2. Self-hosted runner available with TPU
3. Schedule syntax correct
4. Repository Actions enabled

## Cost Management

### Estimated Costs

| Profile | TPU v6e Time | Cost (at $0.64/hr) |
|---------|--------------|---------------------|
| smoke | 3 min | $0.03 |
| standard | 15 min | $0.16 |
| full | 60 min | $0.64 |
| stress (all) | 15 min | $0.16 |

### Monthly Estimates

| Scenario | Runs | Cost |
|----------|------|------|
| Nightly smoke | 30 | $0.90 |
| Weekly standard | 4 | $0.64 |
| Monthly full | 1 | $0.64 |
| **Total** | - | **~$2.20/month** |

### Cost Optimization

1. Use `smoke` profile for regular checks
2. Run `standard`/`full` only when needed
3. Cancel stuck runs promptly
4. Share TPU with other workloads when idle

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_DOWNLOAD_DIR` | `/dev/shm` | Model download directory |
| `BENCHMARK_BASELINE` | `baselines/tpu-v6e-baseline.csv` | Baseline file for CI |
| `REGRESSION_THRESHOLD` | `10` | Percentage threshold for regression |

## Related Documentation

- [RFC-004: Score API Performance Benchmarks](../rfcs/004-score-api-performance-benchmarks.md)
- [Test Plan 004: Performance Benchmarks and Stress Tests](../test-plans/004-performance-benchmarks-and-stress-tests.md)
- [Test Plan 003: JAX Features and Performance](../test-plans/003-jax-features-and-performance.md)
- [v1/benchmark/README.md](../v1/benchmark/README.md) - K8s job infrastructure
