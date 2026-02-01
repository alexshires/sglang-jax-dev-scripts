# Test Plan 003: JAX Features and Performance Benchmarks

| | |
|------------|------|
| **Related** | [RFC-003](../rfcs/003-score-api-comprehensive-test-suite.md) |
| **Phase** | 5-6 |
| **Priority** | P1 (Important but not CI-blocking) |
| **Dependencies** | Test Plan 001 (shared fixtures) |

## Objective

Implement JAX-specific feature tests (dtype stability, sharding) and create performance benchmark tool for throughput/latency tracking and regression detection.

## Deliverables

1. **New file:** `test/srt/test_score_api_jax_features.py` (~300 lines)
2. **New file:** `test/srt/bench_score.py` (~400 lines)
3. **New file:** `test/srt/bench_score_results_baseline.csv` (performance baseline)

## JAX-Specific Tests Specification

### File: `test/srt/test_score_api_jax_features.py`

```python
"""
JAX-specific feature tests for Score API.

These tests validate JAX-specific behavior:
- bfloat16 vs float32 numerical stability
- Multi-device sharding correctness
- Prefix caching integration

NOT run in default CI (requires special setup):
- Multi-device tests need TPU pod
- Dtype tests may need specific hardware
- Caching tests need instrumented engine

Usage:
    # Local/TPU-multi only
    python3 -m unittest test.srt.test_score_api_jax_features -v

    # Or via pytest with skipping
    pytest test/srt/test_score_api_jax_features.py -v
"""

import unittest
import os
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    build_engine,
    get_tokenizer,
    get_label_token_ids,
    skip_if_no_multidevice,
)


class TestScoreAPIJAXFeatures(CustomTestCase):
    """JAX-specific feature tests for Score API"""

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.2-1B-Instruct"
        cls.tokenizer = get_tokenizer(cls.model)

    def test_score_numerical_stability_bf16_vs_fp32(self):
        """
        Test numerical stability between bfloat16 and float32.

        Validates:
        - bf16 and fp32 results are close (within tolerance)
        - Acceptable precision loss for bf16
        - No NaN or Inf values

        Tolerance:
        - Absolute difference: < 0.01 (1%)
        - Relative difference: < 5%

        Note: Requires ability to run with different dtypes.
        May skip if environment doesn't support both.
        """
        try:
            # Build engines with different dtypes
            config_bf16 = ScoreTestConfig(dtype="bfloat16")
            runner_bf16 = build_engine(config_bf16)

            config_fp32 = ScoreTestConfig(dtype="float32")
            runner_fp32 = build_engine(config_fp32)
        except Exception as e:
            self.skipTest(f"Could not build engines with different dtypes: {e}")

        try:
            label_token_ids = get_label_token_ids(
                self.tokenizer, [" to", " of", " for"]
            )

            query = "I pledge allegiance"
            items = [" to the flag", " of the United States", " for which it stands"]

            # Score with bf16
            scores_bf16 = runner_bf16.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )

            # Score with fp32
            scores_fp32 = runner_fp32.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )

            # Compare results
            for i, (bf16_scores, fp32_scores) in enumerate(zip(scores_bf16, scores_fp32)):
                for j, (bf16, fp32) in enumerate(zip(bf16_scores, fp32_scores)):
                    # No NaN or Inf
                    self.assertFalse(math.isnan(bf16))
                    self.assertFalse(math.isnan(fp32))
                    self.assertFalse(math.isinf(bf16))
                    self.assertFalse(math.isinf(fp32))

                    # Absolute difference < 0.01
                    abs_diff = abs(bf16 - fp32)
                    self.assertLess(
                        abs_diff, 0.01,
                        f"Item {i}, label {j}: bf16={bf16}, fp32={fp32}, diff={abs_diff}"
                    )

                    # Relative difference < 5%
                    if fp32 > 0.001:  # Avoid division by near-zero
                        rel_diff = abs_diff / fp32
                        self.assertLess(
                            rel_diff, 0.05,
                            f"Item {i}, label {j}: relative diff {rel_diff*100:.2f}%"
                        )

        finally:
            runner_bf16.shutdown()
            runner_fp32.shutdown()

    @skip_if_no_multidevice()
    def test_score_sharding_correctness(self):
        """
        Test multi-device sharding correctness.

        Validates:
        - Multi-device results match single-device
        - Sharding doesn't affect correctness
        - No device placement errors

        Requires:
        - len(jax.devices()) >= 2
        - Model fits on single device for comparison

        Note: Skipped if < 2 devices available.
        """
        import jax

        num_devices = len(jax.devices())
        self.assertGreaterEqual(num_devices, 2, "Need at least 2 devices")

        # Build single-device engine (tp_size=1)
        config_single = ScoreTestConfig(tp_size=1)
        runner_single = build_engine(config_single)

        # Build multi-device engine (tp_size=num_devices)
        config_multi = ScoreTestConfig(tp_size=num_devices)
        runner_multi = build_engine(config_multi)

        try:
            label_token_ids = get_label_token_ids(
                self.tokenizer, [" to", " of", " for"]
            )

            query = "I pledge allegiance"
            items = [" to the flag", " of the United States", " for which it stands"]

            # Score with single device
            scores_single = runner_single.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )

            # Score with multi-device (sharded)
            scores_multi = runner_multi.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )

            # Results should be identical (bit-exact with deterministic ops)
            for i, (single_scores, multi_scores) in enumerate(
                zip(scores_single, scores_multi)
            ):
                for j, (single, multi) in enumerate(zip(single_scores, multi_scores)):
                    self.assertAlmostEqual(
                        single, multi, places=6,
                        msg=f"Item {i}, label {j}: single={single}, multi={multi}"
                    )

        finally:
            runner_single.shutdown()
            runner_multi.shutdown()

    def test_score_with_prefix_caching(self):
        """
        Test prefix caching integration (if available).

        Validates:
        - Cache hit/miss tracking works
        - Cached results match non-cached
        - Performance benefit from caching

        Note: May require instrumented engine or special build.
        Skips if caching metrics not available.
        """
        # This test is aspirational - requires caching instrumentation
        # that may not exist yet. Skip for now.
        self.skipTest("Prefix caching metrics not yet instrumented")

        # Future implementation:
        # 1. Score with same prefix twice
        # 2. Check cache_miss_count in meta_info
        # 3. Expect miss on first call, hit on second
        # 4. Validate results identical
        # 5. Measure latency improvement


class TestScoreAPIDeterminismAdvanced(CustomTestCase):
    """Advanced determinism tests (cross-session, cross-dtype)"""

    def test_score_determinism_across_dtypes(self):
        """
        Test that results are deterministic within same dtype.

        Even though bf16 != fp32, each should be deterministic
        when run multiple times.
        """
        config = ScoreTestConfig(dtype="bfloat16", seed=42)
        runner = build_engine(config)

        try:
            tokenizer = get_tokenizer(config.model)
            label_token_ids = get_label_token_ids(tokenizer, [" to", " of"])

            query = "I pledge allegiance"
            items = [" to the flag"]

            # Run 3 times with same seed
            results = []
            for _ in range(3):
                scores = runner.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True
                )
                results.append(scores)

            # All 3 runs should be identical
            for i in range(1, 3):
                self.assertEqual(
                    results[0], results[i],
                    f"Run {i+1} differs from run 1"
                )

        finally:
            runner.shutdown()
```

## Performance Benchmark Tool Specification

### File: `test/srt/bench_score.py`

```python
#!/usr/bin/env python3
"""
Score API performance benchmark tool.

Measures throughput, latency, and scaling characteristics.
NOT run in CI - use for manual performance analysis and regression detection.

Usage:
    # Basic benchmark
    python test/srt/bench_score.py

    # Custom batch sizes and label counts
    python test/srt/bench_score.py --batch-sizes 1,2,4,8,16,32 --num-labels 2,4,8,16

    # Save results to CSV
    python test/srt/bench_score.py --output results.csv

    # Compare with baseline
    python test/srt/bench_score.py --baseline baseline.csv --output current.csv

Example output:
    Batch Size | Num Labels | Throughput (items/sec) | p50 Latency (ms) | p95 Latency (ms)
    -----------|------------|------------------------|------------------|------------------
    1          | 2          | 85.3                   | 11.7             | 13.2
    4          | 2          | 312.5                  | 12.8             | 14.5
    8          | 2          | 587.4                  | 13.6             | 16.2
    16         | 2          | 894.2                  | 17.9             | 21.3
"""

import argparse
import time
import statistics
import csv
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    build_engine,
    get_tokenizer,
    get_label_token_ids,
)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    batch_size: int
    num_labels: int
    num_runs: int
    total_items: int
    throughput_items_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float


def run_benchmark(
    runner,
    tokenizer,
    batch_size: int,
    num_labels: int,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> BenchmarkResult:
    """
    Run benchmark for specific configuration.

    Args:
        runner: Engine instance
        tokenizer: Tokenizer instance
        batch_size: Number of items per request
        num_labels: Number of label tokens
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs (not measured)

    Returns:
        BenchmarkResult with metrics
    """
    # Generate test data
    query = "Benchmark query: " * 10  # Longer query for realism
    items = [f" item {i}" for i in range(batch_size)]
    label_tokens = [f" label{i}" for i in range(num_labels)]
    label_token_ids = get_label_token_ids(tokenizer, label_tokens)

    # Warmup
    print(f"  Warming up (batch={batch_size}, labels={num_labels})...")
    for _ in range(warmup_runs):
        runner.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

    # Benchmark
    print(f"  Measuring (batch={batch_size}, labels={num_labels}, runs={num_runs})...")
    latencies = []
    for run in range(num_runs):
        start = time.perf_counter()
        runner.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms

        if (run + 1) % 10 == 0:
            print(f"    Run {run + 1}/{num_runs} complete")

    # Compute metrics
    latencies.sort()
    total_items = batch_size * num_runs
    total_time_sec = sum(latencies) / 1000
    throughput = total_items / total_time_sec

    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return BenchmarkResult(
        batch_size=batch_size,
        num_labels=num_labels,
        num_runs=num_runs,
        total_items=total_items,
        throughput_items_per_sec=throughput,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_mean_ms=mean,
        latency_std_ms=std
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table"""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(
        f"{'Batch':>6} | {'Labels':>7} | {'Throughput':>16} | "
        f"{'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10}"
    )
    print("-"*100)

    for r in results:
        print(
            f"{r.batch_size:6d} | {r.num_labels:7d} | "
            f"{r.throughput_items_per_sec:10.1f} items/s | "
            f"{r.latency_p50_ms:10.2f} | "
            f"{r.latency_p95_ms:10.2f} | "
            f"{r.latency_p99_ms:10.2f}"
        )

    print("="*100 + "\n")


def save_results_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"Results saved to: {output_path}")


def compare_with_baseline(current: List[BenchmarkResult], baseline_path: str):
    """Compare current results with baseline"""
    # Load baseline
    baseline_map = {}
    with open(baseline_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['batch_size']), int(row['num_labels']))
            baseline_map[key] = float(row['throughput_items_per_sec'])

    print("\n" + "="*100)
    print("REGRESSION ANALYSIS")
    print("="*100)
    print(
        f"{'Batch':>6} | {'Labels':>7} | {'Current':>16} | "
        f"{'Baseline':>16} | {'Change':>10} | {'Status':>10}"
    )
    print("-"*100)

    regressions = []
    for r in current:
        key = (r.batch_size, r.num_labels)
        if key in baseline_map:
            baseline_throughput = baseline_map[key]
            current_throughput = r.throughput_items_per_sec
            change_pct = ((current_throughput - baseline_throughput) / baseline_throughput) * 100

            # Consider > 10% slowdown a regression
            status = "REGRESSION" if change_pct < -10 else "OK"
            if change_pct < -10:
                regressions.append((key, change_pct))

            print(
                f"{r.batch_size:6d} | {r.num_labels:7d} | "
                f"{current_throughput:10.1f} items/s | "
                f"{baseline_throughput:10.1f} items/s | "
                f"{change_pct:+9.1f}% | {status:>10}"
            )
        else:
            print(
                f"{r.batch_size:6d} | {r.num_labels:7d} | "
                f"{r.throughput_items_per_sec:10.1f} items/s | "
                f"{'N/A':>16} | {'N/A':>10} | {'NEW':>10}"
            )

    print("="*100 + "\n")

    if regressions:
        print(f"⚠️  FOUND {len(regressions)} PERFORMANCE REGRESSIONS:")
        for (batch, labels), change_pct in regressions:
            print(f"  - Batch {batch}, Labels {labels}: {change_pct:.1f}% slower")
        return False
    else:
        print("✅ No performance regressions detected")
        return True


def main():
    parser = argparse.ArgumentParser(description="Score API performance benchmark")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes (default: 1,2,4,8,16)"
    )
    parser.add_argument(
        "--num-labels",
        default="2,4,8",
        help="Comma-separated label counts (default: 2,4,8)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Number of measurement runs per config (default: 20)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5)"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--baseline",
        help="Baseline CSV file for comparison (optional)"
    )

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    num_labels_list = [int(x) for x in args.num_labels.split(',')]

    print(f"\n{'='*100}")
    print("SCORE API PERFORMANCE BENCHMARK")
    print(f"{'='*100}")
    print(f"Model: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Label counts: {num_labels_list}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"{'='*100}\n")

    # Build engine
    print("Building engine...")
    config = ScoreTestConfig(model=args.model)
    runner = build_engine(config)
    tokenizer = get_tokenizer(args.model)

    try:
        results = []

        # Run benchmarks
        for batch_size in batch_sizes:
            for num_labels in num_labels_list:
                result = run_benchmark(
                    runner=runner,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    num_labels=num_labels,
                    num_runs=args.num_runs,
                    warmup_runs=args.warmup_runs
                )
                results.append(result)

        # Print results
        print_results_table(results)

        # Save to CSV
        if args.output:
            save_results_csv(results, args.output)

        # Compare with baseline
        if args.baseline:
            passed = compare_with_baseline(results, args.baseline)
            if not passed:
                exit(1)  # Exit with error on regression

    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
```

### Creating Baseline

```bash
# First run to establish baseline
cd /Users/kanna/Sandbox/sglang-jax
python test/srt/bench_score.py --output test/srt/bench_score_results_baseline.csv

# Future runs compare against baseline
python test/srt/bench_score.py \
    --baseline test/srt/bench_score_results_baseline.csv \
    --output test/srt/bench_score_results_current.csv
```

## Test Execution Plan

### Step 1: Implement JAX Features Tests

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Create file
touch test/srt/test_score_api_jax_features.py

# Implement dtype stability test
python3 -m unittest test.srt.test_score_api_jax_features.TestScoreAPIJAXFeatures.test_score_numerical_stability_bf16_vs_fp32 -v

# Implement sharding test (requires multi-device)
python3 -m unittest test.srt.test_score_api_jax_features.TestScoreAPIJAXFeatures.test_score_sharding_correctness -v
```

### Step 2: Create Benchmark Tool

```bash
# Create benchmark script
touch test/srt/bench_score.py
chmod +x test/srt/bench_score.py

# Test it works
python test/srt/bench_score.py --batch-sizes 1,2 --num-labels 2 --num-runs 3
```

### Step 3: Establish Baseline

```bash
# Run comprehensive benchmark (takes ~10-15 minutes)
python test/srt/bench_score.py --output test/srt/bench_score_results_baseline.csv

# Commit baseline to repo
git add test/srt/bench_score_results_baseline.csv
git commit -m "Add Score API performance baseline"
```

### Step 4: Document Usage

Add to README or runbook:
- How to run JAX-specific tests
- How to use bench_score.py
- How to interpret results
- When to update baseline

## Success Criteria

- [ ] JAX features tests implemented (2-3 tests)
- [ ] Tests skip gracefully when requirements not met
- [ ] Benchmark tool functional and documented
- [ ] Baseline established on representative hardware
- [ ] Results reproducible (< 5% variance)
- [ ] Regression detection works (< 10% threshold)

## Dependencies

- Test Plan 001 (requires `score_test_utils.py`)
- TPU pod for multi-device tests (optional)
- Multiple dtype support (optional)

## Risks

1. **Hardware availability** - Multi-device tests need TPU pod
   - Mitigation: Skip gracefully with clear message

2. **Performance variance** - Results may vary by hardware
   - Mitigation: Warmup runs, multiple measurement runs, statistical analysis

3. **Baseline staleness** - Baseline becomes outdated
   - Mitigation: Update baseline monthly, document when to update

## When to Run

**JAX Features Tests:**
- Local development when changing dtype or sharding logic
- Nightly CI on multi-device hardware (if available)
- Before releases

**Performance Benchmarks:**
- After any performance-related changes
- Before/after major refactorings
- Monthly for baseline updates
- Never in default CI (too slow)

## Follow-Up

After this phase completes:
- Integrate JAX tests into nightly CI (if multi-device available)
- Set up automated performance tracking (optional)
- Create dashboard for performance trends (optional)
- Document performance optimization guide
