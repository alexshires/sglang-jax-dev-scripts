# Test Plan 004: Performance Benchmarks and Stress Tests

**Related RFC:** RFC-004
**Phase:** 7+ (Post Test Suite Implementation)
**Priority:** P1 (Important for Production Readiness)
**Dependencies:** Test Plan 001 (requires shared fixtures), Test Plan 003 (benchmark foundation)

## Objective

Implement comprehensive performance benchmarking and stress testing infrastructure for the `/v1/score` Scoring API with tiered profiles, regression detection, and CI integration.

## Deliverables

1. **Modified file:** `test/srt/bench_score.py` (~600 lines, expanded from ~400)
2. **New file:** `test/srt/bench_score_stress.py` (~350 lines)
3. **New file:** `test/srt/baselines/tpu-v6e-baseline.csv` (baseline data)
4. **New file:** `test/srt/baselines/baseline-metadata.json` (environment metadata)
5. **New file:** `.github/workflows/nightly-perf.yaml` (~80 lines)

## Benchmark Tool Specification

### File: `test/srt/bench_score.py` (Enhanced)

```python
#!/usr/bin/env python3
"""
Score API performance benchmark tool.

Supports tiered profiles for different use cases:
- smoke: Quick validation (~2-3 minutes)
- standard: Balanced coverage (~10-15 minutes)
- full: Comprehensive analysis (~45-60 minutes)

Usage:
    # Quick smoke test
    python test/srt/bench_score.py --profile smoke

    # Standard benchmark with baseline comparison
    python test/srt/bench_score.py --profile standard \
        --baseline baselines/tpu-v6e-baseline.csv \
        --output results.csv

    # Full analysis for releases
    python test/srt/bench_score.py --profile full \
        --output full-results.csv \
        --metadata-output full-metadata.json

    # Custom configuration
    python test/srt/bench_score.py \
        --batch-sizes 1,2,4,8 \
        --num-labels 2,4 \
        --num-runs 30

Example output:
    ================================================================================
    BENCHMARK RESULTS - Profile: standard
    ================================================================================
    Batch | Labels | Throughput (IPS) | p50 (ms) | p95 (ms) | p99 (ms) | Status
    ------|--------|------------------|----------|----------|----------|--------
        1 |      2 |           85.3   |    11.7  |    13.2  |    14.1  | OK
        4 |      2 |          312.5   |    12.8  |    14.5  |    15.8  | OK
        8 |      2 |          587.4   |    13.6  |    16.2  |    18.3  | WARN
    ================================================================================
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    build_engine,
    get_tokenizer,
    get_label_token_ids,
)


# =============================================================================
# Configuration
# =============================================================================

PROFILES = {
    "smoke": {
        "batch_sizes": [1, 4, 16],
        "num_labels": [2, 4],
        "num_runs": 5,
        "warmup_runs": 2,
        "models": ["meta-llama/Llama-3.2-1B-Instruct"],
        "dtypes": ["bfloat16"],
    },
    "standard": {
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "num_labels": [2, 4, 8],
        "num_runs": 20,
        "warmup_runs": 5,
        "models": ["meta-llama/Llama-3.2-1B-Instruct"],
        "dtypes": ["bfloat16"],
    },
    "full": {
        "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
        "num_labels": [2, 4, 8, 16],
        "num_runs": 50,
        "warmup_runs": 10,
        "models": [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "dtypes": ["bfloat16", "float32"],
    },
}

DEFAULT_REGRESSION_THRESHOLD = 10  # 10% throughput decrease = regression


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark configuration result"""
    batch_size: int
    num_labels: int
    model: str
    dtype: str
    num_runs: int
    warmup_runs: int
    total_items: int
    throughput_ips: float  # Items per second
    throughput_rps: float  # Requests per second
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    status: str = "OK"  # OK, WARN, REGRESSION


@dataclass
class BenchmarkMetadata:
    """Environment and configuration metadata for reproducibility"""
    timestamp: str
    profile: str
    hardware: str
    platform: str
    python_version: str
    commit_hash: str
    model: str
    dtype: str
    tp_size: int
    total_configurations: int
    total_runtime_seconds: float


# =============================================================================
# Environment Detection
# =============================================================================

def detect_hardware() -> str:
    """Detect hardware accelerator type"""
    try:
        import jax
        devices = jax.devices()
        if devices:
            device_type = devices[0].platform
            device_count = len(devices)
            if device_type == "tpu":
                return f"tpu-v6e-{device_count}x"
            elif device_type == "gpu":
                return f"gpu-{device_count}x"
            else:
                return f"cpu-{device_count}x"
    except Exception:
        pass
    return "unknown"


def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def collect_metadata(
    profile: str,
    model: str,
    dtype: str,
    tp_size: int,
    total_configs: int,
    runtime_seconds: float
) -> BenchmarkMetadata:
    """Collect environment metadata for reproducibility"""
    return BenchmarkMetadata(
        timestamp=datetime.utcnow().isoformat() + "Z",
        profile=profile,
        hardware=detect_hardware(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        commit_hash=get_git_commit(),
        model=model,
        dtype=dtype,
        tp_size=tp_size,
        total_configurations=total_configs,
        total_runtime_seconds=runtime_seconds,
    )


# =============================================================================
# Benchmark Execution
# =============================================================================

def run_single_benchmark(
    runner,
    tokenizer,
    batch_size: int,
    num_labels: int,
    model: str,
    dtype: str,
    num_runs: int,
    warmup_runs: int,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.

    Strategy:
    1. Generate test data (query + items + labels)
    2. Warmup runs (not measured) to stabilize JIT
    3. Measurement runs with timing
    4. Compute statistics (p50, p95, p99, mean, std)
    """
    # Generate test data
    query = "Benchmark query for performance testing: " * 5  # ~200 chars
    items = [f" candidate item number {i}" for i in range(batch_size)]
    label_tokens = [f" label{i}" for i in range(num_labels)]
    label_token_ids = get_label_token_ids(tokenizer, label_tokens)

    if verbose:
        print(f"  Config: batch={batch_size}, labels={num_labels}, "
              f"runs={num_runs}, warmup={warmup_runs}")

    # Warmup
    if verbose:
        print(f"    Warming up...")
    for _ in range(warmup_runs):
        runner.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

    # Measurement
    if verbose:
        print(f"    Measuring...")
    latencies_ms = []
    for run in range(num_runs):
        start = time.perf_counter()
        runner.score(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    # Compute statistics
    latencies_ms.sort()
    total_items = batch_size * num_runs
    total_time_sec = sum(latencies_ms) / 1000

    throughput_ips = total_items / total_time_sec
    throughput_rps = num_runs / total_time_sec

    p50_idx = int(len(latencies_ms) * 0.50)
    p95_idx = min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)
    p99_idx = min(int(len(latencies_ms) * 0.99), len(latencies_ms) - 1)

    return BenchmarkResult(
        batch_size=batch_size,
        num_labels=num_labels,
        model=model,
        dtype=dtype,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        total_items=total_items,
        throughput_ips=throughput_ips,
        throughput_rps=throughput_rps,
        latency_p50_ms=latencies_ms[p50_idx],
        latency_p95_ms=latencies_ms[p95_idx],
        latency_p99_ms=latencies_ms[p99_idx],
        latency_mean_ms=statistics.mean(latencies_ms),
        latency_std_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
    )


def run_benchmark_suite(
    profile_name: str,
    custom_config: Optional[Dict] = None,
    verbose: bool = True
) -> tuple[List[BenchmarkResult], float]:
    """
    Run complete benchmark suite for given profile.

    Returns:
        Tuple of (results list, total runtime in seconds)
    """
    config = PROFILES.get(profile_name, PROFILES["standard"])
    if custom_config:
        config = {**config, **custom_config}

    results = []
    start_time = time.time()

    for model in config["models"]:
        for dtype in config["dtypes"]:
            if verbose:
                print(f"\nBuilding engine: model={model}, dtype={dtype}")

            engine_config = ScoreTestConfig(model=model, dtype=dtype)
            runner = build_engine(engine_config)
            tokenizer = get_tokenizer(model)

            try:
                for batch_size in config["batch_sizes"]:
                    for num_labels in config["num_labels"]:
                        result = run_single_benchmark(
                            runner=runner,
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            num_labels=num_labels,
                            model=model,
                            dtype=dtype,
                            num_runs=config["num_runs"],
                            warmup_runs=config["warmup_runs"],
                            verbose=verbose
                        )
                        results.append(result)
            finally:
                runner.shutdown()

    total_runtime = time.time() - start_time
    return results, total_runtime


# =============================================================================
# Baseline Comparison
# =============================================================================

def load_baseline(baseline_path: str) -> Dict[tuple, float]:
    """Load baseline throughput values from CSV"""
    baseline_map = {}
    with open(baseline_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                int(row['batch_size']),
                int(row['num_labels']),
                row.get('model', 'unknown'),
                row.get('dtype', 'bfloat16')
            )
            baseline_map[key] = float(row['throughput_ips'])
    return baseline_map


def compare_with_baseline(
    results: List[BenchmarkResult],
    baseline_path: str,
    threshold_pct: float = DEFAULT_REGRESSION_THRESHOLD
) -> tuple[List[BenchmarkResult], bool]:
    """
    Compare results against baseline and mark regressions.

    Returns:
        Tuple of (updated results with status, passed boolean)
    """
    baseline_map = load_baseline(baseline_path)
    has_regression = False

    for result in results:
        key = (result.batch_size, result.num_labels, result.model, result.dtype)
        if key in baseline_map:
            baseline_throughput = baseline_map[key]
            change_pct = ((result.throughput_ips - baseline_throughput) / baseline_throughput) * 100

            if change_pct < -threshold_pct:
                result.status = "REGRESSION"
                has_regression = True
            elif change_pct < -5:  # Warn at 5%
                result.status = "WARN"
            else:
                result.status = "OK"

    return results, not has_regression


# =============================================================================
# Output Formatters
# =============================================================================

def print_results_table(results: List[BenchmarkResult], profile: str):
    """Print human-readable results table"""
    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS - Profile: {profile}")
    print(f"{'='*100}")
    print(
        f"{'Batch':>6} | {'Labels':>6} | {'IPS':>12} | "
        f"{'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | {'Status':>10}"
    )
    print("-" * 100)

    for r in results:
        status_display = {
            "OK": "OK",
            "WARN": "WARN",
            "REGRESSION": "FAIL"
        }.get(r.status, r.status)

        print(
            f"{r.batch_size:6d} | {r.num_labels:6d} | "
            f"{r.throughput_ips:10.1f}   | "
            f"{r.latency_p50_ms:10.2f} | "
            f"{r.latency_p95_ms:10.2f} | "
            f"{r.latency_p99_ms:10.2f} | "
            f"{status_display:>10}"
        )

    print("=" * 100)

    # Summary
    regressions = [r for r in results if r.status == "REGRESSION"]
    warnings = [r for r in results if r.status == "WARN"]

    if regressions:
        print(f"\n FOUND {len(regressions)} PERFORMANCE REGRESSIONS")
        for r in regressions:
            print(f"  - Batch {r.batch_size}, Labels {r.num_labels}: {r.throughput_ips:.1f} IPS")
    elif warnings:
        print(f"\n Found {len(warnings)} warnings (minor degradation)")
    else:
        print(f"\n All {len(results)} configurations passed")


def save_results_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'batch_size', 'num_labels', 'model', 'dtype',
            'num_runs', 'warmup_runs', 'total_items',
            'throughput_ips', 'throughput_rps',
            'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
            'latency_mean_ms', 'latency_std_ms', 'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"\nResults saved to: {output_path}")


def save_results_json(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to: {output_path}")


def save_metadata_json(metadata: BenchmarkMetadata, output_path: str):
    """Save metadata to JSON file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"Metadata saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score API performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  smoke     Quick validation (~2-3 min)
  standard  Balanced coverage (~10-15 min)
  full      Comprehensive analysis (~45-60 min)

Examples:
  # Quick smoke test
  python bench_score.py --profile smoke

  # Standard with baseline comparison
  python bench_score.py --profile standard --baseline baselines/tpu-v6e-baseline.csv

  # Custom configuration
  python bench_score.py --batch-sizes 1,2,4 --num-labels 2,4 --num-runs 30
        """
    )

    # Profile selection
    parser.add_argument(
        "--profile",
        choices=["smoke", "standard", "full"],
        default="standard",
        help="Benchmark profile (default: standard)"
    )

    # Custom configuration (override profile)
    parser.add_argument(
        "--batch-sizes",
        help="Comma-separated batch sizes (overrides profile)"
    )
    parser.add_argument(
        "--num-labels",
        help="Comma-separated label counts (overrides profile)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Measurement runs per config (overrides profile)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        help="Warmup runs (overrides profile)"
    )
    parser.add_argument(
        "--model",
        help="Model to benchmark (overrides profile)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--json-output",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--metadata-output",
        help="Metadata JSON file path"
    )

    # Baseline comparison
    parser.add_argument(
        "--baseline",
        help="Baseline CSV file for regression comparison"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help=f"Regression threshold percentage (default: {DEFAULT_REGRESSION_THRESHOLD})"
    )

    # Behavior options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Build custom config from args
    custom_config = {}
    if args.batch_sizes:
        custom_config["batch_sizes"] = [int(x) for x in args.batch_sizes.split(',')]
    if args.num_labels:
        custom_config["num_labels"] = [int(x) for x in args.num_labels.split(',')]
    if args.num_runs:
        custom_config["num_runs"] = args.num_runs
    if args.warmup_runs:
        custom_config["warmup_runs"] = args.warmup_runs
    if args.model:
        custom_config["models"] = [args.model]

    # Print header
    profile_config = PROFILES.get(args.profile, PROFILES["standard"])
    if custom_config:
        profile_config = {**profile_config, **custom_config}

    if not args.quiet:
        print(f"\n{'='*100}")
        print("SCORE API PERFORMANCE BENCHMARK")
        print(f"{'='*100}")
        print(f"Profile: {args.profile}")
        print(f"Batch sizes: {profile_config['batch_sizes']}")
        print(f"Label counts: {profile_config['num_labels']}")
        print(f"Runs per config: {profile_config['num_runs']}")
        print(f"Warmup runs: {profile_config['warmup_runs']}")
        print(f"Models: {profile_config['models']}")
        if args.baseline:
            print(f"Baseline: {args.baseline}")
            print(f"Regression threshold: {args.regression_threshold}%")
        print(f"{'='*100}")

    # Run benchmarks
    results, runtime = run_benchmark_suite(
        profile_name=args.profile,
        custom_config=custom_config if custom_config else None,
        verbose=not args.quiet
    )

    # Compare with baseline if provided
    passed = True
    if args.baseline:
        results, passed = compare_with_baseline(
            results,
            args.baseline,
            args.regression_threshold
        )

    # Output results
    print_results_table(results, args.profile)

    if args.output:
        save_results_csv(results, args.output)

    if args.json_output:
        save_results_json(results, args.json_output)

    if args.metadata_output:
        first_result = results[0] if results else None
        metadata = collect_metadata(
            profile=args.profile,
            model=first_result.model if first_result else "unknown",
            dtype=first_result.dtype if first_result else "unknown",
            tp_size=1,  # TODO: Get from engine
            total_configs=len(results),
            runtime_seconds=runtime
        )
        save_metadata_json(metadata, args.metadata_output)

    # Exit with appropriate code
    if not passed:
        print("\n BENCHMARK FAILED: Performance regression detected")
        exit(1)
    else:
        print("\n BENCHMARK PASSED")
        exit(0)


if __name__ == "__main__":
    main()
```

## Stress Test Specification

### File: `test/srt/bench_score_stress.py`

```python
#!/usr/bin/env python3
"""
Score API stress tests.

Validates behavior under extreme conditions:
- Large batch sizes (50, 100, 200 items)
- Concurrent requests (4, 8, 16 threads)
- Extended duration (sustained load)

Usage:
    # Run all stress tests
    python test/srt/bench_score_stress.py

    # Run specific scenario
    python test/srt/bench_score_stress.py --scenario large-batch
    python test/srt/bench_score_stress.py --scenario concurrent

    # Custom configuration
    python test/srt/bench_score_stress.py --scenario large-batch --batch-sizes 50,100,150
"""

import argparse
import concurrent.futures
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    build_engine,
    get_tokenizer,
    get_label_token_ids,
)


# =============================================================================
# Configuration
# =============================================================================

STRESS_SCENARIOS = {
    "large-batch": {
        "description": "Test very large batch sizes",
        "batch_sizes": [50, 100, 200],
        "num_labels": 4,
        "iterations": 5,
        "timeout_seconds": 120,
    },
    "concurrent": {
        "description": "Test concurrent request handling",
        "concurrency_levels": [4, 8, 16],
        "batch_size": 8,
        "num_labels": 4,
        "duration_seconds": 30,
    },
    "sustained": {
        "description": "Test sustained load over time",
        "batch_size": 16,
        "num_labels": 4,
        "duration_seconds": 300,  # 5 minutes
        "target_rps": 10,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StressTestResult:
    """Result of a stress test scenario"""
    scenario: str
    configuration: str
    success: bool
    total_requests: int
    successful_requests: int
    failed_requests: int
    throughput_rps: float
    latency_mean_ms: float
    latency_p99_ms: float
    errors: List[str]
    duration_seconds: float


# =============================================================================
# Stress Test Implementations
# =============================================================================

def run_large_batch_stress(
    runner,
    tokenizer,
    batch_sizes: List[int],
    num_labels: int,
    iterations: int,
    timeout_seconds: float
) -> List[StressTestResult]:
    """
    Test behavior with very large batch sizes.

    Validates:
    - Memory handling (no OOM)
    - Completion within timeout
    - Correct result shape
    """
    results = []
    query = "Stress test query: " * 10
    label_tokens = [f" label{i}" for i in range(num_labels)]
    label_token_ids = get_label_token_ids(tokenizer, label_tokens)

    for batch_size in batch_sizes:
        print(f"\n  Testing batch_size={batch_size}...")
        items = [f" item {i}" for i in range(batch_size)]

        latencies = []
        errors = []
        successful = 0

        for i in range(iterations):
            try:
                start = time.perf_counter()
                scores = runner.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True
                )
                elapsed = time.perf_counter() - start

                # Verify shape
                if len(scores) != batch_size:
                    errors.append(f"Iteration {i}: Wrong shape {len(scores)} != {batch_size}")
                else:
                    latencies.append(elapsed * 1000)
                    successful += 1

                if elapsed > timeout_seconds:
                    errors.append(f"Iteration {i}: Timeout ({elapsed:.1f}s > {timeout_seconds}s)")

            except Exception as e:
                errors.append(f"Iteration {i}: {type(e).__name__}: {str(e)[:100]}")

        total_time = sum(latencies) / 1000 if latencies else 0
        results.append(StressTestResult(
            scenario="large-batch",
            configuration=f"batch_size={batch_size}",
            success=successful == iterations and not errors,
            total_requests=iterations,
            successful_requests=successful,
            failed_requests=iterations - successful,
            throughput_rps=successful / total_time if total_time > 0 else 0,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_p99_ms=latencies[-1] if latencies else 0,  # Sorted by occurrence
            errors=errors,
            duration_seconds=total_time,
        ))

    return results


def run_concurrent_stress(
    runner,
    tokenizer,
    concurrency_levels: List[int],
    batch_size: int,
    num_labels: int,
    duration_seconds: float
) -> List[StressTestResult]:
    """
    Test concurrent request handling.

    Validates:
    - Thread safety
    - Resource contention handling
    - Consistent results under load
    """
    results = []
    query = "Concurrent test query: " * 5
    items = [f" item {i}" for i in range(batch_size)]
    label_tokens = [f" label{i}" for i in range(num_labels)]
    label_token_ids = get_label_token_ids(tokenizer, label_tokens)

    for num_threads in concurrency_levels:
        print(f"\n  Testing concurrency={num_threads}...")

        # Shared state for results
        latencies = []
        errors = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                try:
                    start = time.perf_counter()
                    scores = runner.score(
                        query=query,
                        items=items,
                        label_token_ids=label_token_ids,
                        apply_softmax=True
                    )
                    elapsed = (time.perf_counter() - start) * 1000

                    with lock:
                        if len(scores) == batch_size:
                            latencies.append(elapsed)
                        else:
                            errors.append(f"Wrong shape: {len(scores)}")

                except Exception as e:
                    with lock:
                        errors.append(f"{type(e).__name__}: {str(e)[:50]}")

        # Run workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            time.sleep(duration_seconds)
            stop_event.set()
            concurrent.futures.wait(futures)

        actual_duration = time.time() - start_time
        successful = len(latencies)
        failed = len(errors)

        latencies.sort()
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1) if latencies else 0

        results.append(StressTestResult(
            scenario="concurrent",
            configuration=f"threads={num_threads}",
            success=failed == 0 and successful > 0,
            total_requests=successful + failed,
            successful_requests=successful,
            failed_requests=failed,
            throughput_rps=successful / actual_duration if actual_duration > 0 else 0,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_p99_ms=latencies[p99_idx] if latencies else 0,
            errors=errors[:10],  # Limit to first 10 errors
            duration_seconds=actual_duration,
        ))

    return results


def run_sustained_stress(
    runner,
    tokenizer,
    batch_size: int,
    num_labels: int,
    duration_seconds: float,
    target_rps: float
) -> StressTestResult:
    """
    Test sustained load over extended period.

    Validates:
    - Stability over time
    - No memory leaks
    - Consistent latency
    """
    print(f"\n  Running sustained load for {duration_seconds}s at {target_rps} RPS...")

    query = "Sustained test query: " * 5
    items = [f" item {i}" for i in range(batch_size)]
    label_tokens = [f" label{i}" for i in range(num_labels)]
    label_token_ids = get_label_token_ids(tokenizer, label_tokens)

    interval = 1.0 / target_rps
    latencies = []
    errors = []

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        request_start = time.perf_counter()
        try:
            scores = runner.score(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )
            elapsed = (time.perf_counter() - request_start) * 1000

            if len(scores) == batch_size:
                latencies.append(elapsed)
            else:
                errors.append(f"Wrong shape: {len(scores)}")

        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:50]}")

        # Rate limiting
        sleep_time = interval - (time.perf_counter() - request_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    actual_duration = time.time() - start_time
    successful = len(latencies)

    latencies.sort()
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1) if latencies else 0

    return StressTestResult(
        scenario="sustained",
        configuration=f"duration={duration_seconds}s, target_rps={target_rps}",
        success=len(errors) == 0 and successful > 0,
        total_requests=successful + len(errors),
        successful_requests=successful,
        failed_requests=len(errors),
        throughput_rps=successful / actual_duration if actual_duration > 0 else 0,
        latency_mean_ms=statistics.mean(latencies) if latencies else 0,
        latency_p99_ms=latencies[p99_idx] if latencies else 0,
        errors=errors[:10],
        duration_seconds=actual_duration,
    )


# =============================================================================
# Output
# =============================================================================

def print_results(results: List[StressTestResult]):
    """Print stress test results"""
    print(f"\n{'='*100}")
    print("STRESS TEST RESULTS")
    print(f"{'='*100}")

    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"\n{r.scenario.upper()} - {r.configuration}")
        print(f"  Status: {status}")
        print(f"  Requests: {r.successful_requests}/{r.total_requests} successful")
        print(f"  Throughput: {r.throughput_rps:.1f} RPS")
        print(f"  Latency: mean={r.latency_mean_ms:.1f}ms, p99={r.latency_p99_ms:.1f}ms")
        if r.errors:
            print(f"  Errors ({len(r.errors)}):")
            for err in r.errors[:5]:
                print(f"    - {err}")

    print(f"\n{'='*100}")

    # Summary
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    if failed > 0:
        print(f"\n {passed} passed, {failed} FAILED")
        return False
    else:
        print(f"\n All {passed} stress tests passed")
        return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score API stress tests")
    parser.add_argument(
        "--scenario",
        choices=["large-batch", "concurrent", "sustained", "all"],
        default="all",
        help="Stress test scenario to run"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--batch-sizes",
        help="Custom batch sizes for large-batch scenario (comma-separated)"
    )
    parser.add_argument(
        "--concurrency",
        help="Custom concurrency levels (comma-separated)"
    )

    args = parser.parse_args()

    print(f"\n{'='*100}")
    print("SCORE API STRESS TESTS")
    print(f"{'='*100}")
    print(f"Model: {args.model}")
    print(f"Scenario: {args.scenario}")
    print(f"{'='*100}")

    # Build engine
    print("\nBuilding engine...")
    config = ScoreTestConfig(model=args.model)
    runner = build_engine(config)
    tokenizer = get_tokenizer(args.model)

    results = []

    try:
        if args.scenario in ["large-batch", "all"]:
            print("\n--- Large Batch Stress Test ---")
            scenario_config = STRESS_SCENARIOS["large-batch"]
            batch_sizes = scenario_config["batch_sizes"]
            if args.batch_sizes:
                batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

            results.extend(run_large_batch_stress(
                runner=runner,
                tokenizer=tokenizer,
                batch_sizes=batch_sizes,
                num_labels=scenario_config["num_labels"],
                iterations=scenario_config["iterations"],
                timeout_seconds=scenario_config["timeout_seconds"],
            ))

        if args.scenario in ["concurrent", "all"]:
            print("\n--- Concurrent Request Stress Test ---")
            scenario_config = STRESS_SCENARIOS["concurrent"]
            concurrency_levels = scenario_config["concurrency_levels"]
            if args.concurrency:
                concurrency_levels = [int(x) for x in args.concurrency.split(',')]

            results.extend(run_concurrent_stress(
                runner=runner,
                tokenizer=tokenizer,
                concurrency_levels=concurrency_levels,
                batch_size=scenario_config["batch_size"],
                num_labels=scenario_config["num_labels"],
                duration_seconds=scenario_config["duration_seconds"],
            ))

        if args.scenario in ["sustained", "all"]:
            print("\n--- Sustained Load Stress Test ---")
            scenario_config = STRESS_SCENARIOS["sustained"]
            results.append(run_sustained_stress(
                runner=runner,
                tokenizer=tokenizer,
                batch_size=scenario_config["batch_size"],
                num_labels=scenario_config["num_labels"],
                duration_seconds=scenario_config["duration_seconds"],
                target_rps=scenario_config["target_rps"],
            ))

    finally:
        runner.shutdown()

    # Print results and exit
    passed = print_results(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

## CI Integration Specification

### File: `.github/workflows/nightly-perf.yaml`

```yaml
name: Nightly Performance Benchmarks

on:
  schedule:
    # Run at 2 AM UTC daily
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      profile:
        description: 'Benchmark profile to run'
        required: true
        default: 'smoke'
        type: choice
        options:
          - smoke
          - standard
          - full
      run_stress:
        description: 'Run stress tests'
        required: false
        default: false
        type: boolean

env:
  BENCHMARK_BASELINE: baselines/tpu-v6e-baseline.csv
  REGRESSION_THRESHOLD: 10

jobs:
  benchmark:
    name: Score API Benchmark
    runs-on: [self-hosted, tpu]
    timeout-minutes: 120

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run Benchmark
        id: benchmark
        run: |
          PROFILE="${{ inputs.profile || 'smoke' }}"
          echo "Running $PROFILE benchmark..."

          python test/srt/bench_score.py \
            --profile $PROFILE \
            --baseline $BENCHMARK_BASELINE \
            --regression-threshold $REGRESSION_THRESHOLD \
            --output results/benchmark-${{ github.run_number }}.csv \
            --metadata-output results/metadata-${{ github.run_number }}.json

      - name: Run Stress Tests
        if: inputs.run_stress == true
        run: |
          python test/srt/bench_score_stress.py --scenario all

      - name: Upload Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: perf-results-${{ github.run_number }}
          path: results/
          retention-days: 90

      - name: Create Summary
        if: always()
        run: |
          echo "## Performance Benchmark Results" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "- **Profile:** ${{ inputs.profile || 'smoke' }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Run:** #${{ github.run_number }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Commit:** ${{ github.sha }}" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY

          if [ -f results/benchmark-${{ github.run_number }}.csv ]; then
            echo "### Results" >> $GITHUB_STEP_SUMMARY
            echo "\`\`\`" >> $GITHUB_STEP_SUMMARY
            head -20 results/benchmark-${{ github.run_number }}.csv >> $GITHUB_STEP_SUMMARY
            echo "\`\`\`" >> $GITHUB_STEP_SUMMARY
          fi

  notify:
    name: Notify on Failure
    needs: benchmark
    if: failure()
    runs-on: ubuntu-latest
    steps:
      - name: Create Issue
        uses: actions/github-script@v7
        with:
          script: |
            const title = `Performance regression detected - Run #${{ github.run_number }}`;
            const body = `
            ## Performance Regression Alert

            A performance regression was detected in the nightly benchmark run.

            - **Run:** [#${{ github.run_number }}](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})
            - **Commit:** ${{ github.sha }}
            - **Profile:** ${{ inputs.profile || 'smoke' }}

            Please investigate the benchmark results and determine if this is:
            1. An expected change (update baseline)
            2. A real regression (needs fix)
            3. A flaky result (re-run)
            `;

            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: title,
              body: body,
              labels: ['performance', 'automated']
            });
```

## Test Execution Plan

### Step 1: Implement Enhanced Benchmark Tool

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Update bench_score.py with profile support
# (Apply changes from specification above)

# Test basic functionality
python test/srt/bench_score.py --profile smoke --output /tmp/smoke-test.csv
```

### Step 2: Implement Stress Tests

```bash
# Create stress test file
touch test/srt/bench_score_stress.py

# Test individual scenarios
python test/srt/bench_score_stress.py --scenario large-batch --batch-sizes 20,40
python test/srt/bench_score_stress.py --scenario concurrent --concurrency 2,4
```

### Step 3: Establish Baseline

```bash
# Run on representative hardware (TPU v6e)
python test/srt/bench_score.py \
  --profile standard \
  --output test/srt/baselines/tpu-v6e-baseline.csv \
  --metadata-output test/srt/baselines/tpu-v6e-metadata.json

# Commit baseline
git add test/srt/baselines/
git commit -m "Add TPU v6e performance baseline"
```

### Step 4: Test Regression Detection

```bash
# Should pass (comparing to itself)
python test/srt/bench_score.py \
  --profile smoke \
  --baseline test/srt/baselines/tpu-v6e-baseline.csv \
  --regression-threshold 10
```

### Step 5: Set Up CI

```bash
# Add workflow file
mkdir -p .github/workflows
# Copy nightly-perf.yaml

# Test locally with act (optional)
act workflow_dispatch -W .github/workflows/nightly-perf.yaml
```

## Success Criteria

- [ ] Benchmark tool supports smoke/standard/full profiles
- [ ] Metadata capture includes hardware, commit, timestamp
- [ ] Regression detection works with configurable threshold
- [ ] Stress tests validate large batches (50, 100 items)
- [ ] Stress tests validate concurrent requests (4, 8, 16 threads)
- [ ] Baseline established on TPU v6e
- [ ] CI workflow runs nightly and uploads artifacts
- [ ] Regression failures create GitHub issues

## Dependencies

- Test Plan 001 (requires `score_test_utils.py`)
- Test Plan 003 (builds on existing `bench_score.py`)
- TPU access for baseline establishment
- GitHub Actions self-hosted runner with TPU

## Risks

1. **Hardware variance** - Results differ between TPU types/generations
   - Mitigation: Separate baselines per hardware type

2. **Flaky stress tests** - Concurrent tests may have race conditions
   - Mitigation: Multiple iterations, statistical thresholds

3. **Baseline staleness** - Performance improves but baseline not updated
   - Mitigation: Monthly baseline review, document update process

4. **CI costs** - Nightly TPU runs add up
   - Mitigation: Use smoke profile by default, full only on-demand

## When to Run

**Benchmark Tool:**
- After performance-related changes
- Before releases
- When investigating performance issues

**Stress Tests:**
- Before releases (stability validation)
- After architectural changes
- When debugging production issues

**Nightly CI:**
- Automatically at 2 AM UTC
- On-demand via workflow_dispatch

## Follow-Up

After this plan completes:
- [ ] Monitor nightly runs for 2 weeks
- [ ] Adjust regression threshold if too sensitive
- [ ] Consider adding GPU baseline
- [ ] Evaluate need for performance dashboard
