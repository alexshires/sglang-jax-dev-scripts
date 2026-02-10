#!/usr/bin/env python3
"""
Multi-Item Chunk Size Benchmark

Standalone script to benchmark multi-item scoring at various chunk sizes.
Measures throughput, latency, and identifies recompilation behavior.

Usage:
    python bench_multi_item_chunk_size.py --output results.json
    python bench_multi_item_chunk_size.py --chunk-sizes 2,8,32 --output results.json
    python bench_multi_item_chunk_size.py --skip-serial --output results.json

Requirements:
    - Run from sglang-jax directory with venv activated
    - TPU v6e-1 or compatible hardware
    - Model: Qwen/Qwen3-0.6B (or specify --model)
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

# Ensure we can import from sglang-jax
try:
    import jax
    from transformers import AutoTokenizer
    from sgl_jax.srt.entrypoints.engine import Engine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the sglang-jax directory with venv activated:")
    print("  cd ~/sglang-jax && source .venv/bin/activate")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration matching the target scenario."""
    # Query configuration
    static_prefix_len: int = 100      # System prompt
    dynamic_suffix_len: int = 1900    # User context

    # Item configuration
    num_items: int = 500
    tokens_per_item: int = 20

    # Benchmark settings
    num_warmup_requests: int = 1
    num_timed_requests: int = 5

    # Chunk sizes to test (0 = all items in one pass)
    chunk_sizes: tuple = (2, 8, 32, 64, 128, 256, 500)

    # Model settings
    model_path: str = "Qwen/Qwen3-0.6B"
    delimiter_token_id: int = 128001

    @property
    def query_len(self) -> int:
        return self.static_prefix_len + self.dynamic_suffix_len


@dataclass
class ChunkSizeResult:
    """Results for a single chunk size."""
    chunk_size: int
    num_chunks: int
    tokens_per_chunk: int
    mask_size_mb: float

    # Latency measurements
    first_request_ms: float
    steady_state_mean_ms: float
    steady_state_std_ms: float
    all_latencies_ms: list

    # Throughput
    throughput_items_sec: float

    # Observations
    notes: str = ""


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    metadata: dict
    chunk_size_results: list
    serial_baseline: Optional[dict] = None


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_tokens_per_chunk(query_len: int, chunk_size: int, tokens_per_item: int) -> int:
    """Calculate total tokens in a packed chunk."""
    # query + chunk_size * (item_tokens + delimiter)
    return query_len + chunk_size * (tokens_per_item + 1)


def calculate_mask_size_mb(tokens_per_chunk: int) -> float:
    """Calculate mask size in MB (int32 = 4 bytes per element)."""
    return (tokens_per_chunk ** 2 * 4) / (1024 * 1024)


def calculate_num_chunks(num_items: int, chunk_size: int) -> int:
    """Calculate number of chunks needed."""
    if chunk_size <= 0 or chunk_size >= num_items:
        return 1
    return (num_items + chunk_size - 1) // chunk_size


# =============================================================================
# Benchmark Runner
# =============================================================================

class ChunkSizeBenchmark:
    """Runs chunk size benchmarks."""

    def __init__(self, config: BenchmarkConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.engine = None
        self.tokenizer = None

        # Generate test data once
        self.query_tokens = [1] * config.query_len
        self.item_tokens_list = [
            [2] * config.tokens_per_item
            for _ in range(config.num_items)
        ]
        self.label_token_ids = [198]  # Common token for scoring

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def _create_engine(self, chunk_size: int) -> Engine:
        """Create engine with specific chunk size."""
        self.log(f"  Creating engine with chunk_size={chunk_size}...")

        # Use 0 to disable chunking (all items in one pass)
        effective_chunk_size = 0 if chunk_size >= self.config.num_items else chunk_size

        engine = Engine(
            model_path=self.config.model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=42,
            node_rank=0,
            mem_fraction_static=0.7,
            max_prefill_tokens=32768,
            chunked_prefill_size=-1,  # Disable chunked prefill for multi-item
            download_dir=os.path.expanduser("~/.cache/huggingface"),
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024, 2048, 4096, 8192, 16384],
            page_size=64,
            log_requests=False,
            # Multi-item settings
            multi_item_scoring_delimiter=self.config.delimiter_token_id,
            multi_item_scoring_chunk_size=effective_chunk_size,
            max_multi_item_seq_len=32768,
            disable_radix_cache=True,
        )

        return engine

    def _shutdown_engine(self):
        """Clean up engine resources."""
        if self.engine is not None:
            self.log("  Shutting down engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                self.log(f"  Warning: Engine shutdown error: {e}")
            self.engine = None

        # Force garbage collection and clear JAX caches
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    def run_chunk_size_benchmark(self, chunk_size: int) -> ChunkSizeResult:
        """Run benchmark for a specific chunk size."""
        self.log(f"\n{'='*60}")
        self.log(f"Benchmarking chunk_size={chunk_size}")
        self.log(f"{'='*60}")

        # Calculate expected metrics
        effective_chunk_size = min(chunk_size, self.config.num_items)
        num_chunks = calculate_num_chunks(self.config.num_items, effective_chunk_size)
        tokens_per_chunk = calculate_tokens_per_chunk(
            self.config.query_len, effective_chunk_size, self.config.tokens_per_item
        )
        mask_size_mb = calculate_mask_size_mb(tokens_per_chunk)

        self.log(f"  Expected: {num_chunks} chunks, {tokens_per_chunk} tokens/chunk, {mask_size_mb:.1f} MB mask")

        # Create fresh engine
        self._shutdown_engine()
        self.engine = self._create_engine(chunk_size)

        # Warmup
        self.log(f"  Warming up ({self.config.num_warmup_requests} requests)...")
        for _ in range(self.config.num_warmup_requests):
            self.engine.score(
                query=self.query_tokens,
                items=self.item_tokens_list[:10],  # Small warmup
                label_token_ids=self.label_token_ids,
            )

        # Timed runs
        self.log(f"  Running {self.config.num_timed_requests} timed requests...")
        latencies_ms = []

        for i in range(self.config.num_timed_requests):
            start = time.perf_counter()
            self.engine.score(
                query=self.query_tokens,
                items=self.item_tokens_list,
                label_token_ids=self.label_token_ids,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            self.log(f"    Request {i+1}: {elapsed_ms:.1f} ms")

        # Calculate statistics
        first_request_ms = latencies_ms[0]
        steady_state = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
        steady_state_mean = statistics.mean(steady_state)
        steady_state_std = statistics.stdev(steady_state) if len(steady_state) > 1 else 0.0

        # Throughput based on steady state
        throughput = (self.config.num_items / steady_state_mean) * 1000  # items/sec

        # Check for potential recompilation
        notes = ""
        if first_request_ms > steady_state_mean * 1.5:
            notes = f"First request {first_request_ms/steady_state_mean:.1f}x slower (possible compilation)"

        result = ChunkSizeResult(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            tokens_per_chunk=tokens_per_chunk,
            mask_size_mb=mask_size_mb,
            first_request_ms=first_request_ms,
            steady_state_mean_ms=steady_state_mean,
            steady_state_std_ms=steady_state_std,
            all_latencies_ms=latencies_ms,
            throughput_items_sec=throughput,
            notes=notes,
        )

        self.log(f"  Results: {throughput:.1f} items/sec, {steady_state_mean:.1f}ms mean latency")
        if notes:
            self.log(f"  Note: {notes}")

        return result

    def run_serial_baseline(self) -> dict:
        """Run serial baseline (single-item scoring in batches)."""
        self.log(f"\n{'='*60}")
        self.log("Running Serial Baseline (batch_size=32)")
        self.log(f"{'='*60}")

        # Create engine without multi-item chunking
        self._shutdown_engine()
        self.engine = self._create_engine(chunk_size=1)  # Minimal chunking

        # Warmup
        self.log("  Warming up...")
        self.engine.score(
            query=self.query_tokens,
            items=self.item_tokens_list[:32],
            label_token_ids=self.label_token_ids,
        )

        # Time scoring all items in batches of 32
        self.log(f"  Scoring {self.config.num_items} items in batches of 32...")
        batch_size = 32

        start = time.perf_counter()
        for i in range(0, self.config.num_items, batch_size):
            batch = self.item_tokens_list[i:i + batch_size]
            self.engine.score(
                query=self.query_tokens,
                items=batch,
                label_token_ids=self.label_token_ids,
            )
        total_time = time.perf_counter() - start

        throughput = self.config.num_items / total_time

        result = {
            "method": "serial_batch_32",
            "total_time_sec": total_time,
            "throughput_items_sec": throughput,
            "latency_per_item_ms": (total_time * 1000) / self.config.num_items,
        }

        self.log(f"  Results: {throughput:.1f} items/sec, {total_time:.1f}s total")

        return result

    def run_full_benchmark(
        self,
        chunk_sizes: tuple = None,
        include_serial: bool = True,
    ) -> BenchmarkResults:
        """Run complete benchmark suite."""
        if chunk_sizes is None:
            chunk_sizes = self.config.chunk_sizes

        self.log("=" * 60)
        self.log("Multi-Item Chunk Size Benchmark")
        self.log("=" * 60)
        self.log(f"Query: {self.config.query_len} tokens")
        self.log(f"Items: {self.config.num_items} x {self.config.tokens_per_item} tokens")
        self.log(f"Chunk sizes to test: {chunk_sizes}")
        self.log("")

        # Metadata
        metadata = {
            "model": self.config.model_path,
            "query_tokens": self.config.query_len,
            "num_items": self.config.num_items,
            "tokens_per_item": self.config.tokens_per_item,
            "hardware": "TPU v6e-1",
            "timestamp": datetime.now().isoformat(),
            "num_warmup_requests": self.config.num_warmup_requests,
            "num_timed_requests": self.config.num_timed_requests,
        }

        # Run benchmarks for each chunk size
        chunk_results = []
        for cs in chunk_sizes:
            try:
                result = self.run_chunk_size_benchmark(cs)
                chunk_results.append(asdict(result))
            except Exception as e:
                self.log(f"  ERROR: {e}")
                chunk_results.append({
                    "chunk_size": cs,
                    "error": str(e),
                })

        # Run serial baseline
        serial_result = None
        if include_serial:
            try:
                serial_result = self.run_serial_baseline()
            except Exception as e:
                self.log(f"  Serial baseline ERROR: {e}")
                serial_result = {"error": str(e)}

        # Cleanup
        self._shutdown_engine()

        return BenchmarkResults(
            metadata=metadata,
            chunk_size_results=chunk_results,
            serial_baseline=serial_result,
        )


# =============================================================================
# Report Generation
# =============================================================================

def print_summary_table(results: BenchmarkResults):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Chunk':<8} {'Chunks':<8} {'Tokens':<8} {'Mask':<10} "
          f"{'First(ms)':<12} {'Steady(ms)':<12} {'Items/s':<10} {'Speedup':<8}")
    print("-" * 80)

    # Get serial baseline throughput for speedup calculation
    serial_throughput = None
    if results.serial_baseline and "throughput_items_sec" in results.serial_baseline:
        serial_throughput = results.serial_baseline["throughput_items_sec"]
        print(f"{'Serial':<8} {'-':<8} {'-':<8} {'-':<10} "
              f"{'-':<12} {'-':<12} {serial_throughput:<10.1f} {'1.0x':<8}")
        print("-" * 80)

    # Chunk size results
    for r in results.chunk_size_results:
        if "error" in r:
            print(f"{r['chunk_size']:<8} ERROR: {r['error']}")
            continue

        speedup = "-"
        if serial_throughput:
            speedup = f"{r['throughput_items_sec'] / serial_throughput:.1f}x"

        print(f"{r['chunk_size']:<8} {r['num_chunks']:<8} {r['tokens_per_chunk']:<8} "
              f"{r['mask_size_mb']:<10.1f} {r['first_request_ms']:<12.1f} "
              f"{r['steady_state_mean_ms']:<12.1f} {r['throughput_items_sec']:<10.1f} {speedup:<8}")

    print("=" * 80)

    # Notes
    print("\nNotes:")
    for r in results.chunk_size_results:
        if "notes" in r and r.get("notes"):
            print(f"  chunk_size={r['chunk_size']}: {r['notes']}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-item scoring at various chunk sizes"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="chunk_size_benchmark_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="2,8,32,64,128,256,500",
        help="Comma-separated list of chunk sizes to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model to use for benchmarking"
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=500,
        help="Number of items to score"
    )
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="Skip serial baseline benchmark"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse chunk sizes
    chunk_sizes = tuple(int(x.strip()) for x in args.chunk_sizes.split(","))

    # Create config
    config = BenchmarkConfig(
        model_path=args.model,
        num_items=args.num_items,
        chunk_sizes=chunk_sizes,
    )

    # Run benchmark
    benchmark = ChunkSizeBenchmark(config, verbose=not args.quiet)
    results = benchmark.run_full_benchmark(
        chunk_sizes=chunk_sizes,
        include_serial=not args.skip_serial,
    )

    # Print summary
    print_summary_table(results)

    # Save results
    output_data = {
        "metadata": results.metadata,
        "results": results.chunk_size_results,
        "baseline_serial": results.serial_baseline,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
