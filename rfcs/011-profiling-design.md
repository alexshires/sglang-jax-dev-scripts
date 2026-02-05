# RFC-011: Comprehensive Profiling Framework for sglang-jax

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-05 |
| **Updated** | 2026-02-05 |
| **Related** | [RFC-004](004-score-api-performance-benchmarks.md), [RFC-010](010-cross-backend-benchmarking.md) |

## Summary

Design a comprehensive profiling framework for sglang-jax that enables deep performance analysis of JAX/TPU workloads, with special focus on the Score API. This RFC covers hardware profiling (TPU traces), software profiling (Python/JAX), memory profiling, and provides step-by-step operational guides.

## Table of Contents

1. [Motivation](#motivation)
2. [Current State Analysis](#current-state-analysis)
3. [Profiling Architecture](#profiling-architecture)
4. [JAX/TPU Profiling Tools](#jaxtpu-profiling-tools)
5. [Proposed Enhancements](#proposed-enhancements)
6. [Score API Profiling Strategy](#score-api-profiling-strategy)
7. [Step-by-Step Profiling Guides](#step-by-step-profiling-guides)
8. [Visualization Tools](#visualization-tools)
9. [General Benchmarking Tools](#general-benchmarking-tools)
10. [Distributed Profiling (TP/EP)](#distributed-profiling-tpep)
11. [Advanced Options](#advanced-options)
12. [Known Issues and Troubleshooting](#known-issues-and-troubleshooting)
13. [Implementation Plan](#implementation-plan)
14. [Cost Analysis](#cost-analysis)

---

## Motivation

### Why Profiling Matters

1. **Performance optimization** - Identify bottlenecks in Score API computation
2. **Memory analysis** - Understand HBM usage patterns, detect memory leaks
3. **Compilation analysis** - Verify XLA compilation is cached, identify recompilations
4. **Multi-device scaling** - Analyze TP/EP communication overhead
5. **Regression detection** - Compare performance before/after changes
6. **TPU utilization** - Ensure MXU (Matrix Multiply Unit) is efficiently utilized

### Current Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No Score API-specific profiling | Can't isolate scoring overhead | P0 |
| Limited memory profiling in hot paths | Silent HBM exhaustion | P0 |
| No layer-by-layer breakdown for scoring | Can't optimize individual layers | P1 |
| No automated profiling in CI | Regressions undetected | P1 |
| No cross-backend profiling comparison | Can't compare JAX vs PyTorch | P2 |

### Goals

1. **Enable end-to-end Score API profiling** with single command
2. **Provide layer-by-layer performance breakdown** for scoring workloads
3. **Automate memory profiling** for Score API requests
4. **Integrate profiling into nightly CI** for regression detection
5. **Document step-by-step profiling workflows** for all scenarios

---

## Current State Analysis

### Existing Profiling Infrastructure in sglang-jax

#### 1. HTTP Profiling Client (`python/sgl_jax/profiler.py`)

```python
# Location: sglang-jax/python/sgl_jax/profiler.py
# Purpose: HTTP-based profiling orchestration

def run_profile(url, num_steps, activities, output_dir, profile_name, profile_by_stage):
    """
    Sends POST to /start_profile endpoint.
    Supports activities: CPU, GPU, MEM, RPD
    """
```

**Capabilities:**
- HTTP POST to `/start_profile` endpoint
- Configurable number of steps to profile
- Output directory management
- Server info retrieval

**Usage:**
```bash
python -m sgl_jax.profiler --url http://localhost:30000 --num-steps 5 --output-dir /tmp/profile
```

#### 2. Scheduler Profiler Mixin (`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py`)

```python
# Location: sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py
# Purpose: JAX profiler integration for scheduler

class SchedulerProfilerMixin:
    def start_profile(self, output_dir, start_step, num_steps, host_tracer_level, python_tracer_level, profile_id):
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.host_tracer_level = host_tracer_level
        profiler_options.python_tracer_level = python_tracer_level
        jax.profiler.start_trace(output_dir, profiler_options=profiler_options)
```

**Capabilities:**
- JAX `start_trace`/`stop_trace` integration
- Configurable tracer levels (host, Python)
- Per-request profiling
- Forward step counting for targeted profiling

#### 3. Memory Profiler (`python/sgl_jax/srt/memory_profiler.py`)

```python
# Location: sglang-jax/python/sgl_jax/srt/memory_profiler.py
# Purpose: Memory profiling and analysis

class MemoryProfiler:
    """Context manager for profiling memory usage"""

def memory_profile(stage, layer_id, report_type):
    """Decorator for memory profiling functions"""

def profile_attention(stage, layer_id):
    """Specialized profiler for attention layers"""
```

**Capabilities:**
- Uses `jax.profiler.save_device_memory_profile()`
- Layer-by-layer memory analysis
- JSON/text report generation
- Tensor memory calculation
- Environment variable configuration

**Configuration:**
```bash
export ENABLE_MEMORY_PROFILING=1
export SGL_MEMORY_OUTPUT_DIR=/tmp/memory_profiles
export MEMORY_PROFILING_LAYERS=4  # Profile every 4th layer
```

#### 4. Kernel Performance Utilities (`python/sgl_jax/srt/kernels/utils/perf.py`)

```python
# Location: sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py
# Purpose: Kernel-level performance profiling

def multiple_iteration_timeit_from_trace(compute_func, data_generator, task, tries=5, trace_root="/tmp/trace"):
    """
    Profile multiple iterations and extract per-iteration kernel time from trace.
    Uses jax.profiler.StepTraceAnnotation for step-based tracing.
    """
```

**Capabilities:**
- Extract kernel durations from JAX traces
- Parse gzip-compressed trace JSON
- Step-based trace annotations
- Device duration extraction (picoseconds)

#### 5. Profiling Utilities (`python/sgl_jax/srt/utils/profiling_utils.py`)

```python
# Location: sglang-jax/python/sgl_jax/srt/utils/profiling_utils.py
# Purpose: JAX named scope utilities

def named_scope(name_or_obj):
    """Decorator for wrapping functions with jax.named_scope()"""
```

**Usage in codebase:**
- `jax.named_scope("forward_batch")` in scheduler
- `jax.profiler.TraceAnnotation("run_batch")` in model_runner
- Named scopes throughout attention, MLP layers

### Existing Benchmarks

| Benchmark | Location | Purpose |
|-----------|----------|---------|
| `bench_one_batch.py` | `python/sgl_jax/` | Single batch latency (with `--profile` flag) |
| `bench_serving.py` | `python/sgl_jax/` | Online serving throughput |
| `bench_offline_throughput.py` | `python/sgl_jax/` | Offline batch throughput |
| `bench_flashattention.py` | `benchmark/kernels/` | FlashAttention kernel timing |
| `bench_update_kv_cache.py` | `benchmark/kernels/` | KV cache operations |
| `bench_fused_moe.py` | `benchmark/moe/` | Mixture-of-Experts kernels |

### PyTorch SGLang Profiling (Reference)

PyTorch SGLang has more mature profiling infrastructure we can learn from:

| Feature | PyTorch Implementation | JAX Status |
|---------|----------------------|------------|
| Stage-based profiling (prefill/decode) | `profile_utils.py` with `ProfileManager` | Partial (scheduler mixin) |
| Multi-rank trace merging | `profile_merger.py` | Missing |
| NVTX annotations | `nvtx_pytorch_hooks.py` | N/A (JAX uses named_scope) |
| Nsight Systems integration | `gputrc2graph.py` | N/A (use XProf instead) |
| Memory snapshots | PyTorch memory viz | JAX device memory profile |

---

## Profiling Architecture

### Request Flow with Profiling Points

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Score API Request Flow                               │
└──────────────────────────────────────────────────────────────────────────────┘

[Client Request]
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HTTP Server (/v1/score)                                    [PROFILE POINT 1]│
│  - Request parsing                                          HTTP latency      │
│  - Validation                                               Input size        │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TokenizerManager.score_request()                           [PROFILE POINT 2]│
│  - Tokenization                                             CPU time          │
│  - Prompt concatenation (query + items)                     Memory alloc      │
│  - Batch creation (GenerateReqInput)                                          │
│                                                                               │
│  ⚠️ DEVICE-AGNOSTIC: Uses scipy.special.softmax (ADR-001)                    │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼ (IPC to subprocess)
┌──────────────────────────────────────────────────────────────────────────────┐
│  Scheduler (subprocess)                                     [PROFILE POINT 3]│
│  - Request queuing                                          Queue wait time   │
│  - Batch formation                                          Batch size        │
│  - Memory allocation                                        HBM usage         │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Model Forward Pass (JAX/TPU)                               [PROFILE POINT 4]│
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Embedding Layer                                          HBM access     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Transformer Layers × N                                   MXU utilization│  │
│  │  ├─ Attention (QKV projection, softmax, output proj)    Kernel time    │  │
│  │  ├─ MLP (up_proj, gate, down_proj)                      Memory bandwidth│  │
│  │  └─ LayerNorm, residuals                                                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Logits Computation                                       [PROFILE POINT 5]│
│  │  - lm_head projection                                   Final layer time │
│  │  - Token ID logprob extraction                          Label extraction │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Post-processing (TokenizerManager)                         [PROFILE POINT 6]│
│  - Logprob extraction                                       CPU time          │
│  - Softmax application (scipy)                              Pure Python       │
│  - Response formatting                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
[Client Response]
```

### Profiling Levels

| Level | Scope | Tool | Output | Use Case |
|-------|-------|------|--------|----------|
| **L1: End-to-End** | Full request | `bench_serving.py --profile` | Trace file | Overall latency |
| **L2: Stage** | Prefill/Decode | Scheduler profiler mixin | Stage traces | Identify slow stage |
| **L3: Layer** | Individual layers | Memory profiler | Per-layer report | Layer optimization |
| **L4: Kernel** | Individual ops | `perf.py` utilities | Kernel durations | Kernel optimization |
| **L5: Memory** | HBM usage | `memory_profiler.py` | Memory snapshots | Memory optimization |

---

## JAX/TPU Profiling Tools

### Core JAX Profiling APIs

#### 1. `jax.profiler.trace()` - Full Trace Context

```python
import jax

# Capture full trace of code block
with jax.profiler.trace("/tmp/jax_trace"):
    result = my_jax_function(inputs)
    jax.block_until_ready(result)

# Trace saved to: /tmp/jax_trace/plugins/profile/<timestamp>/
```

**Output:** `.trace.json.gz` file viewable in Perfetto/TensorBoard

#### 2. `jax.profiler.start_trace()` / `stop_trace()` - Manual Control

```python
import jax

# Start profiling
jax.profiler.start_trace("/tmp/jax_trace", create_perfetto_link=True)

# Run workload
for i in range(10):
    result = my_jax_function(inputs)
    jax.block_until_ready(result)

# Stop and save
jax.profiler.stop_trace()
```

#### 3. `jax.named_scope()` - Hierarchical Tracing

```python
import jax

def transformer_layer(x, layer_idx):
    with jax.named_scope(f"layer_{layer_idx}"):
        with jax.named_scope("attention"):
            x = attention(x)
        with jax.named_scope("mlp"):
            x = mlp(x)
    return x
```

**Trace output:**
```
layer_0
├── attention
│   ├── qkv_proj
│   ├── softmax
│   └── output_proj
└── mlp
    ├── up_proj
    ├── gate
    └── down_proj
```

#### 4. `jax.profiler.TraceAnnotation()` - Event Markers

```python
import jax

with jax.profiler.TraceAnnotation("score_batch", batch_size=32, num_items=16):
    scores = score_batch(queries, items)
```

#### 5. `jax.profiler.StepTraceAnnotation()` - Iteration Tracking

```python
import jax

for step in range(100):
    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
        loss = train_step(batch)
```

#### 6. `jax.profiler.save_device_memory_profile()` - Memory Snapshots

```python
import jax

# Take HBM snapshot
jax.profiler.save_device_memory_profile("/tmp/memory.prof")

# View with: go tool pprof -http=:8080 /tmp/memory.prof
```

#### 7. `jax.profiler.ProfileOptions` - Configuration

```python
import jax

options = jax.profiler.ProfileOptions()
options.host_tracer_level = 2  # 0=off, 1=minimal, 2=medium, 3=verbose
options.python_tracer_level = 1  # 0=off, 1=enabled
options.device_tracer_level = 1  # Device-level tracing

jax.profiler.start_trace("/tmp/trace", profiler_options=options)
```

### TPU-Specific Profiling

#### TPU Metrics Available in Traces

| Metric | Description | Unit |
|--------|-------------|------|
| `device_duration_ps` | Kernel execution time | picoseconds |
| `flops` | Floating point operations | count |
| `bytes_accessed` | HBM memory accessed | bytes |
| `MXU_utilization` | Matrix unit utilization | percentage |
| `infeed/outfeed` | Host-device transfer | bytes |

#### TPU Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        TPU v6e Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   MXU       │     │   MXU       │     │   MXU       │       │
│   │ (Matrix     │     │ (Matrix     │     │ (Matrix     │       │
│   │  Multiply)  │     │  Multiply)  │     │  Multiply)  │       │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│          │                   │                   │               │
│          └───────────────────┴───────────────────┘               │
│                              │                                   │
│                     ┌────────┴────────┐                          │
│                     │   Vector Unit   │                          │
│                     │   (softmax,     │                          │
│                     │    activations) │                          │
│                     └────────┬────────┘                          │
│                              │                                   │
│                     ┌────────┴────────┐                          │
│                     │      VMEM       │  ← Scratchpad memory     │
│                     │    (fast)       │                          │
│                     └────────┬────────┘                          │
│                              │                                   │
│                     ┌────────┴────────┐                          │
│                     │      HBM        │  ← High Bandwidth Memory │
│                     │   (16-32 GB)    │     (main memory)        │
│                     └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Proposed Enhancements

### Enhancement 1: Score API Profiling Module

Create a dedicated profiling module for Score API workloads.

**File:** `sglang-jax/python/sgl_jax/srt/score_profiler.py`

```python
"""Score API Profiling Module

Provides comprehensive profiling for /v1/score requests including:
- End-to-end latency breakdown
- Layer-by-layer performance
- Memory usage per request
- Kernel-level timing
"""

import jax
import jax.profiler
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json


@dataclass
class ScoreProfileResult:
    """Profiling result for a Score API request."""

    # Timing breakdown (milliseconds)
    total_latency_ms: float
    tokenization_ms: float
    queue_wait_ms: float
    model_forward_ms: float
    logprob_extraction_ms: float
    softmax_ms: float

    # Batch info
    batch_size: int
    num_items: int
    num_label_tokens: int
    sequence_lengths: List[int]

    # Memory info (optional)
    peak_hbm_mb: Optional[float] = None
    activation_memory_mb: Optional[float] = None

    # Additional metadata
    model_name: Optional[str] = None
    hardware: Optional[str] = None


class ScoreProfiler:
    """Profiler for Score API requests."""

    def __init__(
        self,
        output_dir: str = "/tmp/score_profiles",
        enable_trace: bool = True,
        enable_memory: bool = True,
        enable_layer_breakdown: bool = False,
    ):
        self.output_dir = output_dir
        self.enable_trace = enable_trace
        self.enable_memory = enable_memory
        self.enable_layer_breakdown = enable_layer_breakdown
        self._timings: Dict[str, float] = {}

    @contextmanager
    def profile_request(self, request_id: str, batch_size: int, num_items: int):
        """Context manager for profiling a score request."""
        self._timings = {}
        start_time = time.perf_counter()

        trace_path = f"{self.output_dir}/{request_id}"

        if self.enable_trace:
            jax.profiler.start_trace(trace_path, create_perfetto_link=False)

        try:
            with jax.profiler.TraceAnnotation(
                "score_request",
                batch_size=batch_size,
                num_items=num_items
            ):
                yield self
        finally:
            if self.enable_trace:
                jax.profiler.stop_trace()

            total_time = (time.perf_counter() - start_time) * 1000
            self._timings["total"] = total_time

    @contextmanager
    def time_stage(self, stage_name: str):
        """Time a specific stage of processing."""
        start = time.perf_counter()
        with jax.named_scope(stage_name):
            yield
        self._timings[stage_name] = (time.perf_counter() - start) * 1000

    def get_result(self, **kwargs) -> ScoreProfileResult:
        """Get the profiling result."""
        return ScoreProfileResult(
            total_latency_ms=self._timings.get("total", 0),
            tokenization_ms=self._timings.get("tokenization", 0),
            queue_wait_ms=self._timings.get("queue_wait", 0),
            model_forward_ms=self._timings.get("model_forward", 0),
            logprob_extraction_ms=self._timings.get("logprob_extraction", 0),
            softmax_ms=self._timings.get("softmax", 0),
            **kwargs
        )


# Singleton instance for easy access
_score_profiler: Optional[ScoreProfiler] = None

def get_score_profiler() -> ScoreProfiler:
    """Get or create the global score profiler."""
    global _score_profiler
    if _score_profiler is None:
        _score_profiler = ScoreProfiler()
    return _score_profiler
```

### Enhancement 2: Score API Benchmark with Profiling

Create a dedicated benchmark script that combines benchmarking with profiling.

**File:** `sglang-jax/test/srt/bench_score.py`

```python
#!/usr/bin/env python3
"""Score API Benchmark with Profiling Support

Usage:
    # Quick benchmark
    python test/srt/bench_score.py --profile smoke

    # Standard benchmark with profiling
    python test/srt/bench_score.py --profile standard --enable-trace

    # Full benchmark with memory profiling
    python test/srt/bench_score.py --profile full --enable-memory
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import jax
import jax.profiler


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    batch_sizes: List[int]
    label_counts: List[int]
    num_runs: int
    warmup_runs: int
    model: str
    dtype: str


PROFILES = {
    "smoke": BenchmarkConfig(
        batch_sizes=[1, 4, 16],
        label_counts=[2, 4],
        num_runs=5,
        warmup_runs=2,
        model="meta-llama/Llama-3.2-1B-Instruct",
        dtype="bfloat16",
    ),
    "standard": BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        label_counts=[2, 4, 8],
        num_runs=20,
        warmup_runs=5,
        model="meta-llama/Llama-3.2-1B-Instruct",
        dtype="bfloat16",
    ),
    "full": BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        label_counts=[2, 4, 8, 16],
        num_runs=50,
        warmup_runs=10,
        model="meta-llama/Llama-3.2-1B-Instruct",
        dtype="bfloat16",
    ),
}


async def run_benchmark(
    config: BenchmarkConfig,
    base_url: str,
    enable_trace: bool = False,
    enable_memory: bool = False,
    output_dir: str = "/tmp/score_benchmark",
) -> Dict[str, Any]:
    """Run the benchmark with optional profiling."""
    results = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for batch_size in config.batch_sizes:
        for label_count in config.label_counts:
            # Warmup
            for _ in range(config.warmup_runs):
                await run_single_score_request(base_url, batch_size, label_count)

            # Measurement
            latencies = []

            trace_path = None
            if enable_trace:
                trace_path = f"{output_dir}/trace_b{batch_size}_l{label_count}"
                jax.profiler.start_trace(trace_path)

            for run in range(config.num_runs):
                start = time.perf_counter()
                await run_single_score_request(base_url, batch_size, label_count)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            if enable_trace:
                jax.profiler.stop_trace()

            # Calculate statistics
            latencies.sort()
            results.append({
                "batch_size": batch_size,
                "label_count": label_count,
                "p50_ms": latencies[len(latencies) // 2],
                "p95_ms": latencies[int(len(latencies) * 0.95)],
                "p99_ms": latencies[int(len(latencies) * 0.99)],
                "mean_ms": sum(latencies) / len(latencies),
                "throughput_items_per_sec": batch_size * 1000 / (sum(latencies) / len(latencies)),
                "trace_path": trace_path,
            })

    return {
        "config": config.__dict__,
        "results": results,
        "metadata": get_environment_metadata(),
    }


async def run_single_score_request(base_url: str, batch_size: int, label_count: int):
    """Run a single score request."""
    import aiohttp

    # Generate test data
    query = "Is this a positive review? "
    items = [f"Sample item {i} with some text content." for i in range(batch_size)]
    label_token_ids = list(range(label_count))  # Placeholder

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/score",
            json={
                "query": query,
                "items": items,
                "label_token_ids": label_token_ids,
                "model": "default",
            },
        ) as response:
            return await response.json()


def get_environment_metadata() -> Dict[str, Any]:
    """Get environment metadata for reproducibility."""
    import subprocess

    return {
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "device_count": jax.device_count(),
        "platform": jax.default_backend(),
        "commit": subprocess.getoutput("git rev-parse --short HEAD"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    parser = argparse.ArgumentParser(description="Score API Benchmark")
    parser.add_argument("--profile", choices=PROFILES.keys(), default="smoke")
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--enable-trace", action="store_true")
    parser.add_argument("--enable-memory", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/score_benchmark")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config = PROFILES[args.profile]
    results = asyncio.run(run_benchmark(
        config,
        args.base_url,
        args.enable_trace,
        args.enable_memory,
        args.output_dir,
    ))

    # Print results
    print("\n" + "=" * 80)
    print(f"Score API Benchmark Results ({args.profile} profile)")
    print("=" * 80)
    print(f"{'Batch':>6} {'Labels':>6} {'p50':>10} {'p95':>10} {'p99':>10} {'Throughput':>12}")
    print(f"{'Size':>6} {'Count':>6} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(items/s)':>12}")
    print("-" * 80)

    for r in results["results"]:
        print(f"{r['batch_size']:>6} {r['label_count']:>6} {r['p50_ms']:>10.2f} "
              f"{r['p95_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['throughput_items_per_sec']:>12.1f}")

    # Save JSON output
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    if args.enable_trace:
        print(f"\nTrace files saved to: {args.output_dir}")
        print("View with: https://ui.perfetto.dev or tensorboard --logdir=" + args.output_dir)


if __name__ == "__main__":
    main()
```

### Enhancement 3: Layer-by-Layer Profiling Integration

Add named scopes to the scoring path for fine-grained profiling.

**Integration points in `tokenizer_manager.py`:**

```python
# In score_request() method around line 1210

async def score_request(self, query, items, label_token_ids, apply_softmax, item_first, request):
    with jax.named_scope("score_request"):
        # Tokenization
        with jax.named_scope("tokenization"):
            if isinstance(query, str):
                # Text processing...
                pass

        # Batch creation
        with jax.named_scope("batch_creation"):
            batch_request = GenerateReqInput(...)

        # Model forward (handled by scheduler)
        with jax.named_scope("model_inference"):
            results = await self.generate_request(batch_request, request).__anext__()

        # Post-processing
        with jax.named_scope("post_processing"):
            with jax.named_scope("logprob_extraction"):
                # Extract logprobs...
                pass

            with jax.named_scope("softmax"):
                if apply_softmax:
                    score_list = softmax(score_list).tolist()

        return scores
```

---

## Score API Profiling Strategy

### Profiling Scenarios

#### Scenario 1: Single Request Latency Analysis

**Goal:** Understand where time is spent in a single score request

```bash
# Start server with profiling enabled
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --device tpu \
    --port 30000

# Send profiled request
curl -X POST 'http://localhost:30000/start_profile' \
    -H 'Content-Type: application/json' \
    -d '{"output_dir": "/tmp/score_profile", "num_steps": 1}'

# Send score request
curl -X POST 'http://localhost:30000/v1/score' \
    -H 'Content-Type: application/json' \
    -d '{
        "query": "Is this positive?",
        "items": ["Great product!", "Terrible experience"],
        "label_token_ids": [9454, 2753],
        "apply_softmax": true
    }'

# Stop profiling
curl -X POST 'http://localhost:30000/stop_profile'

# View trace
open https://ui.perfetto.dev  # Upload /tmp/score_profile/*.trace.json.gz
```

#### Scenario 2: Batch Throughput Analysis

**Goal:** Understand scaling behavior with batch size

```bash
# Run benchmark with increasing batch sizes
python test/srt/bench_score.py \
    --profile standard \
    --enable-trace \
    --output-dir /tmp/batch_analysis

# Analyze traces for each batch size
for trace in /tmp/batch_analysis/trace_b*; do
    echo "Analyzing: $trace"
    # Extract kernel times using perf.py utilities
done
```

#### Scenario 3: Memory Profiling

**Goal:** Understand HBM usage patterns

```bash
# Enable memory profiling
export ENABLE_MEMORY_PROFILING=1
export SGL_MEMORY_OUTPUT_DIR=/tmp/memory_profiles

# Run server and send requests
python -m sgl_jax.launch_server --model-path ... &

# Send requests with varying batch sizes
for batch_size in 1 4 16 32 64; do
    curl -X POST 'http://localhost:30000/v1/score' \
        -H 'Content-Type: application/json' \
        -d "{\"query\": \"test\", \"items\": $(python -c "print(['item']*$batch_size)"), ...}"
done

# Analyze memory reports
cat /tmp/memory_profiles/memory_report_*.json | jq '.total_memory_mb'
```

#### Scenario 4: Kernel-Level Analysis

**Goal:** Identify hotspot kernels in scoring

```python
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace

def score_function(*args):
    # Your score computation
    return scores

def data_generator():
    # Generate test data
    return (query, items, label_ids)

# Profile and extract kernel times
durations = multiple_iteration_timeit_from_trace(
    score_function,
    data_generator,
    task="score_forward",
    tries=10,
    trace_root="/tmp/kernel_analysis"
)

print(f"Kernel durations (ms): {durations}")
print(f"Mean: {sum(durations)/len(durations):.2f} ms")
```

### Key Metrics to Capture

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Single item latency (p50) | < 20 ms | > 50 ms |
| Single item latency (p99) | < 50 ms | > 150 ms |
| Batch-8 throughput | > 200 items/sec | < 100 items/sec |
| Batch-32 throughput | > 500 items/sec | < 250 items/sec |
| Tokenization overhead | < 5% of total | > 15% |
| Softmax overhead | < 1% of total | > 5% |
| Peak HBM usage (1B model) | < 8 GB | > 12 GB |

---

## Step-by-Step Profiling Guides

### Guide 1: End-to-End Profiling of Score API

#### Prerequisites

```bash
# Install dependencies
pip install tensorboard xprof jax[tpu]

# Verify JAX can see TPU
python -c "import jax; print(jax.devices())"
```

#### Step 1: Launch Server

```bash
cd sglang-jax

# Launch server with profiling-friendly settings
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --device tpu \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp-size 1 \
    --skip-server-warmup
```

#### Step 2: Start Profiling

```bash
# Option A: Via HTTP API
curl -X POST 'http://localhost:30000/start_profile' \
    -H 'Content-Type: application/json' \
    -d '{
        "output_dir": "/tmp/score_profile",
        "num_steps": 10,
        "host_tracer_level": 2,
        "python_tracer_level": 1
    }'

# Option B: Via Python client
python -m sgl_jax.profiler \
    --url http://localhost:30000 \
    --num-steps 10 \
    --output-dir /tmp/score_profile \
    --cpu --gpu
```

#### Step 3: Send Score Requests

```bash
# Single request
curl -X POST 'http://localhost:30000/v1/score' \
    -H 'Content-Type: application/json' \
    -d '{
        "query": "Is this a positive review? Review: ",
        "items": [
            "Amazing product, works perfectly!",
            "Terrible quality, broke after one day",
            "Average product, nothing special"
        ],
        "label_token_ids": [9454, 2753],
        "apply_softmax": true,
        "model": "meta-llama/Llama-3.2-1B-Instruct"
    }'

# Multiple requests for statistical significance
for i in {1..10}; do
    curl -s -X POST 'http://localhost:30000/v1/score' \
        -H 'Content-Type: application/json' \
        -d '{...}' &
done
wait
```

#### Step 4: Stop Profiling and Collect Traces

```bash
# Stop profiling
curl -X POST 'http://localhost:30000/stop_profile'

# List generated files
ls -la /tmp/score_profile/plugins/profile/*/

# Files you'll see:
# - *.trace.json.gz  (main trace file)
# - *.memory_profile.json.gz (memory profile if enabled)
# - metadata.json (run metadata)
```

#### Step 5: Analyze Traces

**Option A: Perfetto UI (Recommended)**

```bash
# Open in browser
open https://ui.perfetto.dev

# Drag and drop: /tmp/score_profile/plugins/profile/*/*.trace.json.gz
```

**Option B: TensorBoard**

```bash
tensorboard --logdir /tmp/score_profile --port 6006

# Open http://localhost:6006
# Navigate to: Profile > trace_viewer
```

**Option C: XProf (Advanced)**

```bash
# Install XProf
pip install xprof-nightly

# Launch XProf server
xprof --logdir /tmp/score_profile --port 6006

# Open http://localhost:6006
```

#### Step 6: Interpret Results

**In Perfetto/TensorBoard, look for:**

1. **Timeline View:**
   - Look for `score_request` named scope
   - Identify long-running operations
   - Check for gaps (idle time)

2. **Kernel Analysis:**
   - Sort by duration to find hotspots
   - Look for `dot_general` (matmul), `reduce_scatter`, `all_gather`
   - Check MXU utilization percentages

3. **Memory Usage:**
   - Peak HBM allocation
   - Memory allocation patterns
   - Identify memory-bound operations

### Guide 2: Memory Profiling for Score API

#### Step 1: Enable Memory Profiling

```bash
# Set environment variables
export ENABLE_MEMORY_PROFILING=1
export SGL_MEMORY_OUTPUT_DIR=/tmp/memory_analysis
export MEMORY_PROFILING_LAYERS=all  # or "4" for every 4th layer

# Alternative: Profile specific layers
export MEMORY_PROFILING_LAYERS="0,1,2,3"  # First 4 layers only
```

#### Step 2: Run Workload

```bash
# Launch server
python -m sgl_jax.launch_server --model-path ... &

# Wait for server to start
sleep 30

# Send requests with varying batch sizes
for batch_size in 1 4 8 16 32; do
    echo "Testing batch size: $batch_size"

    # Generate items array
    items=$(python -c "import json; print(json.dumps(['item ' + str(i) for i in range($batch_size)]))")

    curl -X POST 'http://localhost:30000/v1/score' \
        -H 'Content-Type: application/json' \
        -d "{
            \"query\": \"classify: \",
            \"items\": $items,
            \"label_token_ids\": [9454, 2753]
        }"

    sleep 2  # Allow memory reports to be written
done
```

#### Step 3: Analyze Memory Reports

```bash
# List generated reports
ls /tmp/memory_analysis/

# View summary
python -c "
import json
import glob

reports = glob.glob('/tmp/memory_analysis/memory_report_*.json')
for report in sorted(reports):
    with open(report) as f:
        data = json.load(f)
    print(f\"{report}: {data['total_memory_mb']:.2f} MB\")
"
```

#### Step 4: Generate Summary Report

```python
from sgl_jax.srt.memory_profiler import generate_summary_report

summary = generate_summary_report("/tmp/memory_analysis")
print(json.dumps(summary, indent=2))
```

### Guide 3: Kernel-Level Profiling

#### Step 1: Identify Target Kernels

Common kernels in Score API:

| Kernel | Description | Location in Trace |
|--------|-------------|-------------------|
| `dot_general` | Matrix multiplication | Attention QKV, MLP |
| `reduce_scatter` | TP communication | After matmuls |
| `all_gather` | TP communication | Before matmuls |
| `softmax` | Attention softmax | Attention layer |
| `rms_norm` | Layer normalization | Every layer |

#### Step 2: Extract Kernel Times from Trace

```python
import gzip
import json
from pathlib import Path


def analyze_trace(trace_path: str):
    """Extract kernel timings from JAX trace."""

    # Load trace
    trace_file = list(Path(trace_path).glob("**/*.trace.json.gz"))[0]
    with gzip.open(trace_file, "rb") as f:
        trace = json.load(f)

    # Extract kernel events
    kernels = {}
    for event in trace.get("traceEvents", []):
        if event.get("cat") == "kernel":
            name = event.get("name", "unknown")
            duration_us = event.get("dur", 0)

            if name not in kernels:
                kernels[name] = []
            kernels[name].append(duration_us / 1000)  # Convert to ms

    # Print summary
    print("Kernel Analysis:")
    print("-" * 60)
    for name, durations in sorted(kernels.items(), key=lambda x: -sum(x[1])):
        total = sum(durations)
        count = len(durations)
        mean = total / count
        print(f"{name[:40]:<40} {count:>5}x  {mean:>8.2f}ms  total: {total:>8.2f}ms")


analyze_trace("/tmp/score_profile")
```

#### Step 3: Profile Specific Operations

```python
import jax
import jax.numpy as jnp
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace


# Profile a specific operation
def attention_forward(q, k, v):
    """Scaled dot-product attention."""
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    scores = jnp.einsum("bqhd,bkhd->bqhk", q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bqhk,bkhd->bqhd", weights, v)
    return output


def data_gen():
    """Generate random attention inputs."""
    batch, seq, heads, dim = 32, 128, 32, 128
    q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, heads, dim))
    k = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, heads, dim))
    v = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, heads, dim))
    return (q, k, v)


# Run profiled iterations
durations = multiple_iteration_timeit_from_trace(
    attention_forward,
    data_gen,
    task="attention",
    tries=20,
    trace_root="/tmp/attention_profile"
)

print(f"Attention kernel times: {durations}")
print(f"Mean: {sum(durations)/len(durations):.3f} ms")
print(f"Std: {(sum((d - sum(durations)/len(durations))**2 for d in durations)/len(durations))**0.5:.3f} ms")
```

### Guide 4: Comparative Profiling (Before/After)

#### Step 1: Establish Baseline

```bash
# Checkout baseline commit
git checkout main

# Run benchmark
python test/srt/bench_score.py \
    --profile standard \
    --enable-trace \
    --output-dir /tmp/baseline_profile \
    --output-json /tmp/baseline_results.json

# Save baseline
cp /tmp/baseline_results.json baselines/score-api-baseline.json
```

#### Step 2: Profile Changes

```bash
# Checkout feature branch
git checkout feature/score-optimization

# Run benchmark
python test/srt/bench_score.py \
    --profile standard \
    --enable-trace \
    --output-dir /tmp/feature_profile \
    --output-json /tmp/feature_results.json
```

#### Step 3: Compare Results

```python
import json

with open("/tmp/baseline_results.json") as f:
    baseline = json.load(f)

with open("/tmp/feature_results.json") as f:
    feature = json.load(f)

print("Performance Comparison:")
print("-" * 80)
print(f"{'Config':<20} {'Baseline p50':>12} {'Feature p50':>12} {'Change':>12}")
print("-" * 80)

for b, f in zip(baseline["results"], feature["results"]):
    config = f"b{b['batch_size']}_l{b['label_count']}"
    change = (f["p50_ms"] - b["p50_ms"]) / b["p50_ms"] * 100
    status = "FASTER" if change < -5 else "SLOWER" if change > 5 else "SAME"
    print(f"{config:<20} {b['p50_ms']:>12.2f} {f['p50_ms']:>12.2f} {change:>+11.1f}% {status}")
```

### Guide 5: CI/CD Integration

#### Nightly Profiling Workflow

```yaml
# .github/workflows/nightly-profiling.yaml
name: Nightly Profiling

on:
  schedule:
    - cron: '0 3 * * *'  # 3 AM UTC daily
  workflow_dispatch:

jobs:
  profile:
    runs-on: [self-hosted, tpu]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install tensorboard xprof-nightly

      - name: Run Profiled Benchmark
        run: |
          # Start server in background
          python -m sgl_jax.launch_server \
            --model-path meta-llama/Llama-3.2-1B-Instruct \
            --device tpu &

          sleep 60  # Wait for server

          # Run benchmark with profiling
          python test/srt/bench_score.py \
            --profile smoke \
            --enable-trace \
            --output-dir /tmp/nightly_profile \
            --output-json /tmp/results.json

      - name: Compare with Baseline
        run: |
          python scripts/compare_profiles.py \
            --baseline baselines/score-api-baseline.json \
            --current /tmp/results.json \
            --threshold 10

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: profile-${{ github.run_number }}
          path: |
            /tmp/nightly_profile/
            /tmp/results.json
          retention-days: 30

      - name: Post Results to PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('/tmp/results.json'));
            // Format and post results...
```

---

## Visualization Tools

### Tool Comparison

| Tool | Best For | Format | Installation |
|------|----------|--------|--------------|
| **Perfetto** | Interactive timeline | `.trace.json.gz` | Web-based (no install) |
| **TensorBoard** | TPU/XLA analysis | Profile plugin | `pip install tensorboard` |
| **XProf** | Detailed TPU metrics | Profile plugin | `pip install xprof-nightly` |
| **pprof** | Memory profiles | `.prof` | `go tool pprof` |
| **Chrome Tracing** | Legacy traces | `.json` | Chrome browser |

### Perfetto Quick Reference

**Loading Trace:**
1. Open https://ui.perfetto.dev
2. Click "Open trace file"
3. Select `*.trace.json.gz`

**Key Views:**
- **Timeline:** Main view showing all events
- **Slices:** Expandable call tree
- **Counters:** Memory, utilization over time
- **Flow Events:** Async operation tracking

**Useful Queries (SQL):**
```sql
-- Top 10 longest kernels
SELECT name, dur/1e6 as ms FROM slice ORDER BY dur DESC LIMIT 10;

-- Total time by operation
SELECT name, SUM(dur)/1e6 as total_ms FROM slice GROUP BY name ORDER BY total_ms DESC;

-- Memory over time
SELECT ts, value FROM counter WHERE name = 'hbm_usage';
```

### TensorBoard Quick Reference

```bash
# Launch
tensorboard --logdir /path/to/profile --port 6006

# Key tabs:
# - Profile > Overview: Summary statistics
# - Profile > Trace Viewer: Timeline
# - Profile > Memory Profile: HBM analysis
# - Profile > Op Profile: Operation breakdown
```

### XProf Quick Reference

```bash
# Launch standalone
xprof --logdir /path/to/profile --port 6006

# Key features:
# - Input pipeline analysis
# - Kernel stats
# - Memory breakdown
# - Roofline analysis
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] Create `sgl_jax/srt/score_profiler.py` module
- [ ] Add named scopes to `score_request()` in tokenizer_manager.py
- [ ] Create `test/srt/bench_score.py` with profiling support
- [ ] Document environment variables and configuration

### Phase 2: Memory Profiling (Week 2-3)

- [ ] Integrate memory profiler into Score API path
- [ ] Add automatic peak HBM tracking
- [ ] Create memory regression tests
- [ ] Document memory profiling workflow

### Phase 3: CI Integration (Week 3-4)

- [ ] Create nightly profiling workflow
- [ ] Set up artifact storage and retention
- [ ] Implement regression comparison script
- [ ] Add Slack/PR notifications for regressions

### Phase 4: Documentation and Tooling (Week 4)

- [ ] Create runbook for profiling operations
- [ ] Add CLI tools for trace analysis
- [ ] Document all profiling scenarios
- [ ] Create dashboard template (optional)

---

## Cost Analysis

### Profiling Overhead

| Activity | Overhead | Acceptable? |
|----------|----------|-------------|
| JAX trace (basic) | 5-10% | Yes |
| JAX trace (full) | 15-25% | During debugging |
| Memory profiling | 10-20% | Yes |
| Named scopes only | < 1% | Yes (always on) |

### CI Costs

| Scenario | TPU Hours/Month | Cost/Month |
|----------|-----------------|------------|
| Nightly smoke profile | 0.5 hr × 30 = 15 hrs | $9.60 |
| Weekly full profile | 1 hr × 4 = 4 hrs | $2.56 |
| On-demand debugging | ~5 hrs | $3.20 |
| **Total** | ~24 hrs | **~$15.36** |

Based on TPU v6e-1 at $0.64/hr.

---

## Open Questions

| Question | Proposed Answer | Status |
|----------|-----------------|--------|
| Should named scopes be always on? | Yes, < 1% overhead | Proposed |
| Trace file retention in CI? | 30 days | Proposed |
| Memory profile frequency? | Every 4th layer by default | Proposed |
| Compare against PyTorch baselines? | Yes, via RFC-010 | Deferred to RFC-010 |

---

## General Benchmarking Tools

This section covers the standard benchmarking tools available in sglang-jax, aligned with the [official sglang documentation](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html).

### bench_one_batch: Static Batch Benchmarking

Benchmark latency of running a single static batch **without a server**. Useful for isolating model inference time from server overhead.

**Location:** `sglang-jax/python/sgl_jax/bench_one_batch.py`

```bash
# Basic usage
python -m sgl_jax.bench_one_batch \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --batch-size 1 4 8 16 \
    --input-len 256 512 \
    --output-len 32

# With profiling enabled
python -m sgl_jax.bench_one_batch \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --batch-size 8 \
    --input-len 512 \
    --output-len 32 \
    --profile \
    --profile-filename-prefix /tmp/one_batch_profile
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--batch-size` | Batch sizes to test (space-separated) |
| `--input-len` | Input sequence lengths (space-separated) |
| `--output-len` | Output tokens to generate |
| `--profile` | Enable JAX profiling |
| `--profile-filename-prefix` | Output directory for traces |
| `--correctness-test` | Run correctness validation |

**Note:** This runs without dynamic batching, so may OOM at batch sizes a real server can handle.

### bench_one_batch_server: Server-Based Single Batch

Similar to `bench_one_batch` but runs against a live server via HTTP.

```bash
# Start server first
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --port 30000

# Run benchmark
python -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --batch-size 32 \
    --input-len 256 \
    --output-len 32
```

### bench_offline_throughput: Offline Throughput

Measures throughput for offline batch processing without server overhead.

**Location:** `sglang-jax/python/sgl_jax/bench_offline_throughput.py`

```bash
python -m sgl_jax.bench_offline_throughput \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --num-prompts 100 \
    --input-len 512 \
    --output-len 128
```

### bench_serving: Online Serving Benchmark

Comprehensive benchmark for online serving with configurable request patterns.

**Location:** `sglang-jax/python/sgl_jax/bench_serving.py`

```bash
# Start server
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --port 30000

# Basic benchmark
python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --num-prompts 100

# With profiling
python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --num-prompts 50 \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --profile \
    --num-steps 10
```

**Recommended benchmark matrix (from docs):**

```bash
#!/bin/bash
# Standard benchmark matrix
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)  # 1 for TTFT measurement
max_concurrencies=(8 16 32 64 128 256)

for input_len in "${input_seq_lens[@]}"; do
    for output_len in "${output_seq_lens[@]}"; do
        for concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((3 * concurrency))
            python -m sgl_jax.bench_serving \
                --backend sgl-jax \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input-len ${input_len} \
                --random-output-len ${output_len} \
                --max-concurrency ${concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

---

## Distributed Profiling (TP/EP)

When running with tensor parallelism (TP) or expert parallelism (EP), profiling requires special handling.

### HTTP API Parameters Reference

The `/start_profile` endpoint accepts these parameters:

```json
{
    "output_dir": "/tmp/profile",       // Trace output directory
    "num_steps": 10,                    // Number of forward steps to profile (optional)
    "start_step": 5,                    // Skip warmup steps before profiling (optional)
    "host_tracer_level": 2,             // 0=off, 1=minimal, 2=medium, 3=verbose
    "python_tracer_level": 1,           // 0=off, 1=enabled
    "profile_id": "my_profile"          // Custom identifier for trace files
}
```

**Note:** If `num_steps` is not specified, profiling continues until `/stop_profile` is called.

### Multi-Device Profiling

sglang-jax automatically handles profiling across TP/EP ranks. Each rank generates its own trace file with a standardized naming convention.

```bash
# Launch with TP=4
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --tp-size 4 \
    --port 30000

# Start profiling - traces generated for each rank
curl -X POST 'http://localhost:30000/start_profile' \
    -H 'Content-Type: application/json' \
    -d '{
        "output_dir": "/tmp/tp4_profile",
        "num_steps": 5,
        "profile_id": "tp4_test"
    }'
```

**Generated files follow this naming convention:**
```
/tmp/tp4_profile/
├── plugins/profile/<timestamp>/
│   ├── tp4_test-TP-0-EP-0.trace.json.gz
│   ├── tp4_test-TP-1-EP-0.trace.json.gz
│   ├── tp4_test-TP-2-EP-0.trace.json.gz
│   ├── tp4_test-TP-3-EP-0.trace.json.gz
│   └── merged-tp4_test.trace.json.gz    # Auto-merged if supported
```

**File naming format:** `{profile_id}-TP-{tp_rank}-EP-{ep_rank}.trace.json.gz`

For setups with expert parallelism (MoE models):
```
# TP=2, EP=2 setup
├── moe_profile-TP-0-EP-0.trace.json.gz
├── moe_profile-TP-0-EP-1.trace.json.gz
├── moe_profile-TP-1-EP-0.trace.json.gz
└── moe_profile-TP-1-EP-1.trace.json.gz
```

### Stage-Based Profiling (Prefill vs Decode)

PyTorch SGLang supports profiling prefill and decode stages separately via `profile_by_stage` parameter. This is useful for identifying stage-specific bottlenecks.

**PyTorch approach (for reference):**
```bash
# Profile prefill and decode separately
curl -X POST 'http://localhost:30000/start_profile' \
    -d '{
        "output_dir": "/tmp/profile",
        "num_steps": 10,
        "profile_by_stage": true,
        "profile_stages": ["prefill", "decode"]
    }'
```

**JAX current status:**

The sglang-jax scheduler profiler mixin (`scheduler_profiler_mixing.py`) supports basic step-based profiling but does not yet have explicit prefill/decode stage separation like PyTorch. However, you can achieve similar analysis by:

1. **Using named scopes in traces** - Look for `forward_batch` scopes and analyze batch types
2. **Filtering by batch size** - Prefill typically has larger batch token counts
3. **Manual separation** - Run separate profiling sessions for prefill-heavy vs decode-heavy workloads

**Score API note:** For Score API workloads, this distinction is less relevant since scoring is **prefill-only** (no token generation). All Score API requests use `max_new_tokens=0`.

### PD Disaggregation Profiling

For prefill-decode disaggregated deployments (separate prefill and decode workers), PyTorch SGLang requires profiling workers separately:

**PyTorch approach:**
```bash
# Profile prefill workers only
python -m sglang.bench_serving \
    --backend sglang-oai \
    --profile \
    --profile-prefill-url http://prefill-worker-0:30000 http://prefill-worker-1:30000

# Profile decode workers only (in separate run)
python -m sglang.bench_serving \
    --backend sglang-oai \
    --profile \
    --profile-decode-url http://decode-worker-0:30010 http://decode-worker-1:30010
```

**Note:** `--profile-prefill-url` and `--profile-decode-url` are mutually exclusive.

**JAX status:** PD disaggregation is supported in sglang-jax (see `schedule_batch.py:327`), but dedicated profiling flags for disaggregated workers are not yet implemented. Profile each worker separately using standard `/start_profile` endpoints.

### Trace Merging for Distributed Runs

For multi-host setups, traces must be collected to shared storage:

```bash
# Ensure shared storage (e.g., NFS, GCS FUSE)
export SGLANG_JAX_PROFILER_DIR=/shared/profiles

# After profiling, merge traces
python -c "
from sgl_jax.srt.utils.trace_merger import merge_traces
merge_traces('/shared/profiles', output='/shared/profiles/merged.trace.json.gz')
"
```

### Analyzing TP Communication Overhead

Look for these kernels in traces to understand TP overhead:

| Kernel | Description | Optimization Target |
|--------|-------------|---------------------|
| `all_gather` | Gather weights/activations | Overlap with compute |
| `reduce_scatter` | Distribute gradients | Batch size tuning |
| `all_reduce` | Collective reduction | TP size vs compute |
| `psum` | JAX collective sum | Sharding strategy |

---

## Advanced Options

### Dummy Weights for Rapid Prototyping

Test profiling setup without loading real model weights:

```bash
# Create minimal config.json in a directory
mkdir -p /tmp/dummy_model
cat > /tmp/dummy_model/config.json << 'EOF'
{
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 4,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000
}
EOF

# Launch with dummy weights
python -m sgl_jax.launch_server \
    --model-path /tmp/dummy_model \
    --load-format dummy \
    --port 30000
```

### Model Architecture Override

Test different model configurations without retraining:

```bash
# Override number of layers for quick profiling
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --json-model-override-args '{"num_hidden_layers": 4}' \
    --port 30000
```

### Environment Variable Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_JAX_PROFILER_DIR` | Default profiler output directory | `/tmp` |
| `ENABLE_MEMORY_PROFILING` | Enable memory profiling | `0` |
| `SGL_MEMORY_OUTPUT_DIR` | Memory profile output directory | `memory_profiles` |
| `MEMORY_PROFILING_LAYERS` | Layers to profile (`all`, `4`, `0,1,2`) | `4` |
| `JAX_COMPILATION_CACHE_DIR` | XLA compilation cache | None |
| `XLA_FLAGS` | XLA compiler flags | None |

---

## Known Issues and Troubleshooting

### Issue 1: Large Trace Files

**Symptom:** Browser cannot open trace file, or Perfetto crashes.

**Cause:** Trace files can be hundreds of MB for long runs.

**Solutions:**
1. Reduce `--num-steps` to profile fewer forward passes
2. Reduce `--num-prompts` in benchmark
3. Use shorter sequences (`--random-input-len 128`)
4. Split into multiple shorter profiling sessions

```bash
# Generate smaller trace (<100MB)
curl -X POST 'http://localhost:30000/start_profile' \
    -d '{"output_dir": "/tmp/profile", "num_steps": 3}'
```

### Issue 2: XLA Recompilation During Profiling

**Symptom:** First profiled run is much slower than subsequent runs.

**Cause:** XLA compiles new shapes on first encounter.

**Solutions:**
1. Warm up with same shapes before profiling
2. Use compilation cache:
   ```bash
   export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
   ```
3. Use `start_step` parameter to skip warmup:
   ```json
   {"output_dir": "/tmp/profile", "num_steps": 5, "start_step": 10}
   ```

### Issue 3: Memory Profiling Not Generating Output

**Symptom:** No memory profile files generated.

**Cause:** Environment variables not set, or wrong directory permissions.

**Solution:**
```bash
# Verify environment
export ENABLE_MEMORY_PROFILING=1
export SGL_MEMORY_OUTPUT_DIR=/tmp/memory
mkdir -p /tmp/memory

# Check permissions
ls -la /tmp/memory
```

### Issue 4: Profile HTTP API Returns Error

**Symptom:** `/start_profile` returns `"Profiling is already in progress"`.

**Solution:** Stop existing profiling session first:
```bash
curl -X POST 'http://localhost:30000/stop_profile'
# Then start new session
curl -X POST 'http://localhost:30000/start_profile' -d '...'
```

### Issue 5: Named Scopes Not Appearing in Trace

**Symptom:** Trace shows flat kernel list without hierarchy.

**Cause:** Named scopes require proper JAX profiler configuration.

**Solution:** Ensure `host_tracer_level` >= 2:
```json
{
    "output_dir": "/tmp/profile",
    "num_steps": 5,
    "host_tracer_level": 2,
    "python_tracer_level": 1
}
```

### Issue 6: OOM During Profiling

**Symptom:** OOM errors only when profiling is enabled.

**Cause:** Profiling adds ~10-20% memory overhead.

**Solutions:**
1. Reduce batch size during profiling
2. Profile smaller model first
3. Use memory profiling to identify peak usage:
   ```bash
   export ENABLE_MEMORY_PROFILING=1
   ```

### Debugging Tips

1. **Check server logs** for profiler initialization:
   ```bash
   grep -i "profil" server.log
   ```

2. **Verify trace files exist:**
   ```bash
   find /tmp/profile -name "*.trace.json.gz" -ls
   ```

3. **Test with minimal config:**
   ```bash
   python -c "
   import jax
   with jax.profiler.trace('/tmp/test_trace'):
       x = jax.numpy.ones((100, 100))
       y = x @ x
       y.block_until_ready()
   print('Trace saved to /tmp/test_trace')
   "
   ```

---

## References

- [Official SGLang Benchmark & Profiling Guide](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html) - PyTorch reference
- [JAX Profiling Documentation](https://docs.jax.dev/en/latest/profiling.html)
- [XProf Documentation](https://github.com/openxla/xprof)
- [Perfetto UI](https://ui.perfetto.dev)
- [Cloud TPU Profiling Guide](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm)
- [How to Scale Your Model - Profiling](https://jax-ml.github.io/scaling-book/profiling/)
- [RFC-004: Performance Benchmarks](004-score-api-performance-benchmarks.md)
- [RFC-010: Cross-Backend Benchmarking](010-cross-backend-benchmarking.md)
- [ADR-001: Pure Python Softmax](../decisions/001-pure-python-softmax.md)
- [sglang-jax Benchmark Docs](../../sglang-jax/docs/developer_guide/benchmark_and_profiling.md)

---

## Appendix A: Quick Reference Commands

### Server Commands

```bash
# Launch server with profiling support
python -m sgl_jax.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --device tpu \
    --port 30000

# Start profiling via HTTP
curl -X POST 'http://localhost:30000/start_profile' \
    -H 'Content-Type: application/json' \
    -d '{"output_dir": "/tmp/profile", "num_steps": 10}'

# Stop profiling
curl -X POST 'http://localhost:30000/stop_profile'
```

### Benchmark Commands

```bash
# Quick smoke test
python test/srt/bench_score.py --profile smoke

# Standard with tracing
python test/srt/bench_score.py --profile standard --enable-trace

# Full with memory profiling
python test/srt/bench_score.py --profile full --enable-memory --output-json results.json
```

### Analysis Commands

```bash
# View in TensorBoard
tensorboard --logdir /tmp/profile --port 6006

# View in XProf
xprof --logdir /tmp/profile --port 6006

# Memory profile analysis
go tool pprof -http=:8080 /tmp/profile/memory.prof
```

### Environment Variables

```bash
# Memory profiling
export ENABLE_MEMORY_PROFILING=1
export SGL_MEMORY_OUTPUT_DIR=/tmp/memory
export MEMORY_PROFILING_LAYERS=4

# Profiler output
export SGLANG_JAX_PROFILER_DIR=/tmp/profiles

# JAX settings
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
```

---

## Appendix B: Trace Event Schema

### JAX Trace Event Format

```json
{
  "traceEvents": [
    {
      "name": "score_request",
      "cat": "jax",
      "ph": "X",
      "ts": 1234567890,
      "dur": 5000,
      "pid": 1,
      "tid": 1,
      "args": {
        "batch_size": 8,
        "num_items": 16,
        "device_duration_ps": 5000000000
      }
    }
  ],
  "metadata": {
    "jax_version": "0.4.x",
    "devices": ["TPU v6e-1"]
  }
}
```

### Memory Report Schema

```json
{
  "report_type": "score_request",
  "stage": "forward",
  "total_memory_mb": 1234.56,
  "largest_tensor": "attention_weights",
  "largest_memory_mb": 256.0,
  "tensors": {
    "query": {"memory_mb": 64.0, "shape": [8, 128, 32, 128], "dtype": "bfloat16"},
    "key": {"memory_mb": 64.0, "shape": [8, 128, 32, 128], "dtype": "bfloat16"},
    "value": {"memory_mb": 64.0, "shape": [8, 128, 32, 128], "dtype": "bfloat16"}
  }
}
```
