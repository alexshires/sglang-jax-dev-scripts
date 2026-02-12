# Multi-Item Scoring GPU Performance Evaluation (2026-02-12)

## Overview

This report documents the performance of SGLang (PyTorch backend) on NVIDIA GPUs for the specialized scoring scenarios defined for the SGL-JAX comparison.

## Test Configuration

### Environment
- **Accelerator 1 (G2)**: NVIDIA L4 (24GB VRAM)
- **Accelerator 2 (G4)**: NVIDIA RTX PRO 6000 (48GB VRAM)
- **Software**: `docker.io/lmsysorg/sglang:latest` (PyTorch backend)
- **Model**: `Qwen/Qwen3-0.6B` (bfloat16)
- **Engine Config**: `mem_fraction_static=0.7`, `tp_size=1`

### Benchmark Scenarios
Each scenario processes **500 candidate items** per request.

- **Scenario 1**: 2000-token static prefix + 20-token dynamic suffix. Items: 20 tokens each.
- **Scenario 2**: 1900-token static prefix + 10-token dynamic suffix. Items: 10 tokens each.

---

## Results

| Accelerator | Scenario | Throughput (items/sec) | Latency/Item (ms) | Total Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L4 (G2)** | Scenario 1 | 112.32 | 8.90 | 4.45 |
| **NVIDIA L4 (G2)** | Scenario 2 | 129.65 | 7.71 | 3.86 |
| **RTX 6000 (G4)** | Scenario 1 | 400.10 | 2.50 | 1.25 |
| **RTX 6000 (G4)** | Scenario 2 | 464.32 | 2.15 | 1.08 |

---

## Infrastructure & Commands

The benchmarks were executed using a set of specialized scripts in `sglang-scripts/debug-score-tests/`.

### 1. Benchmark Script: `test_bench_multi_item_score_gpu.py`
A `pytest`-based script using the `sglang.srt.entrypoints.engine.Engine` API. It approximates the token counts using string multiplication to simulate the target load conditions.

### 2. Execution Script: `run_gpu_tests.sh`
Orchestrates the benchmark by copying the test script to the pod and executing it via `kubectl exec`.

**Usage:**
```bash
cd sglang-scripts/debug-score-tests
./run_gpu_tests.sh <pod_name> <tp_size>
```

### 3. Kubernetes Manifests
- `debug-gpu-l4-pod.yaml`: Targets the `nvidia-l4` pool.
- `debug-gpu-g4-pod.yaml`: Targets the `nvidia-rtx-pro-6000` pool.

---

## Key Observations

1.  **G4 Dominance**: The RTX 6000 (G4) is significantly faster than the L4 (G2), providing nearly **3.5x higher throughput**.
2.  **Efficiency**: PyTorch SGLang achieves very high throughput on these relatively small models (0.6B), with latencies as low as 2.15ms per item on the G4.
3.  **Consistency**: Like the JAX results, Scenario 2 (shorter tokens/shorter prefix) is faster than Scenario 1, though the relative gap is smaller on GPU than observed on TPU.
