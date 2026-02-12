# Multi-Item Scoring TPU Multi-Chip Evaluation (2026-02-12)

## Overview

This report documents the performance evaluation of the `/v1/score` API's multi-item scoring mode across different Google TPU v6e topologies (1x1, 2x2, 2x4). We analyzed the impact of Tensor Parallelism (TP) and chunk size scaling on large-scale reranking throughput.

## Test Configuration

### Environment
- **Accelerators**: TPU v6e
  - **1x1**: 1 chip (TP=1)
  - **2x2**: 4 chips (TP=4)
  - **2x4**: 8 chips (TP=8)
- **Model**: `Qwen/Qwen3-0.6B` (bfloat16)
- **Engine Config**: `mem_fraction_static=0.7`, `chunked_prefill_size=-1`, `max_multi_item_seq_len=65536`

### Scenario: "Large Scale Reranking"
- **Load**: 500 items per request, 20 tokens per item.
- **Prefix**: 2000 tokens total.

---

## Results

### 1. Topology Scaling (Peak Throughput)

| Topology | TP Size | Peak Throughput (items/sec) | Latency/Item (ms) | Improvement vs 1x1 |
| :--- | :--- | :--- | :--- | :--- |
| **1x1** | 1 | 52.17 | 19.17 | 1.0x (Reference) |
| **2x2** | 4 | **70.13** | **14.26** | **1.34x** |
| **2x4** | 8 | 66.94 | 14.94 | 1.28x |

**Observation**: Upgrading from 1x1 to 2x2 (4 chips) provided a **34% throughput boost**. However, moving to 2x4 (8 chips) showed a slight performance regression compared to 2x2 for this specific model (0.6B), likely due to increased inter-chip communication overhead dominating the compute gains on such a small model.

### 2. Chunk Size Scaling (Multi-Chip)

We re-verified the impact of `multi_item_scoring_chunk_size` on the multi-chip configurations.

| Chunk Size | 2x2 Throughput (items/s) | 2x4 Throughput (items/s) |
| :--- | :--- | :--- |
| 32 | 12.80 | 1.87 |
| 64 | 68.01 | 65.51 |
| 128 | 69.29 | 66.26 |
| 256 | 69.78 | - |
| 512 | **70.13** | - |

**Key Finding**: Larger topologies are **extremely sensitive** to small chunk sizes. On 2x4, a chunk size of 32 was nearly unusable (1.87 items/s), while 64 restored performance to expected levels. This confirms that maximizing sequence packing is critical for multi-chip efficiency.

---

## Conclusion

- **Scaling Recommendation**: For the Qwen3-0.6B model, a **2x2 TPU (4 chips)** configuration provides the best balance of throughput and efficiency.
- **Critical Tuning**: On multi-chip setups, `multi_item_scoring_chunk_size` **MUST** be set to at least **64**. Values of 128-512 provide stable, peak performance.
- **Resource Efficiency**: Multi-item scoring successfully leverages multiple chips to reduce per-item latency by ~25% compared to single-chip execution.
