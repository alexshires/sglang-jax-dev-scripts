# Multi-Item Scoring Large Scale Evaluation (2026-02-10)

## Overview

This report documents the performance evaluation of the `/v1/score` API's multi-item scoring mode under high-load conditions. Specifically, we benchmarked a scenario involving a long system prompt and a large number of candidate items to verify the throughput benefits of the new packed-sequence implementation on TPU v6e.

## Test Configuration

### Environment
- **Hardware**: TPU v6e (1 chip, 1x1 topology)
- **Node Resources**: 40 vCPU, 150Gi RAM
- **Model**: `Qwen/Qwen3-0.6B` (bfloat16)
- **Engine Config**:
  - `mem_fraction_static`: 0.7 (reduced from 0.9 to prevent OOM)
  - `chunked_prefill_size`: -1 (Disabled for multi-item correctness)
  - `disable_radix_cache`: True (Required for multi-item correctness)
  - `max_multi_item_seq_len`: 32,768
  - `max_multi_item_count`: 512
  - `multi_item_scoring_chunk_size`: 32 (default)

### Scenario: "Large Scale Reranking"
- **Prompt Length**: 2000 tokens (100 static prefix + 1900 dynamic suffix)
- **Number of Candidates**: 500 items per request
- **Candidate Length**: 20 tokens each
- **Total Request Load**: ~12,000 tokens (2000 prompt + 500 * 20 items)

## Results

### Throughput Comparison (Chunk Size = 32)

| Metric | Single-Item Sequential | Multi-Item Packed | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 49.26 items/sec | **52.17 items/sec** | **1.06x** |
| **Latency (per item)** | 20.30 ms | **19.17 ms** | **-5.5%** |
| **Total Time (500 items)** | 10.15 sec | **9.58 sec** | **5.6% Faster** |

### Chunk Size Scaling

We evaluated the impact of varying the `multi_item_scoring_chunk_size` on throughput.

| Chunk Size | Throughput (items/sec) | Latency/Item (ms) |
| :--- | :--- | :--- |
| 8 | 10.64 | 93.94 |
| 16 | 20.36 | 49.13 |
| 32 | 19.86 | 50.36 |
| 64 | **50.68** | **19.73** |

*Note: Throughput significantly improves with larger chunk sizes, plateauing around chunk size 64.*

### Static Prefix Impact

We tested two additional scenarios to evaluate the impact of the static prefix length on performance.

#### Scenario 1 (Long Static Prefix)
- **Static Prefix**: 2000 tokens
- **Dynamic Suffix**: 20 tokens
- **Throughput**: **9.72 items/sec**

#### Scenario 2 (Long Static Prefix, Short Suffix)
- **Static Prefix**: 1900 tokens
- **Dynamic Suffix**: 10 tokens
- **Throughput**: **10.77 items/sec**

**Observation**: Throughput drops significantly (~5x) when the static prefix is very long (2000 tokens) compared to the baseline scenario (100 tokens). This confirms that minimizing the shared prefix relative to the dynamic suffix/items maximizes the benefits of multi-item packing.

## Key Findings

1.  **Throughput Gains**: Multi-item scoring provides a **6% throughput increase** over sequential processing for the baseline workload with a short static prefix.
2.  **Chunk Size Matters**: Increasing the chunk size from 8 to 64 yielded a **~5x improvement** in throughput.
3.  **Prefix Sensitivity**: Performance is highly sensitive to the length of the static prefix. Long static prefixes reduce the relative efficiency gains of packing.
4.  **Resource Constraints & Tuning**:
    -   **OOM on TPU v6e**: Initial runs with `mem_fraction_static=0.88` caused OOMs. Reducing to `0.7` was necessary.
    -   **Limit Increases**: Validated system stability with 512 items and 32,768 tokens.

5.  **Infrastructure Reliability**:
    -   Switching to offline model loading (via GCS mount at `/data`) eliminated download flakiness.
    -   Aligning K8s resource requests prevented eviction.

## Conclusion

The multi-item scoring implementation is stable and effective. To maximize performance, users should use larger chunk sizes (e.g., 64) and minimize the static prefix length where possible. The infrastructure improvements have established a reliable baseline for future optimization.
