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

### Scenario: "Large Scale Reranking"
- **Prompt Length**: 2000 tokens (100 static prefix + 1900 dynamic suffix)
- **Number of Candidates**: 500 items per request
- **Candidate Length**: 20 tokens each
- **Total Request Load**: ~12,000 tokens (2000 prompt + 500 * 20 items)

## Results

We compared two execution modes:
1.  **Single-Item Sequential**: Processing items in small batches (32) via multiple requests.
2.  **Multi-Item Packed**: Processing all 500 items in a single request, packed into a single attention window with custom masking.

| Metric | Single-Item Sequential | Multi-Item Packed | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 4.35 items/sec | **9.75 items/sec** | **2.24x** |
| **Latency (per item)** | 229.82 ms | **102.54 ms** | **-55%** |
| **Total Time (500 items)** | 114.91 sec | **51.27 sec** | **2.24x Faster** |

## Key Findings

1.  **Significant Speedup**: Multi-item scoring provides a **>2x throughput increase** for this workload. This is primarily due to:
    -   **Shared Prefix Computation**: The 2000-token prompt is processed only once in the packed sequence, whereas sequential requests re-process it (or rely on radix cache hits, which adds management overhead).
    -   **Parallelism**: Scoring 500 items in a single forward pass (subject to chunking limits) maximizes TPU utilization compared to many small, sequential forward passes.

2.  **Resource Constraints & Tuning**:
    -   **OOM on TPU v6e**: Initial runs with `mem_fraction_static=0.88` caused OOMs. Reducing to `0.7` was necessary to leave room for JAX compilation of the large 32k-token shapes.
    -   **Limit Increases**: The default item limit (128) and sequence length limit (8192) were insufficient. We successfully validated the system stability after increasing these to 512 items and 32,768 tokens, respectively.

3.  **Infrastructure Reliability**:
    -   Switching to offline model loading (via GCS mount at `/data`) eliminated download flakiness and significantly improved test startup times.
    -   Aligning K8s resource requests (40 CPU, 150Gi RAM) with node capacity prevented eviction and OOM-kills during the resource-intensive prefill phase.

## Conclusion

The multi-item scoring implementation is stable and highly effective for large-scale reranking tasks on TPU. It successfully handles heavy payloads (500 items) and delivers substantial performance gains over the baseline sequential approach. Future work should focus on further optimizing `chunked_prefill` compatibility to potentially handle even larger batches without increasing peak memory usage.
