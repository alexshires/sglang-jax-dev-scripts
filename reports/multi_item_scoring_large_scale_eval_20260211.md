# Multi-Item Scoring Large Scale Evaluation (2026-02-11)

## Overview

This report documents the follow-up performance evaluation of the `/v1/score` API's multi-item scoring mode on February 11, 2026. Building on previous findings, we expanded the chunk size scaling analysis (up to 128) and re-verified performance under heavy static prefix loads.

## Test Configuration

### Environment
- **Hardware**: TPU v6e (1 chip, 1x1 topology)
- **Node Resources**: 40 vCPU, 150Gi RAM
- **Model**: `Qwen/Qwen3-0.6B` (bfloat16)
- **Engine Config**:
  - `mem_fraction_static`: 0.7
  - `chunked_prefill_size`: -1
  - `disable_radix_cache`: True
  - `max_multi_item_seq_len`: 32,768
  - `max_multi_item_count`: 512

### Scenario: "Large Scale Reranking"
- **Total Request Load**: 500 candidate items per request.
- **Prompt Length**: 2000 tokens total (composition varies by scenario).

## Results

### 1. Chunk Size Scaling

We evaluated the throughput impact of varying the `multi_item_scoring_chunk_size` from 32 to 128.

| Chunk Size | Throughput (items/sec) | Latency/Item (ms) | Speedup vs Chunk=32 |
| :--- | :--- | :--- | :--- |
| 32 | 19.68 | 50.82 | 1.0x |
| 64 | **50.91** | **19.64** | **2.58x** |
| 128 | **51.63** | **19.37** | **2.62x** |

**Observation**: Increasing the chunk size from 32 to 64 yields a massive 2.5x throughput gain. Increasing further to 128 provides diminishing returns (only ~1.4% improvement over 64), suggesting that a chunk size of 64 is the optimal "sweet spot" for this workload on TPU v6e.

### 2. Static Prefix Impact

We benchmarked throughput under different prompt compositions to quantify the cost of long shared prefixes.

| Scenario | Static Prefix | Dynamic Suffix | Throughput (items/sec) | Relative Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 100 tokens | 1900 tokens | **52.13** | 1.0x (Reference) |
| **Scenario 2** | 1900 tokens | 10 tokens | 22.92 | 0.44x |
| **Scenario 1** | 2000 tokens | 20 tokens | 19.89 | 0.38x |

**Observation**: Throughput drops by ~60% when the static prefix increases from 100 to ~2000 tokens. This confirms that while multi-item scoring is efficient, processing very long shared contexts still incurs significant compute cost, reducing the effective items/sec rate.

## Conclusion

- **Optimal Configuration**: For large-scale reranking (500+ items), a `multi_item_scoring_chunk_size` of **64** or **128** is recommended. The default of 32 leaves significant performance on the table.
- **Prefix Sensitivity**: Applications should be aware that multi-item scoring efficiency degrades with very long static prefixes. Where possible, moving shared context to a separate system prompt or shortening the prefix can yield 2x+ throughput gains.
- **Stability**: The system remained stable with 500 items and a 32k token sequence limit, validating the recent limit increases.
