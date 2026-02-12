# Comprehensive Multi-Item Scoring Benchmarking Summary (2026-02-12)

This report summarizes all configurations, performance results, and identified regressions for the SGLang-JAX `/v1/score` API during the development cycle.

## Results Matrix

| Infrastructure | Branch / Backend | Topology | Peak Throughput | Status / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L4 (G2)** | PyTorch (Reference) | 1x GPU | **112.32 items/s** | Baseline reference |
| **RTX 6000 (G4)** | PyTorch (Reference) | 1x GPU | **400.10 items/s** | Performance leader |
| **TPU v6e (1x1)** | `feat/multi-item-scoring` | TP=1 | 52.17 items/s | Stable Baseline |
| **TPU v6e (2x2)** | `feat/multi-item-scoring` | TP=4 | 70.13 items/s | Stable, 1.34x vs 1x1 |
| **TPU v6e (2x4)** | `feat/multi-item-scoring` | TP=8 | 66.94 items/s | Stable, regressed vs 2x2 |
| **TPU v6e (2x2)** | `aashishrampal/batching` (Turn 105) | TP=4 | **141.22 items/s** | **The Performance Target** |
| **TPU v6e (2x2)** | `aashishrampal/batching` (Turn 115) | TP=4 | 70.16 items/s | Regressed (after merge) |
| **TPU v6e (2x2)** | `aashishrampal/batching` (Turn 131) | TP=4 | **FAILED** | `ValueError` in kernel (after merge) |
| **TPU v6e (2x2)** | `aashishrampal/batching` (Turn 142) | TP=1 | ~1 items/s | OOM / Perf cliff on 1 chip |

## Identified Key Behaviours

### 1. The "141 items/s" Architecture (High Performance)
The high-performance state on the `aashishrampal/batching` branch utilized **parallel chunking**.
- **Mechanism**: The 500 items were split into ~16 parallel chunks (e.g., 32 items per chunk).
- **Execution**: These chunks were processed as a single concurrent batch by the engine.
- **Efficiency**: Kept sequence lengths short (~1k-2k tokens), avoiding the $O(N^2)$ attention compute penalty and the host-to-device mask transfer bottleneck.

### 2. The "Single Packed Sequence" Architecture (Regression)
The `feat/multi-item-scoring` branch (and the unapproved merge) introduced a single packed sequence approach.
- **Mechanism**: Forces all 500 items into one 12k+ token sequence.
- **Bottleneck**: hit the attention compute cliff ($O(N^2)$) and required transferring a massive 1GB-4GB attention mask from host to TPU for every forward pass.
- **Kernel Bug**: Introduced a `ValueError` in the `ragged_paged_attention` kernel when running on `TP > 1`, as the kernel's sharding logic failed for GQA models with custom masks.

### 3. Chunk Size Sensitivity
Larger TPU topologies (2x2, 2x4) are extremely sensitive to `multi_item_scoring_chunk_size`.
- **Chunk=32**: Performance is poor (1-2 items/s) on larger meshes due to under-utilization.
- **Chunk=64+**: Performance restores to 60-70 items/s.
- **Parallel Batching**: Splitting into 16 parallel batches of 32 remains the fastest known method (141 items/s).

## Action Plan to Restore Performance

1.  **Branch Reset**: `aashishrampal/batching` has been reset to `ace9ae7` (pre-merge) to restore the core high-performance architecture.
2.  **Explicit Logging**: All benchmark scripts now explicitly print `server_args` to ensures every run is fully reproducible and traceable.
3.  **Kernel Patching**: The `ragged_paged_attention` kernel must be fixed to support `custom_mask` with `TP > 1` for GQA models.
4.  **Batching Optimization**: We will standardize on the "Parallel Sequence Batching" approach for large item counts, as it provides the only path to >100 items/s on TPU v6e.

---
*Summary compiled by Gemini CLI on 2026-02-12*
