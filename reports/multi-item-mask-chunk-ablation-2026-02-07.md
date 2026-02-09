# Report: Multi-Item Mask/Chunk Ablation (2026-02-07)

| | |
|------------|------|
| **Date** | 2026-02-07 |
| **Scope** | RFC-008 follow-up experiment |
| **Environment** | TPU v6e-1 (`us-east5-b`) |
| **Project** | `sglang-jax-tests-1769450780` |
| **Branch** | `feat/multi-item-scoring-experiments` |
| **PR** | [alexshires/sglang-jax#16](https://github.com/alexshires/sglang-jax/pull/16) |

## Why This Follow-up Was Run

After RFC-008 rollout readiness (PR #15), we evaluated whether alternative mask semantics or chunk-size settings would improve behavior.

Experiment tracks:

1. mask variants at fixed chunk size (`2`)
2. chunk-size sweep (`0,1,2,4,8`) at baseline mask

## Results Summary

### Mask variants (chunk=2)

| Variant | Changed-len isolation max | Delim-aligned equiv max | JAX multi vs torch multi-sem max | Speedup N=32 | Speedup N=128 |
|---|---:|---:|---:|---:|---:|
| `prefix_first_delim` | 0.0000 | 0.0148 | 0.0115 | 4.93x | 5.80x |
| `query_only_prefix` | 0.0000 | 0.4579 | 0.4426 | 5.08x | 5.70x |
| `all_delims_prefix` | 0.0000 | 0.0833 | 0.0817 | 4.97x | 5.47x |

Interpretation:

- `query_only_prefix` and `all_delims_prefix` degrade semantic-aligned parity sharply.
- Baseline `prefix_first_delim` remains the only variant with strong semantic consistency.

### Chunk-size sweep (baseline mask)

| Chunk Size | Changed-len isolation max | Delim-aligned equiv max | JAX multi vs torch multi-sem max | Speedup N=32 | Speedup N=128 |
|---|---:|---:|---:|---:|---:|
| `0` | 0.0117 | 0.0148 | 0.0115 | 4.13x | 1.38x |
| `1` | 0.0000 | 0.0363 | 0.0344 | 2.89x | 3.10x |
| `2` | 0.0000 | 0.0148 | 0.0115 | 4.93x | 5.80x |
| `4` | 0.0117 | 0.0148 | 0.0115 | 7.23x | 8.70x |
| `8` | 0.0117 | 0.0148 | 0.0115 | 7.89x | 9.59x |

Interpretation:

- `4/8` increase throughput but reintroduce changed-length leakage.
- `1/2` keep zero changed-length leakage.
- `2` dominates `1` on both semantic parity and throughput.

## Edge Validation Snapshot

Validation checks from follow-up run:

- delimiter collision: HTTP `400`
- too many items (`129`): HTTP `400`
- over max sequence length: HTTP `400`
- valid within-limit request: HTTP `200`

## Final Recommendation

Keep the production rollout defaults from RFC-008:

1. mask semantics: `prefix_first_delim`
2. chunk size: `2`

## Canonical Artifacts (sglang-jax)

- `docs/features/reports/multi_item_mask_chunk_experiments_20260207_v2.md`
- `docs/features/reports/multi_item_mask_chunk_summary_20260207_mask_chunk_exp_v2.json`
- `docs/features/reports/multi_item_edge_checks_20260207_mask_chunk_exp_v2.json`
