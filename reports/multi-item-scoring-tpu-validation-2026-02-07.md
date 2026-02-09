# Report: Multi-Item Scoring TPU Validation (2026-02-07)

| | |
|------------|------|
| **Date** | 2026-02-07 |
| **Scope** | RFC-008 rollout validation |
| **Environment** | TPU v6e-1 (`us-east5-b`) |
| **Project** | `sglang-jax-tests-1769450780` |
| **Branch** | `feat/multi-item-scoring` |
| **PR** | [alexshires/sglang-jax#15](https://github.com/alexshires/sglang-jax/pull/15) |

## Summary

RFC-008 is implemented in `sglang-jax` as a feature-gated MVP. Validation on TPU shows:

- zero changed-length leakage in validated Qwen3 matrix
- stable semantic-aligned parity behavior
- meaningful throughput gains vs serial one-item scoring

## What Was Validated

1. Correctness and isolation:
   - same-length mutation isolation
   - changed-length mutation isolation
2. Equivalence:
   - multi vs serial query-only comparison
   - multi vs serial delimiter-aligned comparison (`query + delimiter + item`)
3. JAX vs PyTorch parity:
   - serial semantic parity
   - multi semantic parity
4. Throughput:
   - speedup vs serial one-item calls at item counts `1, 8, 32, 64, 128`

## Model Matrix Results

| Model | Same-len isolation max | Changed-len isolation max | Delimiter-aligned equiv max | JAX serial vs torch serial max | JAX multi vs torch multi-sem max | Speedup N=8 | Speedup N=32 | Speedup N=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 0.0000 | 0.0000 | 0.0148 | 0.0041 | 0.0115 | 3.45x | 4.69x | 5.30x |
| `Qwen/Qwen3-1.7B` | 0.0000 | 0.0000 | 0.0150 | 0.0118 | 0.0050 | 3.22x | 4.27x | 5.22x |
| `Qwen/Qwen3-4B` | 0.0000 | 0.0000 | 0.0119 | 0.0052 | 0.0041 | 3.07x | 3.85x | 4.29x |

## Key Decision Confirmed

Using default `--multi-item-scoring-chunk-size=2` is the right rollout choice for this branch:

- it eliminated the last observed changed-length drift in validation
- it still preserved strong multi-item throughput gains at larger item counts

## Known Limitation

Tested Qwen2.5 models (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`) failed on first scoring request in the fused-KV path with:

`ValueError: ... reshape((2, -1, 8, 128)) ... size of ref (1024)`

This is tracked as follow-up kernel compatibility work, not an RFC-008 masking correctness issue.

## Canonical Artifacts

Source-of-truth machine-readable artifacts and detailed markdown reports are in `sglang-jax`:

- `docs/features/reports/multi_item_scoring_tpu_eval_20260207_v4.md`
- `docs/features/reports/multi_item_scoring_tpu_model_matrix_20260207_v1.md`
- `docs/features/reports/multi_item_eval_results_20260207_v4.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_v4.json`
- `docs/features/reports/jax_torch_parity_results_20260207_v4.json`
- `docs/features/reports/multi_item_eval_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/jax_torch_parity_results_20260207_qwen3_1_7b_v2.json`
- `docs/features/reports/multi_item_eval_results_20260207_qwen3_4b_v1.json`
- `docs/features/reports/multi_vs_serial_eval_results_20260207_qwen3_4b_v1.json`
- `docs/features/reports/jax_torch_parity_results_20260207_qwen3_4b_v1.json`
