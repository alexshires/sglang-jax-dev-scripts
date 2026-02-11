# Investigation: JAX vs PyTorch Multi-Item Comparison Methodology

| | |
|------------|------|
| **Date** | 2026-02-11 |
| **Status** | **Ready to Run** |
| **Goal** | Evaluate `sglang-jax` multi-item scoring against frozen PyTorch baseline |
| **Related** | [RFC-008](../rfcs/008-multi-item-scoring.md), [Cross-backend RFC](../rfcs/010-cross-backend-benchmarking.md) |

## Objective

Measure how well the new `sglang-jax` multi-item scoring API compares to PyTorch `sglang` without doing PyTorch optimization work.

This investigation defines a reproducible benchmark contract and result schema so conclusions are defensible.

## Scope Boundaries

In scope:
- benchmark harness scripts
- reproducible runbook
- portable and best-native comparison outputs
- correctness drift checks against PyTorch

Out of scope:
- PyTorch code changes
- PyTorch kernel/runtime optimization
- upstream PyTorch PRs

## Core Comparison Rules

1. PyTorch is the frozen reference baseline.
2. JAX development/tuning remains the priority.
3. Conclusions must separate backend behavior from hardware impact.
4. Correctness is a hard gate: performance-only wins are invalid if score drift exceeds threshold.

## Views

### 1) Portable View

Purpose: strict shape parity for backend behavior comparison.

- Same model: `Qwen/Qwen3-0.6B`
- Same pre-tokenized canonical workload
- Same client chunk sizes: `1,2,4,8,16,32,64,128,256,500`
- Same request order and same label tokens (`9454,2753`)
- JAX configured to avoid hidden extra splitting in this view
- PyTorch run with required multi-item delimiter path

### 2) Best-Native View

Purpose: practical deployment-facing comparison.

- JAX uses recommended native settings from prior JAX profiling/ablation
- PyTorch uses best stable client chunk size from matrix (no code changes)
- Compare highest stable performance while preserving correctness

## Workload Contract

- Query length: `2000` tokens
- Item count: `500`
- Tokens per item: `20`
- Warmup runs: `1`
- Timed runs:
  - Sweep: `5`
  - Confirm: `7`
- Metrics:
  - `p50/p95/mean` full 500-item end-to-end latency
  - items/sec
  - success rate + failure reason (`oom`, `timeout`, `http_error`, etc.)
  - first-run penalty ratio
  - optional cost normalization when hourly cost is provided

## Correctness Gate

For comparable rows (or selected best-native rows), compute:
- `max_abs_diff`
- `mean_abs_diff`

Default pass thresholds:
- `max_abs_diff <= 0.02`
- `mean_abs_diff <= 0.01`

Rows that fail correctness are non-eligible for winner selection.

## Selection Logic

1. Eligible rows must satisfy:
- `success_rate == 1.0`
- valid `p95_total_e2e_ms`
- valid `throughput_items_sec`
- correctness pass

2. Guardrail:
- `guardrail_p95 = 1.25 * min_p95(eligible)`

3. Winner:
- highest throughput among rows within guardrail
- tie-break: smaller chunk size

## Scripts

- Canonical workload generator:
  - `investigations/scripts/generate_canonical_score_workload.py`
- JAX matrix runner:
  - `investigations/scripts/run_score_matrix_jax.py`
- PyTorch matrix runner:
  - `investigations/scripts/run_score_matrix_pytorch.py`
- Comparison harness:
  - `investigations/scripts/compare_score_matrix_results.py`
- Final side-by-side report renderer:
  - `investigations/scripts/render_jax_vs_pytorch_final_report.py`

## Output Schemas

### Canonical workload schema (`canonical_score_workload_v1`)

Top-level fields:
- `schema_version`
- `model`
- `query_tokens`
- `num_items`
- `item_tokens`
- `requests[]`

### Per-backend matrix schema (`score_matrix_v1`)

Top-level fields:
- `schema_version`
- `backend`
- `hardware`
- `server_config`
- `workload_ref`
- `results[]`
- `summary`

### Cross-backend comparison schema (`cross_backend_comparison_v1`)

Top-level fields:
- `schema_version`
- `portable_view`
- `best_native_view`
- `correctness`
- `recommendations`
- `notes`

## Risk Notes

- Hardware differs (TPU vs GPU), so portable view is still not strict hardware-isolation.
- Delimiter token collisions are avoided by canonical workload generation using safe token pools.
- If model/tokenizer revisions change, regenerate workload and keep checksum in artifacts.

## Acceptance Checklist

- [ ] All four JSON outputs exist (JAX portable/native, PyTorch portable/native)
- [ ] Comparison JSON + markdown report produced
- [ ] Portable and best-native winners computed
- [ ] Correctness thresholds explicitly reported
- [ ] Final report includes where JAX is better/worse and by how much
