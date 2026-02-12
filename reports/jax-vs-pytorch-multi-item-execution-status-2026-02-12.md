# Execution Status: JAX vs PyTorch Multi-Item Comparison (2026-02-12)

| | |
|---|---|
| **Date** | 2026-02-12 |
| **Scope** | Cross-backend multi-item scoring comparison execution readiness |
| **Project** | `sglang-jax-tests-1769450780` |
| **Account** | `deshkanna@gmail.com` |
| **Status** | **TPU ready; GPU blocked by quota/capacity** |

## Summary

- TPU path is available and runnable for JAX matrix execution.
- GPU path (G4 VM) is currently blocked in this project by quota/capacity constraints.
- Comparison report is therefore in **TPU-only partial state** until GPU quota is available.

## What Was Verified

1. Auth and project context:
- Active project: `sglang-jax-tests-1769450780`
- Active account: `deshkanna@gmail.com`

2. TPU provisioning:
- TPU v6e-1 creation in `us-east5-b` failed due capacity.
- TPU v6e-1 creation in `us-east5-a` succeeded (`mi-tpu-v6e1`).

3. GPU provisioning (G4 target):
- Multiple zone attempts failed.
- Primary blocker: `GPUS_PER_GPU_FAMILY` quota for `NVIDIA_RTX_PRO_6000` reported as `0` in tested regions.
- Additional blocker in one zone: temporary resource availability.

## Impact On Evaluation

- Portable and best-native **JAX** runs can proceed on TPU v6e-1.
- Portable and best-native **PyTorch** runs are pending GPU quota/capacity.
- Cross-backend winner selection remains **deferred** until GPU results are present.

## Explicit Documentation Requirement Applied

- GPU was intentionally skipped for current execution due quota/capacity blocker.
- This skip should be called out in all result summaries until resolved.

## Next Execution Step (When GPU Is Unblocked)

1. Provision G4 VM in an allowed zone/region with available quota.
2. Run:
- `investigations/scripts/run_score_matrix_pytorch.py` (portable)
- `investigations/scripts/run_score_matrix_pytorch.py` (best-native)
3. Rebuild final comparison:
- `investigations/scripts/compare_score_matrix_results.py`
- `investigations/scripts/render_jax_vs_pytorch_final_report.py`

## Artifact Notes

- Local packaging artifacts were prepared for TPU sync under:
- `/Users/kanna/Sandbox/sglang-all/.artifacts/`
- The large docs archive includes `profiling/` history; a minimal archive variant was also produced for faster transfer.
