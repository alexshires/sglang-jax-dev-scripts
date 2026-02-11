# jax-vs-pytorch-multi-item-20260211 Artifacts

This directory stores raw artifacts for the cross-backend multi-item scoring comparison.

Expected files:
- `canonical_workload.json`
- `jax_portable_matrix.json`
- `jax_best_native_matrix.json`
- `pytorch_portable_matrix.json`
- `pytorch_best_native_matrix.json`
- `comparison.json`
- `comparison.md`

Generate these with:
- `investigations/scripts/generate_canonical_score_workload.py`
- `investigations/scripts/run_score_matrix_jax.py`
- `investigations/scripts/run_score_matrix_pytorch.py`
- `investigations/scripts/compare_score_matrix_results.py`
