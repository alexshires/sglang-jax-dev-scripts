# Profiling Tools

## Scripts
- `score_api_tpu_vm.sh`: end‑to‑end TPU VM orchestration for Score API profiling.
- `analyze_score_run.py`: convert and analyze traces into tables + charts.
- `setup_env.sh`: create a local analysis virtualenv.
- `device_tracer_level.patch`: enable device tracing in sglang‑jax.

## Typical Flow
```bash
./profiling/tools/setup_env.sh
source .venv-profile/bin/activate

# Run profiling (TPU VM)
PROJECT=... ./profiling/tools/score_api_tpu_vm.sh all

# Analyze
./profiling/tools/analyze_score_run.py --run-dir profiling/runs/<run-id>
```

## GCS Uploads
`score_api_tpu_vm.sh` now uploads raw traces **directly from the VM** to a default GCS bucket:
```
gs://sglang-jax-profiles-<timestamp>-score-<rand>/<run-id>/artifacts/raw/device/
```
Override with:
```
GCS_BUCKET=gs://my-bucket ./profiling/tools/score_api_tpu_vm.sh all
```
