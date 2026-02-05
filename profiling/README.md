# Profiling (sglang-jax)

This folder contains reproducible profiling workflows, tooling, and run artifacts.

## Folder Layout
- `profiling/runbooks/` — step‑by‑step runbooks
- `profiling/tools/` — scripts for running and analyzing profiles
- `profiling/runs/` — per‑run artifacts, analysis, and reports
- `profiling/archive/` — old or superseded profiling docs (kept for reference)

## Quick Start
```bash
# Set up local analysis env
./profiling/tools/setup_env.sh
source .venv-profile/bin/activate

# Analyze an existing run
./profiling/tools/analyze_score_run.py --run-dir profiling/runs/<timestamp>_score_<model>_<tpu>
```

## Runbooks
- `profiling/runbooks/score_api_tpu_vm.md`

## Tools
- `profiling/tools/score_api_tpu_vm.sh` — end‑to‑end run orchestration on TPU VM
- `profiling/tools/analyze_score_run.py` — generate tables/charts from artifacts
- `profiling/tools/device_tracer_level.patch` — enable device tracing in sglang‑jax
