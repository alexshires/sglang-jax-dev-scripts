# Score API Profiling Runbook (TPU VM, JAX)

This runbook captures a repeatable, end‑to‑end profiling workflow for the Score API (`/v1/score`) on a TPU VM using the `sglang-jax` fork. It assumes the device tracer changes are required (see patch notes below).

## Prerequisites
- `gcloud` CLI authenticated with access to your project.
- Project has TPU VM quota in `us-east5-b` (v6e).
- Local repo contains the device tracer patch:
  - `profiling/tools/device_tracer_level.patch`
- Optional: local analysis env created via `profiling/tools/setup_env.sh`.

## Quick Start (All-In-One Script)
```bash
# From sglang-jax-dev-scripts
export PROJECT=sglang-jax-tests-1769450780
export ZONE=us-east5-b
export TPU_NAME=sglprof-$(date +%Y%m%d-%H%M%S)

./profiling/tools/score_api_tpu_vm.sh all
```

The script defaults the run directory to:
```
profiling/runs/<timestamp>_score_<model>_<tpu>
```
Example:
```
profiling/runs/20260205T201530Z_score_qwen3_0p6b_tpuv6e1
```

The script also defaults a GCS bucket (globally unique) like:
```
gs://sglang-jax-profiles-<timestamp>-score-<rand>
```
Override if needed:
```
GCS_BUCKET=gs://my-bucket ./profiling/tools/score_api_tpu_vm.sh all
```

This will:
- Create the TPU VM
- Clone `sglang-jax`, apply the device tracer patch, and install dependencies
- Start the server
- Warm up and run a short device‑traced profile
- Download artifacts into `profiling/runs/<timestamp>_score_<model>_<tpu>/...`
- Record inputs and run metadata into `profiling/runs/<timestamp>_score_<model>_<tpu>/inputs/`
- Upload raw traces directly from the VM to GCS (faster than local upload).

## Manual Steps (Detailed)

### 1) Create TPU VM
```bash
gcloud compute tpus tpu-vm create sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b \
  --project sglang-jax-tests-1769450780 \
  --accelerator-type v6e-1 \
  --version v6e-ubuntu-2404
```

### 2) Apply Device Tracer Patch
```bash
gcloud compute tpus tpu-vm scp \
  profiling/tools/device_tracer_level.patch \
  sglprof-YYYYMMDD-HHMMSS:/tmp/device_tracer_level.patch \
  --zone us-east5-b --project sglang-jax-tests-1769450780
```

### 3) VM Setup (deps + repo)
```bash
gcloud compute tpus tpu-vm ssh sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b --project sglang-jax-tests-1769450780 --command "set -e; \
  sudo apt-get update; sudo apt-get install -y git python3-venv; \
  git clone https://github.com/alexshires/sglang-jax.git; \
  cd sglang-jax; git checkout a18802ac38d209eacea09e040969262926781b80; \
  git apply /tmp/device_tracer_level.patch; \
  python3 -m venv .venv; source .venv/bin/activate; \
  pip install -U pip; cd python; \
  pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
  pip install tensorboard xprof"
```

### 4) Start Server
```bash
gcloud compute tpus tpu-vm ssh sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b --project sglang-jax-tests-1769450780 --command "set -e; \
  cd ~/sglang-jax; source .venv/bin/activate; export HF_HOME=/tmp/hf; \
  nohup python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 --tp-size 1 > /tmp/sgl_server.log 2>&1 & \
  echo \$! > /tmp/sgl_server.pid"
```

### 5) Wait for Health
```bash
gcloud compute tpus tpu-vm ssh sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b --project sglang-jax-tests-1769450780 --command "\
  for i in {1..60}; do \
    if curl -s http://localhost:30000/health > /dev/null; then echo READY; exit 0; fi; \
    echo -n '.'; sleep 10; \
  done; echo TIMEOUT; exit 1"
```

### 6) Warmup + Profiled Requests
```bash
gcloud compute tpus tpu-vm ssh sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b --project sglang-jax-tests-1769450780 --command "set -e; \
  curl -s -X POST http://localhost:30000/v1/score \
    -H 'Content-Type: application/json' \
    -d '{\"query\": \"Is this positive?\", \"items\": [\"Great product!\", \"Terrible experience\"], \"label_token_ids\": [9834, 902], \"apply_softmax\": true, \"model\": \"Qwen/Qwen3-0.6B\"}'; \
  curl -s -X POST http://localhost:30000/start_profile \
    -H 'Content-Type: application/json' \
    -d '{\"output_dir\": \"/tmp/score_profile_device\", \"num_steps\": 2, \"host_tracer_level\": 2, \"python_tracer_level\": 1, \"device_tracer_level\": 2}'; \
  for i in {1..2}; do \
    curl -s -X POST http://localhost:30000/v1/score \
      -H 'Content-Type: application/json' \
      -d '{\"query\": \"Is this positive?\", \"items\": [\"Great product!\", \"Terrible experience\"], \"label_token_ids\": [9834, 902], \"apply_softmax\": true, \"model\": \"Qwen/Qwen3-0.6B\"}'; \
  done; \
  curl -s -X POST http://localhost:30000/stop_profile || true"
```

Note: the `label_token_ids` values are specific to Qwen3‑0.6B. If you change the model, regenerate them with the tokenizer and update the request payload.

### 7) Download Artifacts
```bash
mkdir -p profiling/runs/<run-id>/artifacts/raw/device

gcloud compute tpus tpu-vm scp --recurse \
  sglprof-YYYYMMDD-HHMMSS:/tmp/score_profile_device \
  profiling/runs/<run-id>/artifacts/raw/device \
  --zone us-east5-b --project sglang-jax-tests-1769450780

gcloud compute tpus tpu-vm scp \
  sglprof-YYYYMMDD-HHMMSS:/tmp/sgl_server.log \
  profiling/runs/<run-id>/logs/server.log \
  --zone us-east5-b --project sglang-jax-tests-1769450780
```

### 8) Local Analysis
```bash
./profiling/tools/setup_env.sh
source .venv-profile/bin/activate

./profiling/tools/analyze_score_run.py --run-dir profiling/runs/<run-id>
```

## Running From a Fresh Machine
1. Install `gcloud` and authenticate: `gcloud auth login` and `gcloud config set project <PROJECT>`.
2. Clone this repo and `cd sglang-jax-dev-scripts`.
3. Run the all‑in‑one script with `PROJECT` and optional `TPU_NAME`:
   - `PROJECT=... ./profiling/tools/score_api_tpu_vm.sh all`
4. Run analysis locally:
   - `./profiling/tools/setup_env.sh`
   - `source .venv-profile/bin/activate`
   - `./profiling/tools/analyze_score_run.py --run-dir profiling/runs/<run-id>`

## Cleanup
```bash
gcloud compute tpus tpu-vm delete sglprof-YYYYMMDD-HHMMSS \
  --zone us-east5-b --project sglang-jax-tests-1769450780 --quiet
```

## Optional: GCS Backup (Minimal)

Recommended minimal backup (keeps report + analysis + images + inputs + logs, and a small raw subset):
- Host trace: `t1v-*.trace.json.gz` + host `xplane.pb`
- Device trace: `t1v-*.trace.json.gz` + `ALL_HOSTS.op_stats.pb`

Skip device `xplane.pb` if upload is too slow; you can always add it later.

Note: the `upload` step in `score_api_tpu_vm.sh` performs a **raw upload from the VM** to GCS. This is usually the fastest path for large `xplane.pb` files.

Example:
```bash
export BUCKET=gs://<your-bucket>/<run-id>
gsutil -m rsync -r profiling/runs/<run-id>/analysis "$BUCKET/analysis"
gsutil -m rsync -r profiling/runs/<run-id>/images "$BUCKET/images"
gsutil -m rsync -r profiling/runs/<run-id>/inputs "$BUCKET/inputs"
gsutil -m rsync -r profiling/runs/<run-id>/logs "$BUCKET/logs"
gsutil -m cp profiling/runs/<run-id>/report.md profiling/runs/<run-id>/checksums.txt "$BUCKET/"

# Minimal raw traces
gsutil cp profiling/runs/<run-id>/artifacts/raw/host/traces/t1v-*.trace.json.gz "$BUCKET/artifacts/raw/host/traces/"
gsutil cp profiling/runs/<run-id>/artifacts/raw/host/traces/t1v-*.xplane.pb "$BUCKET/artifacts/raw/host/traces/"
gsutil cp profiling/runs/<run-id>/artifacts/raw/device/traces-device/**/t1v-*.trace.json.gz "$BUCKET/artifacts/raw/device/traces-device/"
gsutil cp profiling/runs/<run-id>/artifacts/raw/device/traces-device/**/ALL_HOSTS.op_stats.pb "$BUCKET/artifacts/raw/device/traces-device/"
```

## Notes on Device Tracing
The default server code only sets host/python tracer levels. To capture TPU kernel activity you must apply the patch:
- `profiling/tools/device_tracer_level.patch`

Without it, the XPlane trace shows CPU‑only and you cannot attribute device kernel time.
