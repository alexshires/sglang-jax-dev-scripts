# RFC-011 End-to-End JAX TPU Profiling (Single /v1/score)

Date: 2026-02-05

## Summary
Ran a single-request, end-to-end JAX profiling session on a TPU v6e-1 using the existing sglang-jax fork in this workspace (commit `a18802ac38d209eacea09e040969262926781b80`). Collected JAX profiler artifacts (`.trace.json.gz` and `.xplane.pb`), uploaded to GCS, and captured server logs. No code changes were made.

## Key Choices (and Why)
- **Infra:** TPU VM (not preemptible) for lowest friction and deterministic behavior.
- **TPU type:** `v6e-1` (single chip) to keep the run simple and cost-limited.
- **Zone:** `us-east5-b` (v6e availability).
- **Runtime image:** `v6e-ubuntu-2404` (dedicated v6e image).
- **Model:** `Qwen/Qwen3-0.6B` (ungated, small model).
- **Profiling mode:** JAX `start_trace` via `/start_profile` with `num_steps=1`, `host_tracer_level=2`, `python_tracer_level=1`.

## Environment / Versions
- **Project:** `sglang-jax-tests-1769450780`
- **TPU VM name:** `sglprof-20260205-184819`
- **TPU:** v6e-1
- **OS:** Ubuntu 24.04.2 LTS
- **Python:** 3.12.3
- **JAX / jaxlib:** 0.8.1 / 0.8.1
- **libtpu:** 0.0.30 (from JAX TPU extra)
- **sglang-jax repo:** https://github.com/alexshires/sglang-jax @ `a18802ac38d209eacea09e040969262926781b80`

## Commands Executed (Reproducible)

### 1) Create TPU VM
```bash
gcloud compute tpus tpu-vm create sglprof-20260205-184819 \
  --zone us-east5-b \
  --project sglang-jax-tests-1769450780 \
  --accelerator-type v6e-1 \
  --version v6e-ubuntu-2404
```

### 2) VM setup (deps + repo + venv + install)
```bash
gcloud compute tpus tpu-vm ssh sglprof-20260205-184819 --zone us-east5-b \
  --project sglang-jax-tests-1769450780 --command "set -e; \
  sudo apt-get update; sudo apt-get install -y git python3-venv; \
  git clone https://github.com/alexshires/sglang-jax.git; \
  cd sglang-jax; git checkout a18802ac38d209eacea09e040969262926781b80; \
  python3 -m venv .venv; source .venv/bin/activate; \
  pip install -U pip; \
  cd python; \
  pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
  pip install tensorboard xprof"
```

### 3) Start server
```bash
gcloud compute tpus tpu-vm ssh sglprof-20260205-184819 --zone us-east5-b \
  --project sglang-jax-tests-1769450780 --command "set -e; \
  cd ~/sglang-jax; source .venv/bin/activate; export HF_HOME=/tmp/hf; \
  nohup python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp-size 1 > /tmp/sgl_server.log 2>&1 & \
  echo \$! > /tmp/sgl_server.pid"
```

### 4) Wait for health
```bash
gcloud compute tpus tpu-vm ssh sglprof-20260205-184819 --zone us-east5-b \
  --project sglang-jax-tests-1769450780 --command "\
  for i in {1..40}; do \
    if curl -s http://localhost:30000/health > /dev/null; then echo READY; exit 0; fi; \
    echo -n '.'; sleep 10; \
  done; echo TIMEOUT; exit 1"
```

### 5) Token IDs for labels
```bash
python - <<'PY'
from transformers import AutoTokenizer
model='Qwen/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print(' yes', tok.encode(' yes', add_special_tokens=False))
print(' no', tok.encode(' no', add_special_tokens=False))
PY
# Output:
#  yes [9834]
#  no [902]
```

### 6) Warmup score request (not profiled)
```bash
curl -s -X POST http://localhost:30000/v1/score \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Is this positive?",
    "items": ["Great product!", "Terrible experience"],
    "label_token_ids": [9834, 902],
    "apply_softmax": true,
    "model": "Qwen/Qwen3-0.6B"
  }'
```

### 7) Start profiling
```bash
curl -s -X POST http://localhost:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/score_profile","num_steps":1,"host_tracer_level":2,"python_tracer_level":1}'
```

### 8) Profiled score request
```bash
curl -s -X POST http://localhost:30000/v1/score \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Is this positive?",
    "items": ["Great product!", "Terrible experience"],
    "label_token_ids": [9834, 902],
    "apply_softmax": true,
    "model": "Qwen/Qwen3-0.6B"
  }'
```

### 9) Stop profiling (note: auto-stops after num_steps)
```bash
curl -s -X POST http://localhost:30000/stop_profile
# Returns 500 if auto-stopped already (expected with num_steps=1)
```

### 10) Copy artifacts
```bash
# From VM to local workspace
gcloud compute tpus tpu-vm scp --recurse \
  sglprof-20260205-184819:/tmp/score_profile \
  /Users/kanna/Sandbox/sglang-all/profiling-artifacts/sglprof-20260205-184819 \
  --zone us-east5-b --project sglang-jax-tests-1769450780

# Server log
gcloud compute tpus tpu-vm scp \
  sglprof-20260205-184819:/tmp/sgl_server.log \
  /Users/kanna/Sandbox/sglang-all/profiling-artifacts/sglprof-20260205-184819/ \
  --zone us-east5-b --project sglang-jax-tests-1769450780
```

### 11) Upload to GCS (created bucket)
```bash
# Bucket created:
# gs://sglang-jax-profiles-20260205-191153/

gsutil -m cp -r \
  /Users/kanna/Sandbox/sglang-all/profiling-artifacts/sglprof-20260205-184819/score_profile \
  gs://sglang-jax-profiles-20260205-191153/

gsutil cp \
  /Users/kanna/Sandbox/sglang-all/profiling-artifacts/sglprof-20260205-184819/sgl_server.log \
  gs://sglang-jax-profiles-20260205-191153/
```

## Outputs

### /v1/score Response (Warmup + Profiled)
```json
{"scores":[[0.8438951025545426,0.1561048974454574],[0.29421497216298875,0.7057850278370112]],"model":"Qwen/Qwen3-0.6B","usage":null,"object":"scoring"}
```

### Server log snippet (profiling lifecycle)
```
[2026-02-05 19:10:20] INFO:     127.0.0.1:55034 - "POST /v1/score HTTP/1.1" 200 OK
[2026-02-05 19:10:40] INFO:     127.0.0.1:55480 - "POST /start_profile HTTP/1.1" 200 OK
[2026-02-05 19:11:16] INFO:     127.0.0.1:35678 - "POST /v1/score HTTP/1.1" 200 OK
[2026-02-05 19:11:27] INFO:     127.0.0.1:39876 - "POST /stop_profile HTTP/1.1" 500 Internal Server Error
RuntimeError: Profiling is not in progress. Call /start_profile first.
```

**Note:** `/stop_profile` failed because profiling **auto-stopped** after `num_steps=1`. The trace artifacts were already written.

## Artifacts

Traces are stored in GCS (too large for git):

```bash
gsutil cp -r gs://sglang-jax-profiles-20260205-191153/score_profile/plugins/profile/*/* traces/
gsutil cp gs://sglang-jax-profiles-20260205-191153/sgl_server.log server.log
```

| File | Size |
|------|------|
| `t1v-n-3d21a6bb-w-0.trace.json.gz` | 7.2 MB |
| `t1v-n-3d21a6bb-w-0.xplane.pb` | 498 MB |
| `server.log` | 10 KB |

### Checksums (sha256)
```
509dd2c6d0e9c50ffa988238c48cefbe3cf2731bcdea5541044db68ea8db102d  t1v-n-3d21a6bb-w-0.trace.json.gz
e390899f90128598443983fdf9a441618b4a0cdd6d418c48840d5cf027c1b112  t1v-n-3d21a6bb-w-0.xplane.pb
```

### File sizes
- `t1v-n-3d21a6bb-w-0.trace.json.gz`: 7.2 MB
- `t1v-n-3d21a6bb-w-0.xplane.pb`: 498 MB

## Analysis (Host Trace Summary)

Using `trace.json.gz` (host + Python trace):
- **Total events:** 1,000,008
- **Time window:** ~0.385s
- **Event type counts:** `X` (1,000,000), `M` (7)
- **Top host events by total duration (ms):**
  - `$threading.py:637 wait` — 12,085 ms
  - `$threading.py:323 wait` — 12,085 ms
  - `$scheduler.py:645 recv_requests` — 296 ms
  - `$socket.py:961 recv_pyobj` — 268 ms
  - `$error.py:120 __init__` — 105 ms
  - `$error.py:45 __init__` — 60 ms
  - `$<frozen importlib._bootstrap>:1390 _handle_fromlist` — 47.5 ms
  - `$scheduler.py:1025 check_memory` — 32 ms
  - `$scheduler.py:1096 get_next_batch_to_run` — 19 ms
  - `$scheduler.py:670 process_input_requests` — 18 ms

**Interpretation:** The host trace is dominated by Python/ZMQ scheduling and request handling. Device-level TPU execution data is contained in the **xplane** file; use Perfetto/TensorBoard/XProf for kernel-level analysis.

## How To Analyze In UI

### Perfetto (recommended for trace.json.gz)
1. Open Perfetto UI in a browser: https://ui.perfetto.dev
2. Drag and drop `traces/t1v-n-3d21a6bb-w-0.trace.json.gz`
3. Use the "Thread" view to inspect scheduler and request handling.

### TensorBoard / XProf (xplane)
```bash
tensorboard --logdir traces/ --port 6006
# Open http://localhost:6006 > Profile tab
```

## Cost Considerations
- **Compute:** On‑demand TPU v6e‑1 VM billed per hour. This run used ~40–60 minutes of VM uptime plus setup time. If you need lower cost, use `--preemptible` for spot pricing.
- **Storage:** Artifacts total ~505 MB per run. GCS storage and any egress apply if downloaded externally.
- **Suggested mitigation:** Tear down VM immediately after use and keep only needed artifacts.

## Cleanup (optional)
```bash
# Delete TPU VM
gcloud compute tpus tpu-vm delete sglprof-20260205-184819 \
  --zone us-east5-b --project sglang-jax-tests-1769450780

# Delete GCS bucket (if you do not need artifacts)
gsutil rm -r gs://sglang-jax-profiles-20260205-191153
```
