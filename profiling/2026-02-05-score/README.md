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

## Deep Dive Analysis (Host Trace)

Source: `traces/t1v-n-3d21a6bb-w-0.trace.json.gz`

Top events by total duration (ms):
- `$threading.py:637 wait` — 12,085 ms (count=2)
- `$threading.py:323 wait` — 12,085 ms (count=2)
- `$scheduler.py:645 recv_requests` — 296 ms (count=16,949)
- `$socket.py:961 recv_pyobj` — 268 ms (count=33,898)
- `$error.py:120 __init__` — 105 ms (count=33,898)
- `$error.py:45 __init__` — 60 ms (count=33,898)
- `$<frozen importlib._bootstrap>:1390 _handle_fromlist` — 47.5 ms (count=67,796)
- `$scheduler.py:1025 check_memory` — 32 ms (count=16,949)
- `$scheduler.py:1096 get_next_batch_to_run` — 19 ms (count=16,949)
- `$scheduler.py:670 process_input_requests` — 18 ms (count=16,949)

Interpretation:
- The host timeline is dominated by **wait time** (threading wait) and request receive/dispatch work (`recv_requests`, ZMQ socket receive).
- There is **no visible TPU kernel time in this host trace**. This trace mainly shows Python and scheduler activity.

Command used to extract this summary:
```bash
python - <<'PY'
import gzip, json, collections
from pathlib import Path
path = Path('traces/t1v-n-3d21a6bb-w-0.trace.json.gz')
with gzip.open(path, 'rb') as f:
    data = json.load(f)

events = data.get('traceEvents', [])
name_stats = collections.defaultdict(lambda: {'count':0, 'dur_us':0})
for e in events:
    if 'dur' in e and isinstance(e['dur'], (int, float)):
        name = e.get('name', 'unknown')
        name_stats[name]['count'] += 1
        name_stats[name]['dur_us'] += e['dur']

items = sorted(name_stats.items(), key=lambda x: x[1]['dur_us'], reverse=True)[:20]
print('Top by total duration (ms):')
for name, st in items:
    print(f"{name[:60]:60s} count={st['count']:6d} total_ms={st['dur_us']/1000:.2f}")
PY
```

## Device Trace Attempt (XPlane)

Source: `traces/t1v-n-3d21a6bb-w-0.xplane.pb`

I generated additional tool outputs from the XPlane file to try to surface TPU kernel time:
- `overview_page.json`
- `op_profile.json`
- `framework_op_stats.json`
- `trace_viewer.json` (very large: ~4.65 GB)

Results:
- `overview_page.json` reports `hardware_type: CPU_ONLY` and **no step time measured**.
- `op_profile.json` is empty (`deviceType: CPU_ONLY`).
- `framework_op_stats.json` contains only a single row: `IDLE` on Host.
- The derived `trace_viewer.json` contains **33,158,721 events**, all host/Python-level function names. No TPU kernel categories or TPU/XLA-named events were found.

This indicates the XPlane trace did **not capture TPU device kernels** for this run, or the current conversion pipeline could not see them. The most likely reason is that only `host_tracer_level` and `python_tracer_level` were set in the server’s `ProfileOptions` (see `sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py`), so device-level tracing was not explicitly enabled.

Command used to generate tool outputs:
```bash
python - <<'PY'
from xprof.convert import raw_to_tool_data as rtd
from pathlib import Path
xplane = Path('traces/t1v-n-3d21a6bb-w-0.xplane.pb')

# overview
text, _ = rtd.xspace_to_tool_data([str(xplane)], 'overview_page', {'use_saved_result': False})
Path('overview_page.json').write_text(text.decode('utf-8') if isinstance(text, (bytes, bytearray)) else text)

# op_profile
text, _ = rtd.xspace_to_tool_data([str(xplane)], 'op_profile', {'use_saved_result': False, 'group_by': 'program'})
Path('op_profile.json').write_text(text.decode('utf-8') if isinstance(text, (bytes, bytearray)) else text)

# framework_op_stats
text, _ = rtd.xspace_to_tool_data([str(xplane)], 'framework_op_stats', {'use_saved_result': False})
Path('framework_op_stats.json').write_text(text.decode('utf-8') if isinstance(text, (bytes, bytearray)) else text)
PY
```

Command used to generate the large trace viewer JSON:
```bash
python - <<'PY'
from xprof.convert import raw_to_tool_data as rtd
from pathlib import Path
xplane = Path('traces/t1v-n-3d21a6bb-w-0.xplane.pb')
raw, _ = rtd.xspace_to_tool_data([str(xplane)], 'trace_viewer@', {'use_saved_result': False, 'trace_viewer_options': {'resolution': 8000}})
Path('trace_viewer.json').write_bytes(raw if isinstance(raw, (bytes, bytearray)) else raw.encode('utf-8'))
PY
```

## Conclusion: Where Is the Delay?

Based on the available traces for this run:
- **Host-side time** is dominated by scheduler receive/dispatch and thread waiting.
- **Device-side (TPU) kernel time is not visible** in the captured profile output, so we **cannot attribute the scoring latency to specific kernels** with this trace.

In other words, **the trace shows the host mostly waiting**, which implies the real latency is on the device (TPU) side or in compilation—but that device work was not captured by the profiling options used in this run.

## Next Steps (If You Want Device Kernel Attribution)

This would require enabling device tracing in JAX profiling. The current server code only sets host/python tracer levels. To attribute delay to exact TPU kernels, we would need a profiling run that includes device tracer data (likely a small code change to set a device tracer level or a different profiling entrypoint). I did not make any code changes per your requirement.

If you want, I can propose a no-code-change re-run with different parameters (e.g., more steps) but it may still omit device kernels unless device tracing is explicitly enabled.
