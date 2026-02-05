# Score API Profiling Session - 2026-02-05

## Configuration
| Field | Value |
|-------|-------|
| **Model** | Qwen/Qwen3-0.6B |
| **TPU** | v6e-1 (us-east5-b) |
| **Endpoint** | /v1/score |
| **VM Name** | sglang-profiling-v2 |

## Summary

This session attempted to profile the Score API endpoint. While the profiling infrastructure worked correctly, Score API requests returned 400 errors due to a missing required `model` field in the request payload.

### Key Findings

1. **Profiling Infrastructure Works**: The `/start_profile` and `/stop_profile` endpoints work correctly on sglang-jax
2. **Score API Requires `model` Field**: The Score API request must include the `model` field
3. **Traces Were Generated**: Even without successful API calls, the profiler captured TPU activity (7.5MB trace file)

## Trace Files

| File | Size | Format |
|------|------|--------|
| `traces/t1v-n-2e9e6cfb-w-0.trace.json.gz` | 7.5 MB | Chrome Trace Event Format (Perfetto) |
| `traces/t1v-n-2e9e6cfb-w-0.xplane.pb` | 364 MB | XPlane Format (TensorBoard) |

## Correct Score API Request Format

The Score API requires the `model` field:

```python
import requests

response = requests.post("http://localhost:30000/v1/score", json={
    "model": "Qwen/Qwen3-0.6B",  # REQUIRED
    "query": "Is this a positive review?",
    "items": ["Great product!", "Terrible experience"],
    "label_token_ids": [9454, 2753]
})
```

### Error Without Model Field

```json
{
  "object": "error",
  "message": "[{'type': 'missing', 'loc': ('body', 'model'), 'msg': 'Field required', ...}]",
  "type": "Bad Request",
  "code": 400
}
```

## Viewing Traces

### Perfetto (Recommended)
1. Open https://ui.perfetto.dev
2. Drag and drop `t1v-n-2e9e6cfb-w-0.trace.json.gz`
3. Use keyboard shortcuts: W/S (zoom), A/D (pan), F (fit)

### TensorBoard
```bash
pip install tensorboard tensorboard-plugin-profile
tensorboard --logdir ./traces --port 6006
# Open http://localhost:6006 > Profile tab
```

## Reproduce

```bash
# 1. Create TPU VM
export TPU_NAME="sglang-profiling-v2"
export ZONE="us-east5-b"

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e

# 2. Setup Python 3.12 + sglang-jax
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='
sudo apt-get update && sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

git clone https://github.com/alexshires/sglang-jax.git ~/sglang-jax
cd ~/sglang-jax
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e "./python[tpu]"
'

# 3. Start server
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='
cd ~/sglang-jax && source .venv/bin/activate
python -m sgl_jax.launch_server --model-path Qwen/Qwen3-0.6B --port 30000 &
'

# 4. Wait for server (2 min for model loading)
sleep 120

# 5. Run profiling
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='
# Start profile
curl -X POST http://localhost:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir": "/tmp/profile", "num_steps": 20}'

# Send Score API requests (with model field!)
for i in {1..15}; do
  curl -X POST http://localhost:30000/v1/score \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "query": "Test?", "items": ["A", "B"], "label_token_ids": [9454, 2753]}'
done

# Stop profile
curl -X POST http://localhost:30000/stop_profile
'

# 6. Download traces
gcloud compute tpus tpu-vm scp $TPU_NAME:/tmp/profile/* ./traces/ --zone=$ZONE --recurse

# 7. Cleanup (important to avoid costs!)
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet
```

## Next Steps

1. Re-run profiling with correct Score API request format (including `model` field)
2. Analyze trace files to identify Score API performance characteristics
3. Compare with generate endpoint profiling results
