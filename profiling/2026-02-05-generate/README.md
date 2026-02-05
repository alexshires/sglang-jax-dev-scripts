# Profiling Session: 2026-02-05 Generate Endpoint

| | |
|---|---|
| **Date** | 2026-02-05 |
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Hardware** | TPU v6e-1 (single chip) |
| **Zone** | us-east5-b |
| **Endpoint** | /generate |
| **JAX Version** | 0.8.1 |
| **sglang-jax Version** | 0.0.2 |

## Executive Summary

Successfully executed end-to-end profiling of sglang-jax on TPU v6e. Generated comprehensive trace files that can be viewed in Perfetto and TensorBoard. Key findings show the model execution is TPU-efficient, with most time spent in expected operations.

## Trace Files

Traces are stored in GCS (too large for git):

```bash
gsutil cp -r gs://sglang-jax-profiling-results/2026-02-05-tinyllama-tpu-v6e/ traces/
```

| File | Size | Purpose |
|------|------|---------|
| `t1v-n-c2bbebb4-w-0.trace.json` | 64 MB | Perfetto trace (decompressed) |
| `t1v-n-c2bbebb4-w-0.trace.json.gz` | 4.9 MB | Perfetto trace (compressed) |
| `t1v-n-c2bbebb4-w-0.xplane.pb` | 18 MB | TensorBoard XPlane format |

## Trace Analysis Results

### Overview

| Metric | Value |
|--------|-------|
| Total events | 283,235 |
| Unique operations | 2,322 |
| Total traced time | 27,401 ms |
| Event phases | X (complete): 283,201, M (metadata): 33 |

### Top Operations by Total Time

| Operation | Total (ms) | Count | Avg (ms) | Max (ms) |
|-----------|------------|-------|----------|----------|
| queue.py:154 get | 2,923.78 | 88 | 33.22 | 185.35 |
| threading.py:323 wait | 2,923.08 | 60 | 48.72 | 185.35 |
| acquire (lock) | 2,530.69 | 461 | 5.49 | 99.63 |
| profiler.py:356 wrapper | 1,298.95 | 145 | 8.96 | 47.22 |
| scheduler.py:1394 process_batch_result | 1,275.17 | 30 | 42.51 | 47.57 |
| jit_jitted_run_model | 1,254.46 | 27 | 46.46 | 48.82 |
| **broadcast_select_fusion** | **1,054.92** | **28** | **37.68** | **39.02** |

### TPU/XLA Operations

| Operation | Total (ms) | Count | Notes |
|-----------|------------|-------|-------|
| broadcast_select_fusion | 1,054.92 | 28 | Main XLA fusion kernel |
| broadcast_select_fusion.1 | 231.93 | 28 | Secondary fusion |
| TpuLoadedExecutable::Execute | 34.31 | 232 | TPU execution dispatch |
| DeferredTpuAllocator::Allocate | 10.23 | 2,848 | Memory allocation |
| TpuClient::LinearizeIntoImpl | 4.68 | 702 | Data linearization |

### Observations

1. **Queue/Threading Overhead**: Significant time (~6 seconds) spent in queue waiting and thread synchronization. This is expected in a multi-process architecture (TokenizerManager + Scheduler).

2. **Model Execution**: `jit_jitted_run_model` accounts for ~1.25 seconds total across 27 invocations, averaging 46ms per batch.

3. **XLA Fusion**: The `broadcast_select_fusion` kernel is the dominant TPU operation, indicating effective XLA compilation.

4. **Memory Allocation**: 2,848 allocation events totaling only 10ms suggests efficient memory management.

## How to View Traces

### Perfetto (Recommended)

1. Open https://ui.perfetto.dev
2. Drag and drop `traces/t1v-n-c2bbebb4-w-0.trace.json.gz`
3. Keyboard shortcuts:
   - `W/S`: Zoom in/out
   - `A/D`: Pan left/right
   - `F`: Fit to view

### TensorBoard

```bash
pip install tensorboard xprof
tensorboard --logdir traces/ --port 6006
# Open http://localhost:6006 > Profile tab
```

### Download from GCS

```bash
gsutil cp -r gs://sglang-jax-profiling-results/2026-02-05-tinyllama-tpu-v6e/ ./
```

## Reproduction

### Setup Commands

```bash
# 1. Create TPU VM
gcloud compute tpus tpu-vm create sglang-profiling \
  --zone=us-east5-b \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e

# 2. Install Python 3.12 (required for sglang-jax)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# 3. Clone and install sglang-jax
git clone https://github.com/alexshires/sglang-jax.git ~/sglang-jax
cd ~/sglang-jax/python
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[tpu]'
.venv/bin/pip install tensorboard xprof

# 4. Start server
.venv/bin/python -m sgl_jax.launch_server \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 30000
```

### Profiling Workflow

```bash
# Start profiling
curl -X POST 'http://localhost:30000/start_profile' \
  -H 'Content-Type: application/json' \
  -d '{"output_dir": "/tmp/profile", "num_steps": 30, "host_tracer_level": 2}'

# Send requests (15 generate requests)
for i in $(seq 1 15); do
    curl -X POST 'http://localhost:30000/generate' \
      -H 'Content-Type: application/json' \
      -d '{"text": "Hello, how are you?", "sampling_params": {"max_new_tokens": 20}}'
done

# Stop profiling
curl -X POST 'http://localhost:30000/stop_profile'

# Download traces
gcloud compute tpus tpu-vm scp sglang-profiling:/tmp/profile/plugins/profile/*/* ./traces/ --zone=us-east5-b
```

### Cleanup

```bash
# Delete TPU VM to stop charges
gcloud compute tpus tpu-vm delete sglang-profiling --zone=us-east5-b --quiet
```

## Cost Summary

| Resource | Duration | Cost |
|----------|----------|------|
| TPU v6e-1 (on-demand) | ~45 minutes | ~$1.43 |
| GCS storage | 87 MB | ~$0.002/month |

## Related Documentation

- [RFC-011: Profiling Design](../../rfcs/011-profiling-design.md)
- [Runbook: Profiling Infrastructure Setup](../../runbooks/profiling-infrastructure-setup.md)
