# Runbook: Profiling Infrastructure Setup

| | |
|------------|------|
| **Last Updated** | 2026-02-05 |
| **Maintainer** | Engineering Team |
| **Related** | [RFC-011: Profiling Design](../rfcs/011-profiling-design.md), [RFC-009: ARC Runner Setup](../rfcs/009-arc-runner-setup.md) |

## Overview

This runbook covers how to set up infrastructure for profiling sglang-jax workloads. Choose the option that fits your needs:

| Option | Best For | Cost | Setup Time |
|--------|----------|------|------------|
| [TPU VM](#option-1-tpu-vm-simplest) | Quick profiling, one-off analysis | $0.64/hr | 5 min |
| [GKE + TPU](#option-2-gke-cluster-with-tpu) | Team use, CI integration, persistent | $0.64/hr + cluster | 30 min |
| [GKE + GPU](#option-3-gke-cluster-with-gpu) | PyTorch comparison, CUDA profiling | Varies by GPU | 30 min |
| [Local CPU](#option-4-local-cpu-development) | Development, no TPU access | Free | 2 min |

---

## Prerequisites (All Options)

### Required Tools

```bash
# gcloud CLI
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# kubectl (for GKE options)
gcloud components install kubectl

# Visualization tools (local machine)
pip install tensorboard xprof
```

### GCP Quotas

Check your TPU quota:
```bash
gcloud compute tpus tpu-vm list --zone=us-east5-b
# If you get quota errors, request quota increase:
# Console > IAM & Admin > Quotas > Search "TPU v6e"
```

---

## Option 1: TPU VM (Simplest)

Best for quick, one-off profiling sessions.

### Step 1: Create TPU VM

```bash
export TPU_NAME="profiling-$(date +%s)"
export ZONE="us-east5-b"
export TPU_TYPE="v6e-1"

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$TPU_TYPE \
  --version=v2-alpha-tpuv6e \
  --preemptible
```

### Step 2: Setup Environment

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command="
  # Clone and install
  git clone https://github.com/sgl-project/sglang-jax.git ~/sglang-jax
  cd ~/sglang-jax
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e '.[dev]'
  pip install tensorboard xprof

  # Pre-download model (optional, speeds up profiling)
  python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')\"
"
```

### Step 3: Start Server and Profile

```bash
# SSH into TPU
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE

# On TPU VM:
cd ~/sglang-jax && source .venv/bin/activate

# Start server in background
python -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --port 30000 &

# Wait for server to be ready
sleep 60

# Start profiling
curl -X POST 'http://localhost:30000/start_profile' \
  -H 'Content-Type: application/json' \
  -d '{
    "output_dir": "/tmp/profile",
    "num_steps": 10,
    "host_tracer_level": 2
  }'

# Send score requests
for i in {1..10}; do
  curl -X POST 'http://localhost:30000/v1/score' \
    -H 'Content-Type: application/json' \
    -d '{
      "query": "Is this a positive review?",
      "items": ["Great product!", "Terrible experience"],
      "label_token_ids": [9454, 2753]
    }'
done

# Stop profiling
curl -X POST 'http://localhost:30000/stop_profile'
```

### Step 4: Download and View Traces

```bash
# From local machine:
gcloud compute tpus tpu-vm scp \
  $TPU_NAME:/tmp/profile/plugins/profile/*/*.trace.json.gz \
  ./traces/ --zone=$ZONE --recurse

# View in Perfetto (recommended)
# Open https://ui.perfetto.dev and drag-drop the .trace.json.gz file

# Or view in TensorBoard
tensorboard --logdir ./traces --port 6006
# Open http://localhost:6006 > Profile tab
```

### Step 5: Cleanup

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet
```

### Cost

| Duration | Preemptible | On-Demand |
|----------|-------------|-----------|
| 15 min | $0.16 | $0.48 |
| 1 hour | $0.64 | $1.90 |

---

## Option 2: GKE Cluster with TPU

Best for team use, CI integration, or persistent profiling infrastructure.

### Step 1: Create/Use GKE Cluster

**If you have an existing cluster:**
```bash
export CLUSTER_NAME="your-cluster"
export ZONE="us-east5-b"
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
```

**If you need a new cluster:**
```bash
export CLUSTER_NAME="sglang-profiling"
export ZONE="us-east5-b"

# Create cluster with TPU node pool
gcloud container clusters create $CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type=n2-standard-8 \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=3

# Add TPU node pool
gcloud container node-pools create tpu-v6e-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type=ct6e-standard-1t \
  --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=4 \
  --spot
```

### Step 2: Create Profiling Job

Create `profiling-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sglang-profiling
  namespace: default
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: 1x1
      containers:
      - name: profiler
        image: us-docker.pkg.dev/cloud-tpu-images/inference/sax-jax:v1.3.0
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -ex

          # Install sglang-jax
          pip install git+https://github.com/sgl-project/sglang-jax.git
          pip install tensorboard xprof

          # Start server
          python -m sgl_jax.launch_server \
            --model-path meta-llama/Llama-3.2-1B-Instruct \
            --port 30000 &

          # Wait for server
          sleep 120

          # Profile
          curl -X POST 'http://localhost:30000/start_profile' \
            -H 'Content-Type: application/json' \
            -d '{"output_dir": "/tmp/profile", "num_steps": 20}'

          # Send requests
          for i in $(seq 1 20); do
            curl -X POST 'http://localhost:30000/v1/score' \
              -H 'Content-Type: application/json' \
              -d '{"query": "Test", "items": ["A", "B"], "label_token_ids": [9454, 2753]}'
          done

          # Stop and save
          curl -X POST 'http://localhost:30000/stop_profile'

          # Copy to GCS (optional)
          # gsutil -m cp -r /tmp/profile gs://YOUR_BUCKET/profiles/$(date +%Y%m%d-%H%M%S)/

          # Keep pod alive for log collection
          sleep 300
        resources:
          limits:
            google.com/tpu: 1
        env:
        - name: JAX_PLATFORMS
          value: "tpu"
        - name: SGLANG_JAX_PROFILER_DIR
          value: "/tmp/profile"
```

### Step 3: Run Profiling Job

```bash
kubectl apply -f profiling-job.yaml

# Watch progress
kubectl logs -f job/sglang-profiling

# Get pod name for file extraction
POD=$(kubectl get pods -l job-name=sglang-profiling -o jsonpath='{.items[0].metadata.name}')

# Copy traces locally
kubectl cp $POD:/tmp/profile ./traces

# Cleanup
kubectl delete job sglang-profiling
```

### Step 4: View Traces

```bash
# Perfetto (recommended)
# Open https://ui.perfetto.dev, drag-drop traces/*.trace.json.gz

# TensorBoard
tensorboard --logdir ./traces --port 6006
```

### Cost

| Component | Cost |
|-----------|------|
| GKE cluster (n2-standard-8) | $0.30/hr |
| TPU v6e-1 (spot) | $0.19/hr |
| **Total per hour** | **~$0.50/hr** |

---

## Option 3: GKE Cluster with GPU

Best for PyTorch comparison or CUDA profiling.

### Step 1: Add GPU Node Pool

```bash
export CLUSTER_NAME="your-cluster"
export ZONE="us-central1-a"

# Add GPU node pool
gcloud container node-pools create gpu-l4-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=2 \
  --spot
```

### Step 2: Create GPU Profiling Job

Create `gpu-profiling-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sglang-gpu-profiling
spec:
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      containers:
      - name: profiler
        image: nvcr.io/nvidia/pytorch:24.01-py3
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -ex

          # Install PyTorch sglang
          pip install sglang[all]

          # Start server with profiling
          python -m sglang.launch_server \
            --model-path meta-llama/Llama-3.2-1B-Instruct \
            --port 30000 &

          sleep 120

          # Profile using PyTorch profiler
          curl -X POST 'http://localhost:30000/start_profile' \
            -H 'Content-Type: application/json' \
            -d '{"output_dir": "/tmp/profile"}'

          # Send requests
          for i in $(seq 1 20); do
            curl -X POST 'http://localhost:30000/v1/score' \
              -H 'Content-Type: application/json' \
              -d '{"query": "Test", "items": ["A", "B"], "label_token_ids": [9454, 2753]}'
          done

          curl -X POST 'http://localhost:30000/stop_profile'

          sleep 300
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Step 3: Run and Collect

```bash
kubectl apply -f gpu-profiling-job.yaml
kubectl logs -f job/sglang-gpu-profiling

# Copy traces
POD=$(kubectl get pods -l job-name=sglang-gpu-profiling -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/tmp/profile ./gpu-traces
```

### Cost

| GPU Type | Spot Price | On-Demand |
|----------|------------|-----------|
| L4 | $0.22/hr | $0.70/hr |
| A100 40GB | $1.10/hr | $3.67/hr |

---

## Option 4: Local CPU (Development)

Best for development and debugging profiling code. **Not suitable for accurate performance analysis.**

### Step 1: Setup

```bash
cd /path/to/sglang-jax
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install tensorboard xprof

# Force CPU backend
export JAX_PLATFORMS=cpu
export SGLANG_USE_CPU=1
```

### Step 2: Run with Profiling

```bash
# Start server (will be slow on CPU)
python -m sgl_jax.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --port 30000 &

# Wait longer for CPU
sleep 300

# Profile
curl -X POST 'http://localhost:30000/start_profile' \
  -d '{"output_dir": "/tmp/profile", "num_steps": 5}'

curl -X POST 'http://localhost:30000/v1/score' \
  -d '{"query": "Test", "items": ["A"], "label_token_ids": [9454]}'

curl -X POST 'http://localhost:30000/stop_profile'
```

### Limitations

- 10-100x slower than TPU
- Different execution patterns (no MXU, different memory hierarchy)
- Traces won't reflect production behavior
- Use only for debugging profiling setup, not performance analysis

---

## Viewing and Analyzing Traces

### Perfetto (Recommended)

1. Open https://ui.perfetto.dev
2. Drag and drop `.trace.json.gz` file
3. Use keyboard shortcuts:
   - `W/S`: Zoom in/out
   - `A/D`: Pan left/right
   - `F`: Fit to view

### TensorBoard + XProf

```bash
# Install
pip install tensorboard xprof

# Launch
tensorboard --logdir /path/to/traces --port 6006

# Open http://localhost:6006
# Navigate to Profile > trace_viewer
```

### Key Metrics to Look For

| Metric | Where to Find | What to Look For |
|--------|---------------|------------------|
| Kernel time | Perfetto timeline | Long bars = bottlenecks |
| MXU utilization | XProf > Overview | Should be >50% for compute-bound |
| Memory transfers | Perfetto HBM events | Minimize host<->device transfers |
| Step time | XProf > Step-time | Consistent across steps |

---

## Troubleshooting

### "TPU is already in use"

```bash
# Check for orphaned VMs
gcloud compute tpus tpu-vm list --zone=us-east5-b

# Delete orphans
gcloud compute tpus tpu-vm delete <name> --zone=us-east5-b
```

### "No trace files generated"

1. Check profiling started: `curl http://localhost:30000/get_server_info`
2. Ensure `num_steps` is set and requests were sent during profiling
3. Check output directory exists and is writable
4. Look for errors in server logs

### "TensorBoard doesn't show Profile tab"

```bash
# Install correct plugin
pip install tensorboard xprof

# Or for older setups
pip install tensorboard tensorboard-plugin-profile
```

### GKE pod stuck in Pending

```bash
# Check node pool autoscaling
kubectl describe pod <pod-name>

# Look for:
# - Insufficient TPU quota
# - Node pool at max capacity
# - Spot instance unavailable
```

---

## Cost Summary

| Option | Setup Cost | Per-Hour Cost | Best For |
|--------|------------|---------------|----------|
| TPU VM (preemptible) | $0 | $0.64 | Quick one-off profiling |
| TPU VM (on-demand) | $0 | $1.90 | Guaranteed availability |
| GKE + TPU (spot) | ~$0.30/hr base | $0.50 total | Team/CI use |
| GKE + GPU L4 (spot) | ~$0.30/hr base | $0.52 total | PyTorch comparison |
| Local CPU | $0 | $0 | Development only |

---

## Related Documentation

- [RFC-011: Profiling Design](../rfcs/011-profiling-design.md) - Profiling framework design
- [RFC-009: ARC Runner Setup](../rfcs/009-arc-runner-setup.md) - Self-hosted GKE runners
- [Running Performance Benchmarks](running-performance-benchmarks.md) - Benchmark guides
- [Debugging TPU Test Failures](debugging-tpu-test-failures.md) - Troubleshooting
