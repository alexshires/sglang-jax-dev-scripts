# SGLang-JAX Debug Score Testing

This directory contains the infrastructure for debugging and verifying the SGLang-JAX `/v1/score` API on Google Kubernetes Engine (GKE) with TPUs and GPUs.

## Recent Improvements (2026-02-12)

- **Comprehensive Benchmarking**: Support for multi-chip TPU (2x2, 2x4) and NVIDIA GPU (L4, RTX 6000) performance testing.
- **Unified Scripts**: Helper scripts (`restart_pod.sh`, `run_remote_tests.sh`, `run_gpu_tests.sh`) to automate pod lifecycle and test execution.
- **Robust Build Process**: `Dockerfile` and `cloudbuild.yaml` optimized for reliability and caching, using granular steps and offline model support.
- **Standardized Environment**: All pods mount GCS models at `/models`, align with `sglang` defaults, and configure `HF_HUB_OFFLINE=1`.

---

## Workflow

### 1. Build and Push Image (TPU)
Submit a build to Google Cloud Build. This creates the image used by all TPU debug pods.
```bash
gcloud builds submit --config cloudbuild.yaml .
```

### 2. TPU Benchmarks

Use the helper scripts to deploy the pod and run the multi-item scoring benchmark suite.

#### Single Chip (1x1 / TP=1)
```bash
./restart_pod.sh debug-tpu-sglang-score debug-tpu-pod.yaml
./run_remote_tests.sh debug-tpu-sglang-score 1
```

#### Multi-Chip (2x2 / TP=4)
```bash
./restart_pod.sh debug-tpu-sglang-score-2x2 debug-tpu-pod-2x2.yaml
./run_remote_tests.sh debug-tpu-sglang-score-2x2 4
```

#### Multi-Chip (2x4 / TP=8)
```bash
./restart_pod.sh debug-tpu-sglang-score-2x4 debug-tpu-pod-2x4.yaml
./run_remote_tests.sh debug-tpu-sglang-score-2x4 8
```

### 3. GPU Benchmarks (PyTorch Baseline)

These tests use the official `lmsysorg/sglang:latest` image.

#### NVIDIA L4 (G2)
```bash
kubectl apply -f debug-gpu-l4-pod.yaml
kubectl wait --for=condition=Ready pod/debug-gpu-l4-sglang -n eval-serving --timeout=600s
./run_gpu_tests.sh debug-gpu-l4-sglang 1
```

#### NVIDIA RTX 6000 (G4)
```bash
kubectl apply -f debug-gpu-g4-pod.yaml
kubectl wait --for=condition=Ready pod/debug-gpu-g4-sglang -n eval-serving --timeout=600s
./run_gpu_tests.sh debug-gpu-g4-sglang 1
```

---

## Components

- **`Dockerfile`**: Self-contained build for the JAX server and test runner.
- **`cloudbuild.yaml`**: Automates image creation with Docker layer caching.
- **`run_tests.sh`**: Entry point for executing the multi-item scoring test suite inside the container.
- **`run_remote_tests.sh`**: Orchestrates syncing and running tests on a remote TPU pod.
- **`run_gpu_tests.sh`**: Orchestrates syncing and running benchmark scripts on a remote GPU pod.
- **`restart_pod.sh`**: Helper to delete, apply, and wait for a pod.
- **`debug-*-pod.yaml`**: Kubernetes manifests for various accelerator configurations.
- **`test_bench_multi_item_score.py`**: JAX-based benchmark script.
- **`test_bench_multi_item_score_gpu.py`**: PyTorch-based benchmark script for comparison.
