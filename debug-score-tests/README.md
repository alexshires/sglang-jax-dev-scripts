# SGLang-JAX Debug Score Testing

This directory contains the infrastructure for debugging and verifying the SGLang-JAX `/v1/score` API on Google Kubernetes Engine (GKE) with TPUs.

## Recent Improvements (2026-02-10)

The following enhancements were made to improve developer velocity and test reliability:

1.  **Robust Build Process**: The `Dockerfile` now performs a clean, multi-stage-style build. It clones the repository, switches to the feature branch (`feat/multi-item-scoring`), and installs all dependencies in a single optimized layer.
2.  **Efficient Caching**: `cloudbuild.yaml` is configured to pull the previous `latest` image to use as a cache (`--cache-from`), significantly reducing build times for subsequent changes.
3.  **Standardized Dependencies**: A `requirements.txt` file now manages all additional test packages (e.g., `pytest`, `openai`, `torch`, `transformers`), ensuring consistent environments across local and remote runs.
4.  **Integrated Multi-Item Testing**: The test suite now includes dedicated multi-item scoring regression and correctness tests, accessible via the `run_tests.sh` script.
5.  **Resource Alignment**: Kubernetes manifests (`debug-tpu-pod.yaml`) are aligned with TPU v6e node capacities (40 CPU, 150Gi RAM) to prevent OOMs during JAX compilation.
6.  **Offline Model Support**: The debug environment is configured to use models pre-cached in GCS (mounted at `/data`), avoiding expensive downloads during test execution.

---

## Workflow

### 1. Build and Push Image
Submit a build to Google Cloud Build from this directory. This will clone the latest code and push a new image to GCR.
```bash
gcloud builds submit --config cloudbuild.yaml .
```

### 2. Launch Debug Pod
Delete the existing pod and launch a fresh one with the new image.
```bash
kubectl delete pod -n eval-serving debug-tpu-sglang-score --ignore-not-found
kubectl apply -f debug-tpu-pod.yaml
kubectl wait --for=condition=Ready pod/debug-tpu-sglang-score -n eval-serving --timeout=300s
```

### 3. Run Tests
Execute the integrated test suite on the running pod.
```bash
kubectl exec -n eval-serving debug-tpu-sglang-score -c inference-server -- /app/run_tests.sh
```

---

## Components

- **`Dockerfile`**: Self-contained build for the JAX server and test runner.
- **`cloudbuild.yaml`**: Automates image creation with Docker layer caching.
- **`run_tests.sh`**: Entry point for executing the multi-item scoring test suite.
- **`requirements.txt`**: Additional Python packages required for testing.
- **`debug-tpu-pod.yaml`**: Kubernetes manifest for a TPU-enabled debug environment.
- **`test-runner-job.yaml`**: Template for running tests as a Kubernetes Job.
