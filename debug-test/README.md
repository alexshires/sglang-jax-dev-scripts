# sGLANG Scripts & Manifests

This directory contains Kubernetes manifests, Dockerfiles, and utility scripts for deploying, testing, and debugging sGLANG (JAX/Pytorch) on Google Kubernetes Engine (GKE), specifically targeting TPU environments.

## Directory Structure

- **benchmark/**: Contains scripts and manifests for benchmarking sGLANG performance (see [benchmark/README.md](benchmark/README.md)).
- **cloudbuild.yaml**: Google Cloud Build configuration for building the sGLANG JAX Docker image.
- **debug-tpu-pod.yaml**: A standalone Pod manifest for debugging TPU connectivity and environment issues.
- **Dockerfile**: Defines the sGLANG JAX container image.
- **test-runner-job.yaml**: A Kubernetes Job that runs the sGLANG JAX test suite on a remote node.

---

## File Descriptions & Usage

### 1. `test-runner-job.yaml`
A Kubernetes Job designed to run the sGLANG JAX test suite (including specific new features like the Scoring API) on a remote TPU node.

**Usage:**
```bash
# Create a new test run (creates a unique job name each time)
kubectl create -f test-runner-job.yaml -n eval-serving

# View logs
kubectl logs -f job/<job-name> -n eval-serving
```

**Key Configuration:**
- `git checkout fix/score-api-missing-logprobs`: Specifies the branch to test. Update this in the YAML if needed.
- `backoffLimit: 3`: Retries the pod up to 3 times if it fails (e.g., due to node preemption).

### 2. `debug-tpu-pod.yaml`
A simple Pod that keeps a TPU node active ('sleep infinity') to allow interactive debugging via `kubectl exec`.

**Usage:**
```bash
# Deploy the debug pod
kubectl apply -f debug-tpu-pod.yaml -n eval-serving

# Access the pod
kubectl exec -it debug-tpu-sglang-score -n eval-serving -- /bin/bash
```

### 3. `Dockerfile`
Multi-stage Dockerfile for building the sGLANG JAX environment.

**Usage:**
Usually built via Cloud Build, but can be built locally:
```bash
docker build -t sglang-jax:latest -f Dockerfile ..
```

### 4. `cloudbuild.yaml`
Google Cloud Build config to automate building and pushing the Docker image to Google Container Registry (GCR).

**Usage:**
```bash
gcloud builds submit --config cloudbuild.yaml .
```
