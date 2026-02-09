# RFC-010: Profiling Score API Performance on TPU

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-05 |
| **Updated** | 2026-02-05 |
| **Related** | [RFC-002](002-cicd-tpu-testing.md) |

## Summary

This RFC establishes a repeatable workflow for profiling the `sglang-jax` Score API using `py-spy` on TPU v6e infrastructure. It enables deep performance analysis of the server during mixed-workload execution.

## Infrastructure

### Profiling Container
A custom Docker image (`Dockerfile.profiling`) is used:
- Base: `python:3.12`
- Tools: `py-spy`, `requests`
- Code: Clones specific branch of `sglang-jax`

### Kubernetes Job
A K8s Job (`score-profiling-job.yaml`) orchestrates the run:
- Requests 1 TPU v6e chip (spot instance).
- Mounts GCS models via GCS FUSE.
- Requires `SYS_PTRACE` capability for `py-spy`.

## Workflow

### 1. Build Profiling Image

```bash
# Set your branch
BRANCH="serving-fixes"

docker build -t gcr.io/ashires-e7aaot/sglang-jax-profiler:latest 
  -f docker/Dockerfile.profiling 
  --build-arg BRANCH_NAME=$BRANCH 
  .

docker push gcr.io/ashires-e7aaot/sglang-jax-profiler:latest
```

### 2. Run Profiling Job

```bash
kubectl apply -f k8s/score-profiling-job.yaml
```

### 3. Retrieve Results

The job saves profiles to `/outputs`. Since this example uses ephemeral storage, you should either:
- **Stream logs:** `kubectl logs -f job/score-api-profiling-job`
- **Copy files:** `kubectl cp <pod-name>:/outputs/score_api_profile.svg ./local_profile.svg`

## Profiling Strategy

The script `scripts/profile_score_api.py` performs:
1. **Server Launch:** Starts `sglang_jax.launch_server` on TPU.
2. **Health Check:** Waits for `/health`.
3. **Profiler Attachment:** Attaches `py-spy` with `--subprocesses` to capture the multi-process server architecture.
4. **Workload Generation:**
   - Single item requests
   - Small batch requests
   - Large batch requests
5. **Teardown:** Terminates server and saves profile.

## Usage

Use this setup to:
- Identify bottlenecks in the Score API (Python vs JAX dispatch).
- Verify performance improvements from the `serving-fixes` branch.
- Analyze CPU overhead of the tokenizer/scheduler.
