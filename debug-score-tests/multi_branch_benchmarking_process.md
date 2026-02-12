# Multi-Branch Multi-Chip Benchmarking Process

This document outlines the workflow for benchmarking multi-item scoring performance across different git branches (e.g., `feat/multi-item-scoring`, `aashishrampal/batching`) on various TPU configurations (1x1, 2x2, 2x4).

## Overview

Instead of merging benchmark scripts into every feature branch, we maintain a canonical benchmark script in the `debug-score-tests` directory. We deploy debug pods using branch-specific container images and inject the benchmark script at runtime.

## Prerequisites

1.  **Container Images**: Ensure a Docker image for your target branch is built and pushed to Artifact Registry.
    *   Registry: `europe-docker.pkg.dev/ashires-e7aaot/container/sglang-jax`
    *   Tag: typically the branch name (e.g., `feat-multi-item-scoring`, `aashishrampal-batching`)

2.  **Benchmark Script**: The master copy is `sglang-scripts/debug-score-tests/test_bench_multi_item_score_tpu.py`.

## Workflow

### 1. Configure Pod Manifest
Update the `image` field in the relevant pod manifest (e.g., `debug-tpu-pod-2x2.yaml`) to point to your target branch's image.

```yaml
    image: europe-docker.pkg.dev/ashires-e7aaot/container/sglang-jax:aashishrampal-batching
```

### 2. Deploy Pod
Use the `restart_pod.sh` helper to launch the pod with the new image.

```bash
cd sglang-scripts/debug-score-tests
./restart_pod.sh debug-tpu-sglang-score-2x2 debug-tpu-pod-2x2.yaml
```

### 3. Run Benchmark
Use the `run_branch_benchmark.sh` script (created below) to sync the local benchmark file and execute it on the remote pod.

```bash
./run_branch_benchmark.sh debug-tpu-sglang-score-2x2 4
```

## Scripts

- **`restart_pod.sh`**: Deletes and re-applies the pod manifest, waiting for readiness.
- **`run_branch_benchmark.sh`**: Copies `test_bench_multi_item_score_tpu.py` to the pod and runs it with `python3 -m unittest`.
