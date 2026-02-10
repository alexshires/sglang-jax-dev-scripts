# sGLANG Benchmarking Suite

This directory contains tools for benchmarking the sGLANG Scoring API (`/v1/score`) and Generation API on various hardware accelerators (TPU v4/v5/v6, Nvidia GPUs).

## Files

- **benchmark-job.yaml**: A parameterized Kubernetes Job template. **Do not apply this file directly**; use `run_benchmark.sh` to render it with the correct values.
- **run_benchmark.sh**: A helper script to generate a valid Job manifest from the template and submit it to the cluster.

## Usage

### Prerequisites
- `kubectl` configured with access to your GKE cluster.
- Supporting Docker image (default: `europe-docker.pkg.dev/ashires-e7aaot/container/vllm-tpu:pr`).

### 1. Running on TPU (Default)
To benchmark on a TPU v6e slice:

```bash
./run_benchmark.sh --accelerator tpu-v6e-slice --topology 1x1
```

### 2. Running on GPU
To benchmark on Nvidia GPUs (e.g., L4, A100):

```bash
# Example: 1x Nvidia L4
./run_benchmark.sh --accelerator nvidia-tesla-l4 --count 1 --image <your-gpu-image>
```

### 3. Customizing the Benchmark
The actual benchmark logic is in `sglang-jax/python/sgl_jax/bench_score.py`. You can adjust running parameters (requests, concurrency) by editing the `command` section in `benchmark-job.yaml` or creating a custom image.

## Output & Metrics

The benchmark job prints metrics to stdout. Retrieve them using `kubectl logs`:

```bash
# Get the job name from the run_benchmark.sh output
kubectl logs -f job/sglang-benchmark-xxxxx -n eval-serving
```

**Key Metrics:**
- **RPS (Requests per Second)**: Total scoring requests handled per second.
- **IPS (Items per Second)**: Total candidate items scored per second.
- **Latency (ms)**: P50, P90, P99, and Mean latency.
