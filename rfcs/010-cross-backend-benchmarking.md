# RFC-010: Cross-Backend Benchmarking Infrastructure (PyTorch GPU vs JAX TPU)

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-04 |
| **Updated** | 2026-02-04 |
| **Related** | [RFC-004](004-score-api-performance-benchmarks.md), [RFC-009](009-arc-runner-setup.md), [Investigation: v1 Assessment](../investigations/v1-infrastructure-assessment.md) |

## Summary

Propose a unified benchmarking infrastructure to run comparable Score API performance benchmarks across PyTorch/GPU and JAX/TPU backends on the existing GKE cluster. This enables data-driven performance comparison between the two implementations.

## Motivation

### Current State

The `v1/` infrastructure directory contains K8s job templates for JAX/TPU benchmarking, but they are:
- Hardcoded to a single developer's environment (fork, branch, image, GCS bucket)
- 100% JAX/TPU focused — no PyTorch/GPU support
- Referencing a `bench_score.py` script that does not exist in the JAX codebase

PyTorch has a separate benchmark at `sglang/benchmark/prefill_only/bench_score.py`, but it uses hardcoded configuration (no CLI args) and produces incompatible output.

See [Investigation: v1 Infrastructure Assessment](../investigations/v1-infrastructure-assessment.md) for the full gap analysis.

### Problems

1. **No comparable benchmarks** — Cannot produce an apples-to-apples comparison between PyTorch/GPU and JAX/TPU
2. **No parameterization** — Templates require manual editing to change repos, branches, images, or benchmark parameters
3. **No Score API benchmark for JAX** — `bench_score.py` does not exist in sglang-jax
4. **No unified output** — The two backends produce different output formats with different metrics
5. **No results persistence** — Results go to `kubectl logs` and are lost

### Goals

1. Run comparable Score API benchmarks on both PyTorch (GPU) and JAX (TPU) with the same model, prompts, batch sizes, and metrics
2. Parameterized K8s templates that work across environments
3. Unified results format enabling side-by-side comparison
4. Persistent results storage for historical tracking
5. Cost-efficient execution using spot/preemptible instances

## Proposed Solution

### Approach: K8s Jobs on Existing GKE Cluster

Use Kubernetes Jobs on the existing GKE cluster with Kustomize overlays for backend-specific configuration.

**Why K8s Jobs:**
- GKE cluster already exists with TPU node pools (per RFC-009)
- K8s provides reproducible execution, resource management, and log collection
- Job templates are a natural extension of existing v1/ infrastructure

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Benchmark Runner (local)                 │
│                                                          │
│  $ python scripts/bench/compare.py                       │
│    --model meta-llama/Llama-3.2-1B-Instruct              │
│    --profile standard                                    │
│    --runs 3                                              │
│                                                          │
│  1. Renders K8s jobs (Kustomize)                         │
│  2. Submits TPU job + GPU job in parallel                 │
│  3. Waits for completion                                 │
│  4. Collects results from pod logs                       │
│  5. Normalizes to common schema                          │
│  6. Produces comparison report                           │
│  7. Uploads to GCS                                       │
└───────────────┬───────────────────────┬──────────────────┘
                │                       │
        ┌───────▼───────┐       ┌───────▼───────┐
        │   TPU Job     │       │   GPU Job     │
        │               │       │               │
        │ JAX Server    │       │ PyTorch Server│
        │ bench_score   │       │ bench_score   │
        │               │       │               │
        │ TPU v6e-1     │       │ A100 / L4     │
        │ (preemptible) │       │ (spot)        │
        └───────────────┘       └───────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │    GCS Bucket         │
                │  results/             │
                │    2026-02-04/        │
                │      jax-tpu.json     │
                │      pytorch-gpu.json │
                │      comparison.md    │
                └───────────────────────┘
```

### Template Parameterization: Kustomize

Replace hardcoded values with Kustomize overlays:

```
benchmark/
├── base/
│   ├── kustomization.yaml
│   ├── benchmark-job.yaml      # Base job template with placeholders
│   └── common-env.yaml         # Shared env vars (model, benchmark params)
├── overlays/
│   ├── jax-tpu/
│   │   ├── kustomization.yaml  # TPU-specific patches
│   │   ├── resources.yaml      # TPU resource requests
│   │   └── server-args.yaml    # JAX server launch args
│   └── pytorch-gpu/
│       ├── kustomization.yaml  # GPU-specific patches
│       ├── resources.yaml      # GPU resource requests
│       └── server-args.yaml    # PyTorch server launch args
└── profiles/
    ├── smoke.yaml              # Quick validation (2-3 min)
    ├── standard.yaml           # Regular benchmark (10-15 min)
    └── full.yaml               # Comprehensive (45-60 min)
```

### Docker Images

Two images, one per backend:

| Backend | Base Image | Install | Entrypoint |
|---------|-----------|---------|------------|
| JAX/TPU | `python:3.12` | `pip install -e .[tpu]` | `python -m sgl_jax.launch_server` |
| PyTorch/GPU | `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` | `pip install sglang[all]` | `python -m sglang.launch_server` |

The JAX Dockerfile exists at `sglang-jax/Dockerfile`. The PyTorch Dockerfile exists at `sglang/docker/Dockerfile`.

### Input Parity Specification

For a fair comparison, both backends **must** use identical input construction. Without this, results reflect config differences rather than performance.

#### Locked Parameters (must be identical across backends)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Same model and revision on both sides | Eliminates model differences |
| Chat template | Disabled (`--chat-template none` or raw text) | Chat templates may differ between backends |
| Tokenizer | Same HuggingFace tokenizer (from model) | Ensures identical token sequences |
| Prompt construction | Raw token IDs, not text (avoids template divergence) | Byte-exact parity |
| `apply_softmax` | `true` on both | Score post-processing must match |
| `item_first` | `false` on both (or match PyTorch default) | Item ordering affects scoring |
| `dtype` | `bfloat16` on both | Numerical precision parity |
| `max_new_tokens` | `0` on both | Prefill-only scoring (no generation) |
| Label token IDs | Same IDs (e.g., `[9454, 2753]` for Yes/No) | Scoring target parity |
| BOS/EOS tokens | Explicitly controlled (include or exclude on both) | Tokenizer config can differ |

#### Server Args Parity

Server launch args that affect scoring performance must be documented and controlled:

| Arg | JAX Default | PyTorch Default | Benchmark Value |
|-----|-------------|-----------------|-----------------|
| `--tp-size` | 1 (v6e-1) | 1 (single GPU) | 1 on both |
| `--mem-fraction-static` | 0.8 | 0.88 | 0.8 on both |
| `--max-prefill-tokens` | 8192 | 16384 | 8192 on both |
| `--chunked-prefill-size` | varies | varies | Same value or disabled on both |
| `--disable-radix-cache` | no | no | Same on both |
| `--max-running-requests` | varies | varies | Same on both |

Any divergence from these locked values must be documented in the results metadata with justification.

### Unified Benchmark Script

A Score API-specific benchmark script is needed for the JAX side. Two approaches:

**Option A: Create `bench_score.py` for JAX** (recommended)
- New script at `sglang-jax/python/sgl_jax/bench_score.py`
- Mirror PyTorch's approach: HTTP POST to `/v1/score`, measure latency/throughput
- Add CLI arguments (unlike PyTorch's hardcoded version)
- Produce JSON output in a common schema

**Option B: Adapt `bench_serving.py`**
- Add a `--score` mode to the existing serving benchmark
- Less work but muddies the script's purpose

### Common Output Schema

Both benchmark scripts should produce JSON output in this format:

```json
{
  "metadata": {
    "backend": "jax-tpu" | "pytorch-gpu",
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "model_revision": "main",
    "hardware": "tpu-v6e-1" | "nvidia-a100",
    "timestamp": "2026-02-04T10:30:00Z",
    "commit": "abc123",
    "profile": "standard",
    "server_args": {
      "tp_size": 1,
      "mem_fraction_static": 0.8,
      "max_prefill_tokens": 8192,
      "dtype": "bfloat16"
    }
  },
  "config": {
    "batch_sizes": [1, 4, 8, 16, 32],
    "items_per_request": [1, 5, 10],
    "query_tokens": 128,
    "item_tokens": 64,
    "num_requests": 1000,
    "warmup_requests_per_shape": 5,
    "runs": 3,
    "apply_softmax": true,
    "item_first": false,
    "label_token_ids": [9454, 2753],
    "max_new_tokens": 0
  },
  "load_generation": {
    "mode": "closed_loop",
    "concurrency": 1,
    "distribution": "constant",
    "target_rps": null
  },
  "results": [
    {
      "batch_size": 8,
      "items_per_request": 5,
      "concurrency": 1,
      "throughput_rps": 30.6,
      "throughput_ips": 153.2,
      "latency_p50_ms": 32.1,
      "latency_p95_ms": 48.7,
      "latency_p99_ms": 62.3,
      "success_rate": 1.0,
      "runs": 3,
      "stddev_throughput_ips": 5.2
    }
  ]
}
```

#### Throughput Definitions

To avoid ambiguity:

| Metric | Definition | Formula |
|--------|-----------|---------|
| `throughput_rps` | Requests per second | `successful_requests / elapsed_seconds` |
| `throughput_ips` | Items per second | `throughput_rps * items_per_request` |

A request with `items_per_request=5` that completes in 100ms contributes 10 RPS and 50 IPS.

#### Load Generation Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `closed_loop` | Fixed concurrency, next request sent when previous completes | Default. Measures max throughput at given parallelism. |
| `open_loop` | Fixed RPS with Poisson arrivals, independent of completion | Measures latency under controlled load. |

Default is **closed-loop with concurrency=1** (serial requests). This is the simplest and most reproducible mode. Open-loop with Poisson distribution can be used for latency-under-load analysis but both backends must use the same distribution and target RPS.

## Benchmark Matrix

### Dimensions

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Model | Llama-3.2-1B-Instruct | Small enough for single TPU/GPU; gated but widely used |
| Batch size | 1, 4, 8, 16, 32 | Range from latency-sensitive to throughput-optimized |
| Items per request | 1, 5, 10 | Single-item baseline to multi-item scoring |
| Query tokens | 128 | Representative prompt length |
| Item tokens | 64 | Representative item length |
| Concurrent requests | 1, 4, 8 | Serial to moderate parallelism |

### Metrics

| Metric | Unit | Purpose |
|--------|------|---------|
| Throughput | items/sec | Raw scoring speed |
| Latency p50 | ms | Typical request latency |
| Latency p95 | ms | Tail latency |
| Latency p99 | ms | Worst-case latency |
| Success rate | % | Reliability |
| Cost efficiency | items/$ | Business metric |

### Cost Efficiency Calculation

```
items_per_dollar = throughput_ips * 3600 / hourly_cost

# Reference costs (preemptible/spot):
# TPU v6e-1: $0.64/hr
# A100 (spot): ~$1.10/hr
# L4 (spot):   ~$0.35/hr
```

### Warmup Protocol

Fair comparison requires accounting for compilation overhead. This is especially critical for JAX where XLA JIT-compiles a new program for each unique input shape.

#### JAX/TPU: Per-Shape Warmup

JAX recompiles for each new `(batch_size, sequence_length)` combination. Since the benchmark matrix varies batch sizes and items per request, **each shape in the matrix triggers a new compilation**.

**Strategy: Per-shape warmup before measurement**

```
For each (batch_size, items_per_request) in matrix:
    1. Send 5 warmup requests with this exact shape
       → Triggers XLA compilation and caches the compiled program
    2. Wait for compilation to complete (check JAX compilation cache)
    3. Begin measurement phase (compilation is cached)
```

**Alternative: Input padding to fixed shapes**

Pad all inputs to a small set of bucket sizes (e.g., powers of 2) to reduce the number of unique compilations. This trades compute efficiency for compilation stability. If using padding, document the bucket sizes in results metadata.

**What to watch for:**
- `JAX_COMPILATION_CACHE_DIR` must be set (avoids recompilation across server restarts)
- First request at each shape will be 10-100x slower than subsequent requests
- If compilation time is not excluded, it will dominate latency for low-run-count benchmarks

#### PyTorch/GPU: Standard Warmup

| Phase | Requests | Purpose |
|-------|----------|---------|
| CUDA warmup | 3 requests | Kernel initialization, memory allocation |
| `torch.compile` warmup (if used) | 2 requests | Compilation of optimized kernels |

PyTorch is generally shape-agnostic for eager mode. If `torch.compile` is enabled, it may also recompile for new shapes, but this is less common than JAX.

#### Warmup Validation

Both backends should log warmup completion before measurement begins. The benchmark script should verify that the compilation cache is warm by checking that the last warmup request's latency is within 2x of the expected steady-state latency. If not, additional warmup requests are sent.

### Statistical Rigor

- Minimum 3 runs per configuration (5 for publication-quality results)
- Report median and standard deviation
- Discard outliers beyond 2 standard deviations
- Record system state: chip utilization, memory pressure

## Infrastructure Requirements

### Prerequisites

Before any benchmark jobs can run, the following must be in place:

#### Authentication & Secrets

| Requirement | Why | How |
|-------------|-----|-----|
| HuggingFace token (`HF_TOKEN`) | Gated models (Llama-3.2) require authentication | K8s Secret in `eval-serving` namespace, mounted as env var |
| Container registry access | Pull benchmark images | GKE default service account or Workload Identity |

> **Note on gated models:** Llama-3.2-1B-Instruct requires HuggingFace access approval and a valid `HF_TOKEN`. For automated jobs, consider using an ungated model (e.g., `Qwen/Qwen3-0.6B`) to avoid auth fragility. If gated models are used, the token must be stored as a K8s Secret and injected into job pods.

```yaml
# Create HF token secret
kubectl create secret generic hf-token \
  --namespace eval-serving \
  --from-literal=HF_TOKEN=hf_xxxxx
```

#### GPU Node Pool Requirements

GPU node pools on GKE require additional configuration beyond just adding nodes:

| Component | Purpose | Status |
|-----------|---------|--------|
| NVIDIA GPU device plugin | Exposes GPUs to K8s scheduler | Auto-installed by GKE on GPU node pools |
| NVIDIA drivers | GPU kernel drivers | Auto-installed by GKE (Container-Optimized OS) |
| Node taints | Prevent non-GPU workloads on GPU nodes | Add `nvidia.com/gpu=present:NoSchedule` |
| Pod tolerations | Allow benchmark pods on GPU nodes | Must be in job YAML |
| Resource requests | Request GPU allocation | `nvidia.com/gpu: "1"` in resources.limits |

Example GPU node pool creation:
```bash
gcloud container node-pools create gpu-a100-pool \
  --cluster=<cluster-name> \
  --zone=<zone> \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=2 \
  --spot \
  --node-taints=nvidia.com/gpu=present:NoSchedule
```

Example toleration in job YAML:
```yaml
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
```

### GKE Node Pools

| Node Pool | Status | Configuration |
|-----------|--------|---------------|
| TPU v6e-1 (spot) | Exists (RFC-009) | Autoscaling 0-4, preemptible |
| TPU v6e-4 (spot) | Exists (RFC-009) | Autoscaling 0-2, preemptible |
| GPU (A100 or L4) | **Needs creation** | Spot instances, autoscaling 0-2 |
| CPU | Exists (RFC-009) | For benchmark runner orchestration |

### Other Resources

| Resource | Status | Notes |
|----------|--------|-------|
| Namespace `eval-serving` | Exists | Used by all v1/ manifests |
| Service account `eval-serving-sa` | Exists | GCS FUSE permissions |
| GCS bucket for models | Exists | `ashires-e7aaot-model-download-europe-west4` |
| GCS bucket for results | **Needs creation** | For benchmark result persistence |
| Container registry | Exists | `europe-docker.pkg.dev/ashires-e7aaot/container/` |

## Implementation Plan

### Phase 1: Parameterize Existing Templates

- [ ] Replace hardcoded values in v1/ manifests with Kustomize overlays
- [ ] Fix path mismatches in `run_benchmark.sh` and `cloudbuild.yaml`
- [ ] Create `base/benchmark-job.yaml` with configurable repo, branch, image, model
- [ ] Create `overlays/jax-tpu/` with TPU-specific resource definitions
- [ ] Validate: deploy a JAX benchmark job using new templates

### Phase 2: Add PyTorch/GPU Support

- [ ] Create GPU node pool in GKE cluster (A100 or L4)
- [ ] Build and push PyTorch GPU Docker image to container registry
- [ ] Create `overlays/pytorch-gpu/` with GPU resource definitions and server args
- [ ] Create PyTorch benchmark job that runs `bench_score.py` against GPU server
- [ ] Validate: deploy a PyTorch benchmark job using new templates

### Phase 3: Build Comparison Harness

- [ ] Create JAX `bench_score.py` (or decide to adapt `bench_serving.py`)
- [ ] Define common JSON output schema (see above)
- [ ] Implement `compare.py` orchestration script:
  - Render and submit both K8s jobs
  - Poll for completion
  - Collect results from pod logs
  - Normalize to common schema
  - Generate comparison report (markdown table)
  - Upload to GCS
- [ ] Create benchmark profiles (smoke, standard, full)

### Phase 4: Establish Baselines

- [ ] Run standard profile on JAX/TPU v6e-1 (3 runs)
- [ ] Run standard profile on PyTorch/GPU (3 runs)
- [ ] Generate initial comparison report
- [ ] Commit baseline results to repo
- [ ] Document findings and any tuning needed

## Alternatives Considered

### Alternative 1: SkyPilot

**Pros:**
- Unified YAML for both GPU and TPU
- Built-in spot instance recovery
- Multi-cloud support

**Cons:**
- Team already rejected it for reliability issues (ADR-002)
- User feedback: "the sky thing is not very reliable"
- Additional dependency
- TPU support less mature than GPU
- Multi-cloud not needed (GCP only)

**Why rejected:** ADR-002 decision stands. Reliability concerns outweigh convenience.

### Alternative 2: Plain gcloud VM Commands

**Pros:**
- Simplest approach, no K8s needed
- Direct SSH access for debugging
- No cluster overhead

**Cons:**
- Manual VM lifecycle management
- Less reproducible than K8s Jobs
- No built-in retry/cleanup
- GKE cluster already exists — not using it wastes investment

**Why rejected:** The user has a GKE cluster. K8s Jobs are cleaner for repeatable, automated benchmarks and leverage existing infrastructure.

### Alternative 3: Helm Charts

**Pros:**
- More powerful templating than Kustomize
- Package management for releases
- Widely used in K8s ecosystem

**Cons:**
- Heavier tooling for this use case
- Requires Helm installation and chart repository
- Kustomize is built into kubectl (no extra dependency)
- Existing templates are simple enough for Kustomize

**Why rejected:** Kustomize is sufficient and simpler. Helm adds complexity without clear benefit for benchmark jobs.

## Cost Analysis

### Per Comparison Run

| Resource | Duration | Hourly Cost (Spot) | Cost |
|----------|----------|-------------------|------|
| TPU v6e-1 (preemptible) | ~1 hr | $0.64 | $0.64 |
| GPU A100 (spot) | ~1 hr | ~$1.10 | $1.10 |
| GCE orchestrator (n2-standard-4) | ~2 hrs | $0.19 | $0.38 |
| **Total** | | | **~$2.12** |

### Monthly Estimates

| Frequency | Cost/Month |
|-----------|------------|
| Weekly standard runs | ~$8.50 |
| Daily smoke runs | ~$12.00 |
| Weekly standard + daily smoke | ~$20.50 |

### Comparison with Existing Costs

- Current nightly CI: ~$330-500/month (RFC-009)
- This adds: ~$8-20/month (2-6% increase)

## Open Questions

1. **GPU type:** A100 vs L4? A100 is more comparable to TPU v6e in compute class. L4 is cheaper but lower-end.
2. **Model selection:** Llama-3.2-1B-Instruct is gated (requires `HF_TOKEN`, approval). Qwen/Qwen3-0.6B is ungated and is the PyTorch bench_score.py default. For automated jobs, ungated is more reliable. Recommendation: start with Qwen3-0.6B for automation, add Llama-3.2 as optional gated model.
3. **JAX bench_score.py:** Create new script or adapt `bench_serving.py` with a `--score` mode?
4. **Multi-item scoring:** Include multi-item benchmarks now, or defer until RFC-008 is implemented?
5. **Prompt/template parity:** Should the parity spec (see Input Parity Specification section) be enforced programmatically in the benchmark script, or documented as a manual checklist? Recommendation: enforce in script via a shared config file that both backends consume.
6. **Throughput definition:** This RFC defines `throughput_ips` as `items/sec` and `throughput_rps` as `requests/sec`. The PyTorch bench_score.py reports only RPS. Should we report both, or standardize on one? Recommendation: report both, use `throughput_ips` as the primary comparison metric since it normalizes for `items_per_request`.

## References

- [Investigation: v1 Infrastructure Assessment](../investigations/v1-infrastructure-assessment.md) — Full gap analysis
- [RFC-004: Performance Benchmarks](004-score-api-performance-benchmarks.md) — JAX benchmark framework
- [RFC-009: ARC Runner Setup](009-arc-runner-setup.md) — GKE infrastructure
- [ADR-002: No SkyPilot](../decisions/002-no-skypilot-for-unit-tests.md) — SkyPilot rejection
- PyTorch bench_score.py: `sglang/benchmark/prefill_only/bench_score.py`
- PyTorch Dockerfile: `sglang/docker/Dockerfile`
- JAX Dockerfile: `sglang-jax/Dockerfile`
- Kustomize docs: https://kustomize.io/
