# Investigation: v1/ Infrastructure Assessment for Cross-Backend Benchmarking

| | |
|------------|------|
| **Date** | 2026-02-04 |
| **Status** | Complete |
| **Related** | [RFC-004](../rfcs/004-score-api-performance-benchmarks.md), [RFC-009](../rfcs/009-arc-runner-setup.md), [RFC-010](../rfcs/010-cross-backend-benchmarking.md) |

## Summary

Assessment of the `v1/` K8s infrastructure directory to determine readiness for PyTorch (GPU) vs JAX (TPU) performance benchmarking on GKE. The v1/ directory provides a starting point for JAX/TPU benchmarking but has significant gaps that prevent cross-backend comparison.

**Verdict:** v1/ is a useful foundation but is **not sufficient** for cross-backend benchmarking. All manifests are hardcoded to a single developer's environment, no GPU/PyTorch infrastructure exists, and the benchmark script referenced in templates does not exist in the JAX codebase.

## Methodology

**Scope:**
- Audited all 11 files in `sglang-jax-dev-scripts/v1/` (local copy)
- Checked GitHub repo (`alexshires/sglang-jax-dev-scripts`) for files not in local copy
- Cross-referenced with actual code in `sglang-jax/` (JAX fork) and `sglang/` (PyTorch upstream)
- Verified existence of scripts, Docker images, and K8s resources referenced by templates

**Approach:**
- File-by-file audit of v1/ contents
- Grep for hardcoded values (repos, branches, images, buckets)
- Existence checks for referenced scripts and tools
- Comparison of PyTorch vs JAX benchmark tooling

## File Inventory

### v1/ Directory Contents

| File | Purpose | Usable As-Is? | Issues |
|------|---------|---------------|--------|
| `benchmark/benchmark-job.yaml` | K8s Job: clone repo, start JAX server, run bench_score.py | No | Hardcoded fork/branch/image/bucket; references non-existent bench_score.py |
| `benchmark/run_benchmark.sh` | Renders YAML template, submits via kubectl | Partially | GPU path exists but fragile sed; path mismatch to template |
| `benchmark/benchmark-job-rendered.yaml` | Pre-rendered output of run_benchmark.sh | No | Same issues as template; no actual benefit over template |
| `benchmark/README.md` | Docs for benchmarking suite | Partially | References non-existent `bench_score.py`; metrics description is accurate |
| `Dockerfile` | Builds JAX+TPU image | No | Hardcoded `--tp-size=4`; no model path; no GPU variant |
| `cloudbuild.yaml` | Cloud Build pipeline for Docker image | No | Path mismatch (`sglang-scripts/` vs `v1/`) |
| `test-runner-job.yaml` | K8s Job for running Score API tests on TPU | No | Hardcoded fork/branch/image/bucket |
| `debug-tpu-pod.yaml` | Interactive TPU debug shell | Yes | Hardcoded image/bucket, but functional for debugging |
| `README.md` | Overview of all v1/ files | Yes | Accurate directory documentation |
| `implememtation_overview.md` | Architecture deep dive for /v1/score in JAX | Yes | Design doc, not infra (typo in filename) |
| `optimization_context_jan27.md` | Performance optimization findings (Jan 27) | Yes | Historical context; references TPU v5e (not v6e) |

## Critical Findings

### Finding 1: All Manifests Hardcoded to One Developer's Environment

Every K8s manifest references personal development resources:

| Value | Hardcoded To | Files Affected |
|-------|-------------|----------------|
| Repository | `github.com/alexshires/sglang-jax.git` | benchmark-job.yaml, test-runner-job.yaml |
| Branch | `fix/score-api-missing-logprobs` | benchmark-job.yaml, test-runner-job.yaml |
| Container image | `europe-docker.pkg.dev/ashires-e7aaot/container/vllm-tpu:pr` | All manifests |
| GCS bucket | `ashires-e7aaot-model-download-europe-west4` | All manifests with GCS FUSE |

**Impact:** None of these values will work in another environment without manual editing of every file.

### Finding 2: No PyTorch/GPU Infrastructure

The v1/ directory is 100% JAX/TPU focused:
- No PyTorch Dockerfile (upstream's is at `sglang/docker/Dockerfile` — CUDA 12.9.1 based)
- No GPU benchmark job template that works (the `sed` GPU path in `run_benchmark.sh` is incomplete)
- No PyTorch server launch configuration
- No GPU node pool references

### Finding 3: No Cross-Backend Comparison Tooling

The two benchmark ecosystems are completely separate:
- PyTorch: `sglang/benchmark/prefill_only/bench_score.py` with `util.py`
- JAX: No equivalent bench_score.py exists (see Finding 6)
- No unification layer, no shared output format, no comparison reporting

### Finding 4: Path Mismatches

`run_benchmark.sh` references template at `sglang-scripts/benchmark/benchmark-job.yaml` but the actual path is `v1/benchmark/benchmark-job.yaml`. The script will fail without fixing.

Similarly, `cloudbuild.yaml` references `sglang-scripts/Dockerfile` but the file is at `v1/Dockerfile`.

### Finding 5: No Parameterization Mechanism

There is no Helm, Kustomize, envsubst, or other templating system. The only "parameterization" is fragile `sed` replacements in `run_benchmark.sh`, which:
- Uses naive string replacement that can fail with special characters
- Has no validation of rendered YAML
- Doesn't support most configurable values (repo, branch, model, benchmark params)

### Finding 6: JAX bench_score.py Does Not Exist

The benchmark job template (`benchmark-job.yaml`) runs:
```bash
python3 python/sgl_jax/bench_score.py --num-requests 1000 --concurrency 32 ...
```

**This file does not exist in the sglang-jax repository.** A thorough search confirmed no `bench_score.py` anywhere in the JAX codebase. The JAX repo has these benchmark scripts instead:

| Script | Location | Purpose |
|--------|----------|---------|
| `bench_serving.py` | `python/sgl_jax/bench_serving.py` | Online serving benchmarks (throughput, latency, TTFT) |
| `bench_one_batch.py` | `python/sgl_jax/bench_one_batch.py` | Single static batch without server |
| `bench_one_batch_server.py` | `python/sgl_jax/bench_one_batch_server.py` | Single batch via HTTP server |
| `bench_offline_throughput.py` | `python/sgl_jax/bench_offline_throughput.py` | Offline throughput via Engine API |

None of these are Score API-specific benchmarks.

### Finding 7: No Results Collection Pipeline

Benchmark results go to `kubectl logs` and disappear. There is:
- No GCS upload for results
- No CSV/JSON export from K8s jobs
- No historical tracking or comparison reporting
- No regression detection integration

## Benchmark Script Comparison: PyTorch vs JAX

### PyTorch Score Benchmark

**File:** `sglang/benchmark/prefill_only/bench_score.py` (192 lines)

| Aspect | Detail |
|--------|--------|
| **CLI args** | None — all config hardcoded in script |
| **Server connection** | HTTP POST to `http://localhost:30000/v1/score` |
| **Model** | `Qwen/Qwen3-0.6B` (hardcoded) |
| **Output format** | CSV to stdout (per-minute intervals) |
| **Metrics** | RPS, success/fail counts, avg/p50/p90/p99 latency |
| **Config** | RPS=160, duration=60s, 100 unique requests, Poisson distribution |
| **Dependencies** | `util.py` (814 lines) — BenchmarkConfig, request generation, profiler |
| **Label tokens** | `[9454, 2753]` (Yes/No for Qwen) |
| **Items per request** | 10 |

### JAX Score Benchmark

**No equivalent exists.** The closest tools are:

- `bench_serving.py` — General serving benchmark, not Score API specific. Supports `sgl-jax` backend.
- `test/srt/test_bench_score.py` — CI performance gate test (4 tests with thresholds), not a standalone benchmark tool.

### Gap Analysis

| Capability | PyTorch | JAX | Gap |
|------------|---------|-----|-----|
| Score API benchmark script | `bench_score.py` | Does not exist | **Critical** |
| CLI arguments | None (hardcoded) | N/A | Both need work |
| Configurable profiles | No | No | Both need work |
| CSV/JSON output | CSV (basic) | N/A | PyTorch has basic support |
| Regression detection | No | No | Neither has this |
| Cross-backend comparison | No | No | **Not possible today** |

## What Works

Despite the gaps, the v1/ infrastructure has reusable foundations:

1. **K8s Job structure** — The job pattern (clone → install → start server → wait for health → run benchmark → cleanup) is sound and reusable
2. **GCS FUSE integration** — Model mounting via GCS FUSE is well-configured in the templates
3. **Resource definitions** — CPU/memory/TPU resource requests are reasonable (24 CPU, 100Gi RAM, 1 TPU)
4. **debug-tpu-pod.yaml** — Directly usable for interactive TPU debugging
5. **run_benchmark.sh concept** — Template rendering + kubectl submit is the right approach (just needs better implementation)
6. **GPU support scaffolding** — `run_benchmark.sh` has a GPU code path, even if incomplete

## Recommendations

1. **Do not attempt cross-backend benchmarking with v1/ as-is** — Too many gaps and hardcoded values
2. **Parameterize templates** before any other work — Replace hardcoded values with Kustomize overlays or envsubst variables
3. **Create JAX bench_score.py** or adapt an existing benchmark script for Score API-specific testing
4. **Add PyTorch/GPU K8s job template** — Based on upstream's CUDA Dockerfile
5. **Build unified comparison tooling** — Common output format, normalization, reporting

See [RFC-010: Cross-Backend Benchmarking](../rfcs/010-cross-backend-benchmarking.md) for the proposed solution.

## References

- [v1/ directory](../v1/) — Infrastructure templates assessed
- [RFC-004: Performance Benchmarks](../rfcs/004-score-api-performance-benchmarks.md) — Benchmark framework proposal (JAX only)
- [RFC-009: ARC Runner Setup](../rfcs/009-arc-runner-setup.md) — GKE runner infrastructure
- [ADR-002: No SkyPilot](../decisions/002-no-skypilot-for-unit-tests.md) — Decision against SkyPilot
- PyTorch bench_score.py: `sglang/benchmark/prefill_only/bench_score.py`
- JAX benchmark scripts: `sglang-jax/python/sgl_jax/bench_*.py`
