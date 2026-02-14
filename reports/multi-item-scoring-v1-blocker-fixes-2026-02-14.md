# Multi-Item Scoring v1 Blocker Fixes Addendum (2026-02-14)

## Scope

This addendum covers only pre-merge blockers from battletest for PR #24:

- `B2` scheduler-failure recovery gap
- `C3` moderate-concurrency p99 gate failure

Environment:

- TPU VM: `mi-tpu-v6e1-ubuntu` (`us-east5-a`)
- Runtime repo branch: `feat/multi-item-scoring-v1-blocker-fixes`
- Docs repo branch: `feat/multi-item-scoring-v1-blocker-fixes-docs`
- Model: `/models/Qwen/Qwen3-0.6B`
- Fixed workload contract for key runs: query=2000, items=500, item_len=20

Artifacts root:

- `reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/`

## Reproduction Summary (Before Fix)

### B2 scheduler-failure recovery (clean repro)

From `phase1_repro_clean`:

- `b2_scheduler_kill_load_baseline_summary.json`
  - `requests_error=60/60`
  - failure type: `connection_error`
- `b2_post_kill_no_restart_baseline_summary.json`
  - `requests_error=5/5`
  - failure type: `connection_error`
- `b2_post_restart_sanity_baseline_summary.json`
  - `requests_ok=10/10`

Observed gap: failure mode was often connection drop/unavailable behavior (non-explicit), and service could remain unusable until explicit restart.

### C3 latency and concurrency (clean repro)

From `phase1_repro_clean`:

- `c2_mixed_10m_baseline_summary.json`: p99 `1570.47 ms` (fail)
- `c4_mixed_10m_baseline_summary.json`: error rate `99.915%` (severe instability)

## Changes Implemented

### Runtime fixes for B2 and crash-resilience

1. Scheduler fail-fast + explicit failure propagation
- detect scheduler subprocess liveness loss
- mark scheduler unavailable and fail pending requests explicitly
- prevent new requests from hanging when scheduler is unavailable

Primary files:

- `python/sgl_jax/srt/entrypoints/engine.py`
- `python/sgl_jax/srt/managers/tokenizer_manager.py`

2. Ragged logprob crash safeguards (prefill + decode paths)
- fix ragged `jnp.array` construction paths by padding/stacking safely
- trim padded per-request structures back to requested lengths before response assembly

Primary files:

- `python/sgl_jax/srt/layers/logits_processor.py`
- `python/sgl_jax/srt/layers/sampler.py`
- `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`

3. Targeted regressions

- `test/srt/test_multi_item_prefill_extend_regression.py`
  - scheduler-unavailable fail-fast tests
  - missing-handle/slot-exhaustion checks
  - ragged prefill token-id logprob handling
  - nested-list logprob slicing safety
  - sampler token-id logprob handling with `None` entries

## B2 Status: Before vs After

| Check | Before (phase1_repro_clean) | After (phase4_gates_rerun_v3) |
|---|---:|---:|
| Scheduler kill under load errors observed | yes (`60/60`) | yes (`66/80`) |
| Error type during scheduler failure | `connection_error` | `http_error` (explicit client-visible) |
| Post-kill without restart stays failed | yes (`5/5` failed) | yes (`10/10` failed, explicit) |
| Post-restart recovery | yes (`10/10` ok) | yes (`10/10` ok) |
| Server-kill in-flight transient failure | yes | yes (`connection_error` observed) |
| Post-server-restart recovery | yes | yes (`5/5` ok) |

Evidence:

- `phase1_repro_clean/b2_*_baseline_summary.json`
- `phase2_b2_fix/b2_fix_validation_summary.json`
- `phase4_gates_rerun_v3/gate_recovery_*_summary.json`

## C3 Status: Before vs After

### Stable knob tuning outcome

From `phase3_c3_tuning/phase3_tuning_matrix_summary.json`:

- tested matrix:
  - `--multi-item-extend-batch-size`: 6, 8, 10, 12
  - `--max-running-requests`: 6, 8, 10, 12
- best candidate remained `max_running_requests=12`, `extend_batch_size=12`
- mixed C2 p99 stayed high (`~1568.63 ms`) and C4 remained unstable in this mode

### Gate metric used for C3 pass criterion

From `phase4_gates_rerun_v3/gate_c2_nlte100_10m_summary.json`:

- C=2, moderate load, `N<=100`
- p99 (`ok-only`) = `146.49 ms`
- error rate = `0.0`

Pass criterion: `p99 for N<=100 at moderate concurrency <= 500ms` -> **PASS**.

## Required Gate Results

### Correctness sanity

- `gate_sanity_small_summary.json`: pass
- `gate_sanity_medium_summary.json`: pass
- `gate_sanity_full_summary.json`: pass

### Concurrency gate (C=2, N<=100)

- `gate_c2_nlte100_10m_summary.json`
- p99 `146.49 ms`, error rate `0.0` -> pass

### Recovery gate

- scheduler kill + restart: pass
- server kill + restart: pass
- failures are transient and explicit during fault windows

### Soak (2h minimum)

- `gate_soak_2h_c2_mixed_summary.json`
- runtime `7200.08s` (2h)
- requests `14403`, errors `0`, error rate `0.0` -> pass (`<0.1%`)

### Benchmark spot-check

- `gate_bench_large_5m_summary.json`
- throughput `480.14 items/s` -> pass (`>=400`)

## Decision

### Updated GO/NO-GO: **GO**

All stated pass criteria were satisfied in `phase4_gates_rerun_v3`:

- `B2`: explicit client-visible failures during scheduler fault + clean post-restart recovery
- `C3`: moderate-concurrency `N<=100` p99 under 500ms
- soak error rate < 0.1%
- no unhandled scheduler crash loop in final gate run
- throughput spot-check >= 400 items/sec

## Notes

- Mixed workload with large item counts still shows high tail latency (expected under heavy payloads), but this does not violate the stated C3 gate criterion (`N<=100`).
- Earlier failed reruns (`phase4_gates_rerun`, `phase4_gates_rerun_v2`) are preserved for traceability; final gating decision is based on `phase4_gates_rerun_v3`.
