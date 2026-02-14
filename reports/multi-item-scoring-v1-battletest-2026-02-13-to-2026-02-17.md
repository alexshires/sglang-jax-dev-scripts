# Multi-Item Scoring v1 Battletest Report (2026-02-13 to 2026-02-17)

## Scope and Goal
Pre-merge battle testing for `sglang-jax` multi-item scoring v1 on TPU, focused on:
- correctness
- robustness
- reproducibility
- sustained performance

Target merge window: Tuesday, 2026-02-17.

## Environment and Revisions
- TPU VM: `mi-tpu-v6e1-ubuntu`
- Zone: `us-east5-a`
- Model path: `/models/Qwen/Qwen3-0.6B`
- Runtime repo path (TPU): `/home/kanna/work/sglang-jax-bench-f9d3fb5`
- Runtime branch under test: `feat/multi-item-scoring-v1-battletest`
- Runtime commit SHA: `f9d3fb5`
- Docs branch: `feat/multi-item-scoring-v1-battletest-docs`
- Docs baseline commit SHA: `fbb01b7`

Artifact root:
- `reports/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/`

## Fixed Benchmark Contract (Main Table)
- query tokens: `2000`
- items/request: `500`
- tokens/item: `20`

## Phase 0: Harness Prep
Added dedicated battletest tooling:
- `scripts/multi_item/soak_runner.py`
- `scripts/multi_item/server_metrics_sampler.py`
- `scripts/multi_item/launch_mode_server.sh`
- `scripts/multi_item/run_mode_trials.sh`

Implemented features:
- duration parsing (`s/m/h`)
- deterministic seed
- weighted request mix (`small/medium/large`)
- randomized inter-arrival (Poisson)
- configurable concurrency
- per-request CSV/JSONL + summary JSON outputs
- periodic server sampling (RSS/VSZ, `/get_server_info`, optional `/metrics`)
- memory growth and trend estimation (MB/hour)

## Phase 1: Baselines + Sanity
### Runtime tests
- `pytest -q -s test/srt/test_multi_item_prefill_extend_regression.py`
  - Result: `10 passed`
  - Artifact: `phase1/test_multi_item_prefill_extend_regression_20260213T231607Z.log`
- `pytest -q -s test/srt/test_multi_item_prefill_extend.py`
  - Result: `2 passed`
  - Artifact: `phase1/test_multi_item_prefill_extend_20260213T231607Z.log`

### `/v1/score` sanity
- Small (3 items): success
- Medium (30 items): success
- Full (500 items): success

Artifacts:
- `phase1/sanity_small_summary.json`
- `phase1/sanity_medium_summary.json`
- `phase1/sanity_large_summary.json`

### Scheduler crash signature
- No scheduler traceback in phase-1 sanity prefill+extend run logs.

## Phase 2: Reproducible Mode Benchmarks
### Repro protocol (applied to each mode)
1. Full server restart between modes (new process and clean launch script).
2. Warmup: 5 warmup requests, discard metrics.
3. Wait 30 seconds after warmup.
4. Measured trials: 3 independent reruns.

### Exact server flags per mode
Single-item:
```bash
python -m sgl_jax.launch_server --model-path /models/Qwen/Qwen3-0.6B --trust-remote-code --port 30000 --device tpu --dtype bfloat16 --attention-backend fa --mem-fraction-static 0.7 --skip-server-warmup --max-running-requests 32 --chunked-prefill-size 4096 --precompile-token-paddings 1024 4096 --precompile-bs-paddings 1 4 8 16 32
```
Source: `phase2/logs/server_single_item_20260213T233035Z.cmd`

Packed:
```bash
python -m sgl_jax.launch_server --model-path /models/Qwen/Qwen3-0.6B --trust-remote-code --port 30000 --device tpu --dtype bfloat16 --attention-backend fa --mem-fraction-static 0.7 --page-size 64 --skip-server-warmup --max-running-requests 32 --chunked-prefill-size -1 --disable-radix-cache --max-prefill-tokens 32768 --precompile-token-paddings 1024 4096 16384 --precompile-bs-paddings 1 4 8 16 32 --multi-item-scoring-delimiter 151643 --multi-item-scoring-chunk-size 32 --max-multi-item-count 512 --max-multi-item-seq-len 32768 --multi-item-mask-impl dense --multi-item-segment-fallback-threshold 0
```
Source: `phase2/logs/server_packed_20260213T234701Z.cmd`

Prefill+extend:
```bash
python -m sgl_jax.launch_server --model-path /models/Qwen/Qwen3-0.6B --trust-remote-code --port 30000 --device tpu --dtype bfloat16 --attention-backend fa --mem-fraction-static 0.7 --page-size 64 --skip-server-warmup --max-running-requests 12 --chunked-prefill-size -1 --max-prefill-tokens 32768 --precompile-token-paddings 1024 4096 16384 --precompile-bs-paddings 1 2 3 4 5 6 7 8 9 10 11 12 --multi-item-scoring-delimiter 151643 --multi-item-scoring-chunk-size 500 --max-multi-item-count 512 --max-multi-item-seq-len 32768 --multi-item-mask-impl dense --multi-item-segment-fallback-threshold 0 --multi-item-enable-prefill-extend --multi-item-extend-batch-size 12 --multi-item-prefill-extend-cache-timeout 60 --enable-scoring-cache
```
Source: `phase2/logs/server_prefill_extend_20260214T000222Z.cmd`

### Side-by-side mode results (mean over 3 reruns)
| Mode | Throughput (items/s) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | Error rate |
|---|---:|---:|---:|---:|---:|
| Single-item (serial profile, `large-items=1`) | 35.69 | 27.58 | 28.28 | 29.14 | 0.0% |
| Packed | 51.82 | 9595.24 | 9886.51 | 10058.41 | 0.0% |
| Prefill+extend | 506.62 | 987.98 | 1012.14 | 1021.92 | 0.0% |

Artifacts:
- `phase2/single_item_20260213T233035Z/trials_aggregate.json`
- `phase2/packed_20260213T234701Z/trials_aggregate.json`
- `phase2/prefill_extend_20260214T000222Z/trials_aggregate.json`

### Variance table across reruns
| Mode | Throughput stddev | Throughput min | Throughput max |
|---|---:|---:|---:|
| Single-item | 0.216 | 35.40 | 35.91 |
| Packed | 0.209 | 51.64 | 52.12 |
| Prefill+extend | 2.576 | 502.97 | 508.52 |

### Key performance ratios
- Prefill+extend vs packed: `9.78x`
- Prefill+extend vs single-item (serial profile): `14.20x`
- Packed vs single-item (serial profile): `1.45x`

## Phase 2.5: Load Discovery (Ramp)
10-minute ramp before long soak (C=8, 2-minute steps):
- Rates tested: `0.5`, `1.0`, `1.5`, `2.0`, `3.0` req/s

Observed break point:
- At `0.5 req/s`, error rate already `97.44%`.
- At `>=1.0 req/s`, error rate `100%`.
- Dominant error reason: `connection_error`.

Artifacts:
- `phase25/ramp_c8_r0p5_summary.json`
- `phase25/ramp_c8_r1_summary.json`
- `phase25/ramp_c8_r1p5_summary.json`
- `phase25/ramp_c8_r2_summary.json`
- `phase25/ramp_c8_r3_summary.json`

## Phase 3: Concurrency Matrix + Soak
### Concurrency matrix
| Scenario | Requests total | OK | Error | Error rate | Throughput (items/s) | p99 latency (ms, OK only) |
|---|---:|---:|---:|---:|---:|---:|
| C=1 mixed (5m) | 175 | 175 | 0 | 0.00% | 62.20 | 24891.70 |
| C=2 mixed (10m) | 507 | 507 | 0 | 0.00% | 94.42 | 1606.78 |
| C=4 mixed (10m) | 749 | 1 | 748 | 99.87% | 0.005 | 38790.42 |
| C=8 mixed (10m) | 970 | 3 | 967 | 99.69% | 0.015 | 27114.93 |
| C=8 large-only (10m) | 642 | 0 | 642 | 100.00% | 0.0 | 0.0 |

Artifacts:
- `phase3/c1_mixed_5m_summary.json`
- `phase3/c2_mixed_10m_summary.json`
- `phase3/c4_mixed_10m_summary.json`
- `phase3/c8_mixed_10m_summary.json`
- `phase3/c8_large_10m_summary.json`

### Long soak (target: 2h minimum)
Completed:
- Runtime window: `2026-02-14T01:19:39Z` to `2026-02-14T03:19:39Z` (`7200s`)
- Workload: C=2, mixed `small=0.4, medium=0.4, large=0.2`

Soak request outcomes:
- Requests total: `4244`
- Requests OK: `4244`
- Requests error: `0`
- Error rate: `0.0%`
- Throughput: `69.95 items/s`
- Latency: p50 `102.71 ms`, p95 `1129.27 ms`, p99 `1527.72 ms`
- First-run penalty ratio: `109.16x` (first 20 vs last 20 successful requests)

Memory trend outcomes:
- RSS start: `385.63 MB`
- RSS end: `355.76 MB`
- RSS growth: `-29.88 MB` (`-7.75%`)
- RSS trend after warmup: `-12.40 MB/hour`
- VSZ growth: `+2.00 MB` over 2h

Artifacts:
- `phase3/long_soak_c2_summary.json`
- `phase3/long_soak_metrics_summary.json`
- `phase3/long_soak_c2_requests.csv`
- `phase3/long_soak_metrics_samples.csv`

## Failure Signatures Observed (High Priority)
1. Single-item mode crash in logprob collation path:
- `TypeError: Cannot concatenate arrays with different numbers of dimensions`
- Location in traceback: `logits_processor.py:get_token_ids_logprobs`
- Artifact: `phase2/logs/server_single_item_20260213T232802Z.log`

2. Packed mode overflow at chunk-size 500:
- `JaxRuntimeError: INTERNAL: RET_CHECK failure ... allocation_size_words <= int32 max`
- Artifact: `phase2/logs/server_packed_20260213T233303Z.log`
- Mitigation used for stable packed mode runs: `--multi-item-scoring-chunk-size 32`

3. Prefill+extend instability at higher concurrency:
- Same `Cannot concatenate arrays with different numbers of dimensions` path appears under higher-load runs.
- Artifact: `phase2/logs/server_prefill_extend_20260214T000222Z.log`

## Phase 4: Failure Injection / Recovery
Executed explicit chaos checks:

1) Kill server during in-flight 500-item request:
- Load summary: `1/1` request failed (`connection_error`)
- Recovery sanity after restart: `5/5` successful
- Artifacts:
  - `phase4/fi1_inflight_kill_summary.json`
  - `phase4/fi1_recovery_summary.json`

2) Kill scheduler subprocess:
- Initial scripted attempt was inconclusive because the first scheduler PID pattern missed runtime process naming.
- Targeted rerun with actual scheduler process name (`sglang::scheduler`) captured the failure signature:
  - `phase4/fi2b_scheduler_kill_load_summary.json`: `3/3` timed out
  - `phase4/fi2b_recovery_summary.json`: `2/2` timed out after restart attempt
  - `phase4/fi2b_postcheck_summary.json`: still timing out
- Observed behavior: server can remain unhealthy/stuck after scheduler kill unless full process cleanup is forced.
- Scheduler PID evidence: `phase4/fi2b_scheduler_kill.pid`

3) Restart server during mixed load:
- Load summary during restart: `78/80` failed (`connection_error`)
- Post-restart sanity: `10/10` successful in primary run
- Artifacts:
  - `phase4/fi3_restart_during_load_summary.json`
  - `phase4/fi3_post_restart_sanity_summary.json`

4) Missing cache-handle explicit error path:
- `pytest -q -s test/srt/test_multi_item_prefill_extend_regression.py::test_resolve_extend_from_cache_missing_handle_returns_error`
- Result: `1 passed`
- Artifact: `phase4/fi4_missing_cache_handle_pytest.log`

Phase 4 aggregate:
- `phase4/phase4_summary.json`

## Phase 5: Contract Edge Tests
Executed targeted RFC-006 and known-limitation edge suites:
- `pytest -q -s test/srt/test_score_validation.py`
  - Result: `38 passed`
- `pytest -q -s test/srt/test_score_api_edge_cases.py -m "not integration"`
  - Result: `33 passed`, `7 deselected`
- `pytest -q -s test/srt/test_multi_item_prefill_extend_regression.py::test_resolve_extend_from_cache_missing_handle_returns_error`
  - Result: `1 passed`
  - Verifies missing cache handle returns explicit error path (no silent wrong score).
- `pytest -q -s test/srt/test_multi_item_chunking.py`
  - Result: `2 passed`
  - Covers chunking guard behavior for large-total token paths.
- `pytest -q -s test/srt/test_multi_item_scheduler_output.py`
  - Result: `1 passed`
  - Validates per-item logprob alignment semantics.
- `JAX_PLATFORMS=cpu pytest -q -s test/srt/test_multi_item_segment_mask.py`
  - Result: `3 passed`
  - CPU run used because TPU was actively owned by live soak server process.

Artifacts:
- `phase5/test_score_validation_20260214T012844Z.log`
- `phase5/test_score_api_edge_cases_no_integration_20260214T012844Z.log`
- `phase5/test_missing_cache_handle_20260214T012844Z.log`
- `phase5/test_multi_item_chunking_20260214T012844Z.log`
- `phase5/test_multi_item_scheduler_output_20260214T012912Z.log`
- `phase5/test_multi_item_segment_mask_cpu_20260214T012928Z.log`

## Single-Item Regression vs `main`
Executed same setup/profile on both branches:
- Main worktree commit: `c6b52a0`
- Feat branch commit: `f9d3fb5`
- Profile: `single_item` mode, `large-items=1`, warmup excluded, 3 measured reruns

Results:
- Main throughput mean: `31.1317 items/s`
- Feat throughput mean: `30.3139 items/s`
- Regression (`feat` vs `main`): `-2.63%`
- Criterion (`<= 5%` drop): **Pass**

Artifacts:
- `phase_regression_single_item_main_vs_feat/single_item_regression_summary.json`
- `phase_regression_single_item_main_vs_feat/main/.../trials_aggregate.json`
- `phase_regression_single_item_main_vs_feat/feat/.../trials_aggregate.json`
## Acceptance Criteria Tracking (Final)
| Criterion | Status | Notes |
|---|---|---|
| A1. 0 silent fallbacks on cache-miss paths | Pass (current evidence) | No fallback log signatures found; missing-cache-handle regression test returns explicit error. |
| A2. Response count equals item count | Pass (tested slices) | Enforced by harness; phase1 sanity + phase2 stable runs pass. |
| A3. No cross-item contamination | Pass (functional/unit evidence) | Phase1 functional tests + scheduler/segment mask alignment tests pass; no wrong-score contamination observed in successful requests. |
| B1. Soak error rate < 0.1% | Pass | 2h soak error rate `0.0%` (`4244/4244` success). |
| B2. No unhandled scheduler crashes | Fail | Scheduler kill injection can leave service in unhealthy timeout/stuck state. |
| B3. Memory leak threshold met | Pass | RSS trend is negative (`-12.40 MB/hour` after warmup). |
| C1. Prefill+extend >= 400 items/s (target 500) | Pass | 506.62 items/s mean on fixed contract. |
| C2. Prefill+extend >= 8x packed | Pass | 9.78x vs packed. |
| C3. P99 latency for N<=100 <= 500ms | Fail | C1/C2 matrix p99 significantly above 500ms. |
| D. Single-item regression vs main <= 5% | Pass | `-2.63%` vs main on same setup. |

## GO / NO-GO
**NO-GO** for merge in current state.

Primary reasons:
- Fails latency criterion C3 (p99 > 500ms under moderate concurrency).
- Fails stability criterion B2 (scheduler-failure recovery behavior can leave service unhealthy/stuck).
- High-concurrency matrix remains unstable (near-total failures at C>=4).

## Blockers vs Non-Blockers
### Blocking
- High-concurrency instability (C>=4) with near-total failure rates and connection errors.
- P99 latency criterion for N<=100 requests currently unmet.
- Scheduler-kill recovery gap: service can remain stuck with timeouts after scheduler failure injection.

### Non-blocking / informational
- Packed mode requires conservative chunk setting (`32`) for stable runs.
- `/metrics` endpoint may return 404 on current server path (sampler handles this gracefully).

## Exact Command Record (Representative)
Phase-1 tests:
```bash
source ~/work/sglang-jax/.venv/bin/activate
cd ~/work/sglang-jax-bench-f9d3fb5
PYTHONPATH=python pytest -q -s test/srt/test_multi_item_prefill_extend_regression.py
PYTHONPATH=python pytest -q -s test/srt/test_multi_item_prefill_extend.py
```

Phase-2 mode runner:
```bash
bash ~/work/sglang-jax-dev-scripts/scripts/multi_item/run_mode_trials.sh \
  --mode prefill_extend \
  --repo-dir ~/work/sglang-jax-bench-f9d3fb5 \
  --artifact-dir ~/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase2 \
  --trials 3 --requests-per-trial 10 --warmup-requests 5 \
  --concurrency 1 --arrival-rate 100 --large-items 500
```

Phase-3 matrix sample:
```bash
python ~/work/sglang-jax-dev-scripts/scripts/multi_item/soak_runner.py \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 10m --concurrency 2 --arrival-rate 0.8 \
  --mix small=0.4,medium=0.4,large=0.2 \
  --small-items 3 --medium-items 30 --large-items 500 \
  --query-tokens 2000 --tokens-per-item 20 \
  --output-dir ~/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase3 \
  --output-prefix c2_mixed_10m
```

Phase-3 long soak launch (current run):
```bash
python ~/work/sglang-jax-dev-scripts/scripts/multi_item/server_metrics_sampler.py \
  --base-url http://127.0.0.1:30000 --duration 2h --interval 30 --timeout 5 \
  --pid 824178 --refresh-pid --sample-server-info --sample-metrics-endpoint \
  --warmup-minutes 15 \
  --output-dir ~/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase3 \
  --output-prefix long_soak_metrics

python ~/work/sglang-jax-dev-scripts/scripts/multi_item/soak_runner.py \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 2h --concurrency 2 --arrival-rate 0.6 \
  --mix small=0.4,medium=0.4,large=0.2 \
  --small-items 3 --medium-items 30 --large-items 500 \
  --query-tokens 2000 --tokens-per-item 20 \
  --output-dir ~/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase3 \
  --output-prefix long_soak_c2
```

Phase-4 chaos run:
```bash
/home/kanna/work/sglang-jax-dev-scripts/scripts/multi_item/run_phase4_chaos.sh
```

Single-item regression (`main` vs `feat`):
```bash
/home/kanna/work/sglang-jax-dev-scripts/scripts/multi_item/run_single_item_regression_main_vs_feat.sh
```
