# Segment Mask TPU Fix Validation (2026-02-13)

## Scope
Validate the TPU-safe segment mask lowering fix for multi-item scoring in `sglang-jax`:

- fix TPU lowering crash in segment path
- preserve dense fallback behavior
- verify dense vs segment score parity
- verify prefill+extend flow is not regressed
- run dense vs segment A/B benchmarks using `test/srt/test_bench_multi_item_score.py`

## Environment
- Date: February 13, 2026
- TPU VM: `mi-tpu-v6e1-ubuntu` (`us-east5-a`)
- Model: `/models/Qwen/Qwen3-0.6B`
- Repo: `~/work/sglang-jax-segment-fix-v1`
- Branch: `feat/segment-mask-tpu-fix-v1`

## Implementation Summary
Kernel changes in `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py`:

1. Removed unsupported segment gather (`multi_item_row_seg_starts_ref[q_token_indices]`).
2. Loaded per-token segment starts via TPU-safe scalar loads from scalar-prefetch metadata.
3. Rewrote segment allow/mask combine with int32 mask algebra to avoid TPU Mosaic bool truncation failures.
4. Kept dense/custom-mask path unchanged as fallback.

Test changes in `test/srt/test_multi_item_regression.py`:

1. Added TPU segment regression class `TestMultiItemSegmentTPURegression`.
2. Added:
   - `test_segment_mode_runs_on_tpu`
   - `test_segment_matches_dense_scores`
   - `test_segment_prefill_extend_flow_no_regression`
3. Updated dense-vs-segment parity test to instantiate engines sequentially (avoid TPU ownership contention).

## TPU Regression Validation

### 1) Segment mode lowers and runs on TPU
Command:
```bash
PYTHONPATH=python pytest -q -s \
  test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_mode_runs_on_tpu
```
Result: `1 passed` in `84.44s`.

### 2) Dense vs segment score parity
Command:
```bash
PYTHONPATH=python pytest -q -s \
  test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_matches_dense_scores
```
Result: `1 passed` in `168.78s`, with test assertion `max_diff <= 1e-4`.

### 3) Prefill+extend flow no regression
Command:
```bash
PYTHONPATH=python pytest -q -s \
  test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_prefill_extend_flow_no_regression
```
Result: `1 passed` in `202.79s`.

## Benchmark A/B (Dense vs Segment)

### Packed multi-item (`TestMultiItemScorePerformance::test_benchmark_multi_item_packed`)
Commands:
```bash
MULTI_ITEM_MASK_IMPL=dense MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=3 \
PYTHONPATH=python pytest -q -s \
  test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed

MULTI_ITEM_MASK_IMPL=segment MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=3 \
PYTHONPATH=python pytest -q -s \
  test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed
```

Results:
| Mode | Throughput (items/sec) | Latency per item (ms) | Total time (500 items, sec) |
|---|---:|---:|---:|
| Dense | 52.27 | 19.13 | 9.57 |
| Segment | 105.06 | 9.52 | 4.76 |

Packed-path speedup (segment vs dense): **2.01x**.

### Prefill+extend (`TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend`)
Commands:
```bash
MULTI_ITEM_MASK_IMPL=dense MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=2 \
PYTHONPATH=python pytest -q -s \
  test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend

MULTI_ITEM_MASK_IMPL=segment MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=2 \
PYTHONPATH=python pytest -q -s \
  test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend
```

Results:
| Mode | Throughput (items/sec) | Latency per item (ms) | Total time (500 items, sec) |
|---|---:|---:|---:|
| Dense | 518.36 | 1.93 | 0.96 |
| Segment | 513.06 | 1.95 | 0.97 |

Prefill+extend delta (segment vs dense): **-1.02%** (within run noise).

## Recommendation

1. Keep segment path enabled for packed multi-item scoring (`segment` or `auto` with current threshold), since TPU lowering now works and packed throughput improves materially vs dense.
2. Keep dense mode intact as fallback (already preserved).
3. For prefill+extend workloads, either mode is acceptable; measured difference was negligible.
