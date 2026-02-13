# Multi-Item Prefill+Extend TPU Benchmark (2026-02-13)

## Scope
Benchmark `test/srt/test_bench_multi_item_score.py` on TPU v6e-1 for `Qwen/Qwen3-0.6B` with the fixed workload:
- query tokens: 2000
- items: 500
- item tokens: 20

This run focuses on JAX side only.

## Environment
- Date: February 13, 2026
- TPU VM: `mi-tpu-v6e1-ubuntu` (`us-east5-a`)
- Model: `/models/Qwen/Qwen3-0.6B`
- Test file: `test/srt/test_bench_multi_item_score.py`
- JAX commit: `f9d3fb5`
- Timed runs:
  - Prefill+extend: 5
  - Packed: 5
  - Single-item sequential: 1

## Final Stable Result
From one clean TPU rerun on commit `f9d3fb5`:

| Mode | Throughput (items/sec) | Latency per item (ms) | Time for 500 items (sec) |
|---|---:|---:|---:|
| Single-Item Sequential | 10.34 | 96.75 | 48.37 |
| Multi-Item Packed | 52.18 | 19.17 | 9.58 |
| Multi-Item Prefill+Extend | 525.87 | 1.90 | 0.95 |

Observed speedups:
- Prefill+extend vs packed: **10.08x**
- Prefill+extend vs single-item sequential: **50.86x**
- Packed vs single-item sequential: **5.05x**

## Benchmark Profile Used
`test/srt/test_bench_multi_item_score.py` defaults were tuned for stable TPU benchmarking:
- `MULTI_ITEM_MASK_IMPL=dense`
- Prefill+extend defaults:
  - `MULTI_ITEM_EXTEND_BATCH_SIZE=12`
  - `MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS=12`
  - `MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS=1..12`
  - `MULTI_ITEM_BENCH_WARMUP_RUNS=1` for this rerun

## Repro Commands
Run from TPU VM in a clean checkout:

```bash
source .venv/bin/activate

MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=5 \
MULTI_ITEM_MASK_IMPL=dense PYTHONPATH=python \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed

MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=5 \
MULTI_ITEM_MASK_IMPL=dense MULTI_ITEM_EXTEND_BATCH_SIZE=12 \
MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS=12 PYTHONPATH=python \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend

MULTI_ITEM_MASK_IMPL=dense PYTHONPATH=python \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_single_item_sequential
```

## Notes
- This rerun used dense mode for consistency with the PR benchmark path.
- Segment TPU lowering has a separate validation report: [segment fix validation report](./segment-mask-tpu-fix-validation-2026-02-13.md).
- Higher chunk profiles (for example `extend_batch_size=16` and above) can show higher short-run throughput but also occasional long-tail pauses; the selected default profile prioritizes repeatable benchmark outcomes.

## Artifacts
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/final_bench_pair_20260213_032304.log`
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/prefill_extend_bs12_fullpads_w3_t10_20260213_031737.log`
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/prefill_extend_sweep_20260213_012136.log`
