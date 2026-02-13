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
- Timed runs: 10

## Final Stable Result
From one clean run executing both benchmark tests back-to-back:

| Mode | Throughput (items/sec) | Latency per item (ms) | Time for 500 items (sec) |
|---|---:|---:|---:|
| Multi-Item Packed | 52.35 | 19.10 | 9.55 |
| Multi-Item Prefill+Extend | 508.30 | 1.97 | 0.98 |

Observed speedup (Prefill+Extend vs Packed): **9.71x**.

## Benchmark Profile Used
`test/srt/test_bench_multi_item_score.py` defaults were tuned for stable TPU benchmarking:
- `MULTI_ITEM_MASK_IMPL=dense` default in benchmark classes (avoid TPU segment-lowering failure)
- Prefill+extend defaults:
  - `MULTI_ITEM_EXTEND_BATCH_SIZE=12`
  - `MULTI_ITEM_EXTEND_MAX_RUNNING_REQUESTS=12`
  - `MULTI_ITEM_EXTEND_PRECOMPILE_BS_PADDINGS=1..12`
  - `MULTI_ITEM_BENCH_WARMUP_RUNS=3`

## Repro Commands
Run from TPU VM in `~/work/sglang-jax`:

```bash
source .venv/bin/activate

MULTI_ITEM_BENCH_TIMED_RUNS=10 PYTHONPATH=python \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed

MULTI_ITEM_BENCH_TIMED_RUNS=10 PYTHONPATH=python \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend
```

## Notes
- TPU segment mask path is still blocked by TPU lowering (`Cannot do int indexing on TPU`), so dense mask mode is required for reliable benchmark execution.
- Higher chunk profiles (for example `extend_batch_size=16` and above) can show higher short-run throughput but also occasional long-tail pauses; the selected default profile prioritizes repeatable benchmark outcomes.

## Artifacts
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/final_bench_pair_20260213_032304.log`
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/prefill_extend_bs12_fullpads_w3_t10_20260213_031737.log`
- `reports/artifacts/prefill-extend-tpu-v6e1-20260213/prefill_extend_sweep_20260213_012136.log`
