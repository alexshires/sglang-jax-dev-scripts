# PR #24 Description Update (Benchmark + Hardening)

## Summary
This PR delivers production hardening for multi-item prefill+extend and validates benchmark performance on TPU v6e-1.

Core improvements:
- Harden prefill+extend cache handle lifecycle in scheduler/tokenizer paths.
- Replace unsafe cache-miss fallback with explicit request abort when `extend_from_cache` handle is missing.
- Add server-side cache TTL eviction and touch-on-access behavior.
- Harden release path with timeout/error handling and non-fatal cleanup behavior.
- Fix communicator in-flight request handling to remove race-prone state transitions.
- Add regression coverage for cache miss, timeout, eviction, and req-slot liveness.

## Benchmark Evidence (TPU v6e-1, Qwen/Qwen3-0.6B)
Workload contract:
- Query tokens: `2000`
- Items per request: `500`
- Tokens per item: `20`

Measured on `test/srt/test_bench_multi_item_score.py`:

| Mode | Throughput (items/sec) | Latency/item (ms) | Time for 500 items (sec) |
|---|---:|---:|---:|
| Single-item sequential | 10.34 | 96.75 | 48.37 |
| Multi-item packed | 52.18 | 19.17 | 9.58 |
| Multi-item prefill+extend | 525.87 | 1.90 | 0.95 |

Speedups:
- Prefill+extend vs packed: **10.08x**
- Prefill+extend vs single-item sequential: **50.86x**
- Packed vs single-item sequential: **5.05x**

## Validation / Test Results
- `test/srt/test_multi_item_prefill_extend_regression.py`: `10 passed`
- `test/srt/test_multi_item_prefill_extend.py`: `2 passed`
- Serving path sanity (`/v1/score`) on TPU:
  - tiny request (`3` items): success
  - medium request (`30` items): success
  - no scheduler crash signature in logs

## Config Notes Used for Stable TPU Benchmarking
- `--multi-item-enable-prefill-extend`
- `--enable-scoring-cache`
- `--multi-item-extend-batch-size 12`
- `--max-running-requests 12`
- `--chunked-prefill-size -1`
- Benchmark env: `MULTI_ITEM_MASK_IMPL=dense`

## Related Report
Detailed report and reproducible commands:
- `reports/multi-item-prefill-extend-tpu-v6e1-benchmark-2026-02-13.md`
