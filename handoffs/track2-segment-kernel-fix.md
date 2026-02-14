# Track 2: Segment Kernel Fix - Completion Handoff

**Date:** 2026-02-13
**Status:** Completed
**Branch:** `feat/segment-mask-tpu-fix-v1`
**Validation Report:** [Segment Mask TPU Fix Validation (2026-02-13)](../reports/segment-mask-tpu-fix-validation-2026-02-13.md)

## Scope Delivered

Implemented a TPU-safe segment mask lowering path for multi-item scoring in `sglang-jax` without changing dense fallback behavior.

## What Changed

### Kernel
- Updated `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py`:
  1. Removed unsupported dynamic gather for `multi_item_row_seg_starts`.
  2. Switched segment-row access to TPU-safe scalar loads.
  3. Replaced bool-heavy combine with int32 mask algebra in segment branch.
  4. Preserved dense/custom-mask path as-is.

### Tests
- Updated `test/srt/test_multi_item_regression.py`:
  1. Added TPU regression class `TestMultiItemSegmentTPURegression` with:
     - `test_segment_mode_runs_on_tpu`
     - `test_segment_matches_dense_scores`
     - `test_segment_prefill_extend_flow_no_regression`
  2. Made dense-vs-segment parity test sequential (one engine at a time) to avoid TPU ownership contention.

## Validation Results (TPU v6e-1, Qwen3-0.6B)

- Segment path TPU run test: passed
- Dense vs segment parity (`max_diff <= 1e-4`): passed
- Prefill+extend no-regression: passed

## Benchmark Snapshot

From `test/srt/test_bench_multi_item_score.py`:

- Packed path (`test_benchmark_multi_item_packed`):
  - Dense: `52.27` items/sec
  - Segment: `105.06` items/sec
  - Improvement: `2.01x`

- Prefill+extend (`test_benchmark_multi_item_prefill_extend`):
  - Dense: `518.36` items/sec
  - Segment: `513.06` items/sec
  - Difference: `-1.02%` (noise-level)

## Recommendation

1. Keep segment enabled for packed multi-item mode (`segment` / `auto`).
2. Keep dense as fallback (already intact).
3. For prefill+extend mode, either mask mode is acceptable.
