# Investigation: Segment Mask TPU Kernel Lowering Issue

**Date:** 2026-02-12
**Status:** Resolved on 2026-02-13
**RFC:** RFC-013 Track 2 (Segment Kernel Fix)
**Validation Report:** [Segment Mask TPU Fix Validation (2026-02-13)](../reports/segment-mask-tpu-fix-validation-2026-02-13.md)

## Summary

The original segment mask path failed on TPU with:

```
ValueError: Cannot do int indexing on TPU during kernel lowering
```

Root cause was dynamic integer gather inside Pallas kernel lowering:

```python
row_seg_by_token = multi_item_row_seg_starts_ref[q_token_indices]
```

TPU Pallas lowering does not support this gather form.

## Final Fix Implemented

File: `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py`

1. Removed vector gather from segment path.
2. Loaded per-token segment starts via TPU-safe scalar loads from scalar-prefetch metadata.
3. Rewrote segment mask combine to int32 mask algebra (avoids TPU Mosaic bool truncation failure).
4. Kept dense/custom-mask flow unchanged as fallback.

## Validation Outcome

All required TPU checks passed on v6e-1 (`Qwen/Qwen3-0.6B`):

1. Segment path lowers and runs on TPU.
2. Dense vs segment score parity passes with `max_diff <= 1e-4`.
3. Prefill+extend flow no-regression test passes.

See full commands and outputs in:

- [Segment Mask TPU Fix Validation (2026-02-13)](../reports/segment-mask-tpu-fix-validation-2026-02-13.md)

## Benchmark Impact

From `test/srt/test_bench_multi_item_score.py` A/B run:

- Packed multi-item throughput:
  - Dense: `52.27` items/sec
  - Segment: `105.06` items/sec
  - Segment speedup: `2.01x`
- Prefill+extend throughput:
  - Dense: `518.36` items/sec
  - Segment: `513.06` items/sec
  - Delta: `-1.02%` (within run noise)

## Current Recommendation

- Use segment mode for packed multi-item scoring (or `auto` with current threshold).
- Keep dense mode as fallback (already preserved).
- For prefill+extend, both dense and segment are acceptable.
