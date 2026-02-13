# Report: JAX vs PyTorch Multi-Item Comparison (2026-02-11)

| | |
|------------|------|
| **Date** | 2026-02-11 |
| **Status** | **Ready for Execution / Fill-in** |
| **Question** | How well does `sglang-jax` multi-item scoring perform vs frozen PyTorch baseline? |
| **Methodology** | [Investigation](../investigations/jax-vs-pytorch-multi-item-comparison-methodology.md) |
| **Runbook** | [Execution guide](../runbooks/running-jax-vs-pytorch-multi-item-comparison.md) |
| **Execution Status Update** | [2026-02-12 TPU-ready / GPU-blocked note](./jax-vs-pytorch-multi-item-execution-status-2026-02-12.md) |

## Scope

- Compares `sglang-jax` multi-item scoring against PyTorch `sglang`.
- Keeps PyTorch baseline frozen (no implementation changes).
- Reports both:
  - portable view (same client workload contract)
  - best-native view (backend-native settings)

## Known TPU Limitation

**JAX runs must use dense mode only.** The segment mask mode (`--multi-item-mask-impl=segment` or `auto`) fails on TPU with:
```
ValueError: Cannot do int indexing on TPU during kernel lowering
```

All JAX benchmark commands use:
- `--multi-item-mask-impl dense`
- `--multi-item-segment-fallback-threshold 0`

See: [Segment Mask TPU Lowering Issue](../investigations/segment-mask-tpu-lowering-issue.md)

## Workload Contract

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| Query tokens | `2000` |
| Items per request | `500` |
| Tokens per item | `20` |
| Label token IDs | `9454,2753` |
| Client chunk sizes | `1,2,4,8,16,32,64,128,256,500` |
| Warmup runs | `1` |
| Sweep timed runs | `5` |
| Confirm timed runs | `7` |

## Artifact Locations

All raw outputs:
- `reports/artifacts/jax-vs-pytorch-multi-item-20260211/`

Expected files:
- `canonical_workload.json`
- `jax_portable_matrix.json`
- `jax_best_native_matrix.json`
- `pytorch_portable_matrix.json`
- `pytorch_best_native_matrix.json`
- `comparison.json`
- `comparison.md`

## Portable View Results

Source: `reports/artifacts/jax-vs-pytorch-multi-item-20260211/comparison.json` (`portable_view`)

| Chunk | JAX p95 (ms) | JAX items/s | PyTorch p95 (ms) | PyTorch items/s | Max abs diff | Mean abs diff | Correctness Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 128 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 256 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Portable winner:
- **TBD**

## Best-Native View Results

Source: `reports/artifacts/jax-vs-pytorch-multi-item-20260211/comparison.json` (`best_native_view`)

| Backend | Selected chunk | p95 (ms) | items/s | success rate |
|---|---:|---:|---:|---:|
| JAX | TBD | TBD | TBD | TBD |
| PyTorch | TBD | TBD | TBD | TBD |

Best-native correctness gate:
- Max abs diff: **TBD**
- Mean abs diff: **TBD**
- Pass/fail: **TBD**

Best-native winner:
- **TBD**

## Key Findings (Fill After Run)

1. **Where JAX is better:**
- TBD (with concrete % or x-factor deltas).

2. **Where JAX is worse:**
- TBD (with concrete % or x-factor deltas).

3. **Correctness summary:**
- TBD (threshold pass/fail counts, any outlier chunk sizes).

4. **Stability summary:**
- TBD (timeouts/OOM/partial failures by chunk size and backend).

## Final Conclusion (Required)

- Recommendation for current multi-item scoring deployment decision: **TBD**.
- Confidence level: **TBD**.
- Follow-up for JAX optimization priorities: **TBD**.

## Notes

- This report template is intentionally pre-created on 2026-02-11 so results can be dropped in directly from generated JSON.
- Comparison scripts enforce correctness thresholds before selecting winners.
- Interim execution status (quota/capacity blockers and environment readiness) is tracked in `reports/jax-vs-pytorch-multi-item-execution-status-2026-02-12.md`.
- Current TPU known limitation: segment mask path is unstable at kernel lowering; benchmark runs are currently forced to dense mask mode until segment fix lands.
