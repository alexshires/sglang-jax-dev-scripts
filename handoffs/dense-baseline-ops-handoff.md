# Handoff: Dense Baseline Ops

**Date:** 2026-02-12
**Status:** Complete
**Branch:** `feat/multi-item-scoring-v1-benchmark-ops`

## What Changed

Added dense-only benchmark operations layer for stable TPU baseline collection:

### New Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_tpu_dense_smoke.sh` | Quick smoke test (single chunk size=32) |
| `scripts/run_tpu_dense_matrix.sh` | Full matrix sweep (10 chunk sizes) |
| `scripts/collect_tpu_dense_artifacts.sh` | Download artifacts from TPU |

### New Validator

| File | Purpose |
|------|---------|
| `investigations/scripts/validate_score_artifacts.py` | Schema validation for JSON artifacts |

### Updated Files

| File | Change |
|------|--------|
| `scripts/run_all_jax_vs_pytorch_multi_item.sh` | Added TPU limitation comment |
| `runbooks/running-jax-vs-pytorch-multi-item-comparison.md` | Added dense baseline section |
| `reports/jax-vs-pytorch-multi-item-comparison-2026-02-11.md` | Added TPU limitation note |
| `INDEX.md` | Added Scripts section with new files |

## Command Sequences

### Quick Smoke Test

```bash
export PROJECT=my-gcp-project
export TPU_NAME=my-tpu
export TPU_ZONE=us-east5-b

./scripts/run_tpu_dense_smoke.sh
```

### Full Matrix Sweep

```bash
export PROJECT=my-gcp-project
export TPU_NAME=my-tpu
export TPU_ZONE=us-east5-b

./scripts/run_tpu_dense_matrix.sh
```

### Collect Artifacts (after run)

```bash
# List available
./scripts/collect_tpu_dense_artifacts.sh --list

# Collect specific
ARTIFACT_SUBDIR=dense-matrix-20260212T120000Z ./scripts/collect_tpu_dense_artifacts.sh

# Collect all
./scripts/collect_tpu_dense_artifacts.sh --all
```

### Validate Artifacts

```bash
# Validate matrix results
python investigations/scripts/validate_score_artifacts.py \
  --matrix reports/artifacts/dense-matrix-*/matrix_results.json

# Validate all in directory (verbose)
python investigations/scripts/validate_score_artifacts.py \
  --all reports/artifacts/dense-matrix-*/ -v

# JSON output
python investigations/scripts/validate_score_artifacts.py \
  --all reports/artifacts/ --json > validation_report.json
```

## Artifact Directory Structure

After a matrix run, artifacts are stored in:

```
reports/artifacts/dense-matrix-<timestamp>/
├── run_metadata.json       # Run configuration
├── matrix_results.json     # Combined results with summary
├── chunk_1.json           # Per-chunk results
├── chunk_2.json
├── chunk_4.json
├── ...
├── chunk_500.json
├── server.log             # Server output
└── chunk_*.log            # Per-chunk benchmark logs
```

### Key Files

| File | Contents |
|------|----------|
| `matrix_results.json` | Combined results, summary with best chunk size |
| `run_metadata.json` | Model, workload, server config, timestamps |
| `chunk_*.json` | Individual chunk size results |
| `server.log` | Server stdout/stderr |

### matrix_results.json Schema

```json
{
  "schema_version": "1.0",
  "backend": "jax",
  "hardware": "tpu-v6e-1",
  "server_config": {
    "model": "Qwen/Qwen3-0.6B",
    "multi_item_mask_impl": "dense",
    "multi_item_segment_fallback_threshold": 0
  },
  "workload_ref": "canonical-2000-500-20",
  "results": [...],
  "summary": {
    "best_chunk_size": 64,
    "best_throughput": 79.6,
    "total_configs_tested": 10,
    "successful_configs": 8
  }
}
```

## Known Limitations

1. **Dense mode only** - Segment/auto modes fail on TPU due to kernel lowering issue
2. **Large chunks may OOM** - chunk_size ≥128 may fail depending on model size
3. **Requires pre-setup** - TPU must have sglang-jax repo cloned and venv set up

## Next Steps

1. **Run baseline sweep** - Execute matrix on TPU v6e-1 with Qwen3-0.6B
2. **Validate artifacts** - Use validator to confirm schema compliance
3. **Compare with PyTorch** - Use full orchestrator for cross-backend comparison
4. **Track 2 (Segment Fix)** - Fix kernel issue to enable segment mode
5. **Track 3 (Prefill+Extend)** - Implement for bigger performance gains

## Related Documents

- [RFC-013: Multi-Item Scoring v1.0 Optimization](../rfcs/013-multi-item-scoring-v1-optimization.md)
- [Investigation: Segment Mask TPU Lowering Issue](../investigations/segment-mask-tpu-lowering-issue.md)
- [Runbook: Running JAX vs PyTorch Comparison](../runbooks/running-jax-vs-pytorch-multi-item-comparison.md)
