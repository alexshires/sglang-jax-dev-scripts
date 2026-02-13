# Runbook: Running Multi-Item Scoring Validation

| | |
|------------|------|
| **Last Updated** | 2026-02-13 |
| **Maintainer** | Engineering Team |
| **Related** | [RFC-008](../rfcs/008-multi-item-scoring.md), [Validation Report](../reports/multi-item-scoring-tpu-validation-2026-02-07.md), [Segment TPU Fix Validation](../reports/segment-mask-tpu-fix-validation-2026-02-13.md) |

## Overview

This runbook reproduces RFC-008 validation on TPU for:

- multi-item correctness/isolation
- multi vs serial throughput
- JAX vs PyTorch parity

It also includes the focused TPU segment-mask fix validation and dense-vs-segment A/B benchmark commands.

The commands use two local score endpoints:

- `:30010` for multi-item mode
- `:30011` for serial baseline mode

## Segment TPU Fix Validation (Focused)

Run from `sglang-jax` checkout on TPU VM:

```bash
source .venv/bin/activate
export PYTHONPATH=python
```

### Regression tests

```bash
# Segment lowering must compile/run on TPU
pytest -q -s test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_mode_runs_on_tpu

# Dense vs segment parity (asserts max_diff <= 1e-4)
pytest -q -s test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_matches_dense_scores

# Segment path with prefill+extend flow
pytest -q -s test/srt/test_multi_item_regression.py::TestMultiItemSegmentTPURegression::test_segment_prefill_extend_flow_no_regression
```

### Dense vs segment A/B benchmark commands

```bash
# Packed path A/B
MULTI_ITEM_MASK_IMPL=dense MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=3 \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed

MULTI_ITEM_MASK_IMPL=segment MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=3 \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemScorePerformance::test_benchmark_multi_item_packed

# Prefill+extend A/B
MULTI_ITEM_MASK_IMPL=dense MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=2 \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend

MULTI_ITEM_MASK_IMPL=segment MULTI_ITEM_BENCH_WARMUP_RUNS=1 MULTI_ITEM_BENCH_TIMED_RUNS=2 \
pytest -q -s test/srt/test_bench_multi_item_score.py::TestMultiItemPrefillExtendPerformance::test_benchmark_multi_item_prefill_extend
```

## Prerequisites

1. TPU VM access via `gcloud`
2. `sglang-jax` checkout on TPU VM
3. Virtual environment with dependencies installed
4. Model access via Hugging Face

Reference environment used for rollout:

- Project: `sglang-jax-tests-1769450780`
- Zone: `us-east5-b`
- TPU type: `v6e-1`

## Step 1: Connect to TPU VM

```bash
TPU_NAME="<your-tpu-vm-name>"
PROJECT="sglang-jax-tests-1769450780"
ZONE="us-east5-b"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE"
```

## Step 2: Launch Multi-Item Server (`:30010`)

On TPU VM:

```bash
cd ~/sglang-jax
source .venv/bin/activate
export HF_HOME=/tmp/hf-multi-item

pkill -f "python -m sgl_jax.launch_server.*30010" || true

nohup python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30010 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter 151643 \
  --multi-item-scoring-chunk-size 2 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa \
  > /tmp/sgl_multi_server.log 2>&1 < /dev/null &

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30010/health >/dev/null; then
    echo "multi server healthy"
    break
  fi
  sleep 1
done
```

## Step 3: Run Multi Endpoint Evaluation

```bash
cd ~/sglang-jax
source .venv/bin/activate
export PYTHONPATH=python
export TS="$(date +%Y%m%d)_tmp"

python scripts/multi_item/evaluate_score_endpoint.py \
  --mode multi \
  --url http://127.0.0.1:30010/v1/score \
  --model Qwen/Qwen3-0.6B \
  --output-json docs/features/reports/multi_item_eval_results_${TS}.json
```

## Step 4: Launch Serial Baseline Server (`:30011`)

```bash
pkill -f "python -m sgl_jax.launch_server.*30011" || true

nohup python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30011 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --attention-backend fa \
  > /tmp/sgl_single_server.log 2>&1 < /dev/null &

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30011/health >/dev/null; then
    echo "single server healthy"
    break
  fi
  sleep 1
done
```

## Step 5: Run Serial Evaluation + Combined Analysis

```bash
cd ~/sglang-jax
source .venv/bin/activate
export PYTHONPATH=python

python scripts/multi_item/evaluate_score_endpoint.py \
  --mode serial \
  --url http://127.0.0.1:30011/v1/score \
  --model Qwen/Qwen3-0.6B \
  --output-json docs/features/reports/serial_score_eval_results_${TS}.json

python scripts/multi_item/combine_multi_item_eval.py \
  --multi-json docs/features/reports/multi_item_eval_results_${TS}.json \
  --serial-json docs/features/reports/serial_score_eval_results_${TS}.json \
  --output-multi-vs-serial-json docs/features/reports/multi_vs_serial_eval_results_${TS}.json \
  --output-jax-torch-parity-json docs/features/reports/jax_torch_parity_results_${TS}.json
```

## Step 6: Check Pass/Fail Thresholds

Use this quick checker:

```bash
# If you are in a new shell, re-export TS first:
# export TS="<timestamp_suffix_used_above>"

python - <<'PY'
import json
from pathlib import Path
import os

base = Path("docs/features/reports")
ts = os.environ["TS"]
multi_vs_serial = json.loads((base / f"multi_vs_serial_eval_results_{ts}.json").read_text())
parity = json.loads((base / f"jax_torch_parity_results_{ts}.json").read_text())

checks = {
    "same_length_isolation_zero": multi_vs_serial["isolation"]["same_length_mutation_max_abs_diff"] == 0.0,
    "changed_length_isolation_zero": multi_vs_serial["isolation"]["changed_length_mutation_max_abs_diff"] == 0.0,
    "delimiter_aligned_equiv_le_0p02": multi_vs_serial["equivalence_query_plus_delimiter"]["max_abs_diff"] <= 0.02,
    "speedup_n32_ge_3x": multi_vs_serial["performance"]["32"]["speedup_vs_serial_p50"] >= 3.0,
    "jax_serial_vs_torch_serial_le_0p02": parity["max_abs_diff_jax_serial_vs_torch_serial"] <= 0.02,
}

print(checks)
print("all_pass=", all(checks.values()))
PY
```

## Step 7: Run Broader Model Matrix

Repeat Steps 2-6 for at least two additional models (validated set used in rollout):

- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`

Record outputs with model-specific suffixes, then summarize in a matrix table.

## Step 8: Cleanup

```bash
pkill -f "python -m sgl_jax.launch_server" || true
```

## Known Compatibility Note

During rollout, tested Qwen2.5 variants hit fused-KV layout mismatch on first scoring request:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Treat this as separate kernel compatibility follow-up, not a blocker for validated Qwen3 rollout.
