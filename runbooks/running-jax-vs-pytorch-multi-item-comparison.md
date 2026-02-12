# Runbook: Running JAX vs PyTorch Multi-Item Comparison

| | |
|------------|------|
| **Last Updated** | 2026-02-11 |
| **Maintainer** | Engineering Team |
| **Related** | [Methodology](../investigations/jax-vs-pytorch-multi-item-comparison-methodology.md), [Report Template](../reports/jax-vs-pytorch-multi-item-comparison-2026-02-11.md) |

## Overview

This runbook executes the cross-backend comparison for multi-item scoring with:
- portable view (same client workload contract)
- best-native view (backend-native settings)

Outputs are written under:
- `reports/artifacts/jax-vs-pytorch-multi-item-20260211/`

## One-Command Orchestrator (G4-only)

If you want to run the full flow end-to-end from your local machine:

```bash
cd ~/sglang-jax-dev-scripts

PROJECT="<your-project>" \
TPU_ZONE="us-east5-b" \
GPU_ZONE="us-east5-b" \
TPU_NAME="mi-tpu-v6e1" \
GPU_NAME="mi-g4" \
ARTIFACT_SUBDIR="jax-vs-pytorch-multi-item-$(date -u +%Y%m%dT%H%M%SZ)" \
CREATE_RESOURCES=1 \
TEARDOWN_AT_END=0 \
./scripts/run_all_jax_vs_pytorch_multi_item.sh
```

Notes:
- This runner is **strictly G4-only** for the GPU side.
- There is **no L4 fallback** path.
- If G4 quota/capacity is unavailable, the script fails fast.

## Prerequisites

1. TPU VM with `sglang-jax` environment ready.
2. G4 VM with `sglang` (PyTorch) environment ready.
3. Access to this docs repo on both machines (or synced artifacts).
4. Model access for `Qwen/Qwen3-0.6B`.

## Step 1: Generate Canonical Workload (once)

Run from `sglang-jax-dev-scripts`:

```bash
python investigations/scripts/generate_canonical_score_workload.py \
  --model Qwen/Qwen3-0.6B \
  --query-tokens 2000 \
  --num-items 500 \
  --item-tokens 20 \
  --label-token-ids 9454,2753 \
  --delimiter-token-id 151643 \
  --output reports/artifacts/jax-vs-pytorch-multi-item-20260211/canonical_workload.json
```

## Step 2: Run JAX Portable View (TPU)

### 2.1 Launch JAX server (portable profile)

```bash
cd ~/sglang-jax
source .venv/bin/activate

pkill -f "python -m sgl_jax.launch_server" || true

nohup python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30010 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter 151643 \
  --multi-item-scoring-chunk-size 500 \
  --multi-item-mask-impl auto \
  --multi-item-segment-fallback-threshold 32768 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa \
  > /tmp/sgl_jax_portable.log 2>&1 < /dev/null &

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30010/health >/dev/null; then
    echo "jax portable server healthy"
    break
  fi
  sleep 1
done
```

### 2.2 Run portable matrix

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/run_score_matrix_jax.py \
  --base-url http://127.0.0.1:30010 \
  --workload-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/canonical_workload.json \
  --evaluation-view portable \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "JAX portable run" \
  --jax-server-chunk-size 500 \
  --jax-mask-impl auto \
  --jax-segment-fallback-threshold 32768 \
  --output-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_portable_matrix.json \
  --output-markdown reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_portable_matrix.md
```

## Step 3: Run JAX Best-Native View (TPU)

### 3.1 Launch JAX server (native profile)

Set `--multi-item-scoring-chunk-size` to your current JAX recommended value (example: `64`).

```bash
cd ~/sglang-jax
source .venv/bin/activate

pkill -f "python -m sgl_jax.launch_server" || true

nohup python -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30011 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter 151643 \
  --multi-item-scoring-chunk-size 64 \
  --multi-item-mask-impl auto \
  --multi-item-segment-fallback-threshold 32768 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa \
  > /tmp/sgl_jax_native.log 2>&1 < /dev/null &

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30011/health >/dev/null; then
    echo "jax native server healthy"
    break
  fi
  sleep 1
done
```

### 3.2 Run best-native matrix

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/run_score_matrix_jax.py \
  --base-url http://127.0.0.1:30011 \
  --workload-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/canonical_workload.json \
  --evaluation-view best_native \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "JAX best-native run" \
  --jax-server-chunk-size 64 \
  --jax-mask-impl auto \
  --jax-segment-fallback-threshold 32768 \
  --output-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_best_native_matrix.json \
  --output-markdown reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_best_native_matrix.md
```

## Step 4: Run PyTorch Portable View (G4)

### 4.1 Launch PyTorch server (frozen baseline)

```bash
cd ~/sglang

pkill -f "python -m sglang.launch_server" || true

nohup python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30020 \
  --trust-remote-code \
  --tp-size 1 \
  --multi-item-scoring-delimiter 151643 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  > /tmp/sgl_torch_portable.log 2>&1 < /dev/null &

for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:30020/health_generate >/dev/null; then
    echo "pytorch portable server healthy"
    break
  fi
  sleep 1
done
```

### 4.2 Run portable matrix

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/run_score_matrix_pytorch.py \
  --base-url http://127.0.0.1:30020 \
  --workload-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/canonical_workload.json \
  --evaluation-view portable \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "PyTorch portable run (frozen baseline)" \
  --output-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_portable_matrix.json \
  --output-markdown reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_portable_matrix.md
```

## Step 5: Run PyTorch Best-Native View (G4)

Reuse the same frozen server setup; best-native here is selecting PyTorch best stable client chunk size from the measured matrix.

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/run_score_matrix_pytorch.py \
  --base-url http://127.0.0.1:30020 \
  --workload-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/canonical_workload.json \
  --evaluation-view best_native \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "PyTorch best-native run (frozen baseline)" \
  --output-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_best_native_matrix.json \
  --output-markdown reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_best_native_matrix.md
```

## Step 6: Build Cross-Backend Comparison

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/compare_score_matrix_results.py \
  --jax-portable-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_portable_matrix.json \
  --pytorch-portable-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_portable_matrix.json \
  --jax-best-native-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/jax_best_native_matrix.json \
  --pytorch-best-native-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/pytorch_best_native_matrix.json \
  --correctness-threshold-max-abs 0.02 \
  --correctness-threshold-mean-abs 0.01 \
  --output-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/comparison.json \
  --output-markdown reports/artifacts/jax-vs-pytorch-multi-item-20260211/comparison.md
```

## Step 7: Update Final Report

Render the final side-by-side report directly from comparison output:

```bash
cd ~/sglang-jax-dev-scripts

python investigations/scripts/render_jax_vs_pytorch_final_report.py \
  --comparison-json reports/artifacts/jax-vs-pytorch-multi-item-20260211/comparison.json \
  --output-report reports/jax-vs-pytorch-multi-item-comparison-2026-02-11.md \
  --title-date 2026-02-11
```

The generated report includes:
- side-by-side portable table with throughput/latency deltas
- side-by-side best-native summary with deltas
- explicit findings for where JAX is better/worse
- correctness gate outcomes

## Troubleshooting

1. HTTP 400/500 errors:
- Verify delimiter token and model are consistent.
- Confirm `--disable-radix-cache` and `--chunked-prefill-size -1` are set.

2. Timeouts/OOM:
- Keep run; script records row-level failures and continues.
- Use output failure counts to identify unstable chunk sizes.

3. Missing model/tokenizer:
- Confirm Hugging Face access and local cache permissions.

4. Reproducibility mismatch:
- Regenerate canonical workload and verify checksum recorded in JSON.
