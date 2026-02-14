#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/kanna/work/blocker-fix-20260214"
GATE="$ROOT/sglang-jax-dev-scripts-clean/reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/phase4_gates"
source "/home/kanna/work/sglang-jax/.venv/bin/activate"
python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 2h --concurrency 2 --arrival-rate 2 --mix small=0.4,medium=0.4,large=0.2 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$GATE" --output-prefix gate_soak_2h_c2_mixed
touch "$GATE/GATE_SOAK_2H_DONE"
