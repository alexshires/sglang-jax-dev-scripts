#!/usr/bin/env bash
set -euo pipefail

MODE=""
REPO_DIR=""
ARTIFACT_DIR=""
PORT="30000"
MODEL_PATH="/models/Qwen/Qwen3-0.6B"
TRIALS="3"
REQUESTS_PER_TRIAL="20"
WARMUP_REQUESTS="5"
CONCURRENCY="1"
ARRIVAL_RATE="100"
LARGE_ITEMS="500"

usage() {
  cat <<USAGE
Usage: $0 --mode <single_item|packed|prefill_extend> --repo-dir <path> --artifact-dir <path>
          [--trials 3] [--requests-per-trial 20] [--warmup-requests 5]
          [--concurrency 1] [--arrival-rate 100] [--large-items 500]
          [--port 30000] [--model-path path]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"; shift 2 ;;
    --trials)
      TRIALS="$2"; shift 2 ;;
    --requests-per-trial)
      REQUESTS_PER_TRIAL="$2"; shift 2 ;;
    --warmup-requests)
      WARMUP_REQUESTS="$2"; shift 2 ;;
    --concurrency)
      CONCURRENCY="$2"; shift 2 ;;
    --arrival-rate)
      ARRIVAL_RATE="$2"; shift 2 ;;
    --large-items)
      LARGE_ITEMS="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" || -z "$REPO_DIR" || -z "$ARTIFACT_DIR" ]]; then
  usage
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MODE_DIR="$ARTIFACT_DIR/${MODE}_${TS}"
mkdir -p "$MODE_DIR"

source ~/work/sglang-jax/.venv/bin/activate

LAUNCH_SCRIPT="$HOME/work/sglang-jax-dev-scripts/scripts/multi_item/launch_mode_server.sh"
SOAK_SCRIPT="$HOME/work/sglang-jax-dev-scripts/scripts/multi_item/soak_runner.py"

bash "$LAUNCH_SCRIPT" \
  --mode "$MODE" \
  --repo-dir "$REPO_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  | tee "$MODE_DIR/launch.stdout"

BASE_URL="http://127.0.0.1:${PORT}/v1/score"

python "$SOAK_SCRIPT" \
  --url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --duration 10m \
  --concurrency "$CONCURRENCY" \
  --arrival-rate "$ARRIVAL_RATE" \
  --max-requests "$WARMUP_REQUESTS" \
  --mix large=1 \
    --output-dir "$MODE_DIR" \
    --output-prefix "warmup" \
    --query-tokens 2000 \
    --tokens-per-item 20 \
    --small-items 3 \
    --medium-items 30 \
    --large-items "$LARGE_ITEMS" \
    | tee "$MODE_DIR/warmup.stdout"

sleep 30

for idx in $(seq 1 "$TRIALS"); do
  trial_tag=$(printf 'trial_%02d' "$idx")
  python "$SOAK_SCRIPT" \
    --url "$BASE_URL" \
    --model "$MODEL_PATH" \
    --duration 20m \
    --concurrency "$CONCURRENCY" \
    --arrival-rate "$ARRIVAL_RATE" \
    --max-requests "$REQUESTS_PER_TRIAL" \
    --mix large=1 \
    --output-dir "$MODE_DIR" \
    --output-prefix "$trial_tag" \
    --query-tokens 2000 \
    --tokens-per-item 20 \
    --small-items 3 \
    --medium-items 30 \
    --large-items "$LARGE_ITEMS" \
    | tee "$MODE_DIR/${trial_tag}.stdout"
done

python3 - "$MODE_DIR" "$MODE" <<'PY'
import glob
import json
import math
import os
import statistics
import sys

mode_dir, mode = sys.argv[1:3]
summary_paths = sorted(glob.glob(os.path.join(mode_dir, "trial_*_summary.json")))
if not summary_paths:
    raise SystemExit("no trial summary files found")

trials = []
for path in summary_paths:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = data["results"]
    trials.append(
        {
            "summary_json": path,
            "error_rate": result["error_rate"],
            "throughput_items_per_sec": result["throughput_items_per_sec"],
            "p50_latency_ms": result["latency_ms_ok_only"]["p50"],
            "p95_latency_ms": result["latency_ms_ok_only"]["p95"],
            "p99_latency_ms": result["latency_ms_ok_only"]["p99"],
            "requests_total": result["requests_total"],
            "requests_error": result["requests_error"],
        }
    )


def stat(values):
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }

throughputs = [t["throughput_items_per_sec"] for t in trials]
p50s = [t["p50_latency_ms"] for t in trials]
p95s = [t["p95_latency_ms"] for t in trials]
p99s = [t["p99_latency_ms"] for t in trials]
error_rates = [t["error_rate"] for t in trials]

aggregate = {
    "mode": mode,
    "trial_count": len(trials),
    "trials": trials,
    "throughput_items_per_sec": stat(throughputs),
    "p50_latency_ms": stat(p50s),
    "p95_latency_ms": stat(p95s),
    "p99_latency_ms": stat(p99s),
    "error_rate": stat(error_rates),
}

out_path = os.path.join(mode_dir, "trials_aggregate.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(aggregate, f, indent=2)

print(json.dumps({"aggregate_json": out_path, "mode": mode}))
PY
