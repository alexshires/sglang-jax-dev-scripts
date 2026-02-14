#!/usr/bin/env bash
set -euo pipefail

FEAT_REPO_DIR="${HOME}/work/sglang-jax-bench-f9d3fb5"
MAIN_REPO_DIR="${HOME}/work/sglang-jax-main"
ARTIFACT_DIR="${HOME}/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase_regression_single_item_main_vs_feat"
PORT="30000"
MODEL_PATH="/models/Qwen/Qwen3-0.6B"
TRIALS="3"
REQUESTS_PER_TRIAL="500"
WARMUP_REQUESTS="5"

usage() {
  cat <<USAGE
Usage: $0 [--feat-repo-dir <path>] [--main-repo-dir <path>] [--artifact-dir <path>]
          [--trials 3] [--requests-per-trial 500] [--warmup-requests 5]
          [--port 30000] [--model-path <path>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --feat-repo-dir)
      FEAT_REPO_DIR="$2"; shift 2 ;;
    --main-repo-dir)
      MAIN_REPO_DIR="$2"; shift 2 ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"; shift 2 ;;
    --trials)
      TRIALS="$2"; shift 2 ;;
    --requests-per-trial)
      REQUESTS_PER_TRIAL="$2"; shift 2 ;;
    --warmup-requests)
      WARMUP_REQUESTS="$2"; shift 2 ;;
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

mkdir -p "$ARTIFACT_DIR"
source "${HOME}/work/sglang-jax/.venv/bin/activate"

if [[ ! -d "$FEAT_REPO_DIR" ]]; then
  echo "feat repo not found: $FEAT_REPO_DIR" >&2
  exit 1
fi

if [[ ! -d "$MAIN_REPO_DIR/.git" ]]; then
  echo "creating main worktree: $MAIN_REPO_DIR"
  git -C "$FEAT_REPO_DIR" fetch origin main
  git -C "$FEAT_REPO_DIR" worktree add "$MAIN_REPO_DIR" origin/main
fi

RUN_SCRIPT="${HOME}/work/sglang-jax-dev-scripts/scripts/multi_item/run_mode_trials.sh"

run_one_branch() {
  local branch_name="$1"
  local repo_dir="$2"
  local out_dir="$ARTIFACT_DIR/$branch_name"
  mkdir -p "$out_dir"

  bash "$RUN_SCRIPT" \
    --mode single_item \
    --repo-dir "$repo_dir" \
    --artifact-dir "$out_dir" \
    --trials "$TRIALS" \
    --requests-per-trial "$REQUESTS_PER_TRIAL" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --concurrency 1 \
    --arrival-rate 100 \
    --large-items 1 \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    | tee "$out_dir/run.log"
}

run_one_branch "main" "$MAIN_REPO_DIR"
run_one_branch "feat" "$FEAT_REPO_DIR"

python3 - "$ARTIFACT_DIR" <<'PY'
import glob
import json
import pathlib
import sys

artifact_root = pathlib.Path(sys.argv[1])


def find_latest_aggregate(base_dir: pathlib.Path) -> pathlib.Path:
    candidates = sorted(base_dir.glob("single_item_*/trials_aggregate.json"))
    if not candidates:
        raise FileNotFoundError(f"No aggregate files in {base_dir}")
    return candidates[-1]

main_agg_path = find_latest_aggregate(artifact_root / "main")
feat_agg_path = find_latest_aggregate(artifact_root / "feat")

main_agg = json.loads(main_agg_path.read_text(encoding="utf-8"))
feat_agg = json.loads(feat_agg_path.read_text(encoding="utf-8"))

main_tput = float(main_agg["throughput_items_per_sec"]["mean"])
feat_tput = float(feat_agg["throughput_items_per_sec"]["mean"])
regression_pct = ((feat_tput - main_tput) / main_tput * 100.0) if main_tput else None

out = {
    "main_aggregate": str(main_agg_path),
    "feat_aggregate": str(feat_agg_path),
    "main_throughput_items_per_sec_mean": main_tput,
    "feat_throughput_items_per_sec_mean": feat_tput,
    "regression_pct_feat_vs_main": regression_pct,
    "criterion_pass_leq_5pct_drop": (regression_pct is not None and regression_pct >= -5.0),
}

out_path = artifact_root / "single_item_regression_summary.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY
