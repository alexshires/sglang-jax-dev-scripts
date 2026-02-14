#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/work/sglang-jax-bench-f9d3fb5"
ARTIFACT_DIR="${HOME}/artifacts/multi-item-scoring-v1-battletest-20260213-20260217/phase4"
PORT="30000"
MODEL_PATH="/models/Qwen/Qwen3-0.6B"

usage() {
  cat <<USAGE
Usage: $0 [--repo-dir <path>] [--artifact-dir <path>] [--port <30000>] [--model-path <path>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"; shift 2 ;;
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

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"

source "${HOME}/work/sglang-jax/.venv/bin/activate"

LAUNCH_SCRIPT="${HOME}/work/sglang-jax-dev-scripts/scripts/multi_item/launch_mode_server.sh"
SOAK_SCRIPT="${HOME}/work/sglang-jax-dev-scripts/scripts/multi_item/soak_runner.py"

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
  chmod +x "$LAUNCH_SCRIPT"
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

launch_server() {
  local tag="$1"
  log "launch prefill_extend server ($tag)"
  bash "$LAUNCH_SCRIPT" \
    --mode prefill_extend \
    --repo-dir "$REPO_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    | tee "$ARTIFACT_DIR/${tag}_launch.stdout"

  SERVER_PID_FILE="$ARTIFACT_DIR/logs/server_prefill_extend.pid"
  if [[ ! -f "$SERVER_PID_FILE" ]]; then
    echo "Missing server pid file: $SERVER_PID_FILE" >&2
    exit 1
  fi
  SERVER_PID="$(cat "$SERVER_PID_FILE")"
  log "server pid=$SERVER_PID"
}

run_soak() {
  local prefix="$1"
  local duration="$2"
  local max_requests="$3"
  local concurrency="$4"
  local arrival_rate="$5"
  local mix="$6"

  python "$SOAK_SCRIPT" \
    --url "http://127.0.0.1:${PORT}/v1/score" \
    --model "$MODEL_PATH" \
    --duration "$duration" \
    --max-requests "$max_requests" \
    --concurrency "$concurrency" \
    --arrival-rate "$arrival_rate" \
    --mix "$mix" \
    --small-items 3 \
    --medium-items 30 \
    --large-items 500 \
    --query-tokens 2000 \
    --tokens-per-item 20 \
    --output-dir "$ARTIFACT_DIR" \
    --output-prefix "$prefix"
}

run_quick_sanity() {
  local prefix="$1"
  run_soak "$prefix" "2m" "5" "1" "2" "small=1"
}

# Step 1: kill server during in-flight large request, then recover.
launch_server "fi1"
run_soak "fi1_warmup" "2m" "2" "1" "1" "large=1" >"$ARTIFACT_DIR/fi1_warmup.stdout" 2>&1 || true

log "step1: start one large request and kill server during flight"
run_soak "fi1_inflight_kill" "5m" "1" "1" "1" "large=1" >"$ARTIFACT_DIR/fi1_inflight_kill.stdout" 2>&1 &
FI1_PID=$!
sleep 0.2
kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
wait "$FI1_PID" || true

launch_server "fi1_recovery"
run_quick_sanity "fi1_recovery" >"$ARTIFACT_DIR/fi1_recovery.stdout" 2>&1 || true

# Step 2: kill scheduler subprocess under mixed load, then recover.
log "step2: start mixed load and kill scheduler subprocess"
run_soak "fi2_scheduler_kill_load" "2m" "80" "2" "0.8" "small=0.4,medium=0.4,large=0.2" >"$ARTIFACT_DIR/fi2_scheduler_kill_load.stdout" 2>&1 &
FI2_PID=$!
sleep 10
SCHED_PID="$(pgrep -f 'sglang::scheduler' | head -n 1 || true)"
if [[ -z "$SCHED_PID" ]]; then
  SCHED_PID="$(pgrep -f 'sgl_jax.srt.managers.scheduler' | head -n 1 || true)"
fi
if [[ -n "$SCHED_PID" ]]; then
  log "killing scheduler pid=$SCHED_PID"
  kill -9 "$SCHED_PID" >/dev/null 2>&1 || true
else
  log "scheduler pid not found"
fi
wait "$FI2_PID" || true

launch_server "fi2_recovery"
sleep 30
run_quick_sanity "fi2_recovery" >"$ARTIFACT_DIR/fi2_recovery.stdout" 2>&1 || true

# Step 3: restart server during mixed load and verify recovery.
log "step3: restart server during mixed load"
run_soak "fi3_restart_during_load" "2m" "80" "2" "0.8" "small=0.4,medium=0.4,large=0.2" >"$ARTIFACT_DIR/fi3_restart_during_load.stdout" 2>&1 &
FI3_PID=$!
sleep 15
kill -9 "$SERVER_PID" >/dev/null 2>&1 || true

launch_server "fi3_post_restart"
sleep 30
run_soak "fi3_post_restart_sanity" "2m" "10" "1" "2" "small=1" >"$ARTIFACT_DIR/fi3_post_restart_sanity.stdout" 2>&1 || true
wait "$FI3_PID" || true

# Step 4: missing cache handle explicit error test.
log "step4: run missing cache-handle regression test"
cd "$REPO_DIR"
PYTHONPATH=python pytest -q -s \
  test/srt/test_multi_item_prefill_extend_regression.py::test_resolve_extend_from_cache_missing_handle_returns_error \
  2>&1 | tee "$ARTIFACT_DIR/fi4_missing_cache_handle_pytest.log"

python3 - "$ARTIFACT_DIR" <<'PY'
import json
import pathlib
import re
import sys

artifact_dir = pathlib.Path(sys.argv[1])

summary_names = [
    "fi1_inflight_kill_summary.json",
    "fi1_recovery_summary.json",
    "fi2_scheduler_kill_load_summary.json",
    "fi2_recovery_summary.json",
    "fi3_restart_during_load_summary.json",
    "fi3_post_restart_sanity_summary.json",
]

summaries = {}
for name in summary_names:
    path = artifact_dir / name
    if not path.exists():
        summaries[name] = None
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("results", {})
    summaries[name] = {
        "requests_total": res.get("requests_total"),
        "requests_ok": res.get("requests_ok"),
        "requests_error": res.get("requests_error"),
        "error_rate": res.get("error_rate"),
        "error_reason_counts": res.get("error_reason_counts", {}),
        "throughput_items_per_sec": res.get("throughput_items_per_sec"),
        "p99_latency_ms_ok": (res.get("latency_ms_ok_only") or {}).get("p99"),
    }

pytest_log = artifact_dir / "fi4_missing_cache_handle_pytest.log"
missing_cache_handle_test_passed = False
if pytest_log.exists():
    text = pytest_log.read_text(encoding="utf-8", errors="replace")
    missing_cache_handle_test_passed = bool(re.search(r"\b1 passed\b", text))

fi1 = summaries.get("fi1_inflight_kill_summary.json") or {}
fi1_recovery = summaries.get("fi1_recovery_summary.json") or {}
fi2 = summaries.get("fi2_scheduler_kill_load_summary.json") or {}
fi2_recovery = summaries.get("fi2_recovery_summary.json") or {}
fi3 = summaries.get("fi3_restart_during_load_summary.json") or {}
fi3_recovery = summaries.get("fi3_post_restart_sanity_summary.json") or {}

out = {
    "artifact_dir": str(artifact_dir),
    "step_results": {
        "kill_server_inflight": {
            "load_summary": fi1,
            "post_restart_summary": fi1_recovery,
            "client_failure_observed": (fi1.get("requests_error", 0) or 0) > 0,
            "recovery_ok": (fi1_recovery.get("error_rate") == 0),
        },
        "kill_scheduler_subprocess": {
            "load_summary": fi2,
            "post_restart_summary": fi2_recovery,
            "failure_observed": (fi2.get("requests_error", 0) or 0) > 0,
            "recovery_ok": (fi2_recovery.get("error_rate") == 0),
        },
        "restart_during_mixed_load": {
            "load_summary": fi3,
            "post_restart_summary": fi3_recovery,
            "failure_observed": (fi3.get("requests_error", 0) or 0) > 0,
            "recovery_ok": (fi3_recovery.get("error_rate") == 0),
        },
        "missing_cache_handle": {
            "pytest_log": str(pytest_log),
            "test_passed": missing_cache_handle_test_passed,
        },
    },
    "summaries": summaries,
}

out_path = artifact_dir / "phase4_summary.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({"phase4_summary_json": str(out_path)}, indent=2))
PY

log "phase4 complete"
