#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/kanna/work/blocker-fix-20260214"
RUNTIME="$ROOT/sglang-jax-clean"
OUT="$ROOT/sglang-jax-dev-scripts-clean/reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/phase2_b2_fix"
source "/home/kanna/work/sglang-jax/.venv/bin/activate"

bash "$ROOT/harness/launch_mode_server.sh" \
  --mode prefill_extend \
  --repo-dir "$RUNTIME" \
  --artifact-dir "$OUT" \
  --port 30000 \
  --model-path /models/Qwen/Qwen3-0.6B > "$OUT/b2_fix_launch_initial.stdout" 2>&1

python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 2m --max-requests 10 --concurrency 1 --arrival-rate 2 \
  --mix small=0.4,medium=0.4,large=0.2 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_warmup

python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 3m --max-requests 60 --concurrency 2 --arrival-rate 2 \
  --mix small=0.4,medium=0.4,large=0.2 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_scheduler_kill_load > "$OUT/b2_fix_scheduler_kill_load.stdout" 2>&1 &
LOAD_PID=$!
sleep 10
SCHED_PID="$(pgrep -f 'sglang::scheduler' | head -n 1 || true)"
if [[ -z "$SCHED_PID" ]]; then
  SCHED_PID="$(pgrep -f 'sgl_jax.srt.managers.scheduler' | head -n 1 || true)"
fi
echo "$SCHED_PID" > "$OUT/b2_fix_scheduler_kill.pid"
if [[ -n "$SCHED_PID" ]]; then
  kill -9 "$SCHED_PID" || true
fi
wait "$LOAD_PID" || true

python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 1m --max-requests 5 --concurrency 1 --arrival-rate 2 \
  --mix small=1 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_post_kill_no_restart > "$OUT/b2_fix_post_kill_no_restart.stdout" 2>&1 || true

bash "$ROOT/harness/launch_mode_server.sh" \
  --mode prefill_extend \
  --repo-dir "$RUNTIME" \
  --artifact-dir "$OUT" \
  --port 30000 \
  --model-path /models/Qwen/Qwen3-0.6B > "$OUT/b2_fix_launch_recovery.stdout" 2>&1
sleep 30
python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 2m --max-requests 10 --concurrency 1 --arrival-rate 2 \
  --mix small=1 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_post_restart_sanity > "$OUT/b2_fix_post_restart_sanity.stdout" 2>&1 || true

SERVER_PID_FILE="$OUT/logs/server_prefill_extend.pid"
SERVER_PID="$(cat "$SERVER_PID_FILE")"
python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 5m --max-requests 1 --concurrency 1 --arrival-rate 1 \
  --mix large=1 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_kill_server_inflight > "$OUT/b2_fix_kill_server_inflight.stdout" 2>&1 &
INFLIGHT_PID=$!
sleep 1
kill -9 "$SERVER_PID" || true
wait "$INFLIGHT_PID" || true

bash "$ROOT/harness/launch_mode_server.sh" \
  --mode prefill_extend \
  --repo-dir "$RUNTIME" \
  --artifact-dir "$OUT" \
  --port 30000 \
  --model-path /models/Qwen/Qwen3-0.6B > "$OUT/b2_fix_launch_after_server_kill.stdout" 2>&1
sleep 30
python "$ROOT/harness/soak_runner.py" \
  --url http://127.0.0.1:30000/v1/score \
  --model /models/Qwen/Qwen3-0.6B \
  --duration 1m --max-requests 5 --concurrency 1 --arrival-rate 2 \
  --mix small=1 \
  --query-tokens 2000 --tokens-per-item 20 --small-items 3 --medium-items 30 --large-items 500 \
  --output-dir "$OUT" --output-prefix b2_fix_post_server_kill_restart_sanity > "$OUT/b2_fix_post_server_kill_restart_sanity.stdout" 2>&1 || true

python3 - <<'PY'
import json
from pathlib import Path
out=Path('/home/kanna/work/blocker-fix-20260214/sglang-jax-dev-scripts-clean/reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/phase2_b2_fix')

def extract(name):
    p=out/name
    if not p.exists():
        return None
    d=json.loads(p.read_text())
    r=d.get('results',{})
    return {
      'requests_total': r.get('requests_total'),
      'requests_ok': r.get('requests_ok'),
      'requests_error': r.get('requests_error'),
      'error_rate': r.get('error_rate'),
      'error_reason_counts': r.get('error_reason_counts',{}),
      'p99_ms': (r.get('latency_ms_ok_only') or {}).get('p99'),
    }

summary={
  'scheduler_kill_load': extract('b2_fix_scheduler_kill_load_summary.json'),
  'post_kill_no_restart': extract('b2_fix_post_kill_no_restart_summary.json'),
  'post_restart_sanity': extract('b2_fix_post_restart_sanity_summary.json'),
  'server_kill_inflight': extract('b2_fix_kill_server_inflight_summary.json'),
  'post_server_kill_restart_sanity': extract('b2_fix_post_server_kill_restart_sanity_summary.json'),
}
(out/'b2_fix_validation_summary.json').write_text(json.dumps(summary, indent=2))
PY

touch "$OUT/B2_VALIDATION_DONE"
