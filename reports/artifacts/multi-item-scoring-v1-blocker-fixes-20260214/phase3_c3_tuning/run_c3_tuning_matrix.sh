#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/kanna/work/blocker-fix-20260214"
RUNTIME="$ROOT/sglang-jax-clean"
OUT="$ROOT/sglang-jax-dev-scripts-clean/reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/phase3_c3_tuning"
mkdir -p "$OUT"
source "/home/kanna/work/sglang-jax/.venv/bin/activate"

kill_existing() {
  for pattern in \
    "python -m sgl_jax.launch_server" \
    "sgl_jax.srt.managers.scheduler" \
    "sglang::scheduler" \
    "sgl_jax.srt.managers.tp_worker" \
    "sgl_jax.srt.managers.tokenizer_manager" \
    "sgl_jax.srt.managers.detokenizer_manager" \
    "sglang-jax::detokenizer"
  do
    pkill -f "$pattern" || true
  done
  sleep 2
}

launch_server() {
  local mr="$1"
  local eb="$2"
  local tag="$3"
  local log="$OUT/${tag}_server.log"
  local cmd="$OUT/${tag}_server.cmd"
  local pidf="$OUT/${tag}_server.pid"

  kill_existing
  cd "$RUNTIME"
  export PYTHONPATH="$RUNTIME/python${PYTHONPATH:+:$PYTHONPATH}"

  local args=(
    --model-path /models/Qwen/Qwen3-0.6B
    --trust-remote-code
    --port 30000
    --device tpu
    --dtype bfloat16
    --attention-backend fa
    --mem-fraction-static 0.7
    --page-size 64
    --skip-server-warmup
    --max-running-requests "$mr"
    --chunked-prefill-size -1
    --max-prefill-tokens 32768
    --precompile-token-paddings 1024 4096 16384
    --precompile-bs-paddings 1 2 3 4 5 6 7 8 9 10 11 12
    --multi-item-scoring-delimiter 151643
    --multi-item-scoring-chunk-size 500
    --max-multi-item-count 512
    --max-multi-item-seq-len 32768
    --multi-item-mask-impl dense
    --multi-item-segment-fallback-threshold 0
    --multi-item-enable-prefill-extend
    --multi-item-extend-batch-size "$eb"
    --multi-item-prefill-extend-cache-timeout 60
    --enable-scoring-cache
  )

  printf '%q ' python -m sgl_jax.launch_server "${args[@]}" > "$cmd"
  printf '\n' >> "$cmd"

  nohup python -m sgl_jax.launch_server "${args[@]}" > "$log" 2>&1 < /dev/null &
  echo $! > "$pidf"

  local ready=0
  for i in $(seq 1 420); do
    if curl -s "http://127.0.0.1:30000/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$(cat "$pidf")" >/dev/null 2>&1; then
      echo "server_died: $tag" >> "$OUT/tuning_errors.log"
      tail -n 200 "$log" >> "$OUT/tuning_errors.log" || true
      return 1
    fi
    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    echo "server_not_ready: $tag" >> "$OUT/tuning_errors.log"
    tail -n 200 "$log" >> "$OUT/tuning_errors.log" || true
    return 1
  fi

  return 0
}

run_soak() {
  local prefix="$1"
  local duration="$2"
  local maxreq="$3"
  local conc="$4"
  local rate="$5"
  python "$ROOT/harness/soak_runner.py" \
    --url http://127.0.0.1:30000/v1/score \
    --model /models/Qwen/Qwen3-0.6B \
    --duration "$duration" \
    --max-requests "$maxreq" \
    --concurrency "$conc" \
    --arrival-rate "$rate" \
    --mix small=0.4,medium=0.4,large=0.2 \
    --query-tokens 2000 \
    --tokens-per-item 20 \
    --small-items 3 \
    --medium-items 30 \
    --large-items 500 \
    --output-dir "$OUT" \
    --output-prefix "$prefix"
}

for mr in 6 8 10 12; do
  for eb in 6 8 10 12; do
    tag="mr${mr}_eb${eb}"
    launch_server "$mr" "$eb" "$tag"

    # warmup to reduce first-run compile bias per candidate
    run_soak "${tag}_warmup" "2m" "10" "1" "2" > "$OUT/${tag}_warmup.stdout" 2>&1 || true

    run_soak "${tag}_c2" "2m" "240" "2" "2" > "$OUT/${tag}_c2.stdout" 2>&1 || true
    run_soak "${tag}_c4" "2m" "480" "4" "4" > "$OUT/${tag}_c4.stdout" 2>&1 || true
  done
done

python3 - <<'PY'
import json
from pathlib import Path
out=Path('/home/kanna/work/blocker-fix-20260214/sglang-jax-dev-scripts-clean/reports/artifacts/multi-item-scoring-v1-blocker-fixes-20260214/phase3_c3_tuning')
rows=[]
for mr in (6,8,10,12):
  for eb in (6,8,10,12):
    tag=f'mr{mr}_eb{eb}'
    c2p=out/f'{tag}_c2_summary.json'
    c4p=out/f'{tag}_c4_summary.json'
    if not c2p.exists() or not c4p.exists():
      continue
    c2=json.loads(c2p.read_text()).get('results',{})
    c4=json.loads(c4p.read_text()).get('results',{})
    c2lat=(c2.get('latency_ms_ok_only') or {})
    c4lat=(c4.get('latency_ms_ok_only') or {})
    row={
      'max_running_requests': mr,
      'extend_batch_size': eb,
      'c2_error_rate': c2.get('error_rate'),
      'c2_p50_ms': c2lat.get('p50'),
      'c2_p95_ms': c2lat.get('p95'),
      'c2_p99_ms': c2lat.get('p99'),
      'c2_items_per_sec': c2.get('throughput_items_per_sec'),
      'c4_error_rate': c4.get('error_rate'),
      'c4_p50_ms': c4lat.get('p50'),
      'c4_p95_ms': c4lat.get('p95'),
      'c4_p99_ms': c4lat.get('p99'),
      'c4_items_per_sec': c4.get('throughput_items_per_sec'),
    }
    # lower is better; penalize errors heavily
    c2_err = row['c2_error_rate'] if row['c2_error_rate'] is not None else 1.0
    c4_err = row['c4_error_rate'] if row['c4_error_rate'] is not None else 1.0
    c2_p99 = row['c2_p99_ms'] if row['c2_p99_ms'] is not None else 1e9
    c4_p99 = row['c4_p99_ms'] if row['c4_p99_ms'] is not None else 1e9
    row['rank_score'] = (c2_err * 100000.0) + (c4_err * 30000.0) + c2_p99 + (0.2 * c4_p99)
    rows.append(row)

rows.sort(key=lambda x: x['rank_score'])
summary={
  'candidates': rows,
  'best_candidate': rows[0] if rows else None,
}
(out/'phase3_tuning_matrix_summary.json').write_text(json.dumps(summary, indent=2))

# csv for quick scan
header=[
  'max_running_requests','extend_batch_size','c2_error_rate','c2_p50_ms','c2_p95_ms','c2_p99_ms','c2_items_per_sec',
  'c4_error_rate','c4_p50_ms','c4_p95_ms','c4_p99_ms','c4_items_per_sec','rank_score'
]
lines=[','.join(header)]
for r in rows:
  lines.append(','.join(str(r.get(h,'')) for h in header))
(out/'phase3_tuning_matrix_summary.csv').write_text('\n'.join(lines))
PY

touch "$OUT/PHASE3_TUNING_DONE"
