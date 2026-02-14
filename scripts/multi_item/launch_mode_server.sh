#!/usr/bin/env bash
set -euo pipefail

MODE=""
REPO_DIR=""
ARTIFACT_DIR=""
PORT="30000"
MODEL_PATH="/models/Qwen/Qwen3-0.6B"

usage() {
  cat <<USAGE
Usage: $0 --mode <single_item|packed|prefill_extend> --repo-dir <path> --artifact-dir <path> [--port 30000] [--model-path <path>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
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

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR/logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$ARTIFACT_DIR/logs/server_${MODE}_${TS}.log"
PID_PATH="$ARTIFACT_DIR/logs/server_${MODE}.pid"
CMD_PATH="$ARTIFACT_DIR/logs/server_${MODE}_${TS}.cmd"

source ~/work/sglang-jax/.venv/bin/activate
cd "$REPO_DIR"

# Best-effort cleanup to guarantee TPU ownership before restart.
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
for _ in $(seq 1 30); do
  if pgrep -f "sgl_jax.launch_server|sgl_jax.srt.managers.scheduler|sglang::scheduler|sgl_jax.srt.managers.tp_worker|sgl_jax.srt.managers.tokenizer_manager|sgl_jax.srt.managers.detokenizer_manager|sglang-jax::detokenizer" >/dev/null 2>&1; then
    sleep 1
  else
    break
  fi
done

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --port "$PORT"
  --device tpu
  --dtype bfloat16
  --attention-backend fa
  --mem-fraction-static 0.7
  --page-size 64
  --skip-server-warmup
)

MODE_ARGS=()
case "$MODE" in
  single_item)
    MODE_ARGS=(
      --max-running-requests 32
      --chunked-prefill-size 4096
      --precompile-token-paddings 1024 4096
      --precompile-bs-paddings 1 4 8 16 32
    )
    ;;
  packed)
    MODE_ARGS=(
      --max-running-requests 32
      --chunked-prefill-size -1
      --disable-radix-cache
      --max-prefill-tokens 32768
      --precompile-token-paddings 1024 4096 16384
      --precompile-bs-paddings 1 4 8 16 32
      --multi-item-scoring-delimiter 151643
      --multi-item-scoring-chunk-size 32
      --max-multi-item-count 512
      --max-multi-item-seq-len 32768
      --multi-item-mask-impl dense
      --multi-item-segment-fallback-threshold 0
    )
    ;;
  prefill_extend)
    MODE_ARGS=(
      --max-running-requests 12
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
      --multi-item-extend-batch-size 12
      --multi-item-prefill-extend-cache-timeout 60
      --enable-scoring-cache
    )
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

printf '%q ' python -m sgl_jax.launch_server "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" > "$CMD_PATH"
printf '\n' >> "$CMD_PATH"

nohup python -m sgl_jax.launch_server \
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  > "$LOG_PATH" 2>&1 < /dev/null &

echo $! > "$PID_PATH"
PID="$(cat "$PID_PATH")"

READY=0
for i in $(seq 1 420); do
  if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "READY_AFTER_SECONDS=$i"
    break
  fi

  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "SERVER_DIED" >&2
    tail -n 200 "$LOG_PATH" >&2 || true
    exit 1
  fi

  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "SERVER_NOT_READY" >&2
  tail -n 200 "$LOG_PATH" >&2 || true
  exit 1
fi

echo "MODE=$MODE"
echo "PID=$PID"
echo "LOG=$LOG_PATH"
echo "CMD=$CMD_PATH"
echo "PID_FILE=$PID_PATH"
