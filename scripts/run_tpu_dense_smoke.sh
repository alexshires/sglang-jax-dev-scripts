#!/usr/bin/env bash
set -euo pipefail

# TPU Dense Smoke Test
# Quick validation that dense-mode multi-item scoring works on TPU.
# Runs a single chunk size (32) to verify basic functionality.
#
# Usage:
#   export PROJECT=my-gcp-project
#   export TPU_NAME=my-tpu
#   export TPU_ZONE=us-east5-b
#   ./run_tpu_dense_smoke.sh
#
# Required env vars:
#   PROJECT       - GCP project ID
#   TPU_NAME      - Name of the TPU VM
#   TPU_ZONE      - Zone where TPU is located
#
# Optional env vars:
#   MODEL         - Model to use (default: Qwen/Qwen3-0.6B)
#   ARTIFACT_SUBDIR - Subdirectory for artifacts (default: dense-smoke-<timestamp>)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required env vars
PROJECT="${PROJECT:-}"
TPU_NAME="${TPU_NAME:-}"
TPU_ZONE="${TPU_ZONE:-}"

# Optional env vars with defaults
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_SUBDIR="${ARTIFACT_SUBDIR:-dense-smoke-${RUN_ID}}"

# Workload contract (fixed)
QUERY_TOKENS=2000
NUM_ITEMS=500
ITEM_TOKENS=20
LABEL_TOKEN_IDS="9454,2753"
SMOKE_CHUNK_SIZE=32

# Dense-only enforcement
MASK_IMPL="dense"
SEGMENT_FALLBACK_THRESHOLD=0

LOCAL_ARTIFACT_DIR="${ROOT_DIR}/reports/artifacts/${ARTIFACT_SUBDIR}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Missing required command: $1" >&2
    exit 1
  }
}

require_env() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Required env var $var_name is not set" >&2
    echo "Usage: $var_name=value $0" >&2
    exit 1
  fi
}

show_help() {
  cat <<EOF
TPU Dense Smoke Test

Quick validation that dense-mode multi-item scoring works on TPU.

USAGE:
  export PROJECT=my-gcp-project
  export TPU_NAME=my-tpu
  export TPU_ZONE=us-east5-b
  $0

REQUIRED ENV VARS:
  PROJECT       GCP project ID
  TPU_NAME      Name of the TPU VM
  TPU_ZONE      Zone where TPU is located

OPTIONAL ENV VARS:
  MODEL              Model to use (default: Qwen/Qwen3-0.6B)
  ARTIFACT_SUBDIR    Subdirectory for artifacts (default: dense-smoke-<timestamp>)

WORKLOAD (fixed):
  query_tokens=$QUERY_TOKENS
  num_items=$NUM_ITEMS
  item_tokens=$ITEM_TOKENS
  label_token_ids=$LABEL_TOKEN_IDS
  chunk_size=$SMOKE_CHUNK_SIZE

ENFORCEMENT:
  --multi-item-mask-impl=$MASK_IMPL
  --multi-item-segment-fallback-threshold=$SEGMENT_FALLBACK_THRESHOLD

EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

# Validate requirements
require_cmd gcloud
require_cmd ssh
require_env PROJECT
require_env TPU_NAME
require_env TPU_ZONE

log "=========================================="
log "TPU Dense Smoke Test - START"
log "=========================================="
log "PROJECT=$PROJECT"
log "TPU_NAME=$TPU_NAME"
log "TPU_ZONE=$TPU_ZONE"
log "MODEL=$MODEL"
log "ARTIFACT_SUBDIR=$ARTIFACT_SUBDIR"
log "Workload: query=$QUERY_TOKENS, items=$NUM_ITEMS, item_tokens=$ITEM_TOKENS"
log "Chunk size: $SMOKE_CHUNK_SIZE"
log "Mask impl: $MASK_IMPL (segment threshold: $SEGMENT_FALLBACK_THRESHOLD)"
log "=========================================="

# Set GCP project
gcloud config set project "$PROJECT" >/dev/null

# Create local artifact directory
mkdir -p "$LOCAL_ARTIFACT_DIR"
log "Local artifacts will be written to: $LOCAL_ARTIFACT_DIR"

# Build remote script
REMOTE_SCRIPT=$(cat <<'REMOTE_EOF'
#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

log "Starting dense smoke test on TPU"

cd ~/work/sglang-jax || {
  log "ERROR: sglang-jax not found. Please clone the repo first."
  exit 1
}

source .venv/bin/activate 2>/dev/null || {
  log "ERROR: Virtual environment not found. Please set up the repo first."
  exit 1
}

ARTIFACT_DIR="$HOME/artifacts/__ARTIFACT_SUBDIR__"
mkdir -p "$ARTIFACT_DIR"

log "Running smoke test with chunk_size=__CHUNK_SIZE__"

# Start server in background
log "Starting server..."
python -m sgl_jax.launch_server \
  --model-path "__MODEL__" \
  --port 30000 \
  --multi-item-mask-impl __MASK_IMPL__ \
  --multi-item-segment-fallback-threshold __SEGMENT_THRESHOLD__ \
  --disable-radix-cache \
  > "$ARTIFACT_DIR/server.log" 2>&1 &

SERVER_PID=$!
log "Server PID: $SERVER_PID"

# Wait for server to be ready
log "Waiting for server to be ready..."
for i in {1..120}; do
  if curl -s http://localhost:30000/health >/dev/null 2>&1; then
    log "Server is ready"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    log "ERROR: Server process died"
    cat "$ARTIFACT_DIR/server.log"
    exit 1
  fi
  sleep 2
done

# Run benchmark
log "Running benchmark..."
python test/srt/test_bench_multi_item_score.py \
  --chunk-sizes __CHUNK_SIZE__ \
  --num-items __NUM_ITEMS__ \
  --query-tokens __QUERY_TOKENS__ \
  --item-tokens __ITEM_TOKENS__ \
  --output "$ARTIFACT_DIR/smoke_results.json" \
  2>&1 | tee "$ARTIFACT_DIR/benchmark.log"

# Shutdown server
log "Shutting down server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

log "Smoke test complete. Results in $ARTIFACT_DIR"
ls -la "$ARTIFACT_DIR"
REMOTE_EOF
)

# Substitute variables
REMOTE_SCRIPT="${REMOTE_SCRIPT//__ARTIFACT_SUBDIR__/$ARTIFACT_SUBDIR}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MODEL__/$MODEL}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MASK_IMPL__/$MASK_IMPL}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__SEGMENT_THRESHOLD__/$SEGMENT_FALLBACK_THRESHOLD}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__CHUNK_SIZE__/$SMOKE_CHUNK_SIZE}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__NUM_ITEMS__/$NUM_ITEMS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__QUERY_TOKENS__/$QUERY_TOKENS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__ITEM_TOKENS__/$ITEM_TOKENS}"

# Execute on TPU
log "Executing smoke test on TPU..."
echo "$REMOTE_SCRIPT" | gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT" \
  --zone="$TPU_ZONE" \
  --command="bash -s"

# Collect artifacts
log "Collecting artifacts from TPU..."
gcloud compute tpus tpu-vm scp \
  --project="$PROJECT" \
  --zone="$TPU_ZONE" \
  --recurse \
  "$TPU_NAME:~/artifacts/$ARTIFACT_SUBDIR/*" \
  "$LOCAL_ARTIFACT_DIR/"

log "=========================================="
log "TPU Dense Smoke Test - COMPLETE"
log "=========================================="
log "Artifacts written to: $LOCAL_ARTIFACT_DIR"
ls -la "$LOCAL_ARTIFACT_DIR"
log "=========================================="
