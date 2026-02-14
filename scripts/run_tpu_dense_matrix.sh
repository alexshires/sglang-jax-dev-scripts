#!/usr/bin/env bash
set -euo pipefail

# TPU Dense Matrix Benchmark
# Full matrix sweep across all chunk sizes for dense-mode multi-item scoring.
#
# Usage:
#   export PROJECT=my-gcp-project
#   export TPU_NAME=my-tpu
#   export TPU_ZONE=us-east5-b
#   ./run_tpu_dense_matrix.sh
#
# Required env vars:
#   PROJECT       - GCP project ID
#   TPU_NAME      - Name of the TPU VM
#   TPU_ZONE      - Zone where TPU is located
#
# Optional env vars:
#   MODEL         - Model to use (default: Qwen/Qwen3-0.6B)
#   ARTIFACT_SUBDIR - Subdirectory for artifacts (default: dense-matrix-<timestamp>)
#   CHUNK_SIZES   - Comma-separated chunk sizes (default: 1,2,4,8,16,32,64,128,256,500)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required env vars
PROJECT="${PROJECT:-}"
TPU_NAME="${TPU_NAME:-}"
TPU_ZONE="${TPU_ZONE:-}"

# Optional env vars with defaults
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_SUBDIR="${ARTIFACT_SUBDIR:-dense-matrix-${RUN_ID}}"
CHUNK_SIZES="${CHUNK_SIZES:-1,2,4,8,16,32,64,128,256,500}"

# Workload contract (fixed)
QUERY_TOKENS=2000
NUM_ITEMS=500
ITEM_TOKENS=20
LABEL_TOKEN_IDS="9454,2753"

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
TPU Dense Matrix Benchmark

Full matrix sweep across all chunk sizes for dense-mode multi-item scoring.

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
  ARTIFACT_SUBDIR    Subdirectory for artifacts (default: dense-matrix-<timestamp>)
  CHUNK_SIZES        Comma-separated chunk sizes (default: 1,2,4,8,16,32,64,128,256,500)

WORKLOAD (fixed):
  query_tokens=$QUERY_TOKENS
  num_items=$NUM_ITEMS
  item_tokens=$ITEM_TOKENS
  label_token_ids=$LABEL_TOKEN_IDS

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
log "TPU Dense Matrix Benchmark - START"
log "=========================================="
log "PROJECT=$PROJECT"
log "TPU_NAME=$TPU_NAME"
log "TPU_ZONE=$TPU_ZONE"
log "MODEL=$MODEL"
log "ARTIFACT_SUBDIR=$ARTIFACT_SUBDIR"
log "Workload: query=$QUERY_TOKENS, items=$NUM_ITEMS, item_tokens=$ITEM_TOKENS"
log "Chunk sizes: $CHUNK_SIZES"
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

log "Starting dense matrix benchmark on TPU"

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

# Save run metadata
cat > "$ARTIFACT_DIR/run_metadata.json" <<EOF
{
  "schema_version": "1.0",
  "run_id": "__RUN_ID__",
  "backend": "jax",
  "hardware": "tpu-v6e-1",
  "model": "__MODEL__",
  "workload": {
    "query_tokens": __QUERY_TOKENS__,
    "num_items": __NUM_ITEMS__,
    "item_tokens": __ITEM_TOKENS__,
    "label_token_ids": [9454, 2753]
  },
  "server_config": {
    "multi_item_mask_impl": "__MASK_IMPL__",
    "multi_item_segment_fallback_threshold": __SEGMENT_THRESHOLD__,
    "disable_radix_cache": true
  },
  "chunk_sizes": [__CHUNK_SIZES_ARRAY__],
  "timestamp_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

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

cleanup() {
  log "Cleaning up server process..."
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
log "Waiting for server to be ready..."
for i in {1..180}; do
  if curl -s http://localhost:30000/health >/dev/null 2>&1; then
    log "Server is ready"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    log "ERROR: Server process died"
    cat "$ARTIFACT_DIR/server.log"
    exit 1
  fi
  if [[ $i -eq 180 ]]; then
    log "ERROR: Server did not become ready in time"
    cat "$ARTIFACT_DIR/server.log"
    exit 1
  fi
  sleep 2
done

# Run matrix benchmark
log "Running matrix benchmark across chunk sizes: __CHUNK_SIZES__"

# Initialize results file
echo '{"schema_version": "1.0", "results": []}' > "$ARTIFACT_DIR/matrix_results.json"

IFS=',' read -ra CHUNKS <<< "__CHUNK_SIZES__"
for chunk_size in "${CHUNKS[@]}"; do
  log "Testing chunk_size=$chunk_size"

  RESULT_FILE="$ARTIFACT_DIR/chunk_${chunk_size}.json"

  # Run benchmark for this chunk size
  if python test/srt/test_bench_multi_item_score.py \
    --chunk-sizes "$chunk_size" \
    --num-items __NUM_ITEMS__ \
    --query-tokens __QUERY_TOKENS__ \
    --item-tokens __ITEM_TOKENS__ \
    --output "$RESULT_FILE" \
    2>&1 | tee "$ARTIFACT_DIR/chunk_${chunk_size}.log"; then
    log "Chunk size $chunk_size completed successfully"
  else
    log "WARNING: Chunk size $chunk_size failed (possibly OOM)"
    echo '{"chunk_size": '$chunk_size', "status": "failed", "error": "benchmark_failed"}' > "$RESULT_FILE"
  fi
done

# Combine results
log "Combining results..."
python3 - "$ARTIFACT_DIR" <<'COMBINE_EOF'
import json
import sys
import glob
import os

artifact_dir = sys.argv[1]
results = []

for f in sorted(glob.glob(os.path.join(artifact_dir, "chunk_*.json"))):
    try:
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, dict):
                results.append(data)
            elif isinstance(data, list):
                results.extend(data)
    except Exception as e:
        print(f"Warning: Could not parse {f}: {e}")

# Find best result
best = None
for r in results:
    if r.get("status") != "failed" and r.get("throughput_items_per_sec"):
        if best is None or r["throughput_items_per_sec"] > best["throughput_items_per_sec"]:
            best = r

with open(os.path.join(artifact_dir, "matrix_results.json"), "w") as fp:
    json.dump({
        "schema_version": "1.0",
        "backend": "jax",
        "hardware": "tpu-v6e-1",
        "server_config": {
            "model": "__MODEL__",
            "multi_item_mask_impl": "__MASK_IMPL__",
            "multi_item_segment_fallback_threshold": __SEGMENT_THRESHOLD__
        },
        "workload_ref": "canonical-2000-500-20",
        "results": results,
        "summary": {
            "best_chunk_size": best.get("chunk_size") if best else None,
            "best_throughput": best.get("throughput_items_per_sec") if best else None,
            "total_configs_tested": len(results),
            "successful_configs": len([r for r in results if r.get("status") != "failed"])
        }
    }, fp, indent=2)

print(f"Combined {len(results)} results")
COMBINE_EOF

# Update metadata with end time
python3 -c "
import json
with open('$ARTIFACT_DIR/run_metadata.json') as f:
    data = json.load(f)
data['timestamp_end'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
with open('$ARTIFACT_DIR/run_metadata.json', 'w') as f:
    json.dump(data, f, indent=2)
"

log "Matrix benchmark complete. Results in $ARTIFACT_DIR"
ls -la "$ARTIFACT_DIR"

# Print summary
log "Summary:"
cat "$ARTIFACT_DIR/matrix_results.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Configs tested: {data['summary']['total_configs_tested']}\")
print(f\"  Successful: {data['summary']['successful_configs']}\")
print(f\"  Best chunk size: {data['summary']['best_chunk_size']}\")
print(f\"  Best throughput: {data['summary']['best_throughput']} items/sec\")
"
REMOTE_EOF
)

# Substitute variables
REMOTE_SCRIPT="${REMOTE_SCRIPT//__ARTIFACT_SUBDIR__/$ARTIFACT_SUBDIR}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__RUN_ID__/$RUN_ID}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MODEL__/$MODEL}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__MASK_IMPL__/$MASK_IMPL}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__SEGMENT_THRESHOLD__/$SEGMENT_FALLBACK_THRESHOLD}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__CHUNK_SIZES__/$CHUNK_SIZES}"
# Convert comma-separated to JSON array
CHUNK_SIZES_ARRAY=$(echo "$CHUNK_SIZES" | sed 's/,/, /g')
REMOTE_SCRIPT="${REMOTE_SCRIPT//__CHUNK_SIZES_ARRAY__/$CHUNK_SIZES_ARRAY}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__NUM_ITEMS__/$NUM_ITEMS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__QUERY_TOKENS__/$QUERY_TOKENS}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__ITEM_TOKENS__/$ITEM_TOKENS}"

# Execute on TPU
log "Executing matrix benchmark on TPU..."
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
log "TPU Dense Matrix Benchmark - COMPLETE"
log "=========================================="
log "Artifacts written to: $LOCAL_ARTIFACT_DIR"
ls -la "$LOCAL_ARTIFACT_DIR"
log ""
log "Key files:"
log "  - matrix_results.json   Full results with summary"
log "  - run_metadata.json     Run configuration"
log "  - chunk_*.json          Per-chunk results"
log "  - server.log            Server output"
log "=========================================="
