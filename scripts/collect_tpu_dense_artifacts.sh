#!/usr/bin/env bash
set -euo pipefail

# Collect TPU Dense Artifacts
# Downloads benchmark artifacts from TPU VM to local machine.
# Useful for re-collecting artifacts after a run, or for debugging.
#
# Usage:
#   export PROJECT=my-gcp-project
#   export TPU_NAME=my-tpu
#   export TPU_ZONE=us-east5-b
#   export ARTIFACT_SUBDIR=dense-matrix-20260212T120000Z
#   ./collect_tpu_dense_artifacts.sh
#
# Required env vars:
#   PROJECT         - GCP project ID
#   TPU_NAME        - Name of the TPU VM
#   TPU_ZONE        - Zone where TPU is located
#   ARTIFACT_SUBDIR - Subdirectory to collect (e.g., dense-matrix-20260212T120000Z)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required env vars
PROJECT="${PROJECT:-}"
TPU_NAME="${TPU_NAME:-}"
TPU_ZONE="${TPU_ZONE:-}"
ARTIFACT_SUBDIR="${ARTIFACT_SUBDIR:-}"

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
Collect TPU Dense Artifacts

Downloads benchmark artifacts from TPU VM to local machine.

USAGE:
  export PROJECT=my-gcp-project
  export TPU_NAME=my-tpu
  export TPU_ZONE=us-east5-b
  export ARTIFACT_SUBDIR=dense-matrix-20260212T120000Z
  $0

REQUIRED ENV VARS:
  PROJECT         GCP project ID
  TPU_NAME        Name of the TPU VM
  TPU_ZONE        Zone where TPU is located
  ARTIFACT_SUBDIR Subdirectory to collect (e.g., dense-matrix-20260212T120000Z)

OPTIONS:
  -l, --list      List available artifact directories on TPU (don't download)
  -a, --all       Download all artifact directories
  -h, --help      Show this help

EXAMPLES:
  # List available artifacts
  $0 --list

  # Collect specific artifact directory
  ARTIFACT_SUBDIR=dense-matrix-20260212T120000Z $0

  # Collect all artifacts
  $0 --all

EOF
}

list_artifacts() {
  log "Listing artifact directories on TPU..."
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$TPU_ZONE" \
    --command="ls -la ~/artifacts/ 2>/dev/null || echo 'No artifacts directory found'"
}

collect_all() {
  log "Collecting all artifacts from TPU..."
  mkdir -p "${ROOT_DIR}/reports/artifacts"
  gcloud compute tpus tpu-vm scp \
    --project="$PROJECT" \
    --zone="$TPU_ZONE" \
    --recurse \
    "$TPU_NAME:~/artifacts/*" \
    "${ROOT_DIR}/reports/artifacts/" || {
      log "WARNING: Some artifacts may not have been collected"
    }
  log "Artifacts written to: ${ROOT_DIR}/reports/artifacts/"
  ls -la "${ROOT_DIR}/reports/artifacts/"
}

collect_specific() {
  log "Collecting artifacts for: $ARTIFACT_SUBDIR"
  mkdir -p "$LOCAL_ARTIFACT_DIR"

  # Check if directory exists on TPU
  if ! gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$TPU_ZONE" \
    --command="test -d ~/artifacts/$ARTIFACT_SUBDIR"; then
    log "ERROR: Artifact directory not found on TPU: ~/artifacts/$ARTIFACT_SUBDIR"
    log "Available directories:"
    list_artifacts
    exit 1
  fi

  gcloud compute tpus tpu-vm scp \
    --project="$PROJECT" \
    --zone="$TPU_ZONE" \
    --recurse \
    "$TPU_NAME:~/artifacts/$ARTIFACT_SUBDIR/*" \
    "$LOCAL_ARTIFACT_DIR/"

  log "Artifacts written to: $LOCAL_ARTIFACT_DIR"
  ls -la "$LOCAL_ARTIFACT_DIR"
}

# Parse arguments
LIST_ONLY=0
COLLECT_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -l|--list)
      LIST_ONLY=1
      shift
      ;;
    -a|--all)
      COLLECT_ALL=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

# Validate requirements
require_cmd gcloud
require_env PROJECT
require_env TPU_NAME
require_env TPU_ZONE

# Set GCP project
gcloud config set project "$PROJECT" >/dev/null

log "=========================================="
log "Collect TPU Dense Artifacts"
log "=========================================="
log "PROJECT=$PROJECT"
log "TPU_NAME=$TPU_NAME"
log "TPU_ZONE=$TPU_ZONE"

if [[ $LIST_ONLY -eq 1 ]]; then
  list_artifacts
  exit 0
fi

if [[ $COLLECT_ALL -eq 1 ]]; then
  collect_all
  exit 0
fi

# Specific collection requires ARTIFACT_SUBDIR
require_env ARTIFACT_SUBDIR
log "ARTIFACT_SUBDIR=$ARTIFACT_SUBDIR"
log "=========================================="

collect_specific

log "=========================================="
log "Collection complete"
log "=========================================="
