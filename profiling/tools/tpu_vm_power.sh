#!/usr/bin/env bash
# Manage power state for an existing TPU VM used by profiling runs.
#
# Usage examples:
#   ./profiling/tools/tpu_vm_power.sh status
#   ./profiling/tools/tpu_vm_power.sh up
#   ./profiling/tools/tpu_vm_power.sh down
#
# Override target explicitly:
#   PROJECT=... ZONE=us-east5-a TPU_NAME=sglprof-... ./profiling/tools/tpu_vm_power.sh up

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

function usage() {
  cat <<'EOF'
Usage: tpu_vm_power.sh <status|up|down|restart>

Environment variables (optional):
  PROJECT   GCP project id
  ZONE      TPU zone (e.g. us-east5-a)
  TPU_NAME  TPU VM node name

If omitted, defaults are loaded from the latest:
  profiling/runs/*/inputs/run_metadata.json
EOF
}

function latest_metadata_path() {
  ls -1dt "${ROOT_DIR}"/profiling/runs/*/inputs/run_metadata.json 2>/dev/null | head -n1 || true
}

function load_defaults_from_latest_run() {
  local meta_path
  meta_path="$(latest_metadata_path)"
  if [[ -z "${meta_path}" ]]; then
    return 0
  fi

  # Parse only fields we need without introducing jq dependency.
  local parsed
  parsed="$(python3 - <<'PY' "${meta_path}"
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
print(meta.get("project", ""))
print(meta.get("zone", ""))
print(meta.get("tpu_name", ""))
PY
)"
  local meta_project meta_zone meta_name
  meta_project="$(echo "${parsed}" | sed -n '1p')"
  meta_zone="$(echo "${parsed}" | sed -n '2p')"
  meta_name="$(echo "${parsed}" | sed -n '3p')"

  PROJECT="${PROJECT:-${meta_project}}"
  ZONE="${ZONE:-${meta_zone}}"
  TPU_NAME="${TPU_NAME:-${meta_name}}"
}

function load_defaults_from_gcloud() {
  if [[ -z "${PROJECT:-}" ]]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  fi
}

function ensure_required() {
  if [[ -z "${PROJECT:-}" || -z "${ZONE:-}" || -z "${TPU_NAME:-}" ]]; then
    echo "Missing target details. Set PROJECT, ZONE, TPU_NAME or ensure latest run metadata exists." >&2
    usage
    exit 1
  fi
}

function get_state() {
  gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --format='value(state)' 2>/dev/null || true
}

function show_status() {
  gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --format='yaml(name,acceleratorType,runtimeVersion,state,networkEndpoints[0].ipAddress)'
}

function power_up() {
  local state
  state="$(get_state)"
  if [[ "${state}" == "READY" ]]; then
    echo "TPU VM is already READY: ${TPU_NAME} (${ZONE})"
    show_status
    return 0
  fi
  gcloud compute tpus tpu-vm start "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}"
  show_status
}

function power_down() {
  local state
  state="$(get_state)"
  if [[ "${state}" == "STOPPED" ]]; then
    echo "TPU VM is already STOPPED: ${TPU_NAME} (${ZONE})"
    show_status
    return 0
  fi
  gcloud compute tpus tpu-vm stop "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}"
  show_status
}

ACTION="${1:-}"
if [[ -z "${ACTION}" ]]; then
  usage
  exit 1
fi

load_defaults_from_latest_run
load_defaults_from_gcloud
ensure_required

case "${ACTION}" in
  status)
    show_status
    ;;
  up)
    power_up
    ;;
  down)
    power_down
    ;;
  restart)
    power_down
    power_up
    ;;
  *)
    usage
    exit 1
    ;;
esac
