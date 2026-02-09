#!/usr/bin/env bash
# End-to-end Score API profiling on TPU VM.
# Usage:
#   PROJECT=... ZONE=... TPU_NAME=... ./score_api_tpu_vm.sh all
#   ./score_api_tpu_vm.sh create|setup|start|profile|fetch|cleanup

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="${PROJECT:?PROJECT is required}"
ZONE="${ZONE:-us-east5-b}"
TPU_NAME="${TPU_NAME:-sglprof-$(date +%Y%m%d-%H%M%S)}"
TPU_TYPE="${TPU_TYPE:-v6e-1}"
TPU_IMAGE="${TPU_IMAGE:-v6e-ubuntu-2404}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/score_profile_device}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_TAG_RAW="$(echo "${MODEL}" | sed 's#.*/##' | tr '[:upper:]' '[:lower:]')"
MODEL_TAG="$(echo "${MODEL_TAG_RAW}" | sed 's/[.]/p/g; s/[^a-z0-9]/_/g')"
TPU_TAG="$(echo "${TPU_TYPE}" | tr -d '-' | tr '[:upper:]' '[:lower:]')"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/profiling/runs/${RUN_TIMESTAMP}_score_${MODEL_TAG}_${TPU_TAG}}"
RUN_ID="${RUN_ID:-$(basename "${RUN_DIR}")}"
RUN_TS_BUCKET="$(echo "${RUN_TIMESTAMP}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
RAND_SUFFIX="${RAND_SUFFIX:-$(python3 - <<'PY'\nimport secrets\nprint(secrets.token_hex(2))\nPY\n)}"
GCS_BUCKET="${GCS_BUCKET:-gs://sglang-jax-profiles-${RUN_TS_BUCKET}-score-${RAND_SUFFIX}}"
GCS_PREFIX="${GCS_PREFIX:-${GCS_BUCKET}/${RUN_ID}}"
HOST_TRACER_LEVEL="${HOST_TRACER_LEVEL:-2}"
PYTHON_TRACER_LEVEL="${PYTHON_TRACER_LEVEL:-1}"
DEVICE_TRACER_LEVEL="${DEVICE_TRACER_LEVEL:-2}"
NUM_STEPS="${NUM_STEPS:-2}"
PROFILE_REQUESTS="${PROFILE_REQUESTS:-2}"
PATCH_PATH="${PATCH_PATH:-${ROOT_DIR}/profiling/tools/device_tracer_level.patch}"

function record_inputs() {
  mkdir -p "${RUN_DIR}/inputs"
  cat <<EOF > "${RUN_DIR}/inputs/profile_config.json"
{
  "output_dir": "${OUTPUT_DIR}",
  "num_steps": ${NUM_STEPS},
  "host_tracer_level": ${HOST_TRACER_LEVEL},
  "python_tracer_level": ${PYTHON_TRACER_LEVEL},
  "device_tracer_level": ${DEVICE_TRACER_LEVEL}
}
EOF

  cat <<EOF > "${RUN_DIR}/inputs/score_request.json"
{
  "query": "Is this positive?",
  "items": ["Great product!", "Terrible experience"],
  "label_token_ids": [9834, 902],
  "apply_softmax": true,
  "model": "${MODEL}"
}
EOF

  cat <<EOF > "${RUN_DIR}/inputs/run_metadata.json"
{
  "project": "${PROJECT}",
  "zone": "${ZONE}",
  "tpu_name": "${TPU_NAME}",
  "tpu_type": "${TPU_TYPE}",
  "tpu_image": "${TPU_IMAGE}",
  "model": "${MODEL}",
  "run_dir": "${RUN_DIR}",
  "run_id": "${RUN_ID}",
  "gcs_bucket": "${GCS_BUCKET}",
  "gcs_prefix": "${GCS_PREFIX}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

function ensure_bucket() {
  if gsutil ls -b "${GCS_BUCKET}" >/dev/null 2>&1; then
    return 0
  fi
  gsutil mb -p "${PROJECT}" -l US "${GCS_BUCKET}"
}

function create_vm() {
  gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --zone "${ZONE}" \
    --project "${PROJECT}" \
    --accelerator-type "${TPU_TYPE}" \
    --version "${TPU_IMAGE}"
}

function setup_vm() {
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --command "set -e; \
    sudo apt-get update; sudo apt-get install -y git python3-venv; \
    if [ ! -d ~/sglang-jax ]; then git clone https://github.com/alexshires/sglang-jax.git ~/sglang-jax; fi; \
    cd ~/sglang-jax; \
    git fetch --all; \
    git checkout a18802ac38d209eacea09e040969262926781b80; \
    if [ -f /tmp/device_tracer_level.patch ]; then git apply /tmp/device_tracer_level.patch; fi; \
    python3 -m venv .venv; \
    source .venv/bin/activate; \
    python -m pip install -U pip; \
    cd python; \
    pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    pip install tensorboard xprof"
}

function push_patch() {
  if [ -f "${PATCH_PATH}" ]; then
    gcloud compute tpus tpu-vm scp "${PATCH_PATH}" "${TPU_NAME}":/tmp/device_tracer_level.patch \
      --zone "${ZONE}" --project "${PROJECT}"
  else
    echo "Patch not found at ${PATCH_PATH}. Skipping." >&2
  fi
}

function start_server() {
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --command "set -e; \
    cd ~/sglang-jax; source .venv/bin/activate; export HF_HOME=/tmp/hf; \
    nohup python -m sgl_jax.launch_server \
      --model-path ${MODEL} \
      --host 0.0.0.0 --port 30000 \
      --trust-remote-code \
      --dtype bfloat16 \
      --tp-size 1 > /tmp/sgl_server.log 2>&1 & \
    echo \$! > /tmp/sgl_server.pid"

  # Wait for health
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --command "\
    for i in {1..60}; do \
      if curl -s http://localhost:30000/health > /dev/null; then echo READY; exit 0; fi; \
      echo -n '.'; sleep 10; \
    done; echo TIMEOUT; exit 1"
}

function profile() {
  record_inputs
  # Warmup and profile
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --command "set -e; \
    curl -s -X POST http://localhost:30000/v1/score \
      -H 'Content-Type: application/json' \
      -d '{\"query\": \"Is this positive?\", \"items\": [\"Great product!\", \"Terrible experience\"], \"label_token_ids\": [9834, 902], \"apply_softmax\": true, \"model\": \"${MODEL}\"}' > /tmp/score_warmup.json; \
    curl -s -X POST http://localhost:30000/start_profile \
      -H 'Content-Type: application/json' \
      -d '{\"output_dir\": \"${OUTPUT_DIR}\", \"num_steps\": ${NUM_STEPS}, \"host_tracer_level\": ${HOST_TRACER_LEVEL}, \"python_tracer_level\": ${PYTHON_TRACER_LEVEL}, \"device_tracer_level\": ${DEVICE_TRACER_LEVEL}}' > /tmp/score_start_profile.json; \
    for i in $(seq 1 ${PROFILE_REQUESTS}); do \
      curl -s -X POST http://localhost:30000/v1/score \
        -H 'Content-Type: application/json' \
        -d '{\"query\": \"Is this positive?\", \"items\": [\"Great product!\", \"Terrible experience\"], \"label_token_ids\": [9834, 902], \"apply_softmax\": true, \"model\": \"${MODEL}\"}' > /tmp/score_profile_${i}.json; \
    done; \
    curl -s -X POST http://localhost:30000/stop_profile > /tmp/score_stop_profile.json || true"
}

function upload_raw_from_vm() {
  record_inputs
  ensure_bucket
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --command "set -e; \
    gsutil -m cp -r \"${OUTPUT_DIR}\" \"${GCS_PREFIX}/artifacts/raw/device/\""
}

function upload_local_summary() {
  record_inputs
  ensure_bucket
  if [ -d \"${RUN_DIR}/analysis\" ]; then
    gsutil -m rsync -r \"${RUN_DIR}/analysis\" \"${GCS_PREFIX}/analysis\"
  fi
  if [ -d \"${RUN_DIR}/images\" ]; then
    gsutil -m rsync -r \"${RUN_DIR}/images\" \"${GCS_PREFIX}/images\"
  fi
  if [ -d \"${RUN_DIR}/inputs\" ]; then
    gsutil -m rsync -r \"${RUN_DIR}/inputs\" \"${GCS_PREFIX}/inputs\"
  fi
  if [ -d \"${RUN_DIR}/logs\" ]; then
    gsutil -m rsync -r \"${RUN_DIR}/logs\" \"${GCS_PREFIX}/logs\"
  fi
  if [ -f \"${RUN_DIR}/report.md\" ]; then
    gsutil -m cp \"${RUN_DIR}/report.md\" \"${GCS_PREFIX}/\"
  fi
  if [ -f \"${RUN_DIR}/checksums.txt\" ]; then
    gsutil -m cp \"${RUN_DIR}/checksums.txt\" \"${GCS_PREFIX}/\"
  fi
}

function upload() {
  upload_raw_from_vm
  upload_local_summary
}

function fetch() {
  record_inputs
  mkdir -p "${RUN_DIR}/artifacts/raw/device" "${RUN_DIR}/logs"
  gcloud compute tpus tpu-vm scp --recurse \
    "${TPU_NAME}":"${OUTPUT_DIR}" "${RUN_DIR}/artifacts/raw/device" \
    --zone "${ZONE}" --project "${PROJECT}"
  gcloud compute tpus tpu-vm scp \
    "${TPU_NAME}":/tmp/sgl_server.log "${RUN_DIR}/logs/server.log" \
    --zone "${ZONE}" --project "${PROJECT}"
}

function cleanup() {
  gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" --quiet
}

case "${1:-}" in
  create)
    create_vm
    ;;
  setup)
    push_patch
    setup_vm
    ;;
  start)
    start_server
    ;;
  profile)
    profile
    ;;
  upload)
    upload
    ;;
  fetch)
    fetch
    ;;
  cleanup)
    cleanup
    ;;
  all)
    create_vm
    push_patch
    setup_vm
    start_server
    profile
    upload
    fetch
    ;;
  *)
    echo "Usage: $0 {create|setup|start|profile|upload|fetch|cleanup|all}" >&2
    exit 1
    ;;
 esac
