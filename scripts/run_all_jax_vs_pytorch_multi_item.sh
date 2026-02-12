#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for:
# - JAX TPU v6e-1 portable + best-native matrix
# - PyTorch GPU G4 portable + best-native matrix
# - cross-backend comparison + final side-by-side report
#
# This runner is intentionally G4-only (no L4 fallback).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT="${PROJECT:-}"
TPU_ZONE="${TPU_ZONE:-us-east5-b}"
GPU_ZONE="${GPU_ZONE:-us-east5-b}"
TPU_NAME="${TPU_NAME:-mi-tpu-v6e1}"
GPU_NAME="${GPU_NAME:-mi-g4}"

GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-g4-standard-48}"
GPU_IMAGE_PROJECT="${GPU_IMAGE_PROJECT:-deeplearning-platform-release}"
GPU_IMAGE_FAMILY="${GPU_IMAGE_FAMILY:-common-cu128-ubuntu-2204-nvidia-570}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DELIM_TOKEN_ID="${DELIM_TOKEN_ID:-151643}"
JAX_NATIVE_CHUNK_SIZE="${JAX_NATIVE_CHUNK_SIZE:-64}"
JAX_MASK_IMPL="${JAX_MASK_IMPL:-auto}"
JAX_SEGMENT_FALLBACK_THRESHOLD="${JAX_SEGMENT_FALLBACK_THRESHOLD:-32768}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_SUBDIR="${ARTIFACT_SUBDIR:-jax-vs-pytorch-multi-item-${RUN_ID}}"

CREATE_RESOURCES="${CREATE_RESOURCES:-1}"
TEARDOWN_AT_END="${TEARDOWN_AT_END:-0}"

JAX_REPO_URL="${JAX_REPO_URL:-https://github.com/alexshires/sglang-jax.git}"
PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-https://github.com/sgl-project/sglang.git}"

LOCAL_ARTIFACT_DIR="${ROOT_DIR}/reports/artifacts/${ARTIFACT_SUBDIR}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

ensure_project() {
  if [[ -z "$PROJECT" ]]; then
    echo "Set PROJECT env var before running, e.g. PROJECT=my-project" >&2
    exit 1
  fi
  gcloud config set project "$PROJECT" >/dev/null
}

instance_exists() {
  local name="$1"
  local zone="$2"
  gcloud compute instances describe "$name" --zone="$zone" --project="$PROJECT" >/dev/null 2>&1
}

tpu_exists() {
  local name="$1"
  local zone="$2"
  gcloud compute tpus tpu-vm describe "$name" --zone="$zone" --project="$PROJECT" >/dev/null 2>&1
}

create_resources() {
  if [[ "$CREATE_RESOURCES" != "1" ]]; then
    log "Skipping resource creation (CREATE_RESOURCES=$CREATE_RESOURCES)."
    return
  fi

  if tpu_exists "$TPU_NAME" "$TPU_ZONE"; then
    log "TPU already exists: $TPU_NAME ($TPU_ZONE)"
  else
    log "Creating TPU: $TPU_NAME in $TPU_ZONE"
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$TPU_ZONE" \
      --accelerator-type=v6e-1 \
      --version=v2-alpha-tpuv6e
  fi

  if instance_exists "$GPU_NAME" "$GPU_ZONE"; then
    log "GPU VM already exists: $GPU_NAME ($GPU_ZONE)"
  else
    log "Creating GPU VM (G4-only): $GPU_NAME in $GPU_ZONE"
    gcloud compute instances create "$GPU_NAME" \
      --project="$PROJECT" \
      --zone="$GPU_ZONE" \
      --machine-type="$GPU_MACHINE_TYPE" \
      --image-family="$GPU_IMAGE_FAMILY" \
      --image-project="$GPU_IMAGE_PROJECT" \
      --boot-disk-size=200GB \
      --maintenance-policy=TERMINATE
  fi
}

sync_docs_repo() {
  log "Syncing docs repo to TPU VM"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT" --zone="$TPU_ZONE" --command "mkdir -p ~/work"
  gcloud compute tpus tpu-vm scp --project="$PROJECT" --zone="$TPU_ZONE" --recurse "$ROOT_DIR" "$TPU_NAME:~/work/"

  log "Syncing docs repo to GPU VM"
  gcloud compute ssh "$GPU_NAME" --project="$PROJECT" --zone="$GPU_ZONE" --command "mkdir -p ~/work"
  gcloud compute scp --project="$PROJECT" --zone="$GPU_ZONE" --recurse "$ROOT_DIR" "$GPU_NAME:~/work/"
}

run_tpu_stage() {
  local remote_script
  remote_script="$(mktemp)"
  cat > "$remote_script" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update
sudo apt-get install -y git curl python3-venv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

mkdir -p ~/work
cd ~/work

if [[ ! -d sglang-jax ]]; then
  git clone "$JAX_REPO_URL" sglang-jax
fi

cd ~/work/sglang-jax
uv venv --python 3.12 || true
source .venv/bin/activate
uv pip install -e "python[all]"

cd ~/work/sglang-jax-dev-scripts
python -m pip install --upgrade pip
python -m pip install requests transformers
mkdir -p "reports/artifacts/$ARTIFACT_SUBDIR"

python investigations/scripts/generate_canonical_score_workload.py \
  --model "$MODEL" \
  --query-tokens 2000 \
  --num-items 500 \
  --item-tokens 20 \
  --label-token-ids 9454,2753 \
  --delimiter-token-id "$DELIM_TOKEN_ID" \
  --output "reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json"

cd ~/work/sglang-jax
source .venv/bin/activate
pkill -f "python -m sgl_jax.launch_server" || true

nohup python -m sgl_jax.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port 30010 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter "$DELIM_TOKEN_ID" \
  --multi-item-scoring-chunk-size 500 \
  --multi-item-mask-impl "$JAX_MASK_IMPL" \
  --multi-item-segment-fallback-threshold "$JAX_SEGMENT_FALLBACK_THRESHOLD" \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa \
  > /tmp/sgl_jax_portable.log 2>&1 < /dev/null &

for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:30010/health >/dev/null; then
    break
  fi
  sleep 1
done

cd ~/work/sglang-jax-dev-scripts
python investigations/scripts/run_score_matrix_jax.py \
  --base-url http://127.0.0.1:30010 \
  --workload-json "reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json" \
  --evaluation-view portable \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "JAX portable run" \
  --jax-server-chunk-size 500 \
  --jax-mask-impl "$JAX_MASK_IMPL" \
  --jax-segment-fallback-threshold "$JAX_SEGMENT_FALLBACK_THRESHOLD" \
  --output-json "reports/artifacts/$ARTIFACT_SUBDIR/jax_portable_matrix.json" \
  --output-markdown "reports/artifacts/$ARTIFACT_SUBDIR/jax_portable_matrix.md"

cd ~/work/sglang-jax
source .venv/bin/activate
pkill -f "python -m sgl_jax.launch_server" || true

nohup python -m sgl_jax.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port 30011 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 1 \
  --multi-item-scoring-delimiter "$DELIM_TOKEN_ID" \
  --multi-item-scoring-chunk-size "$JAX_NATIVE_CHUNK_SIZE" \
  --multi-item-mask-impl "$JAX_MASK_IMPL" \
  --multi-item-segment-fallback-threshold "$JAX_SEGMENT_FALLBACK_THRESHOLD" \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend fa \
  > /tmp/sgl_jax_native.log 2>&1 < /dev/null &

for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:30011/health >/dev/null; then
    break
  fi
  sleep 1
done

cd ~/work/sglang-jax-dev-scripts
python investigations/scripts/run_score_matrix_jax.py \
  --base-url http://127.0.0.1:30011 \
  --workload-json "reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json" \
  --evaluation-view best_native \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "JAX best-native run" \
  --jax-server-chunk-size "$JAX_NATIVE_CHUNK_SIZE" \
  --jax-mask-impl "$JAX_MASK_IMPL" \
  --jax-segment-fallback-threshold "$JAX_SEGMENT_FALLBACK_THRESHOLD" \
  --output-json "reports/artifacts/$ARTIFACT_SUBDIR/jax_best_native_matrix.json" \
  --output-markdown "reports/artifacts/$ARTIFACT_SUBDIR/jax_best_native_matrix.md"
EOS

  gcloud compute tpus tpu-vm scp --project="$PROJECT" --zone="$TPU_ZONE" "$remote_script" "$TPU_NAME:~/run_mi_tpu.sh"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT" --zone="$TPU_ZONE" \
    --command "chmod +x ~/run_mi_tpu.sh && JAX_REPO_URL='$JAX_REPO_URL' ARTIFACT_SUBDIR='$ARTIFACT_SUBDIR' MODEL='$MODEL' DELIM_TOKEN_ID='$DELIM_TOKEN_ID' JAX_NATIVE_CHUNK_SIZE='$JAX_NATIVE_CHUNK_SIZE' JAX_MASK_IMPL='$JAX_MASK_IMPL' JAX_SEGMENT_FALLBACK_THRESHOLD='$JAX_SEGMENT_FALLBACK_THRESHOLD' bash ~/run_mi_tpu.sh"

  rm -f "$remote_script"
}

copy_canonical_to_gpu() {
  mkdir -p "$LOCAL_ARTIFACT_DIR"
  gcloud compute tpus tpu-vm scp --project="$PROJECT" --zone="$TPU_ZONE" \
    "$TPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json" \
    "$LOCAL_ARTIFACT_DIR/canonical_workload.json"

  gcloud compute scp --project="$PROJECT" --zone="$GPU_ZONE" \
    "$LOCAL_ARTIFACT_DIR/canonical_workload.json" \
    "$GPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json"
}

run_gpu_stage() {
  local remote_script
  remote_script="$(mktemp)"
  cat > "$remote_script" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update
sudo apt-get install -y git curl python3-venv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

mkdir -p ~/work
cd ~/work

if [[ ! -d sglang ]]; then
  git clone "$PYTORCH_REPO_URL" sglang
fi

cd ~/work/sglang
uv venv --python 3.12 || true
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "python"
uv pip install requests transformers

nvidia-smi

cd ~/work/sglang-jax-dev-scripts
mkdir -p "reports/artifacts/$ARTIFACT_SUBDIR"

cd ~/work/sglang
source .venv/bin/activate
pkill -f "python -m sglang.launch_server" || true

nohup python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port 30020 \
  --trust-remote-code \
  --tp-size 1 \
  --multi-item-scoring-delimiter "$DELIM_TOKEN_ID" \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  > /tmp/sgl_torch_portable.log 2>&1 < /dev/null &

for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:30020/health_generate >/dev/null; then
    break
  fi
  sleep 1
done

cd ~/work/sglang-jax-dev-scripts
python investigations/scripts/run_score_matrix_pytorch.py \
  --base-url http://127.0.0.1:30020 \
  --workload-json "reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json" \
  --evaluation-view portable \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "PyTorch portable run (frozen baseline, G4 only)" \
  --output-json "reports/artifacts/$ARTIFACT_SUBDIR/pytorch_portable_matrix.json" \
  --output-markdown "reports/artifacts/$ARTIFACT_SUBDIR/pytorch_portable_matrix.md"

python investigations/scripts/run_score_matrix_pytorch.py \
  --base-url http://127.0.0.1:30020 \
  --workload-json "reports/artifacts/$ARTIFACT_SUBDIR/canonical_workload.json" \
  --evaluation-view best_native \
  --client-chunk-sizes 1,2,4,8,16,32,64,128,256,500 \
  --warmup-runs 1 \
  --timed-runs 5 \
  --timed-runs-confirm 7 \
  --timeout-sec 180 \
  --server-config-note "PyTorch best-native run (frozen baseline, G4 only)" \
  --output-json "reports/artifacts/$ARTIFACT_SUBDIR/pytorch_best_native_matrix.json" \
  --output-markdown "reports/artifacts/$ARTIFACT_SUBDIR/pytorch_best_native_matrix.md"
EOS

  gcloud compute scp --project="$PROJECT" --zone="$GPU_ZONE" "$remote_script" "$GPU_NAME:~/run_mi_gpu.sh"
  gcloud compute ssh "$GPU_NAME" --project="$PROJECT" --zone="$GPU_ZONE" \
    --command "chmod +x ~/run_mi_gpu.sh && PYTORCH_REPO_URL='$PYTORCH_REPO_URL' ARTIFACT_SUBDIR='$ARTIFACT_SUBDIR' MODEL='$MODEL' DELIM_TOKEN_ID='$DELIM_TOKEN_ID' bash ~/run_mi_gpu.sh"

  rm -f "$remote_script"
}

collect_and_compare() {
  mkdir -p "$LOCAL_ARTIFACT_DIR"

  gcloud compute tpus tpu-vm scp --project="$PROJECT" --zone="$TPU_ZONE" \
    "$TPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/jax_portable_matrix.json" \
    "$LOCAL_ARTIFACT_DIR/jax_portable_matrix.json"
  gcloud compute tpus tpu-vm scp --project="$PROJECT" --zone="$TPU_ZONE" \
    "$TPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/jax_best_native_matrix.json" \
    "$LOCAL_ARTIFACT_DIR/jax_best_native_matrix.json"

  gcloud compute scp --project="$PROJECT" --zone="$GPU_ZONE" \
    "$GPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/pytorch_portable_matrix.json" \
    "$LOCAL_ARTIFACT_DIR/pytorch_portable_matrix.json"
  gcloud compute scp --project="$PROJECT" --zone="$GPU_ZONE" \
    "$GPU_NAME:~/work/sglang-jax-dev-scripts/reports/artifacts/$ARTIFACT_SUBDIR/pytorch_best_native_matrix.json" \
    "$LOCAL_ARTIFACT_DIR/pytorch_best_native_matrix.json"

  cd "$ROOT_DIR"
  python3 investigations/scripts/compare_score_matrix_results.py \
    --jax-portable-json "$LOCAL_ARTIFACT_DIR/jax_portable_matrix.json" \
    --pytorch-portable-json "$LOCAL_ARTIFACT_DIR/pytorch_portable_matrix.json" \
    --jax-best-native-json "$LOCAL_ARTIFACT_DIR/jax_best_native_matrix.json" \
    --pytorch-best-native-json "$LOCAL_ARTIFACT_DIR/pytorch_best_native_matrix.json" \
    --correctness-threshold-max-abs 0.02 \
    --correctness-threshold-mean-abs 0.01 \
    --output-json "$LOCAL_ARTIFACT_DIR/comparison.json" \
    --output-markdown "$LOCAL_ARTIFACT_DIR/comparison.md"

  python3 investigations/scripts/render_jax_vs_pytorch_final_report.py \
    --comparison-json "$LOCAL_ARTIFACT_DIR/comparison.json" \
    --output-report "$LOCAL_ARTIFACT_DIR/jax-vs-pytorch-multi-item-comparison-2026-02-11.md" \
    --title-date 2026-02-11

  log "Artifacts generated under: $LOCAL_ARTIFACT_DIR"
}

teardown_resources() {
  if [[ "$TEARDOWN_AT_END" != "1" ]]; then
    return
  fi

  log "Tearing down GPU VM"
  gcloud compute instances delete "$GPU_NAME" --project="$PROJECT" --zone="$GPU_ZONE" -q || true

  log "Tearing down TPU VM"
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT" --zone="$TPU_ZONE" -q || true
}

main() {
  require_cmd gcloud
  require_cmd python3

  ensure_project
  mkdir -p "$LOCAL_ARTIFACT_DIR"

  log "Starting run_all (G4-only, no L4 fallback)."
  log "project=$PROJECT tpu=$TPU_NAME/$TPU_ZONE gpu=$GPU_NAME/$GPU_ZONE"

  create_resources
  sync_docs_repo
  run_tpu_stage
  copy_canonical_to_gpu
  run_gpu_stage
  collect_and_compare
  teardown_resources

  log "Done."
}

main "$@"
