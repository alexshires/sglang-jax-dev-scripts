#!/usr/bin/env bash
# Commands used for the 2026-02-05 run (for reference).

# 1) Create TPU VM
# gcloud compute tpus tpu-vm create sglprof-20260205-184819 \
#   --zone us-east5-b \
#   --project sglang-jax-tests-1769450780 \
#   --accelerator-type v6e-1 \
#   --version v6e-ubuntu-2404

# 2) VM setup (deps + repo + venv + install)
# gcloud compute tpus tpu-vm ssh sglprof-20260205-184819 --zone us-east5-b \
#   --project sglang-jax-tests-1769450780 --command "set -e; \
#   sudo apt-get update; sudo apt-get install -y git python3-venv; \
#   git clone https://github.com/alexshires/sglang-jax.git; \
#   cd sglang-jax; git checkout a18802ac38d209eacea09e040969262926781b80; \
#   python3 -m venv .venv; source .venv/bin/activate; \
#   pip install -U pip; \
#   cd python; \
#   pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
#   pip install tensorboard xprof"

# 3) Start server
# gcloud compute tpus tpu-vm ssh sglprof-20260205-184819 --zone us-east5-b \
#   --project sglang-jax-tests-1769450780 --command "set -e; \
#   cd ~/sglang-jax; source .venv/bin/activate; export HF_HOME=/tmp/hf; \
#   nohup python -m sgl_jax.launch_server \
#     --model-path Qwen/Qwen3-0.6B \
#     --host 0.0.0.0 --port 30000 \
#     --trust-remote-code \
#     --dtype bfloat16 \
#     --tp-size 1 > /tmp/sgl_server.log 2>&1 & \
#   echo \$! > /tmp/sgl_server.pid"

# 4) Start profiling and send /v1/score requests
# curl -s -X POST http://localhost:30000/start_profile \
#   -H 'Content-Type: application/json' \
#   -d '{"output_dir":"/tmp/score_profile_device","num_steps":2,"host_tracer_level":2,"python_tracer_level":1,"device_tracer_level":2}'
# curl -s -X POST http://localhost:30000/v1/score \
#   -H 'Content-Type: application/json' \
#   -d @profiling/runs/20260205T201530Z_score_qwen3_0p6b_tpuv6e1/inputs/score_request.json

# 5) Copy artifacts (device trace)
# gcloud compute tpus tpu-vm scp --recurse \
#   sglprof-20260205-184819:/tmp/score_profile_device \
#   profiling/runs/20260205T201530Z_score_qwen3_0p6b_tpuv6e1/artifacts/raw/device/ \
#   --zone us-east5-b --project sglang-jax-tests-1769450780
