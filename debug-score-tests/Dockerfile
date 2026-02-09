FROM python:3.12

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repository
ARG REPO_URL=https://github.com/alexshires/sglang-jax.git
ARG BRANCH_NAME=feat/scoring-workflow

RUN git clone ${REPO_URL} sglang-jax && \
    cd sglang-jax && \
    git checkout ${BRANCH_NAME}

# Change to sglang-jax directory before installing
WORKDIR /app/sglang-jax

RUN cd python && pip install -e .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

RUN mkdir -p /tmp/jit_cache /tmp/models

ENTRYPOINT ["python", "-u", "-m", "sgl_jax.launch_server", "--host", "0.0.0.0", "--port", "30000", "--dist-init-addr=0.0.0.0:10011", "--nnodes=1", "--tp-size=4", "--node-rank=0", "--mem-fraction-static=0.8", "--max-prefill-tokens=8192", "--download-dir=/tmp", "--dtype=bfloat16", "--skip-server-warmup", "--enable-single-process", "--grammar-backend=none"]
