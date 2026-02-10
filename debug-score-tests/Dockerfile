FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN pip install --no-cache-dir uv

# Copy local config files needed for the build
COPY requirements.txt /app/requirements-debug.txt
COPY run_tests.sh /app/run_tests.sh
RUN chmod +x /app/run_tests.sh

# Clone and setup the repository
# We use a single RUN to ensure the environment is correctly configured
ARG REPO_URL=https://github.com/alexshires/sglang-jax.git
ARG FEATURE_BRANCH=feat/multi-item-scoring

RUN git clone ${REPO_URL} sglang-jax && \
    cd sglang-jax && \
    git checkout main && \
    git checkout ${FEATURE_BRANCH} && \
    cd python && \
    uv pip install --system -e .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    uv pip install --system -r /app/requirements-debug.txt

# Environment setup
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
RUN mkdir -p /tmp/jit_cache /tmp/models

# Use sleep infinity for debugging
ENTRYPOINT ["sleep", "infinity"]
