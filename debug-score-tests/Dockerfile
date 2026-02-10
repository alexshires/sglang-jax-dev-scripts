FROM python:3.12

WORKDIR /app

# Step 1: Install system dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends git curl ca-certificates
RUN rm -rf /var/lib/apt/lists/*

# Step 2: Install uv
RUN pip install --no-cache-dir uv

# Step 3: Setup repository
ARG REPO_URL=https://github.com/alexshires/sglang-jax.git
ARG FEATURE_BRANCH=feat/multi-item-scoring-large

# Cache busting to force fresh clone
ADD "https://api.github.com/repos/alexshires/sglang-jax/commits/${FEATURE_BRANCH}" latest_commit
RUN git clone ${REPO_URL} sglang-jax
WORKDIR /app/sglang-jax
RUN git checkout ${FEATURE_BRANCH}

# Step 4: Install sglang-jax core
WORKDIR /app/sglang-jax/python
RUN uv pip install --system -e .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Step 5: Copy and install debug dependencies
WORKDIR /app/sglang-jax
COPY requirements.txt requirements-debug.txt
RUN uv pip install --system -r requirements-debug.txt

# Step 6: Setup test script
COPY run_tests.sh /app/run_tests.sh
RUN chmod +x /app/run_tests.sh

# Step 7: Final environment setup
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
RUN mkdir -p /tmp/jit_cache /tmp/models

# Entrypoint
ENTRYPOINT ["sleep", "infinity"]
