FROM python:3.12

WORKDIR /app

# Copy the project files
COPY . .

# Install the sglang-jax package with TPU support
# Assuming the build context is the root of the repo (sglang/) or includes sglang-jax/
# If the context is `sglang-scripts`, we might need to adjust. 
# However, usually Cloud Build triggers from root.
# Let's assume the user will run the build from `sglang-jax/` or `sglang/`.
# Given the user asked for files in `sglang-scripts`, but `sglang-jax` is a sibling.
# We should assume the build context will be `sglang-jax` or `sglang`.
# The existing Dockerfile in `sglang-jax` does `COPY . .` then `cd python`.
# So it expects to be in `sglang-jax`.

# If we use this Dockerfile from `sglang-scripts`, we need to ensure we can access `sglang-jax`.
# Best practice: Build from `sglang` root or `sglang-jax` root.
# Let's write this Dockerfile to be compatible with running from `sglang-jax` root, essentially a copy of the other one.
# OR, if running from `sglang/` root:
# COPY sglang-jax/ .
# Let's stick to the existing pattern but put it in the requested folder.
# Actually, if it's in `sglang-scripts`, it might be intended to build `sglang-scripts`?
# No, user said "build the image remotely", implying the SGLang JAX image.
# I will make it assume the build context is `sglang-jax` directory.

COPY . .

# Change to sglang-jax directory before installing
WORKDIR /app/sglang-jax

RUN cd python && pip install -e .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

ENV JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

RUN mkdir -p /tmp/jit_cache /tmp/models

ENTRYPOINT ["python", "-u", "-m", "sgl_jax.launch_server", "--host", "0.0.0.0", "--port", "30000", "--dist-init-addr=0.0.0.0:10011", "--nnodes=1", "--tp-size=4", "--node-rank=0", "--mem-fraction-static=0.8", "--max-prefill-tokens=8192", "--download-dir=/tmp", "--dtype=bfloat16", "--skip-server-warmup", "--enable-single-process", "--grammar-backend=none"]
