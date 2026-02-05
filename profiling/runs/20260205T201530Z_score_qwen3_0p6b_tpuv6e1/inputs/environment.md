# Environment

- Date: 2026-02-05
- Project: sglang-jax-tests-1769450780
- TPU VM: sglprof-20260205-184819
- Zone: us-east5-b
- TPU: v6e-1 (non-preemptible)
- TPU image: v6e-ubuntu-2404
- OS: Ubuntu 24.04.2 LTS
- Python: 3.12.3
- JAX / jaxlib: 0.8.1 / 0.8.1
- libtpu: 0.0.30
- sglang-jax repo: https://github.com/alexshires/sglang-jax @ a18802ac38d209eacea09e040969262926781b80
- Model: Qwen/Qwen3-0.6B
- Endpoint: /v1/score

Notes:
- Device tracing required a local patch (see `profiling/tools/device_tracer_level.patch`).
