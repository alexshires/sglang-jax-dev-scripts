# Optimization Context & Findings (Jan 27, 2026)

## Objective
Maximize scoring API throughput on TPU v5e for `sglang-jax`.
**Target**: Sustain high concurrency (512-1024) to saturate TPU matrix units.

### Current Status (Jan 27)
*   **Best Throughput**: **76.28 req/s** (at 1024 concurrency).
*   **Latency**: ~4.29s (avg).
*   **Key Config**: `max_concurrency=1024`, `chunked-prefill-size=4096`, `precompile-bs=512`, `precompile-len=4096`.
*   **Blockers Resolved**:
    *   JAX backend conflicts (Client CPU vs Server TPU).
    *   OOM at high concurrency (Tuned down precompile buckets).
    *   Python loop overhead (Tokenizer optim).

## Key Findings

### 1. Python Overhead (Solved)
- Initial cpu profile showed `len(tokenizer)` bottleneck.
- Fixed by caching vocab size. Improved throughput 2.5 req/s -> 6.9 req/s.

### 2. Padding Efficiency (Negligible)
- Increasing padding resolution (128 -> 256 bucket) gave **no gain** (remained 6.85 req/s).
- Confirmed that at low concurrency (128), we are **latency-bound** (overhead dominated), not compute-bound.

### 3. Concurrency Scaling (Major Win)
- Increasing `max_running_requests` and `max_concurrency` from 128 -> 512 unlocked the TPU's potential.
- **Result**: Throughput jumped to 16.93 req/s.
- **Constraint**: `context_length` had to be reduced to 4096 to fit 512 concurrent requests in memory (KV cache limits).

### 4. Memory Limits (OOM)
- Attempting **1024 concurrency** caused Out-Of-Memory (OOM) errors.
- 512 is the current stable maximum.

### 5. Prefill "Cliff" (Optimization Target)
- There is a performance cliff when batch size exceeds 2048 or 4096 tokens, causing a jump to the 16384-token kernel.
- **Plan**: Introduce `4096` and `8192` padding buckets and force `chunked-prefill-size=4096` to smooth this out.

## Next Steps (Planned)

1.  **Validate Chunked Prefill**:
    - Current config in `test_bench_score.py` has `chunked-prefill-size=4096` and new buckets.
    - Need to run this on TPU VM to see if it improves latency/throughput further or allows higher concurrency.

2.  **Memory Tuning for 1024**:
    - Try adjusting `mem-fraction-static` or `max-prefill-tokens` to squeeze in 1024 requests?
    - Or accept 512 as the sweet spot and focus on latency.

3.  **Latency Optimization**:
    - Current P99 is likely high >20s.
    - Goal: Reduce latency while maintaining ~17 req/s throughput.

## Scripts & Configs
- **Benchmark**: `test/srt/test_bench_score.py` (Client args updated to 2000 prompts / 1024 concurrency).
- **Server Args**: Updated in `test_bench_score.py` to include `4096`, `8192` buckets.
