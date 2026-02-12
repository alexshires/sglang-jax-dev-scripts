"""
Benchmark for multi-item sequence length scaling (Small sequences).
Compares performance for different max_multi_item_seq_len values.
"""

import os
import time
import unittest
from dataclasses import dataclass

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase

@dataclass
class Result:
    seq_len: int
    chunk_size: int
    throughput: float
    latency: float

class TestSeqLenScaling(CustomTestCase):
    model_name = "/models/Qwen/Qwen3-0.6B"
    NUM_ITEMS = 500
    
    def _run_bench(self, max_seq_len, chunk_size):
        print(f">>> Testing max_seq_len={max_seq_len}, chunk_size={chunk_size}")
        engine = Engine(
            model_path=self.model_name,
            trust_remote_code=True,
            tp_size=int(os.getenv("TP_SIZE", "1")),
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            max_prefill_tokens=max_seq_len,
            max_multi_item_seq_len=max_seq_len,
            multi_item_scoring_chunk_size=chunk_size,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=128001,
            disable_radix_cache=True,
            skip_server_warmup=True,
        )
        print(f"Engine initialized. SERVER ARGS: {engine.server_args}", flush=True)
        
        query = [1] * 2000
        items = [[2] * 20 for _ in range(self.NUM_ITEMS)]
        
        # Warmup
        engine.score(query=query, items=items[:10], label_token_ids=[198])
        
        start = time.perf_counter()
        engine.score(query=query, items=items, label_token_ids=[198])
        end = time.perf_counter()
        
        engine.shutdown()
        
        duration = end - start
        return Result(
            seq_len=max_seq_len,
            chunk_size=chunk_size,
            throughput=self.NUM_ITEMS / duration,
            latency=(duration * 1000) / self.NUM_ITEMS
        )

    def test_scaling(self):
        # We test smaller sequence lengths with batching enabled (chunk_size=32)
        configs = [
            (4096, 32),
            (8192, 32),
            (16384, 32),
        ]
        results = []
        
        for sl, cs in configs:
            try:
                res = self._run_bench(sl, cs)
                results.append(res)
            except Exception as e:
                print(f"FAILED for sl={sl}, cs={cs}: {e}")

        print("\n\n" + "="*60)
        print("SEQUENCE LENGTH SCALING RESULTS (Small Context)")
        print("="*60)
        print(f"{'Max Seq Len':>12} | {'Chunk Size':>12} | {'Throughput':>12} | {'Latency/Item':>15}")
        print("-" * 65)
        for r in results:
            print(f"{r.seq_len:12d} | {r.chunk_size:12d} | {r.throughput:12.2f} | {r.latency:15.2f} ms")
        print("="*60)

if __name__ == "__main__":
    unittest.main()
