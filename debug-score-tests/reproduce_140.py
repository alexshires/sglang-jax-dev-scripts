import os
import time
import unittest
import sys
from sgl_jax.srt.entrypoints.engine import Engine

class TestReproduction(unittest.TestCase):
    def test_run(self):
        # Configuration
        tp_size = 1 # Using 1 chip for stability
        num_items = 100 # Reduced for speed
        chunk_size = 0 # Disable internal chunking
        
        print(f"Initializing Engine with TP_SIZE={tp_size}, chunk_size={chunk_size}...", flush=True)
        engine = Engine(
            model_path="/models/Qwen/Qwen3-0.6B",
            tp_size=tp_size,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            max_prefill_tokens=16384,
            max_multi_item_seq_len=8192,
            multi_item_scoring_chunk_size=chunk_size,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=128001,
            disable_radix_cache=True,
            skip_server_warmup=True,
            trust_remote_code=True,
        )
        print(f"Engine initialized. SERVER ARGS: {engine.server_args}", flush=True)
        
        query = [1] * 2000
        items = [[2] * 20 for _ in range(num_items)]
        
        # Warmup
        print("Warmup starting (10 items)...", flush=True)
        w_start = time.perf_counter()
        engine.score(query=query, items=items[:10], label_token_ids=[198])
        w_end = time.perf_counter()
        print(f"Warmup took: {w_end - w_start:.2f} s", flush=True)
        
        print(f"Benchmark starting ({num_items} items)...", flush=True)
        start = time.perf_counter()
        engine.score(query=query, items=items, label_token_ids=[198])
        end = time.perf_counter()
        
        duration = end - start
        throughput = num_items / duration
        print("\nREPRODUCTION RESULTS (TP=1, chunk=0)", flush=True)
        print("-" * 30, flush=True)
        print(f"Throughput: {throughput:10.2f} items/s", flush=True)
        print(f"Latency:    {(duration*1000)/num_items:10.2f} ms/item", flush=True)
        print(f"Total Time: {duration:10.2f} s", flush=True)
        print("-" * 30, flush=True)
        
        engine.shutdown()

if __name__ == "__main__":
    unittest.main()
