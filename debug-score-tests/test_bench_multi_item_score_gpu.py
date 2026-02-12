"""
Universal Score API multi-item performance benchmark (GPU/PyTorch).
"""

import os
import time
import pytest
import unittest
import logging
import sys
from dataclasses import dataclass

from sglang.srt.entrypoints.engine import Engine

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Force flush on every log
for handler in logger.handlers:
    handler.flush = sys.stdout.flush

# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a score benchmark run."""
    name: str
    total_time_sec: float
    latency_per_item_ms: float
    throughput_items_sec: float
    num_items: int
    prompt_len: int
    candidate_len: int

# =============================================================================
# Test Class
# =============================================================================

class TestMultiItemScorePerformanceGPU(unittest.TestCase):
    """
    Benchmarks comparing single-item scoring with multi-item scoring on GPU.
    """

    model_name = "/models/Qwen/Qwen3-0.6B"
    engine = None
    label_token_ids = [198]
    
    # Target scenario
    NUM_CANDIDATES = 500
    
    @classmethod
    def setUpClass(cls):
        """Initialize engine."""
        logger.info(f"[Benchmark] Loading model: {cls.model_name}")
        
        tp_size = int(os.getenv("TP_SIZE", "1"))
        logger.info(f"[Benchmark] Using TP_SIZE={tp_size}")

        cls.engine = Engine(
            model_path=cls.model_name,
            tp_size=tp_size,
            mem_fraction_static=0.7,
            trust_remote_code=True,
        )
        logger.info("[Benchmark] Engine initialized")
        logger.info(f"[Benchmark] SERVER ARGS: {cls.engine.server_args}")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.engine is not None:
            cls.engine.shutdown()

    def _warmup(self, query, items, count=2):
        """Warm up the engine with a few requests."""
        logger.info(f"  Warmup ({count} requests)...")
        for i in range(count):
            try:
                self.engine.score(
                    query=query, 
                    items=items[:1], 
                    label_token_ids=self.label_token_ids
                )
            except Exception as e:
                logger.info(f"  [Warmup Error] {e}")

    def test_benchmark_scenario_1(self):
        """
        Scenario 1:
        - 500 candidate items per request
        - 20 tokens per candidate
        - 2000-token static prefix
        - 20 token dynamic suffix
        """
        logger.info("\n[Benchmark] Starting Scenario 1")
        static_prefix = "hello " * 2000 
        dynamic_suffix = "world " * 20
        query = static_prefix + dynamic_suffix
        items = ["item " * 20 for _ in range(self.NUM_CANDIDATES)]
        
        # Warmup
        self._warmup(query, items)
        
        start_time = time.perf_counter()
        self.engine.score(query=query, items=items, label_token_ids=self.label_token_ids)
        total_time = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            name="Scenario 1",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=2020,
            candidate_len=20,
        )
        self._report_result(result)

    def test_benchmark_scenario_2(self):
        """
        Scenario 2:
        - 500 candidate items per request
        - 10 tokens per candidate
        - 1900-token static prefix
        - 10 token dynamic suffix
        """
        logger.info("\n[Benchmark] Starting Scenario 2")
        static_prefix = "hello " * 1900
        dynamic_suffix = "world " * 10
        query = static_prefix + dynamic_suffix
        items = ["item " * 10 for _ in range(self.NUM_CANDIDATES)]
        
        # Warmup
        self._warmup(query, items)
        
        start_time = time.perf_counter()
        self.engine.score(query=query, items=items, label_token_ids=self.label_token_ids)
        total_time = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            name="Scenario 2",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=1910,
            candidate_len=10,
        )
        self._report_result(result)

    def _report_result(self, result: BenchmarkResult):
        report = (
            f"  Throughput: {result.throughput_items_sec:.2f} items/sec\n"
            f"  Latency per item: {result.latency_per_item_ms:.2f} ms\n"
            f"  Total time for {result.num_items} items: {result.total_time_sec:.2f} sec\n"
        )
        logger.info(report)

if __name__ == "__main__":
    pytest.main([__file__])
