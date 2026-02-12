"""
Score API multi-item performance benchmark (REPRODUCTION TEST).
"""

import time
import unittest
from dataclasses import dataclass

import jax
from transformers import AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase

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


class TestMultiItemScorePerformance(CustomTestCase):
    """
    Benchmarks comparing single-item scoring with multi-item scoring.
    """

    model_name = "/models/Qwen/Qwen3-0.6B"
    engine = None
    tokenizer = None
    label_token_ids = [198]  # newline token

    # Target scenario
    PROMPT_LEN = 2000
    NUM_CANDIDATES = 500
    CANDIDATE_LEN = 20

    # Multi-item specific
    STATIC_PREFIX_LEN = 100
    DYNAMIC_SUFFIX_LEN = 1900  # 100 + 1900 = 2000
    DELIMITER_TOKEN_ID = 128001  # Specific delimiter for multi-item

    @classmethod
    def setUpClass(cls):
        """Initialize engine with multi-item support."""
        print(f"[Benchmark] Loading model: {cls.model_name}")

        cls.engine = Engine(
            model_path=cls.model_name,
            trust_remote_code=True,
            tp_size=1, # Hardcode to 1 for reproduction
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.7,
            max_prefill_tokens=32768,
            chunked_prefill_size=-1,
            download_dir="/data/huggingface_models",
            dtype="bfloat16",
            precompile_bs_paddings=[1, 4, 8, 16, 32],
            max_running_requests=32,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024, 4096, 16384],
            page_size=64,
            log_requests=False,
            # Enable multi-item delimiter at engine level
            multi_item_scoring_delimiter=cls.DELIMITER_TOKEN_ID,
            disable_radix_cache=True,
            max_multi_item_seq_len=8192, # Use 8k for reproduction
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)
        print("[Benchmark] Engine initialized")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.engine is not None:
            cls.engine.shutdown()
        jax.clear_caches()

    def test_benchmark_multi_item_packed(self):
        """
        Scenario: 500 candidates scored in a single packed sequence.
        Uses 100 token static prefix + 1900 token dynamic suffix + 500 * 20 token items.
        """
        print(
            f"\n[Benchmark] Starting Multi-Item Packed (Items={self.NUM_CANDIDATES}, Prompt={self.PROMPT_LEN})"
        )

        # Generate dummy data
        query_tokens = [1] * (self.STATIC_PREFIX_LEN + self.DYNAMIC_SUFFIX_LEN)
        candidate_tokens_list = [[2] * self.CANDIDATE_LEN for _ in range(self.NUM_CANDIDATES)]

        # 1 warmup request as in Turn 105
        self.engine.score(
            query=query_tokens,
            items=candidate_tokens_list[:10],
            label_token_ids=self.label_token_ids,
        )

        start_time = time.perf_counter()

        # The Engine.score handles the 500 items.
        self.engine.score(
            query=query_tokens,
            items=candidate_tokens_list,
            label_token_ids=self.label_token_ids,
        )

        total_time = time.perf_counter() - start_time

        result = BenchmarkResult(
            name="Multi-Item Packed",
            total_time_sec=total_time,
            latency_per_item_ms=(total_time * 1000) / self.NUM_CANDIDATES,
            throughput_items_sec=self.NUM_CANDIDATES / total_time,
            num_items=self.NUM_CANDIDATES,
            prompt_len=self.PROMPT_LEN,
            candidate_len=self.CANDIDATE_LEN,
        )
        self._report_result(result)

    def _report_result(self, result: BenchmarkResult):
        report = (
            f"  Throughput: {result.throughput_items_sec:.2f} items/sec\n"
            f"  Latency per item: {result.latency_per_item_ms:.2f} ms\n"
            f"  Total time for {result.num_items} items: {result.total_time_sec:.2f} sec\n"
        )
        print(report)

if __name__ == "__main__":
    unittest.main()
