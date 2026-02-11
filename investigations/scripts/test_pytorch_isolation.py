#!/usr/bin/env python3
"""
Test PyTorch SGLang multi-item scoring isolation behavior.

This script determines whether PyTorch SGLang enforces item isolation
in multi-item scoring mode. If items can see previous items (no isolation),
scores will change based on item order and preceding content.

## Decision-Grade Testing

This script supports two modes:
1. **Quick mode (default):** Text prompts, single pass, fast feedback
2. **Decision mode (--decision-grade):** Tokenized inputs, N repeated runs,
   statistical confidence intervals, suitable for making implementation decisions

Usage:
    # Quick feedback
    python test_pytorch_isolation.py --server-url http://localhost:30000

    # Decision-grade with statistical rigor
    python test_pytorch_isolation.py --server-url http://localhost:30000 \
        --decision-grade --runs 10 --model-path Qwen/Qwen3-0.6B

Prerequisites:
    PyTorch SGLang server running with multi-item scoring enabled:

    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --port 30000 \
        --multi-item-scoring-delimiter 1 \
        --disable-radix-cache

See: investigations/pytorch-multi-item-isolation-semantics.md
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install with: pip install requests")
    sys.exit(1)


@dataclass
class StatisticalResult:
    """Statistical summary for repeated measurements."""
    mean: float
    std: float
    min_val: float
    max_val: float
    n_samples: int
    ci_95_lower: float  # 95% confidence interval
    ci_95_upper: float

    @classmethod
    def from_samples(cls, samples: List[float]) -> "StatisticalResult":
        n = len(samples)
        if n == 0:
            return cls(0, 0, 0, 0, 0, 0, 0)
        mean = sum(samples) / n
        variance = sum((x - mean) ** 2 for x in samples) / n if n > 1 else 0
        std = math.sqrt(variance)
        # 95% CI using t-distribution approximation (z=1.96 for large n)
        se = std / math.sqrt(n) if n > 1 else 0
        z = 1.96  # 95% CI
        return cls(
            mean=mean,
            std=std,
            min_val=min(samples),
            max_val=max(samples),
            n_samples=n,
            ci_95_lower=mean - z * se,
            ci_95_upper=mean + z * se,
        )


@dataclass
class IsolationTestResult:
    test_name: str
    passed: bool
    has_isolation: Optional[bool]
    details: Dict[str, Any]
    statistical: Optional[Dict[str, StatisticalResult]] = None


def score_items(
    server_url: str,
    query: str,
    items: List[str],
    label_token_ids: List[int],
    timeout: int = 60,
    use_token_ids: bool = False,
    query_tokens: Optional[List[int]] = None,
    item_tokens_list: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Call /v1/score endpoint and return item->score mapping.

    Args:
        label_token_ids: Token IDs to compute probabilities for (REQUIRED by API)
        use_token_ids: If True, use pre-tokenized inputs for determinism
        query_tokens: Pre-tokenized query (if use_token_ids=True)
        item_tokens_list: Pre-tokenized items (if use_token_ids=True)

    Note: The /v1/score API accepts token IDs directly in the `query` and `items`
    fields - no separate `query_token_ids` or `item_token_ids` parameters exist.
    """
    if use_token_ids and query_tokens is not None and item_tokens_list is not None:
        # Use tokenized inputs for deterministic results
        # API accepts token IDs directly in query/items fields
        payload = {
            "model": "default",
            "query": query_tokens,  # API accepts List[int] directly
            "items": item_tokens_list,  # API accepts List[List[int]] directly
            "label_token_ids": label_token_ids,
            "apply_softmax": False,
        }
    else:
        payload = {
            "model": "default",
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "apply_softmax": False,
        }

    response = requests.post(
        f"{server_url}/v1/score",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    # Response format: scores is List[List[float]] where each inner list
    # contains probabilities for each label_token_id.
    #
    # For isolation testing, use FIRST label only (not sum) to avoid
    # cancellation effects when comparing across runs. The first label
    # is typically "Yes" which gives a stable, interpretable signal.
    scores = {}
    for i, item in enumerate(items):
        key = f"{item}" if items.count(item) == 1 else f"{item}[{i}]"
        # Use first label score only (avoids cancellation in sum)
        scores[key] = data["scores"][i][0]
    return scores


def get_tokenizer(model_path: str):
    """Load tokenizer for pre-tokenized testing."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ImportError:
        print("WARNING: transformers not installed. Cannot use --decision-grade mode.")
        return None


def get_label_token_ids(tokenizer) -> List[int]:
    """
    Get consistent label token IDs for scoring.

    For isolation testing, we use "Yes" and "No" tokens which are commonly
    used for classification tasks. The actual tokens don't matter for isolation
    testing - we just need consistent labels across all calls.
    """
    if tokenizer is None:
        # Fallback: use common token IDs that exist in most vocabularies
        # These are typical IDs for common tokens in Qwen/Llama models
        return [9454, 2753]  # Example: "Yes", "No" in Qwen3

    # Get token IDs for Yes/No
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)
    no_id = tokenizer.encode("No", add_special_tokens=False)

    # Use first token of each (in case they tokenize to multiple tokens)
    return [yes_id[0], no_id[0]]


def test_order_sensitivity(
    server_url: str,
    n_runs: int = 1,
    tokenizer=None,
    label_token_ids: Optional[List[int]] = None,
) -> IsolationTestResult:
    """
    Test if item order affects scores.

    If items are isolated, reordering should not change individual item scores.

    Args:
        n_runs: Number of repeated runs for statistical analysis
        tokenizer: If provided, use pre-tokenized inputs for determinism
        label_token_ids: Token IDs to score (required by API)
    """
    query = "What is the capital of France?"
    items_abc = ["Paris", "London", "Berlin"]
    items_cba = ["Berlin", "London", "Paris"]

    # Get label token IDs if not provided
    if label_token_ids is None:
        label_token_ids = get_label_token_ids(tokenizer)

    # Pre-tokenize if tokenizer provided
    query_tokens = None
    items_abc_tokens = None
    items_cba_tokens = None
    use_tokens = tokenizer is not None

    if use_tokens:
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        items_abc_tokens = [tokenizer.encode(item, add_special_tokens=False) for item in items_abc]
        items_cba_tokens = [tokenizer.encode(item, add_special_tokens=False) for item in items_cba]

    try:
        # Collect samples across runs
        paris_diffs = []
        london_diffs = []
        berlin_diffs = []

        for run in range(n_runs):
            scores_abc = score_items(
                server_url, query, items_abc,
                label_token_ids=label_token_ids,
                use_token_ids=use_tokens,
                query_tokens=query_tokens,
                item_tokens_list=items_abc_tokens,
            )
            scores_cba = score_items(
                server_url, query, items_cba,
                label_token_ids=label_token_ids,
                use_token_ids=use_tokens,
                query_tokens=query_tokens,
                item_tokens_list=items_cba_tokens,
            )

            paris_diffs.append(abs(scores_abc["Paris"] - scores_cba["Paris"]))
            london_diffs.append(abs(scores_abc["London"] - scores_cba["London"]))
            berlin_diffs.append(abs(scores_abc["Berlin"] - scores_cba["Berlin"]))

    except Exception as e:
        return IsolationTestResult(
            test_name="order_sensitivity",
            passed=False,
            has_isolation=None,
            details={"error": str(e)}
        )

    # Statistical analysis
    stats = {
        "Paris": StatisticalResult.from_samples(paris_diffs),
        "London": StatisticalResult.from_samples(london_diffs),
        "Berlin": StatisticalResult.from_samples(berlin_diffs),
    }

    # For single-run mode, use simple threshold
    # For multi-run mode, use confidence interval
    threshold = 0.001

    if n_runs == 1:
        max_diff = max(paris_diffs[0], london_diffs[0], berlin_diffs[0])
        has_isolation = max_diff < threshold
    else:
        # Check if 95% CI upper bound is below threshold for all items
        # This means we're 95% confident the true diff is below threshold
        has_isolation = all(
            s.ci_95_upper < threshold for s in stats.values()
        )
        max_diff = max(s.mean for s in stats.values())

    return IsolationTestResult(
        test_name="order_sensitivity",
        passed=True,
        has_isolation=has_isolation,
        details={
            "n_runs": n_runs,
            "threshold": threshold,
            "max_mean_diff": max_diff,
            "interpretation": "ISOLATED: Order does not affect scores" if has_isolation
                           else f"NO ISOLATION: Order affects scores (max_diff={max_diff:.6f})",
            "statistical_mode": n_runs > 1,
        },
        statistical={k: asdict(v) for k, v in stats.items()} if n_runs > 1 else None,
    )


def test_content_contamination(
    server_url: str,
    n_runs: int = 1,
    tokenizer=None,
    label_token_ids: Optional[List[int]] = None,
) -> IsolationTestResult:
    """
    Test if preceding item content affects subsequent scores.

    If items are isolated, changing item1 should not affect item2's score.
    """
    query = "What is the capital of France?"
    items_neutral = ["Hello world", "Paris"]
    items_contaminated = ["Paris is a beautiful city", "Paris"]

    # Get label token IDs if not provided
    if label_token_ids is None:
        label_token_ids = get_label_token_ids(tokenizer)

    # Pre-tokenize if tokenizer provided
    use_tokens = tokenizer is not None
    query_tokens = None
    items_neutral_tokens = None
    items_contaminated_tokens = None

    if use_tokens:
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        items_neutral_tokens = [tokenizer.encode(item, add_special_tokens=False) for item in items_neutral]
        items_contaminated_tokens = [tokenizer.encode(item, add_special_tokens=False) for item in items_contaminated]

    try:
        diffs = []
        for run in range(n_runs):
            scores_neutral = score_items(
                server_url, query, items_neutral,
                label_token_ids=label_token_ids,
                use_token_ids=use_tokens,
                query_tokens=query_tokens,
                item_tokens_list=items_neutral_tokens,
            )
            scores_contaminated = score_items(
                server_url, query, items_contaminated,
                label_token_ids=label_token_ids,
                use_token_ids=use_tokens,
                query_tokens=query_tokens,
                item_tokens_list=items_contaminated_tokens,
            )

            paris_score_neutral = scores_neutral["Paris"]
            paris_score_contaminated = scores_contaminated["Paris"]
            diffs.append(abs(paris_score_neutral - paris_score_contaminated))

    except Exception as e:
        return IsolationTestResult(
            test_name="content_contamination",
            passed=False,
            has_isolation=None,
            details={"error": str(e)}
        )

    stats = StatisticalResult.from_samples(diffs)
    threshold = 0.001

    if n_runs == 1:
        has_isolation = diffs[0] < threshold
        diff = diffs[0]
    else:
        has_isolation = stats.ci_95_upper < threshold
        diff = stats.mean

    return IsolationTestResult(
        test_name="content_contamination",
        passed=True,
        has_isolation=has_isolation,
        details={
            "n_runs": n_runs,
            "neutral_prefix": "Hello world",
            "contaminating_prefix": "Paris is a beautiful city",
            "target_item": "Paris",
            "diff": diff,
            "threshold": threshold,
            "interpretation": "ISOLATED: Prefix content does not affect target score" if has_isolation
                           else f"NO ISOLATION: Prefix content affects target score (diff={diff:.6f})",
            "statistical_mode": n_runs > 1,
        },
        statistical={"diff": asdict(stats)} if n_runs > 1 else None,
    )


def test_scaling_contamination(
    server_url: str,
    n_runs: int = 1,
    tokenizer=None,
    label_token_ids: Optional[List[int]] = None,
) -> IsolationTestResult:
    """
    Test if contamination scales with number of preceding items.

    If items are isolated, adding more prefix items should not change the target score.
    """
    query = "Rate this restaurant review"
    target_item = "The food was excellent and service was great"
    negative_prefix = "Terrible food, awful service, never coming back"

    # Get label token IDs if not provided
    if label_token_ids is None:
        label_token_ids = get_label_token_ids(tokenizer)

    use_tokens = tokenizer is not None
    query_tokens = None
    target_tokens = None
    prefix_tokens = None

    if use_tokens:
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        target_tokens = tokenizer.encode(target_item, add_special_tokens=False)
        prefix_tokens = tokenizer.encode(negative_prefix, add_special_tokens=False)

    try:
        all_deviations = []
        for run in range(n_runs):
            run_results = []
            for n_prefix in [0, 1, 5, 10]:
                prefix_items = [negative_prefix] * n_prefix
                items = prefix_items + [target_item]

                if use_tokens:
                    item_tokens = [prefix_tokens] * n_prefix + [target_tokens]
                    scores = score_items(
                        server_url, query, items,
                        label_token_ids=label_token_ids,
                        use_token_ids=True,
                        query_tokens=query_tokens,
                        item_tokens_list=item_tokens,
                    )
                else:
                    scores = score_items(
                        server_url, query, items,
                        label_token_ids=label_token_ids,
                    )

                target_key = list(scores.keys())[-1]
                run_results.append({"n_prefix": n_prefix, "score": scores[target_key]})

            # Max deviation from baseline within this run
            baseline = run_results[0]["score"]
            max_dev = max(abs(r["score"] - baseline) for r in run_results)
            all_deviations.append(max_dev)

    except Exception as e:
        return IsolationTestResult(
            test_name="scaling_contamination",
            passed=False,
            has_isolation=None,
            details={"error": str(e)}
        )

    stats = StatisticalResult.from_samples(all_deviations)
    threshold = 0.001

    if n_runs == 1:
        has_isolation = all_deviations[0] < threshold
        max_deviation = all_deviations[0]
    else:
        has_isolation = stats.ci_95_upper < threshold
        max_deviation = stats.mean

    return IsolationTestResult(
        test_name="scaling_contamination",
        passed=True,
        has_isolation=has_isolation,
        details={
            "n_runs": n_runs,
            "target_item": target_item,
            "negative_prefix_item": negative_prefix,
            "max_deviation_from_baseline": max_deviation,
            "threshold": threshold,
            "interpretation": "ISOLATED: Score stable regardless of prefix count" if has_isolation
                           else f"NO ISOLATION: Score changes with prefix count (max_dev={max_deviation:.6f})",
            "statistical_mode": n_runs > 1,
        },
        statistical={"max_deviation": asdict(stats)} if n_runs > 1 else None,
    )


def test_position_in_batch(
    server_url: str,
    n_runs: int = 1,
    tokenizer=None,
    label_token_ids: Optional[List[int]] = None,
) -> IsolationTestResult:
    """
    Test if the same item scores differently at different positions.

    If items are isolated, "Paris" should score the same whether it's
    item 1, item 3, or item 5 in the batch.
    """
    query = "What is the capital of France?"
    filler = "Random text here"
    target = "Paris"

    # Get label token IDs if not provided
    if label_token_ids is None:
        label_token_ids = get_label_token_ids(tokenizer)

    use_tokens = tokenizer is not None
    query_tokens = None
    filler_tokens = None
    target_tokens = None

    if use_tokens:
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
        target_tokens = tokenizer.encode(target, add_special_tokens=False)

    try:
        all_max_diffs = []
        for run in range(n_runs):
            # Target at position 0
            if use_tokens:
                scores_pos0 = score_items(
                    server_url, query, [target],
                    label_token_ids=label_token_ids,
                    use_token_ids=True,
                    query_tokens=query_tokens,
                    item_tokens_list=[target_tokens],
                )
                scores_pos2 = score_items(
                    server_url, query, [filler, filler, target],
                    label_token_ids=label_token_ids,
                    use_token_ids=True,
                    query_tokens=query_tokens,
                    item_tokens_list=[filler_tokens, filler_tokens, target_tokens],
                )
                scores_pos4 = score_items(
                    server_url, query, [filler] * 4 + [target],
                    label_token_ids=label_token_ids,
                    use_token_ids=True,
                    query_tokens=query_tokens,
                    item_tokens_list=[filler_tokens] * 4 + [target_tokens],
                )
            else:
                scores_pos0 = score_items(
                    server_url, query, [target],
                    label_token_ids=label_token_ids,
                )
                scores_pos2 = score_items(
                    server_url, query, [filler, filler, target],
                    label_token_ids=label_token_ids,
                )
                scores_pos4 = score_items(
                    server_url, query, [filler] * 4 + [target],
                    label_token_ids=label_token_ids,
                )

            score_at_0 = list(scores_pos0.values())[-1]
            score_at_2 = list(scores_pos2.values())[-1]
            score_at_4 = list(scores_pos4.values())[-1]

            max_diff = max(
                abs(score_at_0 - score_at_2),
                abs(score_at_0 - score_at_4),
                abs(score_at_2 - score_at_4),
            )
            all_max_diffs.append(max_diff)

    except Exception as e:
        return IsolationTestResult(
            test_name="position_in_batch",
            passed=False,
            has_isolation=None,
            details={"error": str(e)}
        )

    stats = StatisticalResult.from_samples(all_max_diffs)
    threshold = 0.001

    if n_runs == 1:
        has_isolation = all_max_diffs[0] < threshold
        max_diff = all_max_diffs[0]
    else:
        has_isolation = stats.ci_95_upper < threshold
        max_diff = stats.mean

    return IsolationTestResult(
        test_name="position_in_batch",
        passed=True,
        has_isolation=has_isolation,
        details={
            "n_runs": n_runs,
            "target_item": target,
            "filler_item": filler,
            "max_diff": max_diff,
            "threshold": threshold,
            "interpretation": "ISOLATED: Same score regardless of position" if has_isolation
                           else f"NO ISOLATION: Score varies by position (max_diff={max_diff:.6f})",
            "statistical_mode": n_runs > 1,
        },
        statistical={"max_diff": asdict(stats)} if n_runs > 1 else None,
    )


def run_all_tests(
    server_url: str,
    verbose: bool = True,
    n_runs: int = 1,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run all isolation tests and return summary."""
    # Get label token IDs once for all tests (ensures consistency)
    label_token_ids = get_label_token_ids(tokenizer)

    if verbose:
        print(f"Using label_token_ids: {label_token_ids}")

    tests = [
        ("Order Sensitivity", test_order_sensitivity),
        ("Content Contamination", test_content_contamination),
        ("Scaling Contamination", test_scaling_contamination),
        ("Position in Batch", test_position_in_batch),
    ]

    results = []
    for name, test_fn in tests:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            if n_runs > 1:
                print(f"Mode: Decision-grade ({n_runs} runs with statistical analysis)")
            print('='*60)

        result = test_fn(
            server_url,
            n_runs=n_runs,
            tokenizer=tokenizer,
            label_token_ids=label_token_ids,
        )
        results.append(result)

        if verbose:
            if result.passed:
                status = "ISOLATED" if result.has_isolation else "NO ISOLATION"
                print(f"Result: {status}")
                print(f"Details: {result.details.get('interpretation', 'N/A')}")
                if result.statistical:
                    print(f"Statistics: {json.dumps(result.statistical, indent=2)}")
            else:
                print(f"Result: FAILED TO EXECUTE")
                print(f"Error: {result.details.get('error', 'Unknown')}")

    # Summary
    executed_tests = [r for r in results if r.has_isolation is not None]
    if executed_tests:
        all_isolated = all(r.has_isolation for r in executed_tests)
        any_isolated = any(r.has_isolation for r in executed_tests)
        isolation_count = sum(1 for r in executed_tests if r.has_isolation)

        if all_isolated:
            conclusion = "ISOLATED"
        elif any_isolated:
            conclusion = "PARTIAL"
        else:
            conclusion = "NO_ISOLATION"
    else:
        all_isolated = None
        any_isolated = None
        isolation_count = 0
        conclusion = "ERROR"

    # Compute confidence level based on statistical rigor
    # Requirements for "high" confidence:
    # 1. At least 5 runs (minimum for meaningful statistics)
    # 2. All tests have narrow CI (CI width < 2x threshold)
    MIN_RUNS_FOR_HIGH_CONFIDENCE = 5
    CI_WIDTH_THRESHOLD = 0.002  # 2x the 0.001 isolation threshold

    if n_runs >= MIN_RUNS_FOR_HIGH_CONFIDENCE:
        # Check CI widths from statistical results
        ci_widths_ok = True
        for r in executed_tests:
            if r.statistical:
                for stat_key, stat_val in r.statistical.items():
                    ci_width = stat_val.get("ci_95_upper", 0) - stat_val.get("ci_95_lower", 0)
                    if ci_width > CI_WIDTH_THRESHOLD:
                        ci_widths_ok = False
                        break

        if ci_widths_ok:
            confidence = "high (>=5 runs, narrow CIs)"
        else:
            confidence = "medium (>=5 runs, wide CIs - consider more runs)"
    elif n_runs > 1:
        confidence = "low (2-4 runs, insufficient for decision)"
    else:
        confidence = "very low (single pass, not decision-grade)"

    return {
        "server_url": server_url,
        "mode": "decision_grade" if n_runs > 1 else "quick",
        "n_runs": n_runs,
        "uses_tokenized_inputs": tokenizer is not None,
        "tests": [asdict(r) for r in results],
        "summary": {
            "total_tests": len(results),
            "executed_tests": len(executed_tests),
            "tests_showing_isolation": isolation_count,
            "all_tests_show_isolation": all_isolated,
            "any_test_shows_isolation": any_isolated,
            "conclusion": conclusion,
            "confidence": confidence,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test PyTorch SGLang multi-item scoring isolation behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick feedback (single pass, text prompts)
    python test_pytorch_isolation.py --server-url http://localhost:30000

    # Decision-grade testing (10 runs, tokenized inputs, statistical analysis)
    python test_pytorch_isolation.py --server-url http://localhost:30000 \\
        --decision-grade --runs 10 --model-path Qwen/Qwen3-0.6B

    # Save results to file
    python test_pytorch_isolation.py --server-url http://localhost:30000 \\
        --output results.json

    # Quiet mode (JSON only)
    python test_pytorch_isolation.py --server-url http://localhost:30000 --quiet
        """
    )
    parser.add_argument("--server-url", required=True, help="SGLang server URL")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--decision-grade",
        action="store_true",
        help="Enable decision-grade testing with tokenized inputs and statistical analysis"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repeated runs for statistical analysis (default: 10, only used with --decision-grade)"
    )
    parser.add_argument(
        "--model-path",
        help="Model path for tokenizer (required for --decision-grade, e.g., Qwen/Qwen3-0.6B)"
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Load tokenizer if decision-grade mode
    tokenizer = None
    n_runs = 1
    if args.decision_grade:
        if not args.model_path:
            print("ERROR: --model-path is required for --decision-grade mode")
            sys.exit(1)
        tokenizer = get_tokenizer(args.model_path)
        if tokenizer is None:
            print("ERROR: Failed to load tokenizer. Install transformers: pip install transformers")
            sys.exit(1)
        n_runs = args.runs

    if verbose:
        print(f"Testing PyTorch SGLang isolation behavior")
        print(f"Server: {args.server_url}")
        if args.decision_grade:
            print(f"Mode: Decision-grade ({n_runs} runs, tokenized inputs)")
            print(f"Model: {args.model_path}")
        else:
            print(f"Mode: Quick (single pass, text prompts)")

    results = run_all_tests(
        args.server_url,
        verbose=verbose,
        n_runs=n_runs,
        tokenizer=tokenizer,
    )

    if args.quiet:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print('='*60)
        print(json.dumps(results["summary"], indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {args.output}")

    # Print conclusion banner
    conclusion = results["summary"]["conclusion"]
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# CONCLUSION: PyTorch multi-item scoring is {conclusion}")
        print(f"{'#'*60}")

        if conclusion == "ISOLATED":
            print("\nImplication: JAX's custom mask approach matches PyTorch behavior.")
            print("Strategy 2 (causal mode) in RFC-013 is NOT recommended for parity.")
        elif conclusion == "NO_ISOLATION":
            print("\nImplication: JAX has a CORRECTNESS ADVANTAGE over PyTorch.")
            print("Strategy 2 (causal mode) in RFC-013 is viable for users who")
            print("want PyTorch-like behavior and prioritize throughput over isolation.")
        elif conclusion == "PARTIAL":
            print("\nImplication: Mixed results - may depend on backend or configuration.")
            print("Further investigation needed with explicit backend selection.")
        else:
            print("\nTests failed to execute. Check server connectivity and configuration.")

    # Exit code: 0 if all tests executed, 1 if any failed
    sys.exit(0 if results["summary"]["executed_tests"] == results["summary"]["total_tests"] else 1)


if __name__ == "__main__":
    main()
