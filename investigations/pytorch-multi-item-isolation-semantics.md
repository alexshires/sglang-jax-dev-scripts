# Investigation: PyTorch Multi-Item Isolation Semantics

| | |
|------------|------|
| **Date** | 2026-02-11 |
| **Status** | **Ready for Execution** |
| **Goal** | Determine if PyTorch SGLang enforces item isolation in multi-item scoring |
| **Blocks** | [RFC-013](../rfcs/013-multi-item-scoring-v1-optimization.md) Strategy 2 (causal mode) |
| **Related** | [RFC-008](../rfcs/008-multi-item-scoring.md), [Attention Mechanism Investigation](multi-item-attention-mechanism.md) |

## Question

**Does PyTorch SGLang's multi-item scoring enforce strict item isolation?**

In other words: When scoring `[query, item1, item2, item3]`, can item2's score be affected by item1's content?

## Why This Matters

JAX's multi-item scoring (v0.1) uses a custom attention mask to enforce strict isolation:

```
# JAX (current): Isolated scoring
score(item1) = P(item1 | query)
score(item2) = P(item2 | query)           # item1 is invisible
score(item3) = P(item3 | query)           # item1, item2 are invisible
```

If PyTorch uses pure causal attention without custom masks:

```
# PyTorch (suspected): Contaminated scoring
score(item1) = P(item1 | query)
score(item2) = P(item2 | query, item1)    # item1 is visible!
score(item3) = P(item3 | query, item1, item2)  # both visible!
```

**Decision impact:**
- If PyTorch **lacks isolation**: JAX has a correctness advantage; we could offer optional "causal mode" for users who don't need isolation (Strategy 2 in RFC-013)
- If PyTorch **has isolation**: We must match their approach; Strategy 2 is not viable for parity

---

## Preliminary Analysis

### Evidence Suggesting PyTorch Lacks Isolation

From agent exploration of PyTorch codebase:

1. **No custom attention mask construction found** in `tokenizer_manager_multiitem_mixin.py`
2. **FlashInfer is the only backend mentioned** for multi-item isolation support
3. **RFC-008 notes** (line 226): "In PyTorch, multi-item attention isolation is implemented **only** in the FlashInfer backend"
4. **Other backends use causal attention** per RFC-008 (line 232)

### Evidence Suggesting PyTorch Has Isolation

1. **FlashInfer `MultiItemScoringParams`** exists (per RFC-008 references)
2. **PyTorch tests pass** (though coverage of isolation is unclear)
3. **Position encoding resets** may provide partial isolation

### Unknown

- Does PyTorch's default configuration use FlashInfer?
- What happens with other backends (Triton, native)?
- Are the isolation differences measurable in practice?

---

## Test Plan

### Test 1: Order Sensitivity

**Hypothesis:** If items can see previous items, reordering items will change scores.

```python
def test_order_sensitivity():
    query = "What is the capital of France?"
    items = ["Paris", "London", "Berlin"]

    # Request 1: Original order
    scores_abc = score_request(query, items=["Paris", "London", "Berlin"])

    # Request 2: Reversed order
    scores_cba = score_request(query, items=["Berlin", "London", "Paris"])

    # If isolated: scores should be identical regardless of order
    # Paris score in ABC should equal Paris score in CBA
    assert scores_abc["Paris"] == scores_cba["Paris"], "Order affects scores - no isolation"
    assert scores_abc["London"] == scores_cba["London"], "Order affects scores - no isolation"
    assert scores_abc["Berlin"] == scores_cba["Berlin"], "Order affects scores - no isolation"
```

**Expected results:**
- **With isolation:** All assertions pass (scores identical)
- **Without isolation:** At least some scores differ based on position

### Test 2: Content Contamination

**Hypothesis:** If items can see previous items, changing item1 will change item2's score.

```python
def test_content_contamination():
    query = "What is the capital of France?"

    # Request 1: Neutral item1
    scores_neutral = score_request(query, items=["Hello", "Paris"])

    # Request 2: Contaminating item1 (mentions "Paris")
    scores_contaminated = score_request(query, items=["Paris is beautiful", "Paris"])

    # If isolated: item2 ("Paris") score should be identical
    # If not isolated: item2 score may change because it can see item1
    diff = abs(scores_neutral["Paris"] - scores_contaminated["Paris"])

    if diff > 0.001:
        print(f"Content contamination detected: diff={diff}")
        return "NO_ISOLATION"
    else:
        return "ISOLATED"
```

### Test 3: Scaling Contamination

**Hypothesis:** Contamination effect increases with more preceding items.

```python
def test_scaling_contamination():
    query = "Rate this restaurant"
    target_item = "Great food and service"

    results = []
    for n_prefix in [0, 1, 5, 10, 20]:
        prefix_items = ["Terrible experience"] * n_prefix
        items = prefix_items + [target_item]

        scores = score_request(query, items=items)
        target_score = scores[target_item]  # Last item
        results.append((n_prefix, target_score))

    # If isolated: all target scores should be identical
    # If not isolated: target score may degrade with more negative prefix items
    score_variance = variance([r[1] for r in results])

    if score_variance > 0.001:
        print(f"Scaling contamination detected: variance={score_variance}")
        print(f"Scores by prefix length: {results}")
        return "NO_ISOLATION"
    else:
        return "ISOLATED"
```

### Test 4: Direct Backend Comparison

**Hypothesis:** Different PyTorch backends may behave differently.

```python
def test_backend_comparison():
    query = "What is 2+2?"
    items = ["4", "5", "6"]

    # Test with FlashInfer (should have isolation)
    scores_flashinfer = score_request(query, items, backend="flashinfer")

    # Test with Triton (may lack isolation)
    scores_triton = score_request(query, items, backend="triton")

    # Compare item2 and item3 scores
    # If FlashInfer has isolation but Triton doesn't, scores will differ
    diff_item2 = abs(scores_flashinfer["5"] - scores_triton["5"])
    diff_item3 = abs(scores_flashinfer["6"] - scores_triton["6"])

    print(f"Backend diff item2: {diff_item2}, item3: {diff_item3}")
```

---

## Execution Steps

### Prerequisites

1. PyTorch SGLang server running with multi-item scoring enabled:
   ```bash
   python -m sglang.launch_server \
       --model-path Qwen/Qwen3-0.6B \
       --port 30000 \
       --multi-item-scoring-delimiter 1 \
       --disable-radix-cache
   ```

2. Test script location: `sglang-jax-dev-scripts/investigations/scripts/test_pytorch_isolation.py`

### Commands

```bash
# 1. Start PyTorch server (GPU)
cd sglang
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --multi-item-scoring-delimiter 1 \
    --disable-radix-cache

# 2. Run isolation tests
cd ../sglang-jax-dev-scripts
python investigations/scripts/test_pytorch_isolation.py \
    --server-url http://localhost:30000 \
    --output investigations/results/pytorch-isolation-results.json

# 3. Analyze results
python investigations/scripts/analyze_isolation_results.py \
    investigations/results/pytorch-isolation-results.json
```

---

## Test Script

```python
#!/usr/bin/env python3
"""
Test PyTorch SGLang multi-item scoring isolation behavior.

Usage:
    python test_pytorch_isolation.py --server-url http://localhost:30000
"""

import argparse
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class IsolationTestResult:
    test_name: str
    passed: bool
    has_isolation: bool
    details: Dict[str, Any]


def score_items(server_url: str, query: str, items: List[str]) -> Dict[str, float]:
    """Call /v1/score endpoint and return item->score mapping."""
    response = requests.post(
        f"{server_url}/v1/score",
        json={
            "model": "default",
            "query": query,
            "items": items,
            "apply_softmax": False,  # Raw logprobs for precise comparison
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Map items to scores
    scores = {}
    for i, item in enumerate(items):
        scores[item] = data["scores"][i]["score"]
    return scores


def test_order_sensitivity(server_url: str) -> IsolationTestResult:
    """Test if item order affects scores."""
    query = "What is the capital of France?"
    items_abc = ["Paris", "London", "Berlin"]
    items_cba = ["Berlin", "London", "Paris"]

    scores_abc = score_items(server_url, query, items_abc)
    scores_cba = score_items(server_url, query, items_cba)

    # Extract scores for each item regardless of position
    diffs = {
        "Paris": abs(scores_abc["Paris"] - scores_cba["Paris"]),
        "London": abs(scores_abc["London"] - scores_cba["London"]),
        "Berlin": abs(scores_abc["Berlin"] - scores_cba["Berlin"]),
    }

    max_diff = max(diffs.values())
    has_isolation = max_diff < 0.001

    return IsolationTestResult(
        test_name="order_sensitivity",
        passed=True,  # Test executed successfully
        has_isolation=has_isolation,
        details={
            "scores_abc": scores_abc,
            "scores_cba": scores_cba,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": 0.001,
        }
    )


def test_content_contamination(server_url: str) -> IsolationTestResult:
    """Test if preceding item content affects subsequent scores."""
    query = "What is the capital of France?"

    # Neutral prefix
    items_neutral = ["Hello world", "Paris"]
    scores_neutral = score_items(server_url, query, items_neutral)

    # Contaminating prefix (mentions Paris)
    items_contaminated = ["Paris is a beautiful city", "Paris"]
    scores_contaminated = score_items(server_url, query, items_contaminated)

    # The score for "Paris" (item2) should be identical if isolated
    diff = abs(scores_neutral["Paris"] - scores_contaminated["Paris"])
    has_isolation = diff < 0.001

    return IsolationTestResult(
        test_name="content_contamination",
        passed=True,
        has_isolation=has_isolation,
        details={
            "scores_neutral": scores_neutral,
            "scores_contaminated": scores_contaminated,
            "paris_diff": diff,
            "threshold": 0.001,
        }
    )


def test_scaling_contamination(server_url: str) -> IsolationTestResult:
    """Test if contamination scales with number of preceding items."""
    query = "Rate this restaurant review"
    target_item = "The food was excellent and service was great"

    results = []
    for n_prefix in [0, 1, 5, 10]:
        prefix_items = ["Terrible food, awful service, never coming back"] * n_prefix
        items = prefix_items + [target_item]

        scores = score_items(server_url, query, items)
        target_score = scores[target_item]
        results.append({"n_prefix": n_prefix, "score": target_score})

    # Calculate variance in target scores
    scores_only = [r["score"] for r in results]
    mean_score = sum(scores_only) / len(scores_only)
    variance = sum((s - mean_score) ** 2 for s in scores_only) / len(scores_only)

    has_isolation = variance < 0.001

    return IsolationTestResult(
        test_name="scaling_contamination",
        passed=True,
        has_isolation=has_isolation,
        details={
            "results_by_prefix_count": results,
            "score_variance": variance,
            "threshold": 0.001,
        }
    )


def run_all_tests(server_url: str) -> Dict[str, Any]:
    """Run all isolation tests and return summary."""
    tests = [
        test_order_sensitivity,
        test_content_contamination,
        test_scaling_contamination,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn(server_url)
            results.append(result)
        except Exception as e:
            results.append(IsolationTestResult(
                test_name=test_fn.__name__,
                passed=False,
                has_isolation=None,
                details={"error": str(e)}
            ))

    # Summary
    all_isolated = all(r.has_isolation for r in results if r.has_isolation is not None)
    any_isolated = any(r.has_isolation for r in results if r.has_isolation is not None)

    return {
        "server_url": server_url,
        "tests": [
            {
                "name": r.test_name,
                "passed": r.passed,
                "has_isolation": r.has_isolation,
                "details": r.details,
            }
            for r in results
        ],
        "summary": {
            "all_tests_show_isolation": all_isolated,
            "any_test_shows_isolation": any_isolated,
            "conclusion": "ISOLATED" if all_isolated else ("PARTIAL" if any_isolated else "NO_ISOLATION"),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Test PyTorch SGLang isolation behavior")
    parser.add_argument("--server-url", required=True, help="SGLang server URL")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    results = run_all_tests(args.server_url)

    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Exit code based on conclusion
    conclusion = results["summary"]["conclusion"]
    print(f"\n{'='*50}")
    print(f"CONCLUSION: PyTorch multi-item scoring is {conclusion}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

---

## Expected Outcomes

### Outcome A: PyTorch Has Full Isolation

**Evidence:** All tests show `has_isolation=True`

**Implications:**
- JAX's approach is correct and matches PyTorch
- Strategy 2 (causal mode) should NOT be offered for parity
- Focus optimization on Strategy 1 (procedural mask)

### Outcome B: PyTorch Has No Isolation

**Evidence:** All tests show `has_isolation=False`

**Implications:**
- JAX has a **correctness advantage** over PyTorch
- Strategy 2 (causal mode) is viable for users who want PyTorch-like behavior
- Document the semantic difference clearly
- Consider offering both modes

### Outcome C: PyTorch Has Partial Isolation

**Evidence:** Mixed results (some tests pass, some fail)

**Implications:**
- May depend on backend (FlashInfer vs others)
- Need deeper investigation into backend-specific behavior
- May need to test with explicit backend selection

---

## Results

**Status:** Pending execution

| Test | Result | Has Isolation | Notes |
|------|--------|---------------|-------|
| Order Sensitivity | TBD | TBD | TBD |
| Content Contamination | TBD | TBD | TBD |
| Scaling Contamination | TBD | TBD | TBD |

**Conclusion:** TBD

---

## Follow-up Actions

Based on results:

1. **If ISOLATED:** Update RFC-013 to remove Strategy 2, focus on procedural mask
2. **If NO_ISOLATION:**
   - Add ADR documenting semantic difference
   - Implement Strategy 2 (causal mode) as option
   - Update user documentation
3. **If PARTIAL:** Investigate backend-specific behavior, document which backends are safe

---

## References

- [RFC-008: Multi-Item Scoring](../rfcs/008-multi-item-scoring.md) - Line 226 mentions FlashInfer-only isolation
- [RFC-013: v1.0 Optimization](../rfcs/013-multi-item-scoring-v1-optimization.md) - Strategy 2 depends on this investigation
- PyTorch files:
  - `sglang/python/sglang/srt/managers/tokenizer_manager_multiitem_mixin.py`
  - `sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
