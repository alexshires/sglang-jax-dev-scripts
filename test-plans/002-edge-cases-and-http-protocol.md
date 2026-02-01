# Test Plan 002: Edge Cases and HTTP/Protocol Tests

**Related RFC:** [RFC-003](../rfcs/003-score-api-comprehensive-test-suite.md)
**Phase:** 3-4 (Week 3)
**Priority:** P0 (Critical Coverage)
**Dependencies:** Test Plan 001 (requires shared fixtures)

## Objective

Implement comprehensive edge case validation and HTTP/protocol tests to achieve complete API surface coverage and robust error handling.

## Deliverables

1. **New file:** `test/srt/test_score_api_edge_cases.py` (~400 lines)
2. **Expanded:** `test/srt/openai_server/basic/test_openai_server.py` (add 5 HTTP tests)
3. **Expanded:** `test/srt/openai_server/basic/test_protocol.py` (add `TestScoringProtocol`)

## Edge Case Tests Specification

### File: `test/srt/test_score_api_edge_cases.py`

```python
"""
Edge case and validation tests for Score API.

Tests input validation, corner cases, and error handling.
All tests should complete quickly (< 5 seconds each).
"""

import unittest
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.score_test_utils import (
    ScoreTestConfig,
    build_engine,
    get_tokenizer,
    get_label_token_ids,
)


class TestScoreAPIEdgeCases(CustomTestCase):
    """Edge case and validation tests for Score API"""

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.2-1B-Instruct"
        cls.runner = build_engine(ScoreTestConfig(model=cls.model))
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.runner.shutdown()

    def test_score_empty_items(self):
        """
        Test that empty items list raises ValueError.

        Validates:
        - items=[] raises ValueError
        - Error message is clear
        - Fails fast before expensive operations
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" to"])

        with self.assertRaises(ValueError) as cm:
            self.runner.score(
                query="Test query",
                items=[],  # Empty!
                label_token_ids=label_token_ids
            )

        self.assertIn("items cannot be empty", str(cm.exception).lower())

    def test_score_empty_label_token_ids(self):
        """
        Test that empty label_token_ids raises ValueError.

        Validates:
        - label_token_ids=[] raises ValueError
        - Error message mentions "empty" and "label_token_ids"
        - Fails before softmax (which would give cryptic error)
        """
        with self.assertRaises(ValueError) as cm:
            self.runner.score(
                query="Test query",
                items=["item1"],
                label_token_ids=[]  # Empty!
            )

        error_msg = str(cm.exception).lower()
        self.assertIn("label_token_ids", error_msg)
        self.assertIn("empty", error_msg)

    def test_score_negative_token_ids(self):
        """
        Test that negative token IDs raise ValueError.

        Validates:
        - Negative IDs in label_token_ids raise ValueError
        - Error message shows which IDs are negative
        - Never valid in vocabulary
        """
        with self.assertRaises(ValueError) as cm:
            self.runner.score(
                query="Test query",
                items=["item1"],
                label_token_ids=[100, -5, 200, -10]  # Negative IDs!
            )

        error_msg = str(cm.exception).lower()
        self.assertIn("negative", error_msg)
        # Should mention the specific negative IDs
        self.assertTrue("-5" in str(cm.exception) or "-10" in str(cm.exception))

    def test_score_token_ids_exceeds_vocab(self):
        """
        Test that token IDs >= vocab_size raise error.

        Validates:
        - Out-of-vocab IDs detected
        - Error message mentions vocab size
        """
        vocab_size = len(self.tokenizer)

        with self.assertRaises((ValueError, IndexError)) as cm:
            self.runner.score(
                query="Test query",
                items=["item1"],
                label_token_ids=[vocab_size + 1000]  # Way out of range
            )

        error_msg = str(cm.exception).lower()
        # Should mention vocab or out of range
        self.assertTrue(
            "vocab" in error_msg or "range" in error_msg or "index" in error_msg
        )

    def test_score_mixed_input_types_raises(self):
        """
        Test that mixed input types raise ValueError.

        Cases:
        - Text query + token items → ValueError
        - Token query + text items → ValueError

        Validates:
        - Mixed types explicitly rejected
        - Error message explains constraint
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" to"])

        # Case 1: Text query + token items
        with self.assertRaises(ValueError) as cm:
            self.runner.score(
                query="Text query",  # String
                items=[[123, 456]],  # Tokens!
                label_token_ids=label_token_ids
            )
        self.assertIn("both", str(cm.exception).lower())

        # Case 2: Token query + text items
        with self.assertRaises(ValueError) as cm:
            self.runner.score(
                query=[123, 456],  # Tokens!
                items=["text item"],  # String!
                label_token_ids=label_token_ids
            )
        self.assertIn("both", str(cm.exception).lower())

    def test_score_duplicate_label_tokens(self):
        """
        Test with duplicate label_token_ids.

        Validates:
        - Duplicates are allowed (not an error)
        - Returns score for each occurrence
        - Each duplicate gets same score value
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" to", " to"])  # Duplicate

        result = self.runner.score(
            query="I pledge allegiance",
            items=[" to the flag"],
            label_token_ids=label_token_ids,
            apply_softmax=False
        )

        # Should have 2 scores (one per label token ID, even though duplicate)
        self.assertEqual(len(result), 1)  # 1 item
        self.assertEqual(len(result[0]), 2)  # 2 label tokens

        # Both scores should be identical (same token ID)
        self.assertAlmostEqual(result[0][0], result[0][1], places=6)

    def test_score_unicode_handling(self):
        """
        Test unicode characters in query/items.

        Validates:
        - Emoji handled correctly
        - Non-ASCII characters work
        - Multi-byte UTF-8 sequences
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" is"])

        # Test with emoji
        result = self.runner.score(
            query="The capital of France 🇫🇷",
            items=[" is Paris", " is London"],
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        self.assertEqual(len(result), 2)
        # Just validate it didn't crash and returns reasonable values
        for scores in result:
            self.assertEqual(len(scores), 1)
            self.assertGreater(scores[0], 0)

        # Test with Chinese characters
        result = self.runner.score(
            query="东京",  # Tokyo in Chinese
            items=[" is a city", " is a country"],
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        self.assertEqual(len(result), 2)

    def test_score_whitespace_handling(self):
        """
        Test whitespace edge cases.

        Validates:
        - Leading/trailing whitespace preserved (if that's intended behavior)
        - Multiple spaces handled
        - Tabs, newlines
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" to"])

        # Leading/trailing spaces in query
        result = self.runner.score(
            query="  Query with spaces  ",
            items=[" item1"],
            label_token_ids=label_token_ids
        )
        self.assertEqual(len(result), 1)

        # Multiple spaces in items
        result = self.runner.score(
            query="Query",
            items=["  multiple   spaces  "],
            label_token_ids=label_token_ids
        )
        self.assertEqual(len(result), 1)

    def test_score_ordering_preserved(self):
        """
        Test that output order matches input items order.

        Validates:
        - Items returned in same order as input
        - No sorting or reordering
        - Indices align
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" A", " B", " C"])

        items = [" A is first", " B is second", " C is third"]

        result = self.runner.score(
            query="Order test:",
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        # Expect item 0 to score highest for label token " A"
        # (because " A" appears in " A is first")
        self.assertGreater(result[0][0], result[0][1])  # " A" > " B" for item 0
        self.assertGreater(result[0][0], result[0][2])  # " A" > " C" for item 0

        # Similarly for other items
        self.assertGreater(result[1][1], result[1][0])  # " B" > " A" for item 1
        self.assertGreater(result[2][2], result[2][0])  # " C" > " A" for item 2

    def test_score_invalid_types(self):
        """
        Test with invalid types.

        Validates:
        - items not list → TypeError
        - label_token_ids not list → TypeError
        - query not str or list → TypeError
        """
        label_token_ids = get_label_token_ids(self.tokenizer, [" to"])

        # items not a list
        with self.assertRaises(TypeError):
            self.runner.score(
                query="Query",
                items="not a list",  # String instead of list!
                label_token_ids=label_token_ids
            )

        # label_token_ids not a list
        with self.assertRaises(TypeError):
            self.runner.score(
                query="Query",
                items=["item1"],
                label_token_ids=123  # Int instead of list!
            )

        # query not str or list[int]
        with self.assertRaises(TypeError):
            self.runner.score(
                query=12345,  # Int instead of str or list!
                items=["item1"],
                label_token_ids=label_token_ids
            )
```

## HTTP Endpoint Tests Specification

### File: `test/srt/openai_server/basic/test_openai_server.py`

Expand `TestOpenAIV1Score` class:

```python
class TestOpenAIV1Score(CustomTestCase):
    """HTTP endpoint tests for /v1/score"""

    # ... existing test_score_text_input ...

    def test_score_token_input(self):
        """
        Test /v1/score with token inputs.

        Validates:
        - API accepts query/items as token IDs (list[int])
        - Response format matches text input
        - Results are correct
        """
        tokenizer = get_tokenizer(self.model)

        # Tokenize inputs
        query_ids = tokenizer.encode("I pledge allegiance", add_special_tokens=False)
        items_ids = [
            tokenizer.encode(" to", add_special_tokens=False),
            tokenizer.encode(" of", add_special_tokens=False),
        ]
        label_token_ids = get_label_token_ids(tokenizer, [" to", " of"])

        response = self.run_score(
            query=query_ids,
            items=items_ids,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "scoring")
        self.assertEqual(len(data["scores"]), 2)

    def test_score_usage_info(self):
        """
        Test that response includes usage information.

        Validates:
        - usage field present
        - prompt_tokens > 0
        - completion_tokens == 0 (scoring has no completion)
        - total_tokens == prompt_tokens
        """
        tokenizer = get_tokenizer(self.model)
        label_token_ids = get_label_token_ids(tokenizer, [" to", " of"])

        response = self.run_score(
            query="I pledge allegiance",
            items=[" to", " of"],
            label_token_ids=label_token_ids
        )

        data = response.json()
        self.assertIn("usage", data)

        usage = data["usage"]
        self.assertIn("prompt_tokens", usage)
        self.assertIn("completion_tokens", usage)
        self.assertIn("total_tokens", usage)

        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)  # No completion in scoring
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"])

    def test_score_error_handling(self):
        """
        Test error handling and status codes.

        Validates:
        - Invalid label_token_ids → 400 Bad Request
        - Missing required fields → 400 Bad Request
        - Wrong types → 400 Bad Request
        - Error response has proper schema
        """
        # Invalid label_token_ids (empty)
        response = self.run_score(
            query="Test",
            items=["item"],
            label_token_ids=[]  # Empty!
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

        # Missing required field (items)
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "query": "Test",
                # Missing "items"!
                "label_token_ids": [123]
            }
        )
        self.assertEqual(response.status_code, 400)

        # Wrong type for query
        response = self.run_score(
            query=12345,  # Should be str or list[int]
            items=["item"],
            label_token_ids=[123]
        )
        self.assertEqual(response.status_code, 400)

    def test_score_default_fields(self):
        """
        Test default fields in response.

        Validates:
        - "object": "scoring"
        - "model" field present and correct
        - "created" timestamp present
        - "id" field present
        """
        tokenizer = get_tokenizer(self.model)
        label_token_ids = get_label_token_ids(tokenizer, [" to"])

        response = self.run_score(
            query="Test",
            items=["item"],
            label_token_ids=label_token_ids
        )

        data = response.json()

        # Required fields
        self.assertEqual(data["object"], "scoring")
        self.assertIn("model", data)
        self.assertEqual(data["model"], self.model)
        self.assertIn("created", data)
        self.assertIsInstance(data["created"], int)
        self.assertIn("id", data)

    def test_score_ordering(self):
        """
        Test that scores array order matches items order.

        Validates:
        - Response scores[i] corresponds to request items[i]
        - No reordering
        """
        tokenizer = get_tokenizer(self.model)
        label_token_ids = get_label_token_ids(tokenizer, [" A", " B", " C"])

        items = [" A item", " B item", " C item"]

        response = self.run_score(
            query="Prefix:",
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        data = response.json()
        scores = data["scores"]

        # Expect scores[0] (for " A item") to have highest value at index 0 (label " A")
        self.assertGreater(scores[0][0], scores[0][1])
        self.assertGreater(scores[0][0], scores[0][2])

        # Similarly for others
        self.assertGreater(scores[1][1], scores[1][0])
        self.assertGreater(scores[2][2], scores[2][0])
```

## Protocol Tests Specification

### File: `test/srt/openai_server/basic/test_protocol.py`

Add new test class:

```python
class TestScoringProtocol(CustomTestCase):
    """Protocol validation tests for scoring API"""

    def test_scoring_request_validation(self):
        """
        Test request validation (required fields).

        Validates:
        - query is required
        - items is required
        - label_token_ids is required
        - Missing any field → validation error
        """
        from sgl_jax.openai_api_protocol import ScoringRequest
        from pydantic import ValidationError

        # Valid request
        valid = ScoringRequest(
            query="Test",
            items=["item1"],
            label_token_ids=[123]
        )
        self.assertIsNotNone(valid)

        # Missing query
        with self.assertRaises(ValidationError):
            ScoringRequest(
                items=["item1"],
                label_token_ids=[123]
            )

        # Missing items
        with self.assertRaises(ValidationError):
            ScoringRequest(
                query="Test",
                label_token_ids=[123]
            )

        # Missing label_token_ids
        with self.assertRaises(ValidationError):
            ScoringRequest(
                query="Test",
                items=["item1"]
            )

    def test_scoring_request_defaults(self):
        """
        Test default parameter values.

        Validates:
        - apply_softmax defaults to False
        - item_first defaults to False
        """
        from sgl_jax.openai_api_protocol import ScoringRequest

        req = ScoringRequest(
            query="Test",
            items=["item1"],
            label_token_ids=[123]
        )

        self.assertEqual(req.apply_softmax, False)
        self.assertEqual(req.item_first, False)

    def test_scoring_request_accepts_token_and_text(self):
        """
        Test that request accepts both text and token inputs.

        Validates:
        - query can be str or list[int]
        - items can be list[str] or list[list[int]]
        - Both validated correctly
        """
        from sgl_jax.openai_api_protocol import ScoringRequest

        # Text inputs
        req_text = ScoringRequest(
            query="Text query",
            items=["item1", "item2"],
            label_token_ids=[123, 456]
        )
        self.assertIsInstance(req_text.query, str)
        self.assertIsInstance(req_text.items[0], str)

        # Token inputs
        req_tokens = ScoringRequest(
            query=[123, 456],
            items=[[789, 101], [112, 131]],
            label_token_ids=[123, 456]
        )
        self.assertIsInstance(req_tokens.query, list)
        self.assertIsInstance(req_tokens.items[0], list)

    def test_scoring_response_serialization(self):
        """
        Test response serialization.

        Validates:
        - exclude_none=True applied
        - No null fields in JSON
        - All required fields present
        """
        from sgl_jax.openai_api_protocol import ScoringResponse

        response = ScoringResponse(
            id="test-123",
            object="scoring",
            created=1234567890,
            model="test-model",
            scores=[[0.8, 0.2], [0.3, 0.7]],
            usage={"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        )

        # Serialize to JSON
        json_data = response.model_dump(exclude_none=True)

        # Check no None values
        self.assertNotIn(None, json_data.values())

        # Check required fields
        self.assertIn("id", json_data)
        self.assertIn("object", json_data)
        self.assertIn("scores", json_data)
```

## Test Execution Plan

### Step 1: Implement Edge Case Tests

```bash
cd /Users/kanna/Sandbox/sglang-jax

# Create file
touch test/srt/test_score_api_edge_cases.py

# Implement tests one by one
python3 -m unittest test.srt.test_score_api_edge_cases.TestScoreAPIEdgeCases.test_score_empty_items -v

# Continue for all 10 edge case tests
```

### Step 2: Run Edge Case Suite

```bash
# All edge case tests together
python3 -m unittest test.srt.test_score_api_edge_cases -v

# Expected: 10 tests pass, < 2 min runtime
```

### Step 3: Implement HTTP Tests

```bash
# Add tests to existing file
python3 -m unittest test.srt.openai_server.basic.test_openai_server.TestOpenAIV1Score -v

# Expected: 6 tests pass (1 existing + 5 new)
```

### Step 4: Implement Protocol Tests

```bash
# Add TestScoringProtocol class
python3 -m unittest test.srt.openai_server.basic.test_protocol.TestScoringProtocol -v

# Expected: 4 tests pass
```

### Step 5: Run Full Combined Suite

```bash
# All new tests together
python3 -m unittest discover test/srt -p "test_score*.py" -v

# Expected: 12 core + 10 edge + 6 HTTP + 4 protocol = 32 tests pass
```

## Success Criteria

- [ ] 10 edge case tests implemented and passing
- [ ] 5 new HTTP endpoint tests passing
- [ ] 4 protocol validation tests passing
- [ ] All error messages are clear and actionable
- [ ] No test takes > 10 seconds (fast validation)
- [ ] Combined suite passes on TPU
- [ ] Line coverage on score_request() >= 98%

## Dependencies

- Test Plan 001 (requires `score_test_utils.py`)
- Input validation in `tokenizer_manager.py` (from Plan 001)

## Risks

1. **Error message fragility** - Tests check error message content
   - Mitigation: Check for key terms, not exact wording

2. **HTTP server instability** - Server may crash during tests
   - Mitigation: Proper cleanup in tearDown, retry logic

3. **Protocol changes** - Pydantic models may change
   - Mitigation: Import from actual protocol file, not mock

## Follow-Up

After this phase completes:
- Test Plan 003: JAX Features + Perf (advanced tests)
- Update CI configuration to include all new tests
- Document edge cases in API reference
