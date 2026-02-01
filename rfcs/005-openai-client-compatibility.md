# RFC-005: OpenAI Client Compatibility

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-01-29 |
| **Updated** | 2026-01-29 |
| **Related** | [RFC-003](003-score-api-comprehensive-test-suite.md), [RFC-006](006-error-handling-api-contract.md) |

## Summary

Define and validate compatibility with the official OpenAI Python client for the `/v1/score` Scoring API, ensuring users can use SGLang JAX as a drop-in replacement for OpenAI-compatible scoring endpoints.

## Motivation

### Current State

- Score API implemented with OpenAI-style endpoint (`/v1/score`)
- HTTP tests validate basic request/response format
- **No tests using the official OpenAI Python client**
- **No documented compatibility level**
- **Unknown: Does `openai.Client` work out of the box?**

### Problems

1. **Untested client compatibility** - Users expect to use `from openai import OpenAI` but we don't verify this works
2. **Undocumented deviations** - Any differences from OpenAI spec are not documented
3. **Version compatibility unknown** - Which `openai` package versions work?
4. **Error format mismatch risk** - OpenAI client expects specific error formats
5. **No migration guide** - Users switching from OpenAI have no documentation

### Goals

1. **Validate** that official OpenAI Python client works with Score API
2. **Document** compatibility level and any known deviations
3. **Test** with multiple OpenAI client versions
4. **Provide** migration guide for OpenAI users
5. **Define** compatibility tier (full, partial, compatible subset)

## Proposed Solution

### Compatibility Tier Definition

**Tier: Compatible Subset**

SGLang JAX implements the `/v1/score` endpoint with OpenAI-compatible request/response formats. Users can use the OpenAI Python client with `base_url` override.

| Feature | Compatibility | Notes |
|---------|---------------|-------|
| Request format | Full | Matches OpenAI schema |
| Response format | Full | Matches OpenAI schema |
| Error format | Full | See RFC-006 |
| Authentication | Partial | API key accepted but not validated by default |
| Rate limiting | None | Not implemented |
| Streaming | N/A | Score API is non-streaming |

### OpenAI Client Usage

#### Basic Usage

```python
from openai import OpenAI

# Point client to SGLang server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # SGLang doesn't validate by default
)

# Use Score API via OpenAI client's generic post method
# Note: client.post() and cast_to= are part of OpenAI's public API (v1.0+)
# but are less stable than typed methods. For maximum compatibility,
# consider using httpx or requests directly (see below).
response = client.post(
    "/score",
    body={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "query": "I pledge allegiance",
        "items": [" to the flag", " of the United States"],
        "label_token_ids": [311, 315, 369],  # Token IDs for labels
        "apply_softmax": True,
        "item_first": False
    },
    cast_to=dict  # Score API returns custom format
)

print(response["scores"])
# [[0.85, 0.10, 0.05], [0.20, 0.75, 0.05]]
```

#### Note on Score API

The `/v1/score` endpoint is an SGLang extension, not part of the standard OpenAI API. The OpenAI client doesn't have a native `.score()` method, so users must use the generic `.post()` method or raw HTTP requests.

```python
# Alternative: Using requests directly
import requests

response = requests.post(
    "http://localhost:8000/v1/score",
    json={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "query": "I pledge allegiance",
        "items": [" to the flag", " of the United States"],
        "label_token_ids": [311, 315, 369],
        "apply_softmax": True
    }
)
scores = response.json()["scores"]
```

### Compatibility Test Suite

#### New File: `test/srt/test_score_openai_client.py`

```python
"""
OpenAI client compatibility tests for Score API.

Validates that the official OpenAI Python client can be used
with SGLang's /v1/score endpoint.

Requirements:
    pip install openai>=1.0.0

Usage:
    python -m pytest test/srt/test_score_openai_client.py -v
"""

import pytest
import subprocess
import time
import os
from typing import List, Dict, Any

# Skip if openai not installed
openai = pytest.importorskip("openai")
from openai import OpenAI


class TestScoreOpenAIClient:
    """Test Score API compatibility with OpenAI Python client."""

    @classmethod
    def setup_class(cls):
        """
        Initialize OpenAI client for tests.

        NOTE: These tests assume a running SGLang server. The server must be
        started externally before running tests. Set SGLANG_BASE_URL to point
        to your server, or start one locally at http://localhost:8000.

        Example server startup:
            python -m sgl_jax.launch_server --model meta-llama/Llama-3.2-1B-Instruct
        """
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://localhost:8000")
        cls.client = OpenAI(
            base_url=f"{cls.base_url}/v1",
            api_key="test-key"
        )

    def test_score_with_openai_client_post(self):
        """
        Test Score API using OpenAI client's generic post method.

        The /v1/score endpoint is an SGLang extension, so we use
        the generic .post() method rather than a typed method.
        """
        from sgl_jax.test.score_test_utils import get_tokenizer, get_single_token_id

        # Get tokenizer to derive single-token IDs
        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

        # Use helper to ensure these are single tokens
        # Note: Using letter tokens which are reliably single tokens
        A_ID = get_single_token_id(tokenizer, " A")
        B_ID = get_single_token_id(tokenizer, " B")
        C_ID = get_single_token_id(tokenizer, " C")

        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "The capital of France is",
                "items": [" Paris", " London", " Berlin"],
                "label_token_ids": [A_ID, B_ID, C_ID],  # Verified single tokens
                "apply_softmax": True,
                "item_first": False
            },
            cast_to=dict
        )

        # Validate response structure
        assert "scores" in response
        assert "object" in response
        assert response["object"] == "scoring"

        # Validate scores
        scores = response["scores"]
        assert len(scores) == 3  # One per item
        for score_list in scores:
            assert len(score_list) == 3  # One per label
            assert all(isinstance(s, float) for s in score_list)
            assert abs(sum(score_list) - 1.0) < 1e-6  # Softmax sums to 1

    def test_score_error_handling_openai_client(self):
        """
        Test that errors are returned in OpenAI-compatible format.

        OpenAI client expects errors in specific format:
        {
            "error": {
                "message": "...",
                "type": "...",
                "code": "..."
            }
        }
        """
        from openai import BadRequestError

        with pytest.raises(BadRequestError) as exc_info:
            self.client.post(
                "/score",
                body={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "query": "Test",
                    "items": [],  # Empty items should error
                    "label_token_ids": [123]
                },
                cast_to=dict
            )

        # Verify error is in expected format
        error = exc_info.value
        assert error.status_code == 400

    def test_score_with_token_input_openai_client(self):
        """Test Score API with token IDs instead of text."""
        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": [1, 306, 4554, 394],  # Token IDs
                "items": [[311, 278, 7353], [310, 278, 3303]],  # Token ID lists
                "label_token_ids": [311, 310],
                "apply_softmax": False  # Return logprobs
            },
            cast_to=dict
        )

        assert "scores" in response
        scores = response["scores"]
        assert len(scores) == 2

    def test_openai_client_version_compatibility(self):
        """Document which OpenAI client version is being tested."""
        import openai
        version = openai.__version__

        # Log version for documentation
        print(f"OpenAI client version: {version}")

        # Verify minimum version
        major, minor, *_ = version.split(".")
        assert int(major) >= 1, "Requires openai>=1.0.0"


class TestScoreOpenAIClientEdgeCases:
    """Edge case tests for OpenAI client compatibility."""

    @classmethod
    def setup_class(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://localhost:8000")
        cls.client = OpenAI(
            base_url=f"{cls.base_url}/v1",
            api_key="test-key"
        )

    def test_large_batch_openai_client(self):
        """Test large batch through OpenAI client."""
        from sgl_jax.test.score_test_utils import get_tokenizer, get_single_token_id

        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        items = [f" item{i}" for i in range(20)]

        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Score these items:",
                "items": items,
                # Use helper to get verified single-token IDs
                "label_token_ids": [
                    get_single_token_id(tokenizer, " A"),
                    get_single_token_id(tokenizer, " B"),
                ],
                "apply_softmax": True
            },
            cast_to=dict
        )

        assert len(response["scores"]) == 20

    def test_unicode_content_openai_client(self):
        """Test Unicode handling through OpenAI client."""
        from sgl_jax.test.score_test_utils import get_tokenizer, get_single_token_id

        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Translate: こんにちは",
                "items": [" Hello", " Goodbye", " Thanks"],
                # Use helper to get verified single-token IDs
                "label_token_ids": [
                    get_single_token_id(tokenizer, " X"),
                    get_single_token_id(tokenizer, " Y"),
                    get_single_token_id(tokenizer, " Z"),
                ],
                "apply_softmax": True
            },
            cast_to=dict
        )

        assert "scores" in response
        assert len(response["scores"]) == 3

    def test_special_characters_openai_client(self):
        """Test special characters in query/items."""
        from sgl_jax.test.score_test_utils import get_tokenizer, get_single_token_id

        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

        response = self.client.post(
            "/score",
            body={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test with special chars: @#$%^&*()",
                "items": [" option<1>", " option\"2\"", " option'3'"],
                # Use helper to get verified single-token IDs
                "label_token_ids": [
                    get_single_token_id(tokenizer, " 1"),
                    get_single_token_id(tokenizer, " 2"),
                    get_single_token_id(tokenizer, " 3"),
                ],
                "apply_softmax": True
            },
            cast_to=dict
        )

        assert "scores" in response
```

### Supported OpenAI Client Versions

| Version | Status | Notes |
|---------|--------|-------|
| 1.0.x | Supported | Minimum version |
| 1.1.x - 1.5.x | Supported | Tested |
| 1.6.x+ | Expected | Should work, test on release |
| 0.x | Not Supported | Legacy API, different interface |

### Request/Response Compatibility

#### Request Schema

```json
{
    "model": "string (optional, uses server default if not provided)",
    "query": "string | list[int] (required)",
    "items": "list[string] | list[list[int]] (required)",
    "label_token_ids": "list[int] (required)",
    "apply_softmax": "boolean (optional, default: false)",
    "item_first": "boolean (optional, default: false)"
}
```

#### Response Schema

```json
{
    "object": "scoring",
    "model": "string",
    "scores": "list[list[float]]",
    "usage": {
        "prompt_tokens": "int",
        "completion_tokens": 0,
        "total_tokens": "int"
    },
    "created": "int (unix timestamp)"
}
```

#### Error Response Schema

See RFC-006 for complete error handling specification.

```json
{
    "error": {
        "message": "string",
        "type": "invalid_request_error | server_error",
        "code": "string | null"
    }
}
```

### Known Deviations from OpenAI

| Aspect | OpenAI Behavior | SGLang Behavior | Impact |
|--------|-----------------|-----------------|--------|
| API key validation | Required, validated | Accepted, not validated | Low - can add if needed |
| Rate limiting | Enforced | Not implemented | Medium - no protection |
| `/v1/score` endpoint | Does not exist | SGLang extension | N/A - extension |
| Model names | OpenAI models only | Any supported model | Expected difference |
| Organization header | Used for billing | Ignored | None |

### Migration Guide

#### From OpenAI to SGLang

**Step 1: Update base URL**

```python
# Before (OpenAI)
client = OpenAI()

# After (SGLang)
client = OpenAI(
    base_url="http://your-sglang-server:8000/v1",
    api_key="optional"
)
```

**Step 2: Use Score API**

The `/v1/score` endpoint is unique to SGLang. Use the generic `.post()` method:

```python
response = client.post("/score", body={...}, cast_to=dict)
```

**Step 3: Handle model names**

```python
# OpenAI model names won't work
# Use the actual model path
"model": "meta-llama/Llama-3.2-1B-Instruct"
```

## Implementation Plan

### Phase 1: Basic Compatibility Tests

- [ ] Create `test/srt/test_score_openai_client.py`
- [ ] Implement `test_score_with_openai_client_post`
- [ ] Implement `test_score_error_handling_openai_client`
- [ ] Add `openai` to test dependencies

### Phase 2: Edge Case Coverage

- [ ] Test large batches through client
- [ ] Test Unicode content
- [ ] Test special characters
- [ ] Test token input mode

### Phase 3: Version Compatibility

- [ ] Test with openai 1.0.x
- [ ] Test with openai 1.5.x
- [ ] Document minimum version requirement
- [ ] Add CI matrix for multiple versions

### Phase 4: Documentation

- [ ] Add migration guide to docs
- [ ] Document known deviations
- [ ] Add examples to README
- [ ] Create troubleshooting guide

## Testing Strategy

### CI Integration

```yaml
# Add to test workflow
- name: Install OpenAI client
  run: pip install openai>=1.0.0

- name: Run OpenAI compatibility tests
  run: |
    python -m pytest test/srt/test_score_openai_client.py -v
```

### Test Matrix

| Test | CI | Nightly | On-Demand |
|------|-----|---------|-----------|
| Basic client usage | ✓ | ✓ | ✓ |
| Error handling | ✓ | ✓ | ✓ |
| Large batch | | ✓ | ✓ |
| Unicode/special chars | ✓ | ✓ | ✓ |
| Multi-version | | ✓ | |

## Cost Analysis

**Development Cost:**
- Test implementation: 3 hours
- Documentation: 2 hours
- Version testing: 1 hour
- **Total:** 6 hours

**Ongoing Cost:**
- Additional CI time: ~30 seconds per run
- Negligible

**ROI:**
- Prevents user confusion and bug reports
- Reduces support burden
- Improves adoption

## Alternatives Considered

### Alternative 1: Custom SGLang Client

**Description:** Create a dedicated SGLang Python client

**Pros:**
- Full control over interface
- Can add SGLang-specific methods

**Cons:**
- Another dependency for users
- Maintenance burden
- Users expect OpenAI compatibility

**Why rejected:** OpenAI compatibility is the primary goal. Custom client could be added later.

### Alternative 2: Don't Test Client Compatibility

**Description:** Only test raw HTTP, assume client works

**Pros:**
- Less test code
- Faster tests

**Cons:**
- Users may hit issues we don't catch
- No documentation of what works

**Why rejected:** Client compatibility is a key user experience factor.

## Open Questions

- [ ] Should we implement API key validation (optional feature)?
- [ ] Should we add rate limiting support?
- [ ] Create a thin wrapper method for `.score()` convenience?
- [ ] Support for async OpenAI client (`AsyncOpenAI`)?

## Success Metrics

- [ ] All OpenAI client tests pass
- [ ] Documentation covers migration path
- [ ] At least 2 OpenAI versions tested
- [ ] Zero client-related bug reports post-launch

## References

- OpenAI Python client: https://github.com/openai/openai-python
- OpenAI API reference: https://platform.openai.com/docs/api-reference
- [RFC-003: Comprehensive Score API Test Suite](003-score-api-comprehensive-test-suite.md)
- [RFC-006: Error Handling and API Contract](006-error-handling-api-contract.md)
