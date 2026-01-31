# RFC-006: Error Handling and API Contract

**Status:** Draft
**Author:** Engineering Team
**Created:** 2026-01-29
**Updated:** 2026-01-29
**Related RFC:** RFC-003, RFC-005

## Summary

Define a comprehensive error handling specification and API contract for the `/v1/score` Scoring API, including HTTP status codes, error response formats, validation rules, and error messages that align with OpenAI API conventions.

## Motivation

### Current State

- Score API has basic error handling (RFC-001 added validation for empty `output_token_ids_logprobs`)
- RFC-003 proposes edge case tests but doesn't define expected error behavior
- **No documented error response format**
- **No defined HTTP status code semantics**
- **Unclear: When 400 vs 422 vs 500?**

### Problems

1. **Inconsistent error responses** - Different error conditions may return different formats
2. **Undocumented status codes** - Users don't know what 400 vs 422 means
3. **Cryptic error messages** - Internal errors may leak implementation details
4. **No error codes** - Can't programmatically handle specific errors
5. **OpenAI incompatibility** - Error format may not match OpenAI client expectations

### Goals

1. **Define** error response schema (OpenAI-compatible)
2. **Document** HTTP status code usage
3. **Specify** validation rules and corresponding errors
4. **Provide** clear, actionable error messages
5. **Enable** programmatic error handling via error codes

## Proposed Solution

### Error Response Schema

All errors follow the OpenAI error response format:

```json
{
    "error": {
        "message": "Human-readable error description",
        "type": "error_category",
        "param": "field_name_if_applicable",
        "code": "machine_readable_error_code"
    }
}
```

#### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | Human-readable description of the error |
| `type` | string | Yes | Error category (see Error Types) |
| `param` | string | No | Parameter that caused the error |
| `code` | string | No | Machine-readable error code |

### Error Types

| Type | Description | HTTP Status |
|------|-------------|-------------|
| `invalid_request_error` | Request validation failed | 400 |
| `invalid_value_error` | Parameter value is invalid | 400 |
| `missing_parameter_error` | Required parameter missing | 400 |
| `model_error` | Model-related error | 400 or 500 |
| `server_error` | Internal server error | 500 |
| `rate_limit_error` | Rate limit exceeded | 429 |
| `authentication_error` | Auth failed | 401 |

### HTTP Status Codes

#### 400 Bad Request

**When:** Client sent an invalid request that can be fixed by the client.

```json
{
    "error": {
        "message": "items cannot be empty",
        "type": "invalid_request_error",
        "param": "items",
        "code": "empty_items"
    }
}
```

**Use 400 for:**
- Missing required parameters
- Empty arrays that should have items
- Invalid types (string where int expected)
- Negative values where positive required
- Values out of valid range

#### 422 Unprocessable Entity

**When:** Request is syntactically valid but semantically invalid.

```json
{
    "error": {
        "message": "label_token_ids contains token ID 999999 which exceeds vocabulary size 128256",
        "type": "invalid_value_error",
        "param": "label_token_ids",
        "code": "token_id_exceeds_vocab"
    }
}
```

**Use 422 for:**
- Token IDs exceeding vocabulary size
- Incompatible parameter combinations
- Model-specific validation failures

#### 500 Internal Server Error

**When:** Server failed to process a valid request.

```json
{
    "error": {
        "message": "An internal error occurred. Please try again.",
        "type": "server_error",
        "code": "internal_error"
    }
}
```

**Use 500 for:**
- Unexpected exceptions
- Resource exhaustion (OOM)
- Model inference failures
- Infrastructure issues

**Note:** Never expose stack traces or internal details in 500 errors.

#### 429 Too Many Requests

**When:** Rate limit exceeded (if implemented).

```json
{
    "error": {
        "message": "Rate limit exceeded. Please retry after 60 seconds.",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded"
    }
}
```

### Validation Rules and Error Codes

#### Parameter: `query`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Missing | 400 | `missing_parameter_error` | `missing_query` | `query is required` |
| Empty string | 400 | `invalid_value_error` | `empty_query` | `query cannot be empty` |
| Wrong type | 400 | `invalid_request_error` | `invalid_query_type` | `query must be a string or list of integers` |

#### Parameter: `items`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Missing | 400 | `missing_parameter_error` | `missing_items` | `items is required` |
| Empty array | 400 | `invalid_value_error` | `empty_items` | `items cannot be empty. At least one item is required.` |
| Wrong type | 400 | `invalid_request_error` | `invalid_items_type` | `items must be a list of strings or list of token ID lists` |
| Mixed types | 400 | `invalid_request_error` | `mixed_input_types` | `query and items must both be text (str) or both be tokens (list[int])` |

#### Parameter: `label_token_ids`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Missing | 400 | `missing_parameter_error` | `missing_label_token_ids` | `label_token_ids is required` |
| Empty array | 400 | `invalid_value_error` | `empty_label_token_ids` | `label_token_ids cannot be empty. At least one label token ID is required.` |
| Negative value | 400 | `invalid_value_error` | `negative_token_id` | `label_token_ids cannot contain negative values. Got: [-1]` |
| Exceeds vocab | 422 | `invalid_value_error` | `token_id_exceeds_vocab` | `label_token_ids contains token ID {id} which exceeds vocabulary size {vocab_size}` |
| Wrong type | 400 | `invalid_request_error` | `invalid_label_token_ids_type` | `label_token_ids must be a list of integers` |
| Non-integer value | 400 | `invalid_request_error` | `invalid_token_id_type` | `label_token_ids must contain only integers` |

#### Parameter: `apply_softmax`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Wrong type | 400 | `invalid_request_error` | `invalid_apply_softmax_type` | `apply_softmax must be a boolean` |

#### Parameter: `item_first`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Wrong type | 400 | `invalid_request_error` | `invalid_item_first_type` | `item_first must be a boolean` |

#### Parameter: `model`

| Condition | Status | Type | Code | Message |
|-----------|--------|------|------|---------|
| Missing | 400 | `missing_parameter_error` | `missing_model` | `model is required` |
| Not found | 400 | `model_error` | `model_not_found` | `Model '{model}' not found. Available models: [...]` |
| Not loaded | 500 | `model_error` | `model_not_loaded` | `Model '{model}' is not currently loaded` |

### Implementation

#### Validation Layer

```python
# python/sgl_jax/srt/managers/tokenizer_manager.py

class ValidationError(Exception):
    """Raised when request validation fails."""
    def __init__(self, message: str, error_type: str, param: str = None, code: str = None):
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)

    def to_dict(self) -> dict:
        error = {
            "message": self.message,
            "type": self.error_type,
        }
        if self.param:
            error["param"] = self.param
        if self.code:
            error["code"] = self.code
        return {"error": error}


def validate_score_request(
    query: Union[str, List[int]],
    items: Union[List[str], List[List[int]]],
    label_token_ids: List[int],
    vocab_size: int,
    apply_softmax: bool = False,
    item_first: bool = False
) -> None:
    """
    Validate score request parameters.

    Raises:
        ValidationError: If any parameter is invalid
    """
    # Validate query
    if query is None:
        raise ValidationError(
            message="query is required",
            error_type="missing_parameter_error",
            param="query",
            code="missing_query"
        )

    if isinstance(query, str) and len(query) == 0:
        raise ValidationError(
            message="query cannot be empty",
            error_type="invalid_value_error",
            param="query",
            code="empty_query"
        )

    # Validate items
    if items is None:
        raise ValidationError(
            message="items is required",
            error_type="missing_parameter_error",
            param="items",
            code="missing_items"
        )

    if not isinstance(items, list):
        raise ValidationError(
            message="items must be a list of strings or list of token ID lists",
            error_type="invalid_request_error",
            param="items",
            code="invalid_items_type"
        )

    if len(items) == 0:
        raise ValidationError(
            message="items cannot be empty. At least one item is required.",
            error_type="invalid_value_error",
            param="items",
            code="empty_items"
        )

    # Validate type consistency
    query_is_text = isinstance(query, str)
    items_is_text = isinstance(items[0], str) if items else True

    if query_is_text != items_is_text:
        raise ValidationError(
            message=f"query and items must both be text (str) or both be tokens (list[int]). "
                    f"Got query type: {'str' if query_is_text else 'list[int]'}, "
                    f"items[0] type: {'str' if items_is_text else 'list[int]'}",
            error_type="invalid_request_error",
            param="items",
            code="mixed_input_types"
        )

    # Validate label_token_ids
    if label_token_ids is None:
        raise ValidationError(
            message="label_token_ids is required",
            error_type="missing_parameter_error",
            param="label_token_ids",
            code="missing_label_token_ids"
        )

    if not isinstance(label_token_ids, list):
        raise ValidationError(
            message="label_token_ids must be a list of integers",
            error_type="invalid_request_error",
            param="label_token_ids",
            code="invalid_label_token_ids_type"
        )

    if len(label_token_ids) == 0:
        raise ValidationError(
            message="label_token_ids cannot be empty. At least one label token ID is required.",
            error_type="invalid_value_error",
            param="label_token_ids",
            code="empty_label_token_ids"
        )

    # Check for non-integers
    non_integers = [x for x in label_token_ids if not isinstance(x, int)]
    if non_integers:
        raise ValidationError(
            message="label_token_ids must contain only integers",
            error_type="invalid_request_error",
            param="label_token_ids",
            code="invalid_token_id_type"
        )

    # Check for negative values
    negative_ids = [x for x in label_token_ids if x < 0]
    if negative_ids:
        raise ValidationError(
            message=f"label_token_ids cannot contain negative values. Got: {negative_ids}",
            error_type="invalid_value_error",
            param="label_token_ids",
            code="negative_token_id"
        )

    # Check vocabulary bounds
    exceeds_vocab = [x for x in label_token_ids if x >= vocab_size]
    if exceeds_vocab:
        raise ValidationError(
            message=f"label_token_ids contains token ID {exceeds_vocab[0]} "
                    f"which exceeds vocabulary size {vocab_size}",
            error_type="invalid_value_error",
            param="label_token_ids",
            code="token_id_exceeds_vocab"
        )
```

#### HTTP Error Handler

```python
# python/sgl_jax/srt/openai_api_server.py

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Convert ValidationError to OpenAI-compatible error response."""
    status_code = 400
    if exc.code and exc.code.startswith("token_id_exceeds"):
        status_code = 422

    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict()
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors with safe error response."""
    # Log the actual error for debugging
    logger.exception("Unexpected error processing request")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal error occurred. Please try again.",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )
```

### Error Handling Tests

#### New tests in `test/srt/test_score_api_edge_cases.py`:

```python
class TestScoreAPIErrorHandling:
    """Test error responses for invalid inputs."""

    def test_error_empty_items_returns_400(self):
        """Empty items should return 400 with proper error format."""
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [],
                "label_token_ids": [123]
            }
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_value_error"
        assert error["param"] == "items"
        assert error["code"] == "empty_items"
        assert "cannot be empty" in error["message"]

    def test_error_negative_token_id_returns_400(self):
        """Negative token IDs should return 400."""
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [" item"],
                "label_token_ids": [-1, 123]
            }
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "negative_token_id"

    def test_error_token_exceeds_vocab_returns_422(self):
        """Token ID exceeding vocab should return 422."""
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [" item"],
                "label_token_ids": [999999999]  # Way beyond vocab
            }
        )

        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "token_id_exceeds_vocab"

    def test_error_missing_query_returns_400(self):
        """Missing query should return 400."""
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                # "query" missing
                "items": [" item"],
                "label_token_ids": [123]
            }
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "missing_query"

    def test_error_format_matches_openai(self):
        """Error format should match OpenAI specification."""
        response = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "query": "Test",
                "items": [],
                "label_token_ids": [123]
            }
        )

        data = response.json()

        # Must have "error" key
        assert "error" in data

        # Error must have required fields
        error = data["error"]
        assert "message" in error
        assert "type" in error

        # Message should be human-readable
        assert len(error["message"]) > 10
        assert error["message"][0].isupper()  # Proper sentence
```

## Implementation Plan

### Phase 1: Define Contract

- [x] Document error response schema
- [x] Define HTTP status code semantics
- [x] Create validation rules table
- [x] Define error codes

### Phase 2: Implement Validation

- [ ] Create `ValidationError` exception class
- [ ] Implement `validate_score_request()` function
- [ ] Add validation to `score_request()` method
- [ ] Test validation locally

### Phase 3: HTTP Integration

- [ ] Add exception handlers to FastAPI app
- [ ] Ensure 500 errors don't leak details
- [ ] Test HTTP responses match spec

### Phase 4: Test Coverage

- [ ] Add error handling tests to `test_score_api_edge_cases.py`
- [ ] Test each error code
- [ ] Verify OpenAI client compatibility (RFC-005)
- [ ] Add to CI

### Phase 5: Documentation

- [ ] Add error handling section to API docs
- [ ] Document error codes in README
- [ ] Add troubleshooting guide

## Cost Analysis

**Development Cost:**
- Implementation: 4 hours
- Testing: 2 hours
- Documentation: 1 hour
- **Total:** 7 hours

**Ongoing Cost:**
- Negligible (validation adds microseconds)

**ROI:**
- Reduces debugging time for users
- Fewer support requests
- Better developer experience

## Alternatives Considered

### Alternative 1: Use 422 for All Validation Errors

**Pros:**
- Clearer semantic meaning ("valid JSON but invalid content")
- Some frameworks prefer this

**Cons:**
- OpenAI uses 400 for most validation errors
- Less compatible with OpenAI client

**Why rejected:** OpenAI compatibility is primary goal.

### Alternative 2: Simple Error Messages Without Codes

**Pros:**
- Less implementation work
- Simpler API

**Cons:**
- Can't programmatically handle errors
- Brittle string matching

**Why rejected:** Error codes enable better error handling.

### Alternative 3: Return Errors in Request Body (200 OK)

**Pros:**
- Some APIs do this
- Simpler client handling

**Cons:**
- Non-standard
- Breaks HTTP semantics
- OpenAI doesn't do this

**Why rejected:** Violates HTTP standards and OpenAI compatibility.

## Open Questions

- [ ] Should we include request ID in error responses for debugging?
- [ ] Add `retry_after` header for rate limit errors?
- [ ] Include documentation links in error messages?
- [ ] Support error localization (i18n)?

## Success Metrics

- [ ] All validation errors return proper format
- [ ] No 500 errors leak internal details
- [ ] OpenAI client handles all errors correctly
- [ ] Error codes documented and tested
- [ ] Zero error format complaints from users

## References

- OpenAI Error Handling: https://platform.openai.com/docs/guides/error-codes
- RFC-003: Comprehensive Score API Test Suite
- RFC-005: OpenAI Client Compatibility
- HTTP Status Codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
