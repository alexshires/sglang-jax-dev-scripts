# RFC-012: CI/CD Optimization Strategy

| | |
|------------|------|
| **Status** | Draft |
| **Author** | Engineering Team |
| **Created** | 2026-02-05 |
| **Updated** | 2026-02-05 |
| **Related** | [RFC-002](002-cicd-tpu-testing.md) |

## Summary

This RFC defines a strategy to optimize the resource usage of the sglang-jax CI/CD pipeline, specifically reducing TPU consumption.

## Problem

The current CI runs expensive TPU tests on every PR commit, even if there are syntax errors or linting failures. Additionally, running the full test suite (E2E, Accuracy, Performance) consumes significant TPU resources and time, which is inefficient for rapid development cycles.

## Proposed Changes

### 1. Linter Gate

Introduce a `lint` job running on standard Ubuntu (CPU) runners as a dependency for all TPU jobs.

- **Tools:** `ruff`, `black`, `mypy` (as configured in `pyproject.toml`).
- **Behavior:** If `lint` fails, no TPU jobs (`unit-test`, `e2e-test`, etc.) will start.
- **Benefit:** Saves TPU time on commits with basic style/syntax errors.

### 2. Temporary Test Reduction (Dev Mode)

For the current development phase where we want fast feedback on specific components (like the Score API or core utils), we will temporarily disable the heavy "all-or-nothing" test suites.

- **Action:** Comment out `e2e-test`, `accuracy-test`, `performance-test` in `pr-test.yml`.
- **Action:** Retain `unit-test-1-tpu` but restrict it to a minimal set of smoke tests (e.g., `python/sgl_jax/test/test_utils.py`).
- **Action:** Retain `tpu-score-api-test.yml` as the primary focus for the current sprint.

## Workflow Dependencies

```mermaid
graph TD
    A[Push/PR] --> B[Lint (CPU)]
    B -->|Success| C[Unit Test (TPU v6e-1)]
    B -->|Success| D[Score API Test (TPU v6e-1)]
    B -.->|Disabled| E[E2E Test]
    B -.->|Disabled| F[Accuracy Test]
```

## Implementation Plan

1.  **Modify `pr-test.yml`**:
    - Add `lint` job.
    - Add `needs: [lint]` to all other jobs.
    - Comment out resource-heavy jobs.
2.  **Modify `tpu-score-api-test.yml`**:
    - Add `lint` job (or reuse logic).
    - Add `needs: [lint]`.

## Future Work

- Re-enable full suites on `main` branch pushes or via a specific label (`run-full-ci`).
- Use GitHub Actions `concurrency` groups more aggressively to cancel stale runs.
