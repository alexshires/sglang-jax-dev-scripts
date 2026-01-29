# Document Index

Quick reference for all design documents, decisions, and guides in this repository.

**Last Updated:** 2026-01-29

## RFCs (Request for Comments)

### Active

- **[RFC-001: Score API Comprehensive Tests](rfcs/001-score-api-comprehensive-tests.md)**
  - Status: Implemented
  - Comprehensive test suite for `/v1/score` API
  - 4 Tier 1 tests covering numerical correctness, batching, optimization, HTTP integration
  - Discovered and fixed 3 critical bugs

- **[RFC-002: CI/CD for TPU Testing](rfcs/002-cicd-tpu-testing.md)**
  - Status: Draft
  - Automated TPU testing pipeline using GitHub Actions + gcloud
  - Three-tier strategy: CPU tests (every PR), nightly TPU, on-demand TPU
  - Cost: ~$1/month

- **[RFC-003: Comprehensive Score API Test Suite](rfcs/003-score-api-comprehensive-test-suite.md)**
  - Status: Draft
  - Expands test coverage from 4 to 30+ tests
  - Shared fixtures, CI vs local/perf gating, edge cases, JAX features
  - Behavior validation decisions (empty inputs, negative IDs, mixed types)

### Templates

- **[RFC Template](rfcs/template.md)**
  - Standard format for new RFCs

## ADRs (Architecture Decision Records)

- **[ADR-001: Pure Python Softmax in TokenizerManager](decisions/001-pure-python-softmax.md)**
  - Date: 2026-01-29
  - Status: Accepted
  - Use pure Python softmax instead of JAX to avoid device conflicts
  - TokenizerManager runs in main process, must be device-agnostic

- **[ADR-002: Use gcloud Directly for Unit Tests](decisions/002-no-skypilot-for-unit-tests.md)**
  - Date: 2026-01-29
  - Status: Accepted
  - Use gcloud commands directly instead of SkyPilot for local TPU testing
  - Simpler, more reliable, matches CI approach

### Templates

- **[ADR Template](decisions/template.md)**
  - Standard format for new ADRs

## Investigations

- **[TokenizerManager Architecture](investigations/tokenizer-manager-architecture.md)**
  - Deep dive into multi-process architecture
  - Why TokenizerManager must be device-agnostic
  - Communication flow between main process and Scheduler subprocess

- **[Score API: PyTorch vs JAX Comparison](investigations/score-api-pytorch-vs-jax.md)**
  - Comprehensive comparison of implementations
  - Identified 3 bugs in JAX version by comparing with PyTorch
  - Test coverage gap analysis (17 tests vs 0 → 4)

## Runbooks

- **[Debugging TPU Test Failures](runbooks/debugging-tpu-test-failures.md)**
  - Step-by-step troubleshooting guide
  - Common failure scenarios and fixes
  - CI/CD specific debugging
  - Useful commands and diagnostics

- **[Running Score API Tests](runbooks/running-score-api-tests.md)**
  - How to run tests in CI (automatic)
  - How to run tests on TPU (gcloud-based script)
  - How to run tests locally on CPU
  - Cost management and debugging

## Test Plans

- **[Test Plan 001: Shared Fixtures and Core Tests](test-plans/001-shared-fixtures-and-core-tests.md)**
  - Phase 1-2 (Weeks 1-2), P0 Foundation
  - Create `score_test_utils.py` shared fixtures module
  - Expand core engine tests from 3 to 12 tests
  - Add input validation (empty labels, negative IDs, mixed types)

- **[Test Plan 002: Edge Cases and HTTP/Protocol](test-plans/002-edge-cases-and-http-protocol.md)**
  - Phase 3-4 (Week 3), P0 Critical Coverage
  - 10 edge case validation tests
  - 5 HTTP endpoint tests
  - 4 protocol validation tests

- **[Test Plan 003: JAX Features and Performance](test-plans/003-jax-features-and-performance.md)**
  - Phase 5-6 (Week 4), P1 Important
  - JAX-specific tests (bf16 stability, multi-device sharding)
  - Performance benchmark tool (`bench_score.py`)
  - Baseline establishment and regression detection

## Scripts

*No utility scripts yet*

## Document Relationships

### Score API Implementation Chain

```
RFC-001 (Test Strategy)
    ↓
ADR-001 (Softmax Decision)
    ↓
Investigation: Architecture
    ↓
Investigation: PyTorch vs JAX
    ↓
Runbook: Debugging
```

### CI/CD Chain

```
RFC-002 (CI/CD Plan)
    ↓
Runbook: Debugging
```

## Quick Links by Topic

### Testing
- [RFC-001: Score API Tests](rfcs/001-score-api-comprehensive-tests.md)
- [Debugging Runbook](runbooks/debugging-tpu-test-failures.md)
- [RFC-002: CI/CD](rfcs/002-cicd-tpu-testing.md)

### Architecture
- [ADR-001: Pure Python Softmax](decisions/001-pure-python-softmax.md)
- [TokenizerManager Architecture](investigations/tokenizer-manager-architecture.md)

### Comparisons
- [PyTorch vs JAX Score API](investigations/score-api-pytorch-vs-jax.md)

### Operations
- [Debugging TPU Tests](runbooks/debugging-tpu-test-failures.md)
- [RFC-002: CI/CD Setup](rfcs/002-cicd-tpu-testing.md)

## Contributing

See [README.md](README.md) for document workflow and best practices.

### Document Lifecycle

1. **Draft RFC** → Get feedback → **Accepted** → **Implementing** → **Implemented**
2. **Create ADR** when making key decision (reference RFC)
3. **Write investigation** for deep dives and comparisons
4. **Create runbook** for operational procedures
5. **Update INDEX.md** when adding new documents

## Numbering Scheme

- RFCs: `001-999` (chronological)
- ADRs: `001-999` (chronological)
- Investigations: descriptive names (no numbers)
- Runbooks: descriptive names (no numbers)

## Status Legend

### RFC Status
- **Draft:** Initial proposal, seeking feedback
- **Review:** Under team review
- **Accepted:** Approved, ready to implement
- **Implementing:** Work in progress
- **Implemented:** Complete
- **Rejected:** Not moving forward

### ADR Status
- **Proposed:** Under consideration
- **Accepted:** Decision made and active
- **Superseded:** Replaced by newer decision
- **Deprecated:** No longer applicable

## Recent Updates

- **2026-01-29 (Late Evening):** CI/CD Integration Guide
  - Runbook: Running Score API Tests (CI + local TPU + CPU)
  - ADR-002: Use gcloud directly (no SkyPilot)
  - IMPLEMENTATION_CHECKLIST.md: Step-by-step integration guide
  - Ready to integrate into sglang-jax repo

- **2026-01-29 (Evening):** Added comprehensive test suite plan
  - RFC-003: Comprehensive Score API Test Suite (Draft)
  - Test Plan 001: Shared Fixtures and Core Tests
  - Test Plan 002: Edge Cases and HTTP/Protocol
  - Test Plan 003: JAX Features and Performance
  - Complete implementation roadmap with 30+ tests

- **2026-01-29 (Morning):** Created repository with initial RFCs, ADRs, investigations, and runbook
  - RFC-001: Score API comprehensive tests (Implemented)
  - RFC-002: CI/CD for TPU testing (Draft)
  - ADR-001: Pure Python softmax decision (Accepted)
  - Investigation: TokenizerManager architecture
  - Investigation: PyTorch vs JAX comparison
  - Runbook: Debugging TPU test failures
