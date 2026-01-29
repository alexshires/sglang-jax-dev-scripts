# Quick Start Guide

Quick reference for navigating the sglang-jax-dev-scripts documentation.

## I want to...

### Understand the Score API test implementation
→ **[RFC-001: Score API Comprehensive Tests](rfcs/001-score-api-comprehensive-tests.md)**
- What tests were implemented
- What bugs were found and fixed
- Test results and coverage

### Understand why we use pure Python softmax
→ **[ADR-001: Pure Python Softmax](decisions/001-pure-python-softmax.md)**
- Device conflict problem explained
- Alternatives considered
- Implementation details

### Understand the architecture (main process vs subprocess)
→ **[Investigation: TokenizerManager Architecture](investigations/tokenizer-manager-architecture.md)**
- Process layout diagram
- Why TokenizerManager must be device-agnostic
- Communication flow

### Compare JAX vs PyTorch implementation
→ **[Investigation: Score API PyTorch vs JAX](investigations/score-api-pytorch-vs-jax.md)**
- Side-by-side comparison
- Bugs found by comparison
- Test coverage gap analysis

### Debug a TPU test failure
→ **[Runbook: Debugging TPU Test Failures](runbooks/debugging-tpu-test-failures.md)**
- Common failure scenarios
- Step-by-step troubleshooting
- Useful commands

### Set up CI/CD for TPU testing
→ **[RFC-002: CI/CD for TPU Testing](rfcs/002-cicd-tpu-testing.md)**
- Three-tier testing strategy
- GitHub Actions workflow
- Cost analysis (~$1/month)

### Expand test coverage to 30+ tests
→ **[RFC-003: Comprehensive Test Suite](rfcs/003-score-api-comprehensive-test-suite.md)**
- Overall strategy and file structure
- CI vs local/perf gating
- Behavior validation decisions

### Implement shared test fixtures (Phase 1)
→ **[Test Plan 001: Shared Fixtures and Core Tests](test-plans/001-shared-fixtures-and-core-tests.md)**
- `score_test_utils.py` specification
- 12 core engine tests
- Input validation implementation

### Implement edge cases and HTTP tests (Phase 2)
→ **[Test Plan 002: Edge Cases and HTTP/Protocol](test-plans/002-edge-cases-and-http-protocol.md)**
- 10 edge case validation tests
- 5 HTTP endpoint tests
- 4 protocol validation tests

### Implement JAX features and benchmarks (Phase 3)
→ **[Test Plan 003: JAX Features and Performance](test-plans/003-jax-features-and-performance.md)**
- bf16 vs fp32 stability
- Multi-device sharding
- Performance benchmark tool

### Create a new RFC
→ **[RFC Template](rfcs/template.md)**
- Copy template
- Fill in sections
- Follow numbering scheme (RFC-004, RFC-005, etc.)

### Document a key decision
→ **[ADR Template](decisions/template.md)**
- Copy template
- Document what, why, alternatives
- ADRs are immutable (supersede if needed)

## Document Relationships

### Score API Testing Journey

```
RFC-001 (Initial 4 tests)
    ↓
Bugs Found
    ↓
ADR-001 (Pure Python Softmax)
    ↓
Investigation (Architecture)
    ↓
Investigation (PyTorch vs JAX)
    ↓
RFC-003 (Comprehensive Suite)
    ↓
Test Plans 001-003 (Implementation)
```

### CI/CD Journey

```
RFC-001 (Manual testing)
    ↓
RFC-002 (Automated CI/CD)
    ↓
Runbook (Debugging)
```

## File Count Summary

- **RFCs:** 3 (+ template)
- **ADRs:** 1 (+ template)
- **Investigations:** 2
- **Runbooks:** 1
- **Test Plans:** 3
- **Supporting:** README, INDEX, QUICK_START

**Total:** 14 documentation files

## Reading Order for New Contributors

1. **[README.md](README.md)** - Understand the workflow
2. **[INDEX.md](INDEX.md)** - See what's available
3. **[RFC-001](rfcs/001-score-api-comprehensive-tests.md)** - Real example of test implementation
4. **[ADR-001](decisions/001-pure-python-softmax.md)** - Real example of decision record
5. **[RFC-003](rfcs/003-score-api-comprehensive-test-suite.md)** - Future direction
6. Dive into test plans as needed

## Common Tasks

### Starting a new feature
```bash
# 1. Create RFC
cp rfcs/template.md rfcs/004-my-feature.md
# 2. Fill in RFC
# 3. Get team review
# 4. Update status: Draft → Review → Accepted
# 5. Start implementation
```

### Documenting a decision
```bash
# 1. Create ADR
cp decisions/template.md decisions/002-my-decision.md
# 2. Fill in ADR (what, why, alternatives)
# 3. Status: Accepted
# 4. Never edit (supersede if needed)
```

### Finding related docs
```bash
# Search by keyword
grep -r "softmax" . --include="*.md"

# See all RFCs
ls rfcs/*.md

# See all ADRs
ls decisions/*.md
```

## Status Meanings

### RFC Status
- **Draft** - Initial proposal, seeking feedback
- **Review** - Under team review
- **Accepted** - Approved, ready to implement
- **Implementing** - Work in progress
- **Implemented** - Complete
- **Rejected** - Not moving forward

### ADR Status
- **Proposed** - Under consideration
- **Accepted** - Decision made and active
- **Superseded** - Replaced by newer decision (link to new ADR)
- **Deprecated** - No longer applicable

## Getting Help

- Check **[INDEX.md](INDEX.md)** for all documents
- Check **[README.md](README.md)** for workflow
- Search docs with `grep -r "keyword" .`
- Ask in team chat with link to relevant doc

## Contributing New Docs

1. Use appropriate template
2. Follow naming convention (NNN-descriptive-name.md)
3. Add to INDEX.md
4. Link related documents
5. Get at least one review
6. Keep status updated
