# Score API Launch Plan

| | |
|------------|------|
| **Status** | Active |
| **Owner** | Engineering Team |
| **Created** | 2026-01-30 |
| **Updated** | 2026-02-01 |
| **Target** | Upstream contribution to sglang-jax |

This document is the master plan for completing the Score API implementation and contributing it upstream. It consolidates implementation priority, PR strategy, launch criteria, and risk assessment.

## Executive Summary

The `/v1/score` Scoring API is implemented and functional in sglang-jax. This plan covers:
1. Adding performance regression testing (the primary gap)
2. Expanding test coverage
3. Contributing changes upstream via structured PRs

**MVP:** Performance benchmark in CI (`test_bench_score.py`)
**Full Launch:** Comprehensive test suite + OpenAI client compatibility

---

## Implementation Priority Matrix

### Priority 1: CI Performance Gate (The Gap)

| Item | RFC | Status | Blocks |
|------|-----|--------|--------|
| `test_bench_score.py` | RFC-002 | **Done** (pending PR) | Nothing |
| Add to `performance-test-tpu-v6e-1` suite | RFC-002 | **Done** (pending PR) | Above |
| Establish baseline thresholds | RFC-002 | **Done** (in code) | Above |

**Why first:** This is the only significant gap vs PyTorch SGLang. Upstream has functional tests but no Score API performance benchmark.

### Priority 2: Expanded Test Coverage

| Item | RFC | Status | Blocks |
|------|-----|--------|--------|
| Shared test fixtures (`score_test_utils.py`) | RFC-003 | Not Started | Nothing |
| Edge case tests (empty inputs, bounds) | RFC-003 | Not Started | Fixtures |
| Input validation tests | RFC-003 | Not Started | Fixtures |
| Large batch tests (20+ items) | RFC-003 | Not Started | Fixtures |

**Why second:** Increases confidence in correctness. Uses shared fixtures to reduce duplication.

### Priority 3: Client Compatibility

| Item | RFC | Status | Blocks |
|------|-----|--------|--------|
| OpenAI client tests | RFC-005 | Not Started | Nothing |
| Error format validation | RFC-006 | Not Started | Nothing |
| Migration documentation | RFC-005 | Not Started | Tests pass |

**Why third:** Nice-to-have for users, but core functionality works without it.

### Priority 4: Deep Performance Analysis

| Item | RFC | Status | Blocks |
|------|-----|--------|--------|
| `bench_score.py` with profiles | RFC-004 | Not Started | Nothing |
| Stress tests (`bench_score_stress.py`) | RFC-004 | Not Started | Above |
| Baseline CSV + metadata | RFC-004 | Not Started | Tool ready |
| Nightly workflow | RFC-004 | Not Started | Tool ready |

**Why fourth:** Valuable for optimization work, but not blocking launch.

---

## Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │  Priority 1: CI Performance     │
                    │  test_bench_score.py            │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │  Priority 2: Test       │     │  Priority 3: Client     │
    │  Coverage Expansion     │     │  Compatibility          │
    │  (score_test_utils.py)  │     │  (OpenAI client tests)  │
    └─────────────────────────┘     └─────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Priority 4: Deep Performance   │
                    │  bench_score.py + stress tests  │
                    └─────────────────────────────────┘
```

**Key insight:** Priorities 2 and 3 can be done in parallel after Priority 1.

---

## PR Strategy

### Development Model: Fork-First

All development happens on the fork for the initial phase. Once the Score API implementation is mature and well-tested, changes will be contributed upstream in batches.

```
Feature Branch → PR → Fork Main → (later) PR → Upstream Main
```

**Why fork-first:**
- Iterate quickly without upstream review cycles
- Build up comprehensive test coverage
- Validate thresholds with real TPU runs
- Bundle related changes for cleaner upstream PRs

### Principle: Small, Focused PRs

Each PR should do ONE thing. This makes review easier and rollback simpler.

### Phase 1: PRs to Fork

Development PRs merged to fork's main branch. Each PR corresponds to a feature branch.

#### Branch Naming Convention

```
feat/   - New features or enhancements
test/   - Test additions or improvements
ci/     - CI/CD configuration changes
```

#### Foundation Branches (Sequential - must merge in order)

| Branch | Description | Depends On | ~Size | Status |
|--------|-------------|------------|-------|--------|
| `feat/score-test-fixtures` | Shared fixtures (`score_test_utils.py`) | - | ~150 LOC | Not started |
| `feat/score-validation` | Input validation (RFC-006) | fixtures | ~200 LOC | Not started |

#### Test Branches (Parallel - after Foundation)

| Branch | Description | Depends On | ~Size | Status |
|--------|-------------|------------|-------|--------|
| `test/score-synthetic` | RFC-007 synthetic unit tests | fixtures | ~300 LOC | Not started |
| `test/score-edge-cases` | RFC-003 edge case tests | fixtures, validation | ~200 LOC | Not started |
| `test/score-openai-client` | RFC-005 OpenAI client tests | fixtures | ~150 LOC | Not started |
| `test/score-core-expansion` | RFC-003 core engine tests | fixtures | ~200 LOC | Not started |

#### Feature Branches (After tests pass)

| Branch | Description | Depends On | ~Size | Status |
|--------|-------------|------------|-------|--------|
| `feat/multi-item-scoring` | RFC-008 implementation | validation | ~400 LOC | Not started |
| `test/multi-item-scoring` | RFC-008 tests | multi-item-scoring | ~200 LOC | Not started |

#### Tooling Branches (Can parallel with Features)

| Branch | Description | Depends On | ~Size | Status |
|--------|-------------|------------|-------|--------|
| `feat/bench-score-profiles` | RFC-004 benchmark profiles | fixtures | ~300 LOC | Not started |
| `ci/nightly-perf` | RFC-002 CI integration | bench-score-profiles | ~100 LOC | Not started |

#### Merge Order

```
1. feat/score-test-fixtures       ──► Foundation
2. feat/score-validation          ──► Foundation
   ├── 3a. test/score-synthetic       (parallel)
   ├── 3b. test/score-edge-cases      (parallel)
   ├── 3c. test/score-openai-client   (parallel)
   └── 3d. test/score-core-expansion  (parallel)
4. feat/multi-item-scoring        ──► Feature
5. test/multi-item-scoring        ──► Feature tests
6. feat/bench-score-profiles      ──► Tooling (can start after step 1)
7. ci/nightly-perf                ──► Tooling
```

#### Legacy PRs (Already Complete)

| PR | Description | Status |
|----|-------------|--------|
| Score API perf benchmark | `test_bench_score.py` + suite update | Ready |

### Phase 2: PRs to Upstream (Future)

After fork development is stable, contribute to upstream in batches:

### Proposed Upstream PR Sequence

#### PR 1: Score API Performance Benchmark (Priority 1)
**Files:**
- `test/srt/test_bench_score.py` (NEW)
- `test/srt/run_suite.py` (ADD to performance-test-tpu-v6e-1)

**Size:** ~200 lines
**Review complexity:** Low (self-contained)
**Risk:** Low (adds tests, doesn't change production code)

#### PR 2: Shared Test Fixtures (Priority 2a)
**Files:**
- `python/sgl_jax/test/score_test_utils.py` (NEW)
- `test/srt/test_score_api.py` (REFACTOR to use fixtures)

**Size:** ~150 lines
**Review complexity:** Low
**Risk:** Low (refactoring only)

#### PR 3: Extended Test Coverage (Priority 2b)
**Files:**
- `test/srt/test_score_api.py` (EXPAND with edge cases)

**Size:** ~300 lines
**Review complexity:** Medium (many test cases)
**Risk:** Low (tests only)

#### PR 4: OpenAI Client Compatibility (Priority 3)
**Files:**
- `test/srt/test_score_openai_client.py` (NEW)

**Size:** ~200 lines
**Review complexity:** Low
**Risk:** Low (tests only, optional dependency)

#### PR 5: Performance Analysis Tool (Priority 4a)
**Files:**
- `test/srt/bench_score.py` (EXPAND with profiles)
- `test/srt/baselines/` (NEW directory)

**Size:** ~400 lines
**Review complexity:** Medium
**Risk:** Low (tooling only)

#### PR 6: Stress Tests (Priority 4b)
**Files:**
- `test/srt/bench_score_stress.py` (NEW)

**Size:** ~250 lines
**Review complexity:** Low
**Risk:** Low

### PR Best Practices

1. **One logical change per PR** - Don't mix features
2. **Include tests** - Every PR should have test coverage
3. **Update docs** - Keep README/docs in sync
4. **Link to RFC** - Reference the design doc in PR description
5. **Small diffs** - Aim for <500 lines per PR
6. **Self-review first** - Read your own PR before submitting

### PR Template

```markdown
## Summary
[1-2 sentences describing the change]

## Related RFC
[Link to RFC in sglang-jax-dev-scripts]

## Changes
- [Bullet list of changes]

## Testing
- [ ] Local tests pass
- [ ] TPU tests pass (if applicable)
- [ ] No regressions in existing tests

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or documented in description)
```

---

## Launch Criteria (Definition of Done)

### MVP Launch (Priority 1 Complete)

- [ ] `test_bench_score.py` merged to upstream
- [ ] Basic performance benchmarks running in `performance-test-tpu-v6e-1` suite (every PR)
- [ ] Baseline thresholds established and documented
- [ ] No regressions in existing tests

**Note:** Basic perf benchmarks run on every PR via upstream's performance suite. Detailed profiling (smoke/standard/full from RFC-004) runs nightly/on-demand only.

**Success metric:** Score API performance regressions caught by CI.

### Phase 2 Launch (Priorities 1-2 Complete)

- [ ] MVP Launch criteria met
- [ ] Shared test fixtures in place
- [ ] Edge case tests implemented
- [ ] Test coverage at parity with PyTorch SGLang (17+ tests)

**Success metric:** Equal or better test coverage vs PyTorch.

### Full Launch (All Priorities Complete)

- [ ] Phase 2 Launch criteria met
- [ ] OpenAI client compatibility validated
- [ ] Error handling matches OpenAI spec
- [ ] Performance analysis tooling available
- [ ] Stress tests validate large batch handling
- [ ] All documentation complete

**Success metric:** Score API is production-ready with full test coverage.

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance thresholds too tight | Medium | Low | Start conservative (50ms), adjust based on data |
| Large batch OOM | Medium | Medium | Document limits, add graceful failure |
| OpenAI client version incompatibility | Low | Low | Test multiple versions, document minimum |
| Upstream rejects PR structure | Low | Medium | Discuss with maintainers before starting |

### Process Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Slow upstream review | Medium | Low | Keep PRs small, be responsive to feedback |
| Merge conflicts | Medium | Low | Rebase frequently, stay in sync with main |
| CI flakiness | Medium | Medium | Use warmup runs, set reasonable timeouts |

### Unknowns

| Unknown | How to Resolve |
|---------|----------------|
| Upstream appetite for perf tests | Ask maintainers before PR 1 |
| OpenAI client version support | Test with 1.0.x, 1.5.x, document results |

**Resolved:**
- ~~Exact performance thresholds~~ → Done: p50 < 50ms, p99 < 150ms, throughput > 100 items/sec (in `test_bench_score.py`)

---

## Success Metrics

### Quantitative

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Test count | 17+ tests | Count test methods |
| CI coverage | Score API in e2e + perf suites | Check run_suite.py |
| Regression catch rate | >0 regressions caught | Track CI failures |
| PR merge rate | 100% PRs merged | Track PR status |

### Qualitative

- Upstream maintainers approve approach
- Documentation is clear and complete
- No user-reported issues with Score API
- Easy for future contributors to add tests

---

## Timeline Milestones

**Note:** No time estimates per guidelines. These are sequenced milestones.

1. **Milestone 1:** PR 1 merged (Score API perf benchmark in CI)
2. **Milestone 2:** PR 2-3 merged (Expanded test coverage)
3. **Milestone 3:** PR 4 merged (OpenAI compatibility)
4. **Milestone 4:** PR 5-6 merged (Performance tooling)
5. **Milestone 5:** All documentation complete

---

## Open Questions

| Question | Owner | Status |
|----------|-------|--------|
| What are the exact latency/throughput thresholds? | Engineering | Needs baseline run |
| Will upstream accept perf tests in CI? | Engineering | Ask maintainers |
| Should we add Score API to accuracy tests? | Engineering | Decide after MVP |

---

## Implementation Log

Track completed implementations with links to PRs/commits.

| Date | Item | PR/Commit | Files | Notes |
|------|------|-----------|-------|-------|
| 2026-01-31 | Score API Performance Benchmark | [PR #2](https://github.com/alexshires/sglang-jax/pull/2) | `test/srt/test_bench_score.py`, `test/srt/run_suite.py` | 4 benchmark tests with latency/throughput thresholds |

### How to Update This Log

After completing an implementation:
1. Add a row to the table above with date, item, PR link, files changed
2. Check off the corresponding item in Launch Criteria section
3. Update the RFC status if all items in that RFC are complete
4. Update INDEX.md RFC status entry

---

## References

- [RFC-000: Score API Design](rfcs/000-score-api-design.md)
- [RFC-002: CI/CD Strategy](rfcs/002-cicd-tpu-testing.md)
- [RFC-003: Comprehensive Test Suite](rfcs/003-score-api-comprehensive-test-suite.md)
- [RFC-004: Performance Benchmarks](rfcs/004-score-api-performance-benchmarks.md)
- [RFC-005: OpenAI Compatibility](rfcs/005-openai-client-compatibility.md)
- [RFC-006: Error Handling](rfcs/006-error-handling-api-contract.md)
- [Implementation Checklist](IMPLEMENTATION_CHECKLIST.md)
