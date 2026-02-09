# RFC-012: CI/CD Optimization Strategy
# RFC-012: CI/CD Pipeline Optimization

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
| **Related** | RFC-002, RFC-009 |

## Summary

This RFC proposes optimizations to reduce CI pipeline execution time, particularly for PRs that don't require full TPU test validation. Current CI runs 14 TPU jobs for every PR touching `python/sgl_jax/**`, even for documentation or test utility changes that have no functional impact.

## Motivation

### Current State

The `pr-test.yml` workflow triggers on any change to:
- `python/**`
- `scripts/**`
- `test/**`
- `.github/workflows/pr-test.yml`

When triggered, it runs **14 parallel TPU jobs**:

| Job | Runner | Partitions | Timeout |
|-----|--------|------------|---------|
| unit-test-1-tpu | v6e-1 | 2 | 30 min |
| unit-test-4-tpu | v6e-4 | 2 | 30 min |
| e2e-test-1-tpu | v6e-1 | 1 | 120 min |
| e2e-test-4-tpu | v6e-4 | 2 | 120 min |
| accuracy-test-1-tpu | v6e-1 | 1 | 20 min |
| accuracy-test-4-tpu | v6e-4 | 1 | 50 min |
| performance-test-1-tpu | v6e-1 | 1 | 30 min |
| performance-test-4-tpu | v6e-4 | 2 | 50 min |
| pallas-kernel-benchmark | v6e-1 | 2 | 120 min |

### Problems

1. **Test utility PRs trigger full suite**: Adding `score_test_utils.py` (a file not imported by any test) runs all 14 TPU jobs
2. **Long feedback loops**: PRs wait 30-120 minutes for tests unrelated to their changes
3. **Resource waste**: TPU time costs money; unnecessary runs add up
4. **Blocked merges**: Even trivial changes must wait for full validation

### Goals

1. **Reduce CI time for utility/docs PRs** from 30+ min to <5 min
2. **Maintain full validation** for production code changes
3. **Minimize workflow modifications** (synced from upstream)
4. **Reduce TPU costs** without sacrificing test coverage

## Proposed Solution

### Tier 1: Immediate Actions (No Workflow Changes)

#### 1.1 Use Draft PR Workflow

The workflow already skips TPU tests for draft PRs (lines 44-48):

```yaml
- name: Fail if the PR is a draft
  if: github.event_name == 'pull_request' && github.event.pull_request.draft == true
  run: exit 1
```

**Usage Pattern:**
```bash
# Create PR as draft
gh pr create --draft --title "Add test utilities"

# Work on PR, only lint runs
# When ready for full validation:
gh pr ready <PR-number>
```

**Impact:** Only `lint` + `check-changes` run (~2 min) until PR is marked ready.

#### 1.2 Enable `run-ci` Label (Already in Workflow)

Lines 38-42 have commented-out label logic:

```yaml
# TODO: use run-ci label when sglang-jax-bot exists
# - name: Fail if the PR does not have the 'run-ci' label
#   if: github.event_name == 'pull_request' && !contains(github.event.pull_request.labels.*.name, 'run-ci')
#   run: exit 1
```

**Action:** Uncomment to require explicit `run-ci` label for TPU tests.

**Usage:**
- PRs without `run-ci` label: Only lint runs
- Add `run-ci` label when ready for full validation
- Maintainers add label during review

### Tier 2: Path-Based Filtering (Workflow Modification)

Add granular path filters to skip tests for non-functional changes:

```yaml
# In check-changes job
- name: Detect file changes
  id: filter
  uses: dorny/paths-filter@v3
  with:
    filters: |
      main_package:
        - "python/sgl_jax/**"
        - "python/*.toml"
        - "scripts/**"
        - "test/**"
        - ".github/workflows/pr-test.yml"

      # NEW: Test utilities only (no functional impact)
      test_utils_only:
        - "python/sgl_jax/test/**_utils.py"
        - "python/sgl_jax/test/**_fixtures.py"
        - "python/sgl_jax/test/conftest.py"

      # NEW: Documentation only
      docs_only:
        - "**/*.md"
        - "**/*.rst"
        - "docs/**"

      pallas_kernel:
        - "python/sgl_jax/kernels/**"
        - "benchmark/kernels/**"
```

Then modify test job conditions:

```yaml
unit-test-1-tpu:
  needs: [check-changes]
  if: >
    github.event.pull_request.draft == false &&
    needs.check-changes.outputs.main_package == 'true' &&
    needs.check-changes.outputs.test_utils_only != 'true' &&
    needs.check-changes.outputs.docs_only != 'true'
```

**Impact:** Test utility and docs PRs skip TPU tests entirely.

### Tier 3: Reduce PR Test Matrix

Run only essential tests on PRs, full matrix on merge:

| Test | On PR | On Merge to Main |
|------|-------|------------------|
| unit-test-1-tpu | ✅ 1 partition | ✅ 2 partitions |
| unit-test-4-tpu | ❌ Skip | ✅ 2 partitions |
| e2e-test-1-tpu | ✅ | ✅ |
| e2e-test-4-tpu | ❌ Skip | ✅ |
| accuracy-test-* | ❌ Skip | ✅ |
| performance-test-* | ❌ Skip | ✅ |

**Implementation:**
```yaml
unit-test-4-tpu:
  if: github.event_name == 'push'  # Only on merge, not PR
```

**Impact:** Reduces PR jobs from 14 → 4, saving ~70% TPU time.

### Tier 4: Infrastructure Optimizations

#### 4.1 Dependency Caching

Add pip/uv cache to reduce install time:

```yaml
- name: Cache dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/uv
      ~/.cache/pip
      .venv
    key: ${{ runner.os }}-py3.12-${{ hashFiles('python/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-py3.12-
```

**Impact:** Saves 1-2 min per job (~14-28 min total per PR).

#### 4.2 Model Pre-Caching

Mount shared model storage instead of downloading per-job:

```yaml
env:
  HF_HOME: /mnt/shared-models/huggingface
  TRANSFORMERS_CACHE: /mnt/shared-models/transformers
```

**Requires:** GCS FUSE or Filestore mount on ARC runners (see RFC-009).

**Impact:** Saves 2-5 min per job on model downloads.

#### 4.3 Pre-Warmed Runner Pool

Keep runners warm with dependencies pre-installed:

```yaml
runs-on: [self-hosted, tpu-v6e-1, warm]  # Warm runner label
```

**Requires:** Runner configuration to maintain warm pool.

**Impact:** Eliminates 3-5 min cold start per job.

#### 4.4 Parallel Test Execution

Use pytest-xdist for parallel test execution within jobs:

```bash
pip install pytest-xdist
pytest -n auto test/  # Use all available cores
```

**Impact:** 2-4x speedup for CPU-bound test discovery/setup.

## Alternatives Considered

### Alternative 1: Nightly-Only TPU Tests

**Approach:** Run all TPU tests nightly, only lint on PRs.

**Pros:**
- Maximum PR speed
- Simple implementation

**Cons:**
- Bugs discovered late (next day)
- Merge-to-main becomes risky
- Developers lose fast feedback

**Why rejected:** Delayed feedback is worse than slower PRs.

### Alternative 2: Required vs Optional Checks

**Approach:** Make only `lint` + `unit-test-1-tpu` required, others optional.

**Pros:**
- PRs can merge after essential tests
- Full validation still runs

**Cons:**
- Optional tests often ignored
- Bugs slip through to main

**Why rejected:** All tests should pass before merge.

### Alternative 3: Separate CI Workflow for Forks

**Approach:** Create lighter workflow for fork PRs.

**Pros:**
- Fork development is faster
- Upstream workflow unchanged

**Cons:**
- Workflow divergence causes issues
- Different behavior confuses contributors

**Why rejected:** Single workflow is easier to maintain.

## Implementation Plan

### Phase 1: Immediate (No Changes Required)
- [x] Document draft PR workflow for utility PRs
- [ ] Train team on draft PR pattern
- [ ] Add to CONTRIBUTING.md

### Phase 2: Quick Wins (Minimal Changes)
- [ ] Uncomment `run-ci` label logic
- [ ] Add dependency caching to workflow
- [ ] Update branch protection rules

### Phase 3: Path Filtering (Moderate Changes)
- [ ] Add `test_utils_only` and `docs_only` filters
- [ ] Update job conditions
- [ ] Test with sample PRs

### Phase 4: Infrastructure (Significant Changes)
- [ ] Set up shared model storage (RFC-009)
- [ ] Configure pre-warmed runner pool
- [ ] Add pytest-xdist to test runner

## Testing Strategy

1. **Validate draft PR workflow:** Create draft PR, verify only lint runs
2. **Test path filters:** Create PRs touching only test utils, verify skip
3. **Benchmark improvements:** Measure time before/after each optimization
4. **Regression check:** Ensure main branch still runs full suite

## Cost Analysis

**Current State:**
- 14 TPU jobs per PR
- ~45 min average wall time
- TPU cost: ~$0.50-1.00 per PR

**After Optimization:**
- Utility PRs: 2 min (lint only)
- Code PRs: 4-7 jobs, ~20 min
- TPU cost: ~$0.20-0.50 per PR

**Savings:**
- 50-70% reduction in TPU usage for PRs
- Developer time saved: 25+ min per utility PR

## Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | Implement Phase 1 (documentation) |
| Week 1 | Implement Phase 2 (caching, labels) |
| Week 2 | Implement Phase 3 (path filters) |
| Week 3-4 | Implement Phase 4 (infrastructure) |

## Open Questions

1. **Upstream sync:** How to handle workflow changes when syncing from upstream sglang?
2. **Label automation:** Should a bot auto-add `run-ci` label based on file paths?
3. **Runner costs:** What's the actual TPU cost per job to prioritize optimizations?

## References

- [pr-test.yml workflow](https://github.com/alexshires/sglang-jax/blob/main/.github/workflows/pr-test.yml)
- [dorny/paths-filter action](https://github.com/dorny/paths-filter)
- [GitHub Actions caching](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [RFC-002: CI/CD Strategy](002-cicd-tpu-testing.md)
- [RFC-009: ARC Runner Setup](009-arc-runner-setup.md)
