# Document Index

Quick reference for all design documents, decisions, and guides in this repository.

**Last Updated:** 2026-02-06

---

## 🚀 Start Here

- **[Score API Launch Plan](SCORE_API_LAUNCH_PLAN.md)** - Master plan for implementation
  - Implementation priority matrix and dependencies
  - **Branch strategy with specific branch names and merge order**
  - PR strategy for upstream contribution
  - Launch criteria (definition of done)
  - Risk assessment

---

## RFCs (Request for Comments)

### Foundational

- **[RFC-000: Score API Design and Architecture](rfcs/000-score-api-design.md)**
  - Status: Accepted
  - Foundational design document for `/v1/score` API
  - Architecture, design principles, request flow
  - Use cases, performance characteristics, error handling

### Active

- **[RFC-001: Score API Comprehensive Tests](rfcs/001-score-api-comprehensive-tests.md)**
  - Status: Implemented
  - Comprehensive test suite for `/v1/score` API
  - 4 Tier 1 tests covering numerical correctness, batching, optimization, HTTP integration
  - Discovered and fixed 3 critical bugs

- **[RFC-002: CI/CD Strategy for Score API Testing](rfcs/002-cicd-tpu-testing.md)**
  - Status: **Implementing**
  - Fork-aware CI/CD strategy for Score API development
  - Local TPU testing via gcloud on-demand (~$2/month)
  - **Key gap addressed:** Score API performance benchmark with thresholds
  - `test_bench_score.py` implemented, pending upstream PR

- **[RFC-003: Comprehensive Score API Test Suite](rfcs/003-score-api-comprehensive-test-suite.md)**
  - Status: Draft
  - Expands test coverage from 4 to 30+ tests
  - Shared fixtures, CI vs local/perf gating, edge cases, JAX features
  - Behavior validation decisions (empty inputs, negative IDs, mixed types)

- **[RFC-004: Score API Performance Benchmarks and Stress Tests](rfcs/004-score-api-performance-benchmarks.md)**
  - Status: Draft
  - Performance benchmarking framework with tiered profiles (smoke/standard/full)
  - Stress tests for large batches, concurrent requests, sustained load
  - Baseline workflow, regression detection, CI integration

- **[RFC-005: OpenAI Client Compatibility](rfcs/005-openai-client-compatibility.md)**
  - Status: Draft
  - Validate compatibility with official OpenAI Python client
  - Test using `openai.Client` with SGLang as backend
  - Migration guide from OpenAI to SGLang

- **[RFC-006: Error Handling and API Contract](rfcs/006-error-handling-api-contract.md)**
  - Status: Draft
  - Error response schema (OpenAI-compatible)
  - HTTP status code semantics (400 vs 422 vs 500)
  - Validation rules and error codes

- **[RFC-007: Synthetic Unit Tests](rfcs/007-synthetic-unit-tests.md)**
  - Status: Draft
  - Fast, deterministic tests without model inference
  - Shift correctness, mask validation, continuation boundary tests
  - JAX compilation caching verification
  - Fuzz/property testing with hypothesis

- **[RFC-008: Multi-Item Scoring](rfcs/008-multi-item-scoring.md)**
  - Status: **Ready for Implementation**
  - Score N items in single forward pass (vs N passes currently)
  - Matches PyTorch implementation for performance parity
  - Uses delimiter tokens + `custom_mask` in `ragged_paged_attention` for attention isolation
  - All decisions resolved, all open questions answered, Phase 0 prerequisites complete
  - Supporting investigations: [attention mechanism](investigations/multi-item-attention-mechanism.md), [compilation overhead](investigations/multi-item-compilation-overhead.md)

- **[RFC-009: Self-Hosted ARC Runners with TPU](rfcs/009-arc-runner-setup.md)**
  - Status: Draft
  - Set up GitHub Actions self-hosted runners on GKE with TPU
  - Matches upstream sgl-project infrastructure
  - Includes GCS/Filestore model storage setup
  - Cost analysis and autoscaling configuration

- **[RFC-010: Cross-Backend Benchmarking (PyTorch GPU vs JAX TPU)](rfcs/010-cross-backend-benchmarking.md)**
  - Status: Draft
  - Unified benchmarking infrastructure for comparing PyTorch/GPU and JAX/TPU
  - K8s Jobs on existing GKE cluster with Kustomize overlays
  - Common output schema, comparison reporting, GCS result storage
  - Cost analysis: ~$2.12 per comparison run

- **[RFC-011: Comprehensive Profiling Framework](rfcs/011-profiling-design.md)**
  - Status: Draft
  - End-to-end profiling framework for JAX/TPU workloads
  - Focus on Score API profiling with layer-by-layer breakdown
  - Memory profiling, kernel analysis, CI integration
  - Step-by-step guides for all profiling scenarios
  - Tools: JAX profiler, XProf, TensorBoard, Perfetto

- **[RFC-012: CI/CD Pipeline Optimization](rfcs/012-ci-optimization.md)**
  - Status: Draft
  - Reduce CI execution time for utility/docs PRs
  - Path-based filtering, draft PR workflow, dependency caching
  - Infrastructure optimizations: model pre-caching, warm runner pools
  - 4-tier implementation plan with cost analysis

### Templates

- **[RFC Template](rfcs/template.md)**
  - Standard format for new RFCs

## ADRs (Architecture Decision Records)

- **[ADR-001: SciPy Softmax in TokenizerManager](decisions/001-pure-python-softmax.md)**
  - Date: 2026-01-29
  - Status: Implemented
  - Use SciPy softmax instead of JAX to avoid device conflicts
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

- **[v1/ Infrastructure Assessment](investigations/v1-infrastructure-assessment.md)**
  - Gap analysis of v1/ K8s templates for cross-backend benchmarking
  - Audited all 11 files, cross-referenced with actual code in both repos
  - 7 critical gaps identified (hardcoded values, no GPU support, missing bench_score.py)
  - Benchmark script comparison: PyTorch vs JAX tooling

- **[Multi-Item Attention Mechanism](investigations/multi-item-attention-mechanism.md)**
  - Investigation for RFC-008 Decision 8: which JAX API for shared-prefix + block-diagonal masking
  - Evaluated 6 candidates: existing custom_mask, splash attention, Pallas flash attention, jax.nn.dot_product_attention, Kvax, custom Pallas kernel
  - Key finding: `segment_ids` across ALL APIs cannot express the shared-prefix pattern
  - Recommendation: reuse existing `custom_mask` in `ragged_paged_attention` (zero kernel changes)
  - Decision matrix with correctness, memory, dev effort, TPU optimization criteria

- **[Multi-Item Compilation Overhead](investigations/multi-item-compilation-overhead.md)**
  - Investigation for RFC-008 Decision 7: JIT compilation overhead from multi-item scoring
  - Traced pytree chain: ForwardBatch → FlashAttention → FlashAttentionMetadata → custom_mask
  - Key finding: `custom_mask` None→Array changes pytree structure, adding +8 EXTEND compilations (one per token bucket)
  - Item count does NOT affect compilation — only mask values, not shape
  - Previous "5 item-count compilations × token × batch = multiplicative" claim was incorrect
  - Recommendation: lazy JIT compilation for MVP, no precompilation needed

## Runbooks

- **[Debugging TPU Test Failures](runbooks/debugging-tpu-test-failures.md)**
  - Step-by-step troubleshooting guide
  - Common failure scenarios and fixes
  - CI/CD specific debugging
  - Useful commands and diagnostics

- **[Profiling Infrastructure Setup](runbooks/profiling-infrastructure-setup.md)** ← **NEW**
  - 4 infrastructure options: TPU VM, GKE+TPU, GKE+GPU, Local CPU
  - Step-by-step setup for each option
  - Cost comparison and trade-offs
  - Trace viewing with Perfetto and TensorBoard

- **[Running Score API Tests](runbooks/running-score-api-tests.md)**
  - How to run tests in CI (automatic)
  - How to run tests on TPU (gcloud-based script)
  - How to run tests locally on CPU
  - Cost management and debugging

- **[Running Performance Benchmarks](runbooks/running-performance-benchmarks.md)**
  - Benchmark profiles: smoke, standard, full
  - Stress tests: large batch, concurrent, sustained
  - Baseline management and regression detection
  - CI/CD integration and cost management

## Reports

- **[Profiling Session 2026-02-05](reports/profiling-session-2026-02-05.md)** ← **NEW**
  - End-to-end profiling of sglang-jax on TPU v6e
  - TinyLlama 1.1B model, 15 generate requests
  - Trace analysis with 283K events, 27s total traced time
  - Artifacts in GCS: `gs://sglang-jax-profiling-results/2026-02-05-tinyllama-tpu-v6e/`

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

- **[Test Plan 004: Performance Benchmarks and Stress Tests](test-plans/004-performance-benchmarks-and-stress-tests.md)**
  - Phase 7+ (Post Test Suite), P1 Production Readiness
  - Enhanced `bench_score.py` with profiles and regression detection
  - Stress tests (`bench_score_stress.py`): large batch, concurrent, sustained
  - CI integration with nightly workflow

## Scripts

*No utility scripts yet*

## Document Relationships

### Score API Design Chain

```
RFC-000 (Design & Architecture)  ← START HERE
    ↓
RFC-008 (Multi-Item Scoring)  ← NEW: Major optimization
    ↓
RFC-006 (Error Handling)
    ↓
RFC-005 (OpenAI Compatibility)
    ↓
ADR-001 (Softmax Decision)
    ↓
Investigation: Architecture
```

### Score API Testing Chain

```
RFC-001 (Initial Tests)
    ↓
RFC-003 (Comprehensive Test Suite)
    ↓
RFC-007 (Synthetic Unit Tests)  ← NEW: Fast tests, no model needed
    ↓
Test Plan 001 (Shared Fixtures)
    ↓
Test Plan 002 (Edge Cases)
    ↓
Test Plan 003 (JAX Features)
    ↓
RFC-004 (Performance Benchmarks)
    ↓
Test Plan 004 (Benchmarks & Stress)
    ↓
Runbook: Running Performance Benchmarks
```

### CI/CD Chain (Fork Development)

```
RFC-002 (Fork CI/CD Strategy)
    ↓
RFC-012 (CI Optimization)  ← NEW: Speed up PR validation
    ↓
RFC-009 (ARC Runner Setup)  ← Self-hosted TPU runners
    ↓
RFC-004 (Performance Benchmarks)
    ↓
RFC-010 (Cross-Backend Benchmarking)  ← PyTorch GPU vs JAX TPU
    ↓
RFC-011 (Profiling Framework)  ← Comprehensive profiling
    ↓
Investigation: v1 Infrastructure Assessment  ← Gap analysis
    ↓
Runbook: Running Score API Tests
    ↓
Runbook: Debugging
```

## Quick Links by Topic

### Getting Started
- [Score API Launch Plan](SCORE_API_LAUNCH_PLAN.md) ← **Start here for implementation**
- [RFC-000: Score API Design](rfcs/000-score-api-design.md) ← Start here for design

### Design & API Contract
- [RFC-008: Multi-Item Scoring](rfcs/008-multi-item-scoring.md) ← Performance optimization
- [RFC-006: Error Handling](rfcs/006-error-handling-api-contract.md)
- [RFC-005: OpenAI Compatibility](rfcs/005-openai-client-compatibility.md)

### Testing
- [RFC-001: Score API Tests](rfcs/001-score-api-comprehensive-tests.md)
- [RFC-003: Comprehensive Test Suite](rfcs/003-score-api-comprehensive-test-suite.md)
- [RFC-007: Synthetic Unit Tests](rfcs/007-synthetic-unit-tests.md) ← Fast tests, no model
- [Debugging Runbook](runbooks/debugging-tpu-test-failures.md)
- [RFC-002: CI/CD](rfcs/002-cicd-tpu-testing.md)

### Performance
- [RFC-004: Performance Benchmarks](rfcs/004-score-api-performance-benchmarks.md)
- [RFC-010: Cross-Backend Benchmarking](rfcs/010-cross-backend-benchmarking.md) ← PyTorch GPU vs JAX TPU
- [RFC-011: Comprehensive Profiling Framework](rfcs/011-profiling-design.md) ← NEW: Profiling guides
- [v1/ Infrastructure Assessment](investigations/v1-infrastructure-assessment.md)
- [Test Plan 004: Benchmarks and Stress Tests](test-plans/004-performance-benchmarks-and-stress-tests.md)
- [Running Performance Benchmarks](runbooks/running-performance-benchmarks.md)

### Architecture
- [ADR-001: SciPy Softmax](decisions/001-pure-python-softmax.md)
- [TokenizerManager Architecture](investigations/tokenizer-manager-architecture.md)

### Comparisons
- [PyTorch vs JAX Score API](investigations/score-api-pytorch-vs-jax.md)
- [v1/ Infrastructure Assessment](investigations/v1-infrastructure-assessment.md)

### Operations
- [Profiling Infrastructure Setup](runbooks/profiling-infrastructure-setup.md) ← **NEW**
- [Debugging TPU Tests](runbooks/debugging-tpu-test-failures.md)
- [RFC-002: Fork CI/CD Strategy](rfcs/002-cicd-tpu-testing.md)
- [Running Score API Tests](runbooks/running-score-api-tests.md)

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

- **2026-02-05:** RFC-012: CI/CD Pipeline Optimization
  - Reduce CI execution time for utility/docs PRs (14 TPU jobs → lint only)
  - 4-tier implementation: draft PR workflow, run-ci label, path filtering, infra
  - Immediate actions require no workflow changes (draft PR, labels)
  - Infrastructure optimizations: dependency caching, model pre-caching, warm runners
  - Path-based filters to skip TPU tests for test utilities and documentation

- **2026-02-05:** RFC-011: Comprehensive Profiling Framework for sglang-jax
  - End-to-end profiling framework for JAX/TPU workloads with Score API focus
  - Layer-by-layer performance breakdown using JAX named scopes
  - Memory profiling integration with `jax.profiler.save_device_memory_profile()`
  - 5 detailed step-by-step guides: End-to-end, Memory, Kernel-level, Comparative, CI/CD
  - Visualization tools: Perfetto, TensorBoard, XProf comparison
  - Analysis of existing profiling infrastructure in both sglang (PyTorch) and sglang-jax

- **2026-02-04:** Investigation: v1 Infrastructure Assessment + RFC-010: Cross-Backend Benchmarking
  - **Investigation:** Audited all 11 files in v1/ directory for cross-backend benchmarking readiness
    - 7 critical gaps: hardcoded values, no GPU support, missing bench_score.py, path mismatches
    - Benchmark script comparison: PyTorch has bench_score.py, JAX does not
    - Verdict: v1/ is a foundation but not sufficient for PyTorch vs JAX comparison
  - **RFC-010:** Proposes unified benchmarking infrastructure on existing GKE cluster
    - K8s Jobs with Kustomize overlays (base + jax-tpu + pytorch-gpu)
    - Common JSON output schema for cross-backend comparison
    - 4-phase implementation: parameterize templates → add GPU → comparison harness → baselines
    - Cost: ~$2.12 per comparison run

- **2026-02-03:** RFC-009: Self-Hosted ARC Runners with TPU
  - Complete guide to setting up GitHub Actions self-hosted runners on GKE
  - TPU v6e node pools with autoscaling (scale-to-zero)
  - Model storage via GCS FUSE or Filestore at `/models/`
  - Cost analysis: ~$330-500/month for medium usage with spot instances
  - Enables full CI parity with upstream sgl-project/sglang-jax

- **2026-02-01:** Branch Strategy Added to Launch Plan
  - Added detailed branch naming convention (feat/, test/, ci/)
  - Foundation branches: `feat/score-test-fixtures`, `feat/score-validation`
  - Test branches (parallel): `test/score-synthetic`, `test/score-edge-cases`, etc.
  - Feature branches: `feat/multi-item-scoring`
  - Tooling branches: `feat/bench-score-profiles`, `ci/nightly-perf`
  - Documented merge order and parallelization opportunities

- **2026-02-01:** RFC-008: Multi-Item Scoring + RFC-007: Synthetic Unit Tests
  - **RFC-008:** Multi-item scoring design - score N items in 1 forward pass
    - Matches PyTorch's new optimization (added since initial investigation)
    - Estimated 10-60x speedup for large batch scoring workloads
    - Implementation deferred - design documented for future work
  - **RFC-007:** Fast, deterministic tests that run without model inference
    - Covers gaps: shift correctness, mask validation, continuation boundary
    - JAX compilation caching tests (no-recompile verification)
    - Fuzz/property testing framework with hypothesis
  - Updated investigation doc to note PyTorch has moved ahead

- **2026-01-31:** Score API Performance Benchmark Implemented
  - Created `test/srt/test_bench_score.py` in sglang-jax repo
  - 4 benchmark tests: single item latency, batch throughput, large batch, scaling
  - Added to `performance-test-tpu-v6e-1` suite in `run_suite.py`
  - Thresholds: p50 < 50ms, p99 < 150ms, throughput > 100 items/sec
  - RFC-002 status updated to Implementing

- **2026-01-30:** Score API Launch Plan and Engineering Standards
  - Created SCORE_API_LAUNCH_PLAN.md - master plan for implementation
  - Implementation priority matrix with dependency graph
  - PR strategy: 6 focused PRs for upstream contribution
  - Launch criteria and risk assessment
  - Updated CLAUDE.md with staff engineer guidelines

- **2026-01-30:** RFC-002 Revised - Fork-Aware CI/CD Strategy
  - Complete rewrite of RFC-002 based on upstream analysis
  - Key insight: Upstream already runs TPU tests on every PR via self-hosted runners
  - Focus shifted to: Score API performance benchmark (the real gap)
  - Local development via gcloud on-demand TPUs
  - Contribution workflow to upstream documented

- **2026-01-29 (Late Night):** Foundational and API Contract RFCs
  - RFC-000: Score API Design and Architecture (foundational document)
  - RFC-005: OpenAI Client Compatibility
  - RFC-006: Error Handling and API Contract
  - Complete API specification with error codes and validation rules

- **2026-01-29 (Night):** Performance Benchmarks and Stress Tests
  - RFC-004: Score API Performance Benchmarks and Stress Tests
  - Test Plan 004: Performance Benchmarks and Stress Tests
  - Runbook: Running Performance Benchmarks
  - Tiered profiles (smoke/standard/full), stress tests, CI integration

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
  - ADR-001: SciPy softmax decision (Implemented)
  - Investigation: TokenizerManager architecture
  - Investigation: PyTorch vs JAX comparison
  - Runbook: Debugging TPU test failures
