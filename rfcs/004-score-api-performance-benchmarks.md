# RFC-004: Score API Performance Benchmarks and Stress Tests

**Status:** Draft
**Author:** Engineering Team
**Created:** 2026-01-29
**Updated:** 2026-01-29
**Related RFC:** RFC-003

## Summary

Establish a comprehensive performance benchmarking and stress testing framework for the `/v1/score` Scoring API that integrates with existing K8s job infrastructure, provides reproducible regression detection, and supports both local and cluster execution.

## Motivation

### Current State

- **Existing infrastructure:** K8s benchmark job templates (`v1/benchmark/`), `bench_score.py` tool path
- **Existing test coverage:** RFC-003 covers functional tests, but performance is only mentioned
- **Gaps:**
  - No standardized benchmark matrix (batch sizes, label counts, dtypes)
  - No defined profiles (smoke vs standard vs full)
  - No baseline workflow or regression thresholds
  - No CI integration strategy for performance tests
  - No stress tests for edge cases (very large batches, concurrent requests)

### Problems

1. **No regression detection** - Performance changes go unnoticed until production
2. **Ad-hoc benchmarking** - No standardized methodology, results not comparable
3. **Missing stress validation** - Unknown behavior under high load (50+ items, concurrent requests)
4. **No reproducibility** - Benchmarks don't capture environment metadata
5. **No tiered execution** - Can't do quick smoke tests vs comprehensive analysis

### Goals

1. **Standardized benchmark matrix** - Consistent configurations across local/cluster runs
2. **Tiered profiles** - Smoke (quick), standard (balanced), full (comprehensive)
3. **Reproducible results** - Fixed seeds, environment metadata, warmup/measurement strategy
4. **Regression detection** - Baselines with thresholds, automated comparison
5. **CI integration** - Non-blocking nightly runs, artifact publishing
6. **Stress testing** - Validate behavior at scale (large batches, high concurrency)

## Proposed Solution

### Benchmark Matrix

#### Profile: Smoke (Quick Validation)

**Purpose:** Fast sanity check after changes
**Runtime:** ~2-3 minutes

| Parameter | Values |
|-----------|--------|
| Batch sizes | 1, 4, 16 |
| Label counts | 2, 4 |
| Num runs | 5 |
| Warmup runs | 2 |
| Model | meta-llama/Llama-3.2-1B-Instruct |

#### Profile: Standard (Default)

**Purpose:** Balanced coverage for regular testing
**Runtime:** ~10-15 minutes

| Parameter | Values |
|-----------|--------|
| Batch sizes | 1, 2, 4, 8, 16, 32 |
| Label counts | 2, 4, 8 |
| Num runs | 20 |
| Warmup runs | 5 |
| Model | meta-llama/Llama-3.2-1B-Instruct |

#### Profile: Full (Comprehensive)

**Purpose:** Deep analysis for releases and investigations
**Runtime:** ~45-60 minutes

| Parameter | Values |
|-----------|--------|
| Batch sizes | 1, 2, 4, 8, 16, 32, 64 |
| Label counts | 2, 4, 8, 16 |
| Num runs | 50 |
| Warmup runs | 10 |
| Models | Llama-3.2-1B, Llama-3.2-3B |
| Dtypes | bfloat16, float32 |

### Stress Test Configurations

#### Large Batch Stress Test

**Purpose:** Validate behavior with very large batch sizes
**Configurations:**

| Batch Size | Label Count | Expected Behavior |
|------------|-------------|-------------------|
| 50 | 4 | Complete without OOM |
| 100 | 4 | Complete without OOM |
| 200 | 4 | Complete or graceful failure |

#### Concurrent Request Stress Test

**Purpose:** Validate thread safety and resource contention
**Configurations:**

| Concurrent Requests | Batch Size | Duration |
|--------------------|------------|----------|
| 4 | 8 | 30 seconds |
| 8 | 8 | 30 seconds |
| 16 | 4 | 30 seconds |

### Metrics and Output

#### Core Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Throughput (IPS) | Items scored per second | items/sec |
| Throughput (RPS) | Requests per second | requests/sec |
| Latency p50 | 50th percentile latency | ms |
| Latency p95 | 95th percentile latency | ms |
| Latency p99 | 99th percentile latency | ms |
| Latency mean | Mean latency | ms |
| Latency std | Standard deviation | ms |

#### Environment Metadata

Captured for reproducibility:

```json
{
  "hardware": "tpu-v6e-1x1",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "model_hash": "sha256:abc123...",
  "commit": "abc1234",
  "dtype": "bfloat16",
  "tp_size": 1,
  "timestamp": "2026-01-29T12:00:00Z",
  "profile": "standard"
}
```

#### Output Formats

1. **Human-readable table** - Printed to stdout
2. **CSV** - For automation and tracking
3. **JSON** - For programmatic consumption

### Baseline Workflow

#### Establishing Baseline

```bash
# Run comprehensive benchmark on representative hardware
python test/srt/bench_score.py \
  --profile standard \
  --output baselines/tpu-v6e-baseline.csv \
  --metadata-output baselines/tpu-v6e-metadata.json

# Commit to repository
git add baselines/
git commit -m "Add TPU v6e performance baseline"
```

#### Comparing Against Baseline

```bash
# Run benchmark and compare
python test/srt/bench_score.py \
  --profile standard \
  --baseline baselines/tpu-v6e-baseline.csv \
  --output results/current.csv \
  --regression-threshold 10
```

#### Regression Rules

| Metric | Threshold | Action |
|--------|-----------|--------|
| Throughput | > 10% decrease | FAIL |
| Latency p95 | > 15% increase | FAIL |
| Latency p99 | > 20% increase | WARN |

### CI Integration

#### Three-Tier Strategy (Per RFC-002)

1. **Default CI (every PR):** No performance tests (too slow)
2. **Nightly:** Run smoke profile, publish artifacts
3. **On-demand:** Run standard/full via label trigger

#### Nightly Workflow

```yaml
# .github/workflows/nightly-perf.yaml
name: Nightly Performance
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:
    inputs:
      profile:
        description: 'Benchmark profile'
        default: 'smoke'
        type: choice
        options: [smoke, standard, full]

jobs:
  benchmark:
    runs-on: [self-hosted, tpu]
    steps:
      - uses: actions/checkout@v4
      - name: Run Benchmark
        run: |
          python test/srt/bench_score.py \
            --profile ${{ inputs.profile || 'smoke' }} \
            --baseline baselines/tpu-v6e-baseline.csv \
            --output results/nightly-${{ github.run_number }}.csv \
            --regression-threshold 10
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: perf-results-${{ github.run_number }}
          path: results/
      - name: Comment on Regression
        if: failure()
        run: |
          # Post summary to Slack/PR/Issue
          echo "Performance regression detected!"
```

#### Artifact Storage

- **In-repo:** `baselines/` directory (checked-in, version-controlled)
- **CI artifacts:** Nightly results (90-day retention)
- **Optional:** Dashboard integration (Grafana, custom)

### File Structure

```
test/srt/
├── bench_score.py                    # EXPAND: Add profiles, stress tests
├── bench_score_stress.py             # NEW: Stress test runner
└── baselines/
    ├── tpu-v6e-baseline.csv          # NEW: TPU v6e baseline
    ├── gpu-a100-baseline.csv         # NEW: GPU A100 baseline (future)
    └── baseline-metadata.json        # NEW: Environment metadata

sglang-jax-dev-scripts/
├── rfcs/
│   └── 004-score-api-performance-benchmarks.md  # THIS FILE
├── test-plans/
│   └── 004-performance-benchmarks-and-stress-tests.md  # NEW
├── runbooks/
│   └── running-performance-benchmarks.md        # NEW
└── v1/benchmark/
    └── README.md                     # UPDATE: Reference new tools
```

## Implementation Plan

### Phase 1: Benchmark Tool Enhancement

- [ ] Add profile support to `bench_score.py` (smoke/standard/full)
- [ ] Add metadata capture (hardware, model, commit, timestamp)
- [ ] Add JSON output format
- [ ] Add regression comparison with configurable threshold
- [ ] Add environment detection (auto-detect TPU/GPU/CPU)

### Phase 2: Stress Tests

- [ ] Create `bench_score_stress.py` for stress scenarios
- [ ] Implement large batch stress test (50, 100, 200 items)
- [ ] Implement concurrent request stress test
- [ ] Add graceful failure handling and reporting
- [ ] Document expected behavior at various scales

### Phase 3: Baseline Establishment

- [ ] Run full profile on TPU v6e
- [ ] Create baseline CSV and metadata JSON
- [ ] Commit baselines to repository
- [ ] Document baseline update process

### Phase 4: CI Integration

- [ ] Create nightly workflow (`.github/workflows/nightly-perf.yaml`)
- [ ] Configure artifact upload and retention
- [ ] Set up regression notification (Slack/PR comment)
- [ ] Add on-demand trigger with profile selection
- [ ] Test full workflow end-to-end

### Phase 5: Documentation

- [ ] Create Test Plan 004 (detailed specs)
- [ ] Create Runbook (operational guide)
- [ ] Update v1/benchmark/README.md
- [ ] Update INDEX.md

## Alternatives Considered

### Alternative 1: Performance Tests in Default CI

**Description:** Run performance benchmarks on every PR

**Pros:**
- Catch regressions immediately
- No separate workflow needed

**Cons:**
- 10-15+ minute CI time increase
- Results meaningless without dedicated hardware
- Flaky due to resource contention

**Why rejected:** Performance tests need dedicated hardware and consistent environment. Nightly runs provide sufficient coverage without blocking developer workflow.

### Alternative 2: External Benchmarking Service

**Description:** Use service like Codspeed, Bencher.dev

**Pros:**
- Managed infrastructure
- Built-in dashboards and comparison
- No maintenance burden

**Cons:**
- Cost ($50-200/month)
- May not support TPU
- Less control over configuration
- Vendor lock-in

**Why rejected:** TPU support is critical, and in-repo tooling provides better control. Can revisit if maintenance burden becomes significant.

### Alternative 3: Dashboard-First Approach

**Description:** Store all results in time-series database, visualize in Grafana

**Pros:**
- Rich visualization
- Historical trends
- Alerting capabilities

**Cons:**
- Infrastructure overhead
- Overkill for current scale
- Delayed feedback (not inline with CI)

**Why rejected:** Start simple with CSV baselines. Add dashboard later if needed.

### Alternative 4: 5% Regression Threshold

**Description:** Use stricter 5% threshold for regression detection

**Pros:**
- Catch smaller regressions
- Higher quality bar

**Cons:**
- High false positive rate
- Measurement noise often > 5%
- Alert fatigue

**Why rejected:** 10% is industry standard for meaningful regressions. Tighten after understanding variance.

## Testing Strategy

### Validation

**Tool functionality:**
```bash
# Profile execution
python test/srt/bench_score.py --profile smoke --output /tmp/test.csv
# Verify CSV has expected columns and rows

# Regression detection
python test/srt/bench_score.py --profile smoke \
  --baseline baselines/tpu-v6e-baseline.csv \
  --regression-threshold 10
# Verify exit code reflects regression status

# Stress test
python test/srt/bench_score_stress.py --scenario large-batch
# Verify completes or fails gracefully
```

**CI integration:**
```bash
# Trigger nightly manually
gh workflow run nightly-perf.yaml --field profile=smoke
# Verify workflow completes, artifacts uploaded
```

### Monitoring

- **Nightly success rate:** > 95%
- **Variance:** < 5% between runs
- **Baseline staleness:** Update monthly

## Cost Analysis

**Development Cost:**
- Phase 1 (tool enhancement): 4 hours
- Phase 2 (stress tests): 3 hours
- Phase 3 (baselines): 2 hours
- Phase 4 (CI integration): 3 hours
- Phase 5 (documentation): 2 hours
- **Total:** ~14 hours

**Ongoing Cost:**
- Nightly runs: 10 min × 30 days × $0.64/hr = $3.20/month (TPU v6e)
- Maintenance: ~2 hours/month
- Baseline updates: 1 hour/month

**ROI:**
- Early regression detection: 4+ hours saved per incident
- Expected incidents caught: 1-2/month
- **Monthly ROI:** 8-16 hours saved vs 3 hours maintenance

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Primary baseline hardware? | **TPU v6e** - most common deployment target |
| Score API only vs generation? | **Score API only** - scope this RFC narrowly |
| Nightly vs on-demand? | **Nightly** - catches gradual degradation |
| Regression threshold? | **10%** - balance sensitivity vs noise |
| Baseline storage? | **Checked-in CSV** - simple, version-controlled |

## Success Metrics

**Phase 1-3 Complete:**
- [ ] Benchmark tool supports all three profiles
- [ ] Metadata capture working
- [ ] Baseline established on TPU v6e
- [ ] Regression detection functional

**Phase 4-5 Complete:**
- [ ] Nightly workflow operational
- [ ] Artifacts published and retained
- [ ] Documentation complete
- [ ] At least one regression caught and fixed

## References

- RFC-003: Comprehensive Score API Test Suite
- RFC-002: CI/CD for TPU Testing
- Test Plan 003: JAX Features and Performance
- v1/benchmark/README.md: Existing K8s benchmark infrastructure
