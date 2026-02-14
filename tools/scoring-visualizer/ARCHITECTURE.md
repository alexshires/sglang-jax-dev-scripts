# Design and Architecture Notes

## 1) Data Model

The visualizer separates **stage metadata** from **replay events**.

- Stage model (`modes.<mode>.stages[]`):
  - `id`, `name`, `latency_ms`, `status`
  - explanatory fields: `description`, `reused`, `recomputed`, `throughput_impact`
- Replay event model (`modes.<mode>.events[]`):
  - `id`, `stage_id`, `label`, `duration_ms`
  - `active_components[]`, `parallel_items[]`

Why both:
- Stages drive explanation cards and metrics bars.
- Events drive deterministic timeline playback and stepping.

Primary sample dataset:
- `/Users/kanna/Sandbox/sglang-all/sglang-jax-dev-scripts/tools/scoring-visualizer/data/scoring_api_modes_sample.json`

Normalization and validation:
- `/Users/kanna/Sandbox/sglang-all/sglang-jax-dev-scripts/tools/scoring-visualizer/js/data-model.js`

## 2) State Machine and Replay Model

Replay is deterministic and data-driven:

- Engine:
  - `/Users/kanna/Sandbox/sglang-all/sglang-jax-dev-scripts/tools/scoring-visualizer/js/replay-engine.js`
- Core state:
  - `events[]`, `positionMs`, `durationMs`, `speed`, `isPlaying`
- Tick model:
  - fixed-step interval (`tickMs=20`) with `position += tickMs * speed`
  - avoids frame-rate-dependent drift from CSS-only animations
- Controls:
  - play/pause, step forward/back, reset, speed slider
- Derived snapshot on each tick:
  - active event
  - active stage
  - global progress
  - per-stage completion state

The UI consumes snapshots to update:
- timeline progress
- active stage cards
- explainability overlay text
- buffer-view highlights

## 3) Swapping in Real Benchmark JSON

The app supports three input paths:

1. Full schema payload (best)
- Include `example`, `modes`, `stages`, `events`, and `scores`.
- This is the recommended integration target for future benchmark exporters.

2. Lightweight patch payload
- Provide mode-level metrics only:
  - `packed.total_latency_ms`, `packed.items_per_sec`
  - `prefill_extend.total_latency_ms`, `prefill_extend.items_per_sec`
- Optional cache fields:
  - `cache_hits`, `cache_misses`
- The app patches these onto the embedded template.

3. `results[]` payload
- Array entries with `mode` keys (`packed`, `prefill_extend`) and metrics.

Adapter path:
- `/Users/kanna/Sandbox/sglang-all/sglang-jax-dev-scripts/tools/scoring-visualizer/js/data-model.js`
  - `adaptExternalDataset(...)`
  - `normalizeDataset(...)`

## 4) Visual Strategy Choices

- Two synchronized views:
  - `Timeline`: stage latency and event progression
  - `Buffer View`: conceptual physical token/KV layout
- 8-item constraint handling:
  - all 8 items retained in scores and metrics
  - request list visually collapses items 4..8 to reduce clutter
- Accessibility:
  - keyboard-navigable controls
  - focus-visible rings
  - high-contrast palette
  - `prefers-reduced-motion` fallback

## 5) Scope Boundaries

- No changes to runtime serving implementation.
- No dependency/bootstrap overhead; build-step free static artifact.
- Mode comparison is explicitly conceptual + benchmark-derived for docs presentation.
