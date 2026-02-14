# Scoring API Interactive Visualizer

Self-contained interactive explainer for JAX TPU scoring internals with two mode views:
- `Packed` (conceptual packed multi-item path)
- `Prefill+extend` (conceptual prefix-cache path)

This artifact is documentation/product-only and does not modify runtime serving code.

## Run Locally

From this folder:

```bash
cd /Users/kanna/Sandbox/sglang-all/sglang-jax-dev-scripts/tools/scoring-visualizer
python3 -m http.server 8080
```

Open:

- [http://localhost:8080](http://localhost:8080)

Notes:
- The app is build-step free (`HTML/CSS/ES modules`).
- It loads `./data/scoring_api_modes_sample.json` first, and falls back to embedded sample data if file load fails.

## Controls

- Mode toggle: `Packed` / `Prefill+extend`
- Replay: `Play`, `Step -`, `Step +`, `Reset`
- Speed slider: deterministic replay rate
- View toggle: `Timeline` / `Buffer View`
- Overlay toggle:
  - `What is reused?`
  - `What is recomputed?`
  - `Why this impacts throughput`

Keyboard shortcuts:
- `Space`: play/pause
- `Right Arrow`: step forward
- `Left Arrow`: step backward
- `R`: reset
- `1` / `2`: packed / prefill+extend
- `T` / `B`: timeline / buffer view

## Data Input Options

1. Embedded/offline sample (default fallback)
2. Local JSON upload (file chooser)
3. Endpoint fetch (URL returning JSON)

Supported JSON payloads:
1. Full app schema (`example`, `modes`, `stages`, `events`, `scores`)
2. Lightweight patch payload:
   - `packed.total_latency_ms`, `packed.items_per_sec`
   - `prefill_extend.total_latency_ms`, `prefill_extend.items_per_sec`
   - optional `cache_hits` / `cache_misses`
3. Results array payload with `mode` keys (`packed`, `prefill_extend`)

## Screenshot and GIF Export

### Screenshot

Use browser capture or Playwright (optional):

```bash
npx playwright screenshot http://localhost:8080 /tmp/scoring-visualizer.png
```

### GIF

Record a short replay in the browser (QuickTime or screen recorder), then convert:

```bash
ffmpeg -i /path/to/replay.mov -vf "fps=12,scale=1280:-1:flags=lanczos" /tmp/scoring-visualizer.gif
```

## File Layout

```text
tools/scoring-visualizer/
├── index.html
├── styles.css
├── data/
│   └── scoring_api_modes_sample.json
└── js/
    ├── app.js
    ├── data-model.js
    ├── embedded-sample.js
    └── replay-engine.js
```
