import {
  MODE_ORDER,
  OVERLAY_KEYS,
  TECHNICAL_TERMS,
  adaptExternalDataset,
  buildStageEvents,
  cacheBadge,
  modeDisplayName,
  normalizeDataset,
} from "./data-model.js";
import { EMBEDDED_SAMPLE_DATASET } from "./embedded-sample.js";
import { ReplayEngine } from "./replay-engine.js";

const state = {
  dataset: null,
  mode: "packed",
  view: "timeline",
  overlay: "reused",
  events: [],
  snapshot: null,
};

const ui = {
  sourceSummary: document.getElementById("sourceSummary"),
  modeButtons: Array.from(document.querySelectorAll("[data-mode]")),
  viewButtons: Array.from(document.querySelectorAll("[data-view]")),
  overlayButtons: Array.from(document.querySelectorAll("[data-overlay]")),
  playPauseBtn: document.getElementById("playPauseBtn"),
  stepBackBtn: document.getElementById("stepBackBtn"),
  stepForwardBtn: document.getElementById("stepForwardBtn"),
  resetBtn: document.getElementById("resetBtn"),
  speedSlider: document.getElementById("speedSlider"),
  speedValue: document.getElementById("speedValue"),
  jsonUpload: document.getElementById("jsonUpload"),
  endpointInput: document.getElementById("endpointInput"),
  fetchEndpointBtn: document.getElementById("fetchEndpointBtn"),
  reloadSampleBtn: document.getElementById("reloadSampleBtn"),
  dataStatus: document.getElementById("dataStatus"),
  queryText: document.getElementById("queryText"),
  labelTokenIds: document.getElementById("labelTokenIds"),
  itemList: document.getElementById("itemList"),
  timelineView: document.getElementById("timelineView"),
  bufferView: document.getElementById("bufferView"),
  activeStageLabel: document.getElementById("activeStageLabel"),
  replayClock: document.getElementById("replayClock"),
  timelineProgress: document.getElementById("timelineProgress"),
  timelineTrack: document.getElementById("timelineTrack"),
  stageCards: document.getElementById("stageCards"),
  bufferGrid: document.getElementById("bufferGrid"),
  bufferNote: document.getElementById("bufferNote"),
  metricTotalLatency: document.getElementById("metricTotalLatency"),
  metricItemsPerSec: document.getElementById("metricItemsPerSec"),
  metricCacheBadge: document.getElementById("metricCacheBadge"),
  metricCompilerCache: document.getElementById("metricCompilerCache"),
  stageMetrics: document.getElementById("stageMetrics"),
  overlayStageName: document.getElementById("overlayStageName"),
  overlayText: document.getElementById("overlayText"),
  scoreRows: document.getElementById("scoreRows"),
};

const replay = new ReplayEngine({ onUpdate: onReplayUpdate });

bootstrap().catch((error) => {
  setDataStatus(`Initialization error: ${error.message}`);
});

async function bootstrap() {
  wireTermTooltips();
  wireModeControls();
  wireViewControls();
  wireOverlayControls();
  wireReplayControls();
  wireDataControls();
  wireTimelineSeek();
  wireKeyboardShortcuts();

  await loadInitialDataset();
}

async function loadInitialDataset() {
  setDataStatus("Loading embedded sample...");

  try {
    const response = await fetch("./data/scoring_api_modes_sample.json", {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const raw = await response.json();
    applyDataset(raw, "sample-json");
    setDataStatus("Loaded local sample dataset.");
  } catch (_error) {
    applyDataset(EMBEDDED_SAMPLE_DATASET, "embedded-fallback");
    setDataStatus("Loaded embedded fallback dataset.");
  }
}

function applyDataset(rawDataset, sourceTag) {
  let normalized;
  if (rawDataset?.example && rawDataset?.modes) {
    normalized = normalizeDataset(rawDataset);
  } else {
    normalized = adaptExternalDataset(rawDataset, EMBEDDED_SAMPLE_DATASET);
  }

  state.dataset = normalized;

  const sourceModel =
    normalized?.source?.environment?.model || "model-not-specified";
  ui.sourceSummary.textContent = `${sourceTag} | v${normalized.dataset_version} | ${sourceModel}`;

  renderExampleRequest();
  setMode(state.mode, { resetReplay: true });
}

function wireTermTooltips() {
  const termButtons = document.querySelectorAll(".term[data-term]");
  for (const button of termButtons) {
    const term = button.getAttribute("data-term");
    const tooltip = TECHNICAL_TERMS[term] || "Definition unavailable.";
    button.setAttribute("data-tooltip", tooltip);
    button.setAttribute("aria-label", `${button.textContent?.trim()}: ${tooltip}`);
  }
}

function wireModeControls() {
  for (const button of ui.modeButtons) {
    button.addEventListener("click", () => {
      const mode = button.getAttribute("data-mode");
      if (mode && MODE_ORDER.includes(mode)) {
        setMode(mode, { resetReplay: true });
      }
    });
  }
}

function wireViewControls() {
  for (const button of ui.viewButtons) {
    button.addEventListener("click", () => {
      const view = button.getAttribute("data-view");
      if (!view) return;
      setView(view);
    });
  }
}

function wireOverlayControls() {
  for (const button of ui.overlayButtons) {
    button.addEventListener("click", () => {
      const overlay = button.getAttribute("data-overlay");
      if (!overlay || !OVERLAY_KEYS.includes(overlay)) {
        return;
      }
      state.overlay = overlay;
      for (const overlayButton of ui.overlayButtons) {
        const selected = overlayButton === button;
        overlayButton.classList.toggle("is-selected", selected);
        overlayButton.setAttribute("aria-checked", selected ? "true" : "false");
      }
      updateOverlayText();
    });
  }
}

function wireReplayControls() {
  ui.playPauseBtn.addEventListener("click", () => {
    if (replay.getSnapshot().isPlaying) {
      replay.pause();
    } else {
      replay.play();
    }
  });

  ui.stepBackBtn.addEventListener("click", () => replay.stepBackward());
  ui.stepForwardBtn.addEventListener("click", () => replay.stepForward());
  ui.resetBtn.addEventListener("click", () => replay.reset());

  ui.speedSlider.addEventListener("input", () => {
    const speed = Number(ui.speedSlider.value);
    replay.setSpeed(speed);
  });
}

function wireDataControls() {
  ui.jsonUpload.addEventListener("change", async (event) => {
    const input = event.target;
    const file = input.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      applyDataset(parsed, `upload:${file.name}`);
      setDataStatus(`Loaded uploaded dataset: ${file.name}`);
    } catch (error) {
      setDataStatus(`Upload failed: ${error.message}`);
    } finally {
      input.value = "";
    }
  });

  ui.fetchEndpointBtn.addEventListener("click", async () => {
    const endpoint = ui.endpointInput.value.trim();
    if (!endpoint) {
      setDataStatus("Enter an endpoint URL first.");
      return;
    }

    setDataStatus("Fetching endpoint JSON...");
    try {
      const response = await fetch(endpoint, { method: "GET" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const parsed = await response.json();
      applyDataset(parsed, `endpoint:${endpoint}`);
      setDataStatus(`Loaded endpoint dataset: ${endpoint}`);
    } catch (error) {
      setDataStatus(`Endpoint fetch failed: ${error.message}`);
    }
  });

  ui.reloadSampleBtn.addEventListener("click", () => {
    applyDataset(EMBEDDED_SAMPLE_DATASET, "embedded-fallback");
    setDataStatus("Restored embedded sample dataset.");
  });
}

function wireTimelineSeek() {
  ui.timelineTrack.addEventListener("click", (event) => {
    const segment = event.target.closest(".timeline-segment");
    if (!segment) return;
    const index = Number(segment.getAttribute("data-event-index"));
    if (!Number.isInteger(index)) return;
    const selected = state.events[index];
    if (!selected) return;
    replay.pause();
    replay.seek(selected.start_ms);
  });

  ui.stageCards.addEventListener("click", (event) => {
    const card = event.target.closest(".stage-card");
    if (!card) return;
    const stageId = card.getAttribute("data-stage-id");
    if (!stageId) return;
    const selected = state.events.find((row) => row.stage_id === stageId);
    if (!selected) return;
    replay.pause();
    replay.seek(selected.start_ms);
  });
}

function wireKeyboardShortcuts() {
  document.addEventListener("keydown", (event) => {
    const tagName = event.target?.tagName?.toLowerCase();
    if (["input", "textarea", "select"].includes(tagName)) {
      return;
    }

    const key = event.key.toLowerCase();
    if (event.code === "Space") {
      event.preventDefault();
      if (replay.getSnapshot().isPlaying) {
        replay.pause();
      } else {
        replay.play();
      }
      return;
    }

    if (event.key === "ArrowRight") {
      event.preventDefault();
      replay.stepForward();
      return;
    }

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      replay.stepBackward();
      return;
    }

    if (key === "r") {
      event.preventDefault();
      replay.reset();
      return;
    }

    if (key === "1") {
      event.preventDefault();
      setMode("packed", { resetReplay: true });
      return;
    }

    if (key === "2") {
      event.preventDefault();
      setMode("prefill_extend", { resetReplay: true });
      return;
    }

    if (key === "t") {
      event.preventDefault();
      setView("timeline");
      return;
    }

    if (key === "b") {
      event.preventDefault();
      setView("buffer");
    }
  });
}

function setMode(mode, options = { resetReplay: false }) {
  if (!state.dataset || !state.dataset.modes[mode]) {
    return;
  }

  state.mode = mode;

  for (const button of ui.modeButtons) {
    const selected = button.getAttribute("data-mode") === mode;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-checked", selected ? "true" : "false");
  }

  const modeData = getCurrentModeData();
  state.events = buildStageEvents(modeData);

  renderModeViews();

  if (options.resetReplay) {
    replay.setTimeline(state.events);
    replay.setSpeed(Number(ui.speedSlider.value));
  }
}

function setView(view) {
  state.view = view;
  const showTimeline = view === "timeline";

  ui.timelineView.classList.toggle("is-hidden", !showTimeline);
  ui.bufferView.classList.toggle("is-hidden", showTimeline);

  for (const button of ui.viewButtons) {
    const selected = button.getAttribute("data-view") === view;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  }
}

function renderExampleRequest() {
  const { example } = state.dataset;

  ui.queryText.textContent = example.query_text;
  ui.labelTokenIds.textContent = `[${example.label_token_ids.join(", ")}]`;

  ui.itemList.innerHTML = example.items
    .map((item, index) => {
      const collapsed = index >= 3;
      return `
        <article class="item-card ${collapsed ? "is-collapsed" : ""}">
          <small>${item.id.replace("_", " ")}${collapsed ? " (collapsed)" : ""}</small>
          <p>${escapeHtml(item.text)}</p>
        </article>
      `;
    })
    .join("");
}

function renderModeViews() {
  const modeData = getCurrentModeData();
  renderTimeline(modeData);
  renderStageCards(modeData);
  renderMetrics(modeData);
  renderScores(modeData);
  renderBufferView(modeData);

  const viewLabel = modeDisplayName(state.mode);
  setDataStatus(`Mode set to ${viewLabel}.`);
}

function renderTimeline(modeData) {
  const totalMs = modeData.summary_metrics.total_latency_ms;
  ui.timelineTrack.innerHTML = state.events
    .map((event, index) => {
      const weight = Math.max(event.duration_ms, totalMs * 0.04);
      return `
        <button
          class="timeline-segment"
          data-event-index="${index}"
          style="--weight: ${weight};"
          type="button"
        >
          <strong>${escapeHtml(event.label)}</strong>
          <span>${formatMs(event.duration_ms)}</span>
        </button>
      `;
    })
    .join("");
}

function renderStageCards(modeData) {
  ui.stageCards.innerHTML = modeData.stages
    .map((stage) => {
      const skipped = stage.status === "skipped";
      return `
        <button class="stage-card" data-stage-id="${stage.id}" type="button">
          <header>
            <strong>${escapeHtml(stage.name)}</strong>
            <span>${formatMs(stage.latency_ms)}</span>
          </header>
          <p>${escapeHtml(stage.description)}</p>
          <span class="status-chip ${skipped ? "skipped" : ""}">
            ${skipped ? "skipped" : "active"}
          </span>
        </button>
      `;
    })
    .join("");
}

function renderMetrics(modeData) {
  const summary = modeData.summary_metrics;
  const cache = cacheBadge(modeData);

  ui.metricTotalLatency.textContent = formatMs(summary.total_latency_ms);
  ui.metricItemsPerSec.textContent = `${summary.items_per_sec.toFixed(1)}`;
  ui.metricCacheBadge.textContent = cache.label;
  ui.metricCacheBadge.className = cache.tone;
  ui.metricCompilerCache.textContent = summary.compiler_cache;

  const maxLatency = Math.max(
    1,
    ...modeData.stages.map((stage) => stage.latency_ms),
  );

  ui.stageMetrics.innerHTML = modeData.stages
    .map((stage) => {
      const widthPercent = (stage.latency_ms / maxLatency) * 100;
      return `
        <div class="metric-row">
          <label>${escapeHtml(stage.name)}</label>
          <div class="metric-bar"><span style="width: ${widthPercent.toFixed(1)}%"></span></div>
          <code>${formatMs(stage.latency_ms)}</code>
        </div>
      `;
    })
    .join("");
}

function renderScores(modeData) {
  const itemById = new Map(
    state.dataset.example.items.map((item) => [item.id, item]),
  );

  ui.scoreRows.innerHTML = modeData.scores
    .map((row) => {
      const item = itemById.get(row.item_id);
      const relevantPct = row.relevant * 100;
      return `
        <article class="score-row">
          <p><strong>${escapeHtml(row.item_id)}</strong> - ${escapeHtml(item?.text ?? "")}</p>
          <div class="score-bar"><span style="width: ${relevantPct.toFixed(1)}%"></span></div>
          <code>relevant ${relevantPct.toFixed(1)}%</code>
        </article>
      `;
    })
    .join("");
}

function renderBufferView(modeData) {
  const isPacked = state.mode === "packed";

  if (isPacked) {
    ui.bufferGrid.innerHTML = [
      createBufferCell("query", "Shared Query", "1 logical prefix", 8),
      ...state.dataset.example.items.map((_item, index) =>
        createBufferCell("item", `Item ${index + 1}`, "packed segment", 1),
      ),
      createBufferCell("mask", "Mask Matrix", "block-diagonal isolation", 8),
    ].join("");
    ui.bufferNote.textContent =
      "Packed mode materializes one long sequence buffer and enforces isolation with a block mask. Throughput depends heavily on packed shape and mask work.";
  } else {
    ui.bufferGrid.innerHTML = [
      createBufferCell("query", "Query Prefill", "run once", 8),
      createBufferCell("cache", "KV Cache", "prefix reused by all items", 8),
      ...state.dataset.example.items.map((_item, index) =>
        createBufferCell("item", `Item ${index + 1}`, "extend window", 1),
      ),
      createBufferCell("mask", "Isolation Windows", "per-item extend context", 8),
    ].join("");
    ui.bufferNote.textContent =
      "Prefill+extend separates prefix from item tails: create KV cache once, then run short extends that reuse cached prefix state.";
  }

  const activeStageId = state.snapshot?.activeEvent?.stage_id;
  highlightBufferCells(activeStageId);
}

function onReplayUpdate(snapshot) {
  state.snapshot = snapshot;

  ui.playPauseBtn.textContent = snapshot.isPlaying ? "Pause" : "Play";
  ui.speedValue.textContent = `${snapshot.speed.toFixed(2)}x`;

  ui.timelineProgress.style.width = `${(snapshot.progress * 100).toFixed(2)}%`;
  ui.replayClock.textContent = `${formatMs(snapshot.positionMs)} / ${formatMs(snapshot.durationMs)}`;
  ui.activeStageLabel.textContent = `Stage: ${snapshot.activeEvent?.label || "-"}`;

  const activeStageId = snapshot.activeEvent?.stage_id;

  const segmentNodes = ui.timelineTrack.querySelectorAll(".timeline-segment");
  segmentNodes.forEach((segment, index) => {
    const eventData = state.events[index];
    if (!eventData) return;
    const complete = snapshot.positionMs >= eventData.end_ms && eventData.duration_ms > 0;
    const active =
      snapshot.positionMs >= eventData.start_ms &&
      snapshot.positionMs < eventData.end_ms + 1e-5 &&
      snapshot.activeEvent?.id === eventData.id;

    segment.classList.toggle("is-complete", complete);
    segment.classList.toggle("is-active", active);
  });

  const cardNodes = ui.stageCards.querySelectorAll(".stage-card");
  cardNodes.forEach((card) => {
    const stageId = card.getAttribute("data-stage-id");
    const stageEvents = state.events.filter((event) => event.stage_id === stageId);
    const stageEnd = Math.max(...stageEvents.map((event) => event.end_ms), 0);
    const isComplete = snapshot.positionMs >= stageEnd && stageEnd > 0;

    card.classList.toggle("is-complete", isComplete);
    card.classList.toggle("is-active", stageId === activeStageId);
  });

  updateOverlayText();
  highlightBufferCells(activeStageId);
}

function updateOverlayText() {
  const modeData = getCurrentModeData();
  if (!modeData) return;

  const stageId = state.snapshot?.activeEvent?.stage_id || modeData.stages[0]?.id;
  const stage = modeData.stages.find((value) => value.id === stageId) || modeData.stages[0];

  ui.overlayStageName.textContent = stage?.name || "-";
  ui.overlayText.textContent = stage?.[state.overlay] || "-";
}

function highlightBufferCells(stageId) {
  const cells = ui.bufferGrid.querySelectorAll(".buffer-cell");
  cells.forEach((cell) => cell.classList.remove("is-active"));

  if (!stageId) return;

  const modeByStage = {
    input_tokenization: ["query", "item"],
    prefix_query_handling: ["query", "item"],
    cache_creation: ["cache", "query"],
    extend_batch_processing: ["item", "cache"],
    attention_isolation: ["mask", "item"],
    score_extraction: ["item"],
  };

  const classes = modeByStage[stageId] || [];
  for (const targetClass of classes) {
    ui.bufferGrid.querySelectorAll(`.buffer-cell.${targetClass}`).forEach((cell) => {
      cell.classList.add("is-active");
    });
  }
}

function createBufferCell(type, title, subtitle, span) {
  return `
    <article class="buffer-cell ${type}" style="grid-column: span ${span};">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(subtitle)}</span>
    </article>
  `;
}

function getCurrentModeData() {
  if (!state.dataset) return null;
  return state.dataset.modes[state.mode];
}

function setDataStatus(message) {
  ui.dataStatus.textContent = message;
}

function formatMs(value) {
  return `${Number(value || 0).toFixed(1)} ms`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
