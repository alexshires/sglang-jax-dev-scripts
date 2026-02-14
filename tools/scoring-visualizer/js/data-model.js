export const MODE_ORDER = ["packed", "prefill_extend"];

export const STAGE_ORDER = [
  "input_tokenization",
  "prefix_query_handling",
  "cache_creation",
  "extend_batch_processing",
  "attention_isolation",
  "score_extraction",
];

export const OVERLAY_KEYS = [
  "reused",
  "recomputed",
  "throughput_impact",
];

export const TECHNICAL_TERMS = {
  prefill: "The forward pass over prefix tokens to build model state before scoring.",
  extend: "The forward pass over item-tail tokens after prefix handling.",
  kv_cache: "Key/value attention state saved from prefill so later extends can reuse prefix work.",
  block_mask: "A block-diagonal attention mask that isolates item-to-item attention in packed mode.",
  items_per_sec: "Effective throughput: scored items divided by end-to-end request latency.",
  cache_hit: "A request can reuse an existing prefix cache entry rather than recomputing it.",
};

export function normalizeDataset(raw) {
  if (!raw || typeof raw !== "object") {
    throw new Error("Dataset must be a JSON object.");
  }

  if (!raw.example || typeof raw.example !== "object") {
    throw new Error("Dataset missing 'example' section.");
  }

  if (!Array.isArray(raw.example.items) || raw.example.items.length === 0) {
    throw new Error("Dataset example.items must be a non-empty array.");
  }

  if (!raw.modes || typeof raw.modes !== "object") {
    throw new Error("Dataset missing 'modes' section.");
  }

  const normalized = {
    dataset_version: raw.dataset_version || "unknown",
    generated_on: raw.generated_on || "unknown",
    source: raw.source || {},
    example: {
      request_id: raw.example.request_id || "sample-request",
      query_text: raw.example.query_text || "",
      items: raw.example.items.map((item, index) => ({
        id: item.id || `item_${index + 1}`,
        text: item.text || "",
      })),
      label_token_ids: Array.isArray(raw.example.label_token_ids)
        ? raw.example.label_token_ids
        : [],
      label_names: Array.isArray(raw.example.label_names)
        ? raw.example.label_names
        : ["label_a", "label_b"],
    },
    modes: {},
  };

  for (const modeKey of MODE_ORDER) {
    const modeRaw = raw.modes[modeKey];
    if (!modeRaw || typeof modeRaw !== "object") {
      throw new Error(`Dataset missing mode '${modeKey}'.`);
    }

    const stageById = new Map();
    const stages = Array.isArray(modeRaw.stages) ? modeRaw.stages : [];
    for (const stage of stages) {
      if (stage && typeof stage.id === "string") {
        stageById.set(stage.id, stage);
      }
    }

    const normalizedStages = STAGE_ORDER.map((stageId) => {
      const stage = stageById.get(stageId) || {};
      return {
        id: stageId,
        name: stage.name || stageId,
        latency_ms: finiteNumber(stage.latency_ms, 0),
        status: stage.status || "active",
        description: stage.description || "",
        reused: stage.reused || "No reuse details available.",
        recomputed: stage.recomputed || "No recompute details available.",
        throughput_impact:
          stage.throughput_impact || "No throughput impact details available.",
      };
    });

    const normalizedEvents = normalizeEvents(modeRaw.events, normalizedStages);
    const latencySum = normalizedStages.reduce(
      (acc, stage) => acc + stage.latency_ms,
      0,
    );
    const eventDurationSum = normalizedEvents.reduce(
      (acc, event) => acc + event.duration_ms,
      0,
    );

    const summary = modeRaw.summary_metrics || {};
    const totalLatencyMs =
      finiteNumber(summary.total_latency_ms, 0) > 0
        ? finiteNumber(summary.total_latency_ms, 0)
        : eventDurationSum > 0
          ? eventDurationSum
          : latencySum;

    const itemsCount = normalized.example.items.length;
    const derivedIps = totalLatencyMs > 0 ? (itemsCount * 1000) / totalLatencyMs : 0;

    const scoresById = new Map();
    if (Array.isArray(modeRaw.scores)) {
      for (const scoreRow of modeRaw.scores) {
        if (scoreRow && typeof scoreRow.item_id === "string") {
          scoresById.set(scoreRow.item_id, scoreRow);
        }
      }
    }

    const normalizedScores = normalized.example.items.map((item) => {
      const row = scoresById.get(item.id) || {};
      const relevant = clamp01(finiteNumber(row.relevant, 0.5));
      const notRelevant = clamp01(
        finiteNumber(row.not_relevant, 1 - relevant),
      );
      return {
        item_id: item.id,
        relevant,
        not_relevant: notRelevant,
      };
    });

    normalized.modes[modeKey] = {
      title: modeRaw.title || modeKey,
      cache: {
        supported: Boolean(modeRaw.cache?.supported),
        status: modeRaw.cache?.status || "unknown",
        hits: finiteNumber(modeRaw.cache?.hits, 0),
        misses: finiteNumber(modeRaw.cache?.misses, 0),
      },
      summary_metrics: {
        total_latency_ms: totalLatencyMs,
        items_per_sec:
          finiteNumber(summary.items_per_sec, 0) > 0
            ? finiteNumber(summary.items_per_sec, 0)
            : derivedIps,
        compiler_cache: summary.compiler_cache || "unknown",
      },
      stages: normalizedStages,
      events: normalizedEvents,
      scores: normalizedScores,
    };
  }

  return normalized;
}

export function buildStageEvents(modeData) {
  const stageById = new Map(modeData.stages.map((stage) => [stage.id, stage]));
  const sourceEvents =
    Array.isArray(modeData.events) && modeData.events.length > 0
      ? modeData.events
      : modeData.stages.map((stage, index) => ({
          id: `event_${index + 1}`,
          stage_id: stage.id,
          label: stage.name,
          duration_ms: stage.latency_ms,
          active_components: [],
          parallel_items: [],
        }));

  let cursor = 0;
  return sourceEvents.map((event, index) => {
    const stage = stageById.get(event.stage_id) || modeData.stages[index] || null;
    const start_ms = cursor;
    const duration_ms = Math.max(0, finiteNumber(event.duration_ms, 0));
    const end_ms = start_ms + duration_ms;
    cursor = end_ms;
    return {
      ...stage,
      ...event,
      index,
      start_ms,
      end_ms,
      duration_ms,
    };
  });
}

export function modeDisplayName(modeKey) {
  return modeKey === "prefill_extend" ? "Prefill+extend" : "Packed";
}

export function cacheBadge(modeData) {
  if (!modeData.cache.supported) {
    return {
      label: "Cache N/A",
      tone: "neutral",
      details: "Packed mode computes scores without reusable prefix cache.",
    };
  }

  const total = modeData.cache.hits + modeData.cache.misses;
  const hitRate = total > 0 ? (modeData.cache.hits / total) * 100 : 0;

  if (modeData.cache.hits > 0 && modeData.cache.misses > 0) {
    return {
      label: `Miss->Hit ${Math.round(hitRate)}%`,
      tone: "good",
      details: `${modeData.cache.hits} hits / ${modeData.cache.misses} misses in this request flow.`,
    };
  }

  if (modeData.cache.hits > 0) {
    return {
      label: `Hit ${Math.round(hitRate)}%`,
      tone: "good",
      details: `${modeData.cache.hits} hits and ${modeData.cache.misses} misses.`,
    };
  }

  return {
    label: "Cache Miss",
    tone: "warn",
    details: `${modeData.cache.hits} hits and ${modeData.cache.misses} misses.`,
  };
}

export function adaptExternalDataset(raw, fallbackDataset) {
  if (!raw || typeof raw !== "object") {
    throw new Error("Uploaded payload is not valid JSON object data.");
  }

  if (raw.example && raw.modes) {
    return normalizeDataset(raw);
  }

  if (raw.packed && raw.prefill_extend) {
    const cloned = structuredClone(fallbackDataset);
    patchMode(cloned, "packed", raw.packed);
    patchMode(cloned, "prefill_extend", raw.prefill_extend);
    return normalizeDataset(cloned);
  }

  if (Array.isArray(raw.results)) {
    const packed = raw.results.find((r) => r.mode === "packed");
    const prefill = raw.results.find((r) => r.mode === "prefill_extend");
    if (packed && prefill) {
      const cloned = structuredClone(fallbackDataset);
      patchMode(cloned, "packed", packed);
      patchMode(cloned, "prefill_extend", prefill);
      return normalizeDataset(cloned);
    }
  }

  throw new Error(
    "Unsupported JSON shape. Expected this app schema or a results payload with packed/prefill_extend metrics.",
  );
}

function patchMode(dataset, modeKey, patch) {
  if (!dataset?.modes?.[modeKey]) {
    return;
  }

  const mode = dataset.modes[modeKey];
  if (isFinite(patch.total_latency_ms)) {
    mode.summary_metrics.total_latency_ms = Number(patch.total_latency_ms);
  }
  if (isFinite(patch.items_per_sec)) {
    mode.summary_metrics.items_per_sec = Number(patch.items_per_sec);
  }
  if (isFinite(patch.cache_hits)) {
    mode.cache.hits = Number(patch.cache_hits);
    mode.cache.supported = true;
  }
  if (isFinite(patch.cache_misses)) {
    mode.cache.misses = Number(patch.cache_misses);
    mode.cache.supported = true;
  }
}

function normalizeEvents(rawEvents, stages) {
  if (!Array.isArray(rawEvents) || rawEvents.length === 0) {
    return stages.map((stage, index) => ({
      id: `event_${index + 1}`,
      stage_id: stage.id,
      label: stage.name,
      duration_ms: stage.latency_ms,
      active_components: [],
      parallel_items: [],
    }));
  }

  const stageIds = new Set(stages.map((stage) => stage.id));
  return rawEvents.map((event, index) => {
    const fallbackStage = stages[Math.min(index, stages.length - 1)];
    const stageId =
      typeof event.stage_id === "string" && stageIds.has(event.stage_id)
        ? event.stage_id
        : fallbackStage.id;
    const fallbackDuration = fallbackStage?.latency_ms ?? 0;
    return {
      id: typeof event.id === "string" ? event.id : `event_${index + 1}`,
      stage_id: stageId,
      label:
        typeof event.label === "string" && event.label.length > 0
          ? event.label
          : fallbackStage.name,
      duration_ms: Math.max(
        0,
        finiteNumber(event.duration_ms, fallbackDuration),
      ),
      active_components: Array.isArray(event.active_components)
        ? event.active_components.map((value) => String(value))
        : [],
      parallel_items: Array.isArray(event.parallel_items)
        ? event.parallel_items
            .map((value) => Number(value))
            .filter((value) => Number.isFinite(value))
        : [],
    };
  });
}

function finiteNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp01(value) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}
