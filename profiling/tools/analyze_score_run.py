#!/usr/bin/env python3
"""Analyze a profiling run folder and generate tables + charts.

Expected run directory layout:
  runs/<run-id>/
    artifacts/raw/host/traces/**/t1v-*.trace.json.gz
    artifacts/raw/host/traces/**/t1v-*.xplane.pb
    artifacts/raw/device/traces-device/**/t1v-*.xplane.pb
    artifacts/derived/host/*.json (optional)
    artifacts/derived/device/*.json (optional)
    analysis/
    images/

If derived JSONs are missing, the script attempts to generate them via xprof.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = list(root.rglob(pattern))
    return matches[0] if matches else None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_md_table(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    _write_text(path, "\n".join(lines) + "\n")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_trace_events(trace_gz: Path):
    with gzip.open(trace_gz, "rb") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def _maybe_xprof_convert(xplane: Path, out_dir: Path, suffix: str, force: bool) -> Dict[str, Path]:
    """Convert xplane to overview/op_profile/framework_op_stats JSONs."""
    _ensure_dir(out_dir)
    out = {
        f"overview_page{suffix}.json": out_dir / f"overview_page{suffix}.json",
        f"op_profile{suffix}.json": out_dir / f"op_profile{suffix}.json",
        f"framework_op_stats{suffix}.json": out_dir / f"framework_op_stats{suffix}.json",
    }

    if not force and all(p.exists() for p in out.values()):
        return out

    try:
        from xprof.convert import raw_to_tool_data as rtd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xprof is required to generate derived JSONs. Install with: "
            "pip install xprof tensorboard tensorboard-plugin-profile"
        ) from exc

    # overview
    text, _ = rtd.xspace_to_tool_data([str(xplane)], "overview_page", {"use_saved_result": False})
    out[f"overview_page{suffix}.json"].write_text(
        text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text,
        encoding="utf-8",
    )

    # op_profile
    text, _ = rtd.xspace_to_tool_data(
        [str(xplane)], "op_profile", {"use_saved_result": False, "group_by": "program"}
    )
    out[f"op_profile{suffix}.json"].write_text(
        text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text,
        encoding="utf-8",
    )

    # framework_op_stats
    text, _ = rtd.xspace_to_tool_data([str(xplane)], "framework_op_stats", {"use_saved_result": False})
    out[f"framework_op_stats{suffix}.json"].write_text(
        text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text,
        encoding="utf-8",
    )

    return out


def _summarize_host_trace(trace_gz: Path, out_path: Path, out_png: Path | None = None) -> float | None:
    events = _load_trace_events(trace_gz)
    name_stats: Dict[str, Dict[str, float]] = {}
    ts_min = None
    ts_max = None

    for e in events:
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)

        dur = e.get("dur")
        if not isinstance(dur, (int, float)):
            continue
        name = e.get("name", "unknown")
        stat = name_stats.setdefault(name, {"count": 0, "dur_us": 0.0})
        stat["count"] += 1
        stat["dur_us"] += dur

    window_ms: float | None = None
    if ts_min is not None and ts_max is not None:
        window_ms = (ts_max - ts_min) / 1000.0

    top = sorted(name_stats.items(), key=lambda x: x[1]["dur_us"], reverse=True)[:20]
    rows = []
    for name, st in top:
        rows.append([
            name.replace("|", "/")[:80],
            str(int(st["count"])) ,
            f"{st['dur_us']/1000.0:.3f}",
        ])

    text = []
    text.append("# Host Trace Summary\n")
    text.append(f"Trace file: `{trace_gz.name}`\n")
    if window_ms is not None:
        text.append(f"Time window: **{window_ms:.3f} ms**\n")
    text.append("Top events by total duration (ms):\n")
    text.append("| Event | Count | Total ms |")
    text.append("|---|---|---|")
    text.extend([f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows])
    _write_text(out_path, "\n".join(text) + "\n")

    if out_png is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return window_ms

        labels = [name.replace("|", "/")[:40] for name, _ in top][::-1]
        values = [st["dur_us"] / 1000.0 for _, st in top][::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color="#d95f02")
        plt.xlabel("Total time (ms)")
        plt.title("Top Host Events by Total Duration")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)

    return window_ms


def _load_framework_op_stats(path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    data = _load_json(path)
    # format: list of tables
    table = data[0]
    cols = table["cols"]
    col_index = {c["id"]: i for i, c in enumerate(cols)}
    rows = table["rows"]
    return rows, col_index


def _extract_non_idle(rows: List[Dict], col_index: Dict[str, int]):
    non_idle = []
    idle_total = 0.0
    for row in rows:
        c = row["c"]
        op_type = c[col_index["type"]]["v"]
        op_name = c[col_index["operation"]]["v"]
        total_time = c[col_index["total_time"]]["v"]
        if op_type == "IDLE" or op_name == "IDLE":
            idle_total += total_time
            continue
        non_idle.append((op_type, op_name, total_time, row))
    return non_idle, idle_total


def _write_op_type_breakdown(rows, col_index, out_md, out_png):
    from collections import defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    non_idle, idle_total = _extract_non_idle(rows, col_index)
    agg = defaultdict(float)
    for op_type, _, total_time, _ in non_idle:
        agg[op_type] += total_time
    total_non_idle = sum(agg.values())

    # markdown table
    items = [(k, v) for k, v in agg.items() if v > 0]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    md_rows = []
    for op_type, total in items:
        pct = (total / total_non_idle * 100.0) if total_non_idle else 0.0
        md_rows.append([op_type, f"{total:.1f}", f"{pct:.1f}%"])
    _write_md_table(out_md, ["Op type", "Total time (us)", "% of non-idle"], md_rows)

    # chart
    top = items[:10]
    labels = [k for k, _ in top][::-1]
    values = [v for _, v in top][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="#1b9e77")
    plt.xlabel("Total time (us)")
    plt.title("Top Device Op Types (Non-idle)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

    return total_non_idle, idle_total


def _write_top_ops(rows, col_index, out_md, out_png, limit=20):
    non_idle, _ = _extract_non_idle(rows, col_index)
    total_non_idle = sum(t for _, _, t, _ in non_idle)

    # Sort by total time
    non_idle_sorted = sorted(non_idle, key=lambda x: x[2], reverse=True)[:limit]
    md_rows = []
    for op_type, op_name, total_time, row in non_idle_sorted:
        occurrences = row["c"][col_index["occurrences"]]["v"]
        avg_time = row["c"][col_index["avg_time"]]["v"]
        pct = (total_time / total_non_idle * 100.0) if total_non_idle else 0.0
        md_rows.append([
            op_type,
            op_name.replace("|", "/")[:90],
            f"{occurrences}",
            f"{total_time:.3f}",
            f"{avg_time:.3f}",
            f"{pct:.2f}%",
        ])

    _write_md_table(out_md, ["Op type", "Operation", "Occurrences", "Total us", "Avg us", "% of non-idle"], md_rows)

    # Chart (top 10)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    top = non_idle_sorted[:10]
    labels = [f"{op_type}: {op_name[:30]}" for op_type, op_name, _, _ in top][::-1]
    values = [t for _, _, t, _ in top][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color="#2c7fb8")
    plt.xlabel("Total time (us)")
    plt.title("Top Device Ops (Non-idle)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def _component_class(op_type: str, op_name: str) -> str:
    name = op_name.lower()
    if "flashattention" in name or "ragged_paged_attention" in name or "rpa-" in name:
        return "attention.flash"
    if "qwen3attention" in name or "qwen3attention" in op_name.lower() or "attention" in name:
        if op_type == "dot_general":
            return "attention.matmul"
        return "attention.misc"
    if "qwen3mlp" in name or "mlp" in name:
        if op_type == "dot_general":
            return "mlp.matmul"
        return "mlp.misc"
    if "sampler" in name or "logitsprocessor" in name:
        return "sampler/logits"
    if op_type == "Unknown" or "while" in name or "conditional" in name:
        return "control_flow"
    return "other"


def _write_component_breakdown(rows, col_index, out_md, out_png):
    from collections import defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    non_idle, _ = _extract_non_idle(rows, col_index)
    agg = defaultdict(float)
    for op_type, op_name, total_time, _ in non_idle:
        comp = _component_class(op_type, op_name)
        agg[comp] += total_time

    total_non_idle = sum(agg.values())
    items = [(k, v) for k, v in agg.items() if v > 0]
    items = sorted(items, key=lambda x: x[1], reverse=True)

    md_rows = []
    for comp, total in items:
        pct = (total / total_non_idle * 100.0) if total_non_idle else 0.0
        md_rows.append([comp, f"{total:.1f}", f"{pct:.1f}%"])
    _write_md_table(out_md, ["Component", "Total time (us)", "% of non-idle"], md_rows)

    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="#7570b3")
    plt.xlabel("Total time (us)")
    plt.title("Device Time by Component (Non-idle)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def _write_idle_vs_non_idle(idle_total, non_idle_total, out_md, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = idle_total + non_idle_total
    idle_pct = (idle_total / total * 100.0) if total else 0.0
    non_idle_pct = (non_idle_total / total * 100.0) if total else 0.0

    md = [
        "# Device Idle vs Non-idle\n",
        f"Total device time (trace window): **{total:.1f} us**\n",
        "| Category | Time (us) | % of device time |",
        "|---|---|---|",
        f"| Idle | {idle_total:.1f} | {idle_pct:.2f}% |",
        f"| Non-idle | {non_idle_total:.1f} | {non_idle_pct:.2f}% |",
    ]
    _write_text(out_md, "\n".join(md) + "\n")

    plt.figure(figsize=(5, 4))
    plt.bar(["Idle", "Non-idle"], [idle_total, non_idle_total], color=["#d95f02", "#1b9e77"])
    plt.ylabel("Time (us)")
    plt.title("Device Idle vs Non-idle")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to runs/<id> directory")
    parser.add_argument("--force-xprof", action="store_true", help="Regenerate derived JSONs")
    parser.add_argument("--skip-xprof", action="store_true", help="Skip xprof conversion even if missing")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    analysis_dir = run_dir / "analysis"
    images_dir = run_dir / "images"
    _ensure_dir(analysis_dir)
    _ensure_dir(images_dir)

    # Host trace summary
    host_trace = _find_first(run_dir / "artifacts/raw/host", "*.trace.json.gz")
    host_window_ms = None
    if host_trace:
        host_window_ms = _summarize_host_trace(
            host_trace,
            analysis_dir / "host_trace_summary.md",
            images_dir / "host_top_events.png",
        )

    # Derived JSONs
    derived_host = run_dir / "artifacts/derived/host"
    derived_device = run_dir / "artifacts/derived/device"
    _ensure_dir(derived_host)
    _ensure_dir(derived_device)

    if not args.skip_xprof:
        host_xplane = _find_first(run_dir / "artifacts/raw/host", "*.xplane.pb")
        if host_xplane:
            _maybe_xprof_convert(host_xplane, derived_host, "", args.force_xprof)

        device_xplane = _find_first(run_dir / "artifacts/raw/device", "*.xplane.pb")
        if device_xplane:
            _maybe_xprof_convert(device_xplane, derived_device, "_device", args.force_xprof)

    # Device op stats analysis
    device_framework = derived_device / "framework_op_stats_device.json"
    if device_framework.exists():
        rows, col_index = _load_framework_op_stats(device_framework)
        _write_top_ops(
            rows,
            col_index,
            analysis_dir / "device_top_ops.md",
            images_dir / "top_ops_device.png",
            limit=20,
        )
        non_idle_total, idle_total = _write_op_type_breakdown(
            rows, col_index, analysis_dir / "op_type_breakdown.md", images_dir / "optype_breakdown.png"
        )
        _write_component_breakdown(
            rows, col_index, analysis_dir / "component_breakdown.md", images_dir / "component_breakdown.png"
        )
        _write_idle_vs_non_idle(
            idle_total, non_idle_total, analysis_dir / "idle_vs_non_idle.md", images_dir / "idle_vs_non_idle.png"
        )

        # Save summary JSON
        summary = {
            "device_idle_us": idle_total,
            "device_non_idle_us": non_idle_total,
            "device_total_us": idle_total + non_idle_total,
        }

        if host_window_ms is not None:
            summary["host_trace_window_ms"] = host_window_ms

        # Optional server log counters
        log_path = run_dir / "logs" / "server-device.log"
        if not log_path.exists():
            log_path = run_dir / "logs" / "server.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            summary["score_requests"] = text.count("POST /v1/score")
            summary["start_profile_calls"] = text.count("POST /start_profile")
            summary["stop_profile_calls"] = text.count("POST /stop_profile")

        _write_text(analysis_dir / "summary.json", json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
