#!/usr/bin/env python3
"""Render final side-by-side JAX vs PyTorch report from comparison JSON."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def pct_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if float(b) == 0.0:
        return None
    return ((float(a) - float(b)) / float(b)) * 100.0


def choose_portable_highlights(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = []
    for row in rows:
        j = row.get("jax") or {}
        p = row.get("pytorch") or {}
        c = row.get("correctness") or {}
        if not c.get("pass"):
            continue
        if j.get("throughput_items_sec") is None or p.get("throughput_items_sec") is None:
            continue
        if j.get("p95_total_e2e_ms") is None or p.get("p95_total_e2e_ms") is None:
            continue
        comparable.append(row)

    if not comparable:
        return {
            "jax_better_chunks": 0,
            "pytorch_better_chunks": 0,
            "best_jax_throughput_edge": None,
            "best_pytorch_throughput_edge": None,
        }

    jax_better = []
    pyt_better = []

    for row in comparable:
        j_thr = float(row["jax"]["throughput_items_sec"])
        p_thr = float(row["pytorch"]["throughput_items_sec"])
        if j_thr > p_thr:
            jax_better.append(row)
        elif p_thr > j_thr:
            pyt_better.append(row)

    def best_edge(rows_in: List[Dict[str, Any]], winner: str) -> Optional[Dict[str, Any]]:
        if not rows_in:
            return None

        def edge_value(row: Dict[str, Any]) -> float:
            j_thr = float(row["jax"]["throughput_items_sec"])
            p_thr = float(row["pytorch"]["throughput_items_sec"])
            if winner == "jax":
                return j_thr - p_thr
            return p_thr - j_thr

        best = sorted(rows_in, key=edge_value, reverse=True)[0]
        return {
            "chunk": int(best["client_chunk_size"]),
            "jax_thr": float(best["jax"]["throughput_items_sec"]),
            "pytorch_thr": float(best["pytorch"]["throughput_items_sec"]),
            "delta_items_sec": abs(float(best["jax"]["throughput_items_sec"]) - float(best["pytorch"]["throughput_items_sec"])),
            "delta_percent_vs_pytorch": pct_delta(
                best["jax"]["throughput_items_sec"], best["pytorch"]["throughput_items_sec"]
            ),
        }

    return {
        "jax_better_chunks": len(jax_better),
        "pytorch_better_chunks": len(pyt_better),
        "best_jax_throughput_edge": best_edge(jax_better, "jax"),
        "best_pytorch_throughput_edge": best_edge(pyt_better, "pytorch"),
    }


def build_report(
    comparison: Dict[str, Any],
    output_path: Path,
    title_date: str,
) -> None:
    portable = comparison.get("portable_view") or {}
    best_native = comparison.get("best_native_view") or {}
    rows = portable.get("rows") or []

    portable_winner = ((portable.get("winner") or {}).get("winner")) or None
    native_winner = (((best_native.get("winner") or {}).get("winner")) or None)

    native_jax = best_native.get("jax_selected") or None
    native_pytorch = best_native.get("pytorch_selected") or None
    native_correctness = best_native.get("correctness") or {}

    highlights = choose_portable_highlights(rows)

    lines: List[str] = []
    lines.append(f"# Report: JAX vs PyTorch Multi-Item Comparison ({title_date})")
    lines.append("")
    lines.append("| | |")
    lines.append("|------------|------|")
    lines.append(f"| **Generated** | {utc_now_iso()} |")
    lines.append("| **Method** | Portable view + Best-native view |")
    lines.append("| **Baseline policy** | PyTorch frozen reference (no implementation changes) |")
    lines.append("")

    lines.append("## Portable View (Side-by-Side)")
    lines.append("")
    lines.append("| Chunk | JAX p95 (ms) | PyTorch p95 (ms) | p95 Delta (JAX-PyTorch, ms) | JAX items/s | PyTorch items/s | Throughput Delta (JAX-PyTorch, %) | Correctness Pass |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        j = row.get("jax") or {}
        p = row.get("pytorch") or {}
        c = row.get("correctness") or {}

        j_p95 = j.get("p95_total_e2e_ms")
        p_p95 = p.get("p95_total_e2e_ms")
        j_thr = j.get("throughput_items_sec")
        p_thr = p.get("throughput_items_sec")

        p95_delta = None
        if j_p95 is not None and p_p95 is not None:
            p95_delta = float(j_p95) - float(p_p95)

        thr_delta_pct = pct_delta(j_thr, p_thr)

        lines.append(
            "| {chunk} | {jp95} | {pp95} | {p95d} | {jthr} | {pthr} | {thrd} | {cp} |".format(
                chunk=row.get("client_chunk_size"),
                jp95=fmt_num(j_p95),
                pp95=fmt_num(p_p95),
                p95d=fmt_num(p95_delta),
                jthr=fmt_num(j_thr),
                pthr=fmt_num(p_thr),
                thrd=(f"{thr_delta_pct:.2f}" if thr_delta_pct is not None else "-"),
                cp="yes" if c.get("pass") else "no",
            )
        )

    lines.append("")
    if portable_winner:
        lines.append(
            "Portable winner: **{backend}** at chunk `{chunk}` (throughput `{thr}` items/s, p95 `{p95}` ms).".format(
                backend=portable_winner["backend"],
                chunk=portable_winner["client_chunk_size"],
                thr=fmt_num(portable_winner["throughput_items_sec"]),
                p95=fmt_num(portable_winner["p95_total_e2e_ms"]),
            )
        )
    else:
        lines.append("Portable winner: **none** (no eligible candidates after gating).")

    lines.append("")
    lines.append("## Best-Native View (Side-by-Side)")
    lines.append("")
    lines.append("| Metric | JAX | PyTorch | Delta (JAX-PyTorch) |")
    lines.append("|---|---:|---:|---:|")

    j_chunk = native_jax.get("client_chunk_size") if native_jax else None
    p_chunk = native_pytorch.get("client_chunk_size") if native_pytorch else None
    lines.append(f"| Selected chunk | {j_chunk if j_chunk is not None else '-'} | {p_chunk if p_chunk is not None else '-'} | - |")

    j_p95 = native_jax.get("p95_total_e2e_ms") if native_jax else None
    p_p95 = native_pytorch.get("p95_total_e2e_ms") if native_pytorch else None
    p95_delta = (float(j_p95) - float(p_p95)) if (j_p95 is not None and p_p95 is not None) else None
    lines.append(f"| p95 latency (ms) | {fmt_num(j_p95)} | {fmt_num(p_p95)} | {fmt_num(p95_delta)} |")

    j_thr = native_jax.get("throughput_items_sec") if native_jax else None
    p_thr = native_pytorch.get("throughput_items_sec") if native_pytorch else None
    thr_delta_pct = pct_delta(j_thr, p_thr)
    lines.append(
        f"| Throughput (items/s) | {fmt_num(j_thr)} | {fmt_num(p_thr)} | {(f'{thr_delta_pct:.2f}%' if thr_delta_pct is not None else '-')} |"
    )

    j_sr = native_jax.get("success_rate") if native_jax else None
    p_sr = native_pytorch.get("success_rate") if native_pytorch else None
    lines.append(f"| Success rate | {fmt_num(j_sr, 3)} | {fmt_num(p_sr, 3)} | - |")

    lines.append("")
    lines.append(
        "Best-native correctness gate: status=`{status}`, max_abs_diff=`{max_abs}`, mean_abs_diff=`{mean_abs}`, pass=`{passed}`.".format(
            status=native_correctness.get("status", "unknown"),
            max_abs=fmt_num(native_correctness.get("max_abs_diff"), 6),
            mean_abs=fmt_num(native_correctness.get("mean_abs_diff"), 6),
            passed="yes" if native_correctness.get("pass") else "no",
        )
    )

    if native_winner:
        lines.append(
            "Best-native winner: **{backend}** at chunk `{chunk}` (throughput `{thr}` items/s, p95 `{p95}` ms).".format(
                backend=native_winner["backend"],
                chunk=native_winner["client_chunk_size"],
                thr=fmt_num(native_winner["throughput_items_sec"]),
                p95=fmt_num(native_winner["p95_total_e2e_ms"]),
            )
        )
    else:
        lines.append("Best-native winner: **none** (no eligible candidates after gating).")

    lines.append("")
    lines.append("## Findings")
    lines.append("")

    lines.append(
        "1. Portable view chunk-level comparison: JAX better on `{}` chunks, PyTorch better on `{}` chunks (among correctness-pass rows).".format(
            highlights["jax_better_chunks"], highlights["pytorch_better_chunks"]
        )
    )

    j_edge = highlights.get("best_jax_throughput_edge")
    if j_edge:
        lines.append(
            "2. Largest JAX portable throughput edge: chunk `{}` with `+{:.2f}` items/s (`{:.2f}%` vs PyTorch).".format(
                j_edge["chunk"],
                j_edge["delta_items_sec"],
                j_edge["delta_percent_vs_pytorch"] if j_edge["delta_percent_vs_pytorch"] is not None else 0.0,
            )
        )
    else:
        lines.append("2. Largest JAX portable throughput edge: none observed.")

    p_edge = highlights.get("best_pytorch_throughput_edge")
    if p_edge:
        lines.append(
            "3. Largest PyTorch portable throughput edge: chunk `{}` with `+{:.2f}` items/s (JAX lagging by `{:.2f}%` at that chunk).".format(
                p_edge["chunk"],
                p_edge["delta_items_sec"],
                -p_edge["delta_percent_vs_pytorch"] if p_edge["delta_percent_vs_pytorch"] is not None else 0.0,
            )
        )
    else:
        lines.append("3. Largest PyTorch portable throughput edge: none observed.")

    if native_winner:
        lines.append(
            "4. Best-native decision currently favors **{}** under the configured correctness and latency guardrails.".format(
                native_winner["backend"]
            )
        )
    else:
        lines.append("4. Best-native decision is pending because no eligible winner passed all gates.")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if native_winner:
        lines.append(
            "Use the best-native winner (**{}**) as primary deployment recommendation, while keeping portable results as backend-behavior evidence.".format(
                native_winner["backend"]
            )
        )
    elif portable_winner:
        lines.append(
            "Use the portable winner (**{}**) as interim recommendation until best-native eligibility issues are resolved.".format(
                portable_winner["backend"]
            )
        )
    else:
        lines.append("No deployment recommendation yet; rerun after addressing failures/correctness gate issues.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument(
        "--output-report",
        default="reports/jax-vs-pytorch-multi-item-comparison-2026-02-11.md",
    )
    parser.add_argument("--title-date", default="2026-02-11")
    args = parser.parse_args()

    comparison = load_json(Path(args.comparison_json))
    build_report(comparison, Path(args.output_report), args.title_date)
    print(f"Wrote report: {args.output_report}")


if __name__ == "__main__":
    main()
