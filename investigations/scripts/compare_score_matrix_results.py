#!/usr/bin/env python3
"""Compare JAX and PyTorch score matrix results and produce final recommendations."""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "cross_backend_comparison_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_chunk(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        out[int(row["client_chunk_size"])] = row
    return out


def compute_score_diff(
    scores_a: Optional[List[List[float]]],
    scores_b: Optional[List[List[float]]],
) -> Dict[str, Any]:
    if not isinstance(scores_a, list) or not isinstance(scores_b, list):
        return {
            "status": "missing_scores",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "num_compared": 0,
            "detail": "One or both reference_scores are missing",
        }

    if len(scores_a) != len(scores_b):
        return {
            "status": "shape_mismatch",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "num_compared": 0,
            "detail": f"Item length mismatch: {len(scores_a)} vs {len(scores_b)}",
        }

    abs_diffs: List[float] = []

    for idx, (item_a, item_b) in enumerate(zip(scores_a, scores_b)):
        if not isinstance(item_a, list) or not isinstance(item_b, list):
            return {
                "status": "shape_mismatch",
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "num_compared": 0,
                "detail": f"Non-list item scores at index {idx}",
            }
        if len(item_a) != len(item_b):
            return {
                "status": "shape_mismatch",
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "num_compared": 0,
                "detail": f"Label length mismatch at item {idx}: {len(item_a)} vs {len(item_b)}",
            }
        for a, b in zip(item_a, item_b):
            abs_diffs.append(abs(float(a) - float(b)))

    if not abs_diffs:
        return {
            "status": "missing_scores",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "num_compared": 0,
            "detail": "No scores to compare",
        }

    return {
        "status": "ok",
        "max_abs_diff": max(abs_diffs),
        "mean_abs_diff": statistics.mean(abs_diffs),
        "num_compared": len(abs_diffs),
        "detail": "",
    }


def row_successful(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return float(row.get("success_rate", 0.0)) == 1.0 and row.get("status") == "ok"


def select_winner(candidates: List[Dict[str, Any]], guardrail_ratio: float) -> Dict[str, Any]:
    eligible = [
        c
        for c in candidates
        if c.get("success_rate") == 1.0
        and c.get("correctness_pass") is True
        and c.get("p95_total_e2e_ms") is not None
        and c.get("throughput_items_sec") is not None
    ]

    if not eligible:
        return {
            "eligible_count": 0,
            "guardrail_ratio": guardrail_ratio,
            "guardrail_p95_ms": None,
            "winner": None,
        }

    min_p95 = min(float(c["p95_total_e2e_ms"]) for c in eligible)
    guardrail_p95 = min_p95 * guardrail_ratio
    within_guardrail = [
        c for c in eligible if float(c["p95_total_e2e_ms"]) <= guardrail_p95
    ]

    winner = sorted(
        within_guardrail,
        key=lambda c: (
            -float(c["throughput_items_sec"]),
            c["backend"],
            int(c["client_chunk_size"]),
        ),
    )[0]

    return {
        "eligible_count": len(eligible),
        "guardrail_ratio": guardrail_ratio,
        "guardrail_p95_ms": guardrail_p95,
        "winner": {
            "backend": winner["backend"],
            "client_chunk_size": winner["client_chunk_size"],
            "throughput_items_sec": winner["throughput_items_sec"],
            "p95_total_e2e_ms": winner["p95_total_e2e_ms"],
        },
    }


def extract_recommended_row(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected = ((doc.get("summary") or {}).get("final_recommendation") or {}).get(
        "selected"
    )
    if not selected:
        return None
    chunk = int(selected["client_chunk_size"])

    for source in [doc.get("confirm_results") or [], doc.get("results") or []]:
        for row in source:
            if int(row.get("client_chunk_size", -1)) == chunk:
                return row
    return None


def build_portable_view(
    jax_doc: Dict[str, Any],
    pytorch_doc: Dict[str, Any],
    guardrail_ratio: float,
    max_abs_threshold: float,
    mean_abs_threshold: float,
) -> Dict[str, Any]:
    jax_rows = rows_by_chunk(jax_doc.get("results") or [])
    pyt_rows = rows_by_chunk(pytorch_doc.get("results") or [])

    common_chunks = sorted(set(jax_rows.keys()) & set(pyt_rows.keys()))

    table_rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    for chunk in common_chunks:
        jax_row = jax_rows[chunk]
        pyt_row = pyt_rows[chunk]

        diff = compute_score_diff(
            jax_row.get("reference_scores"),
            pyt_row.get("reference_scores"),
        )

        correctness_pass = (
            diff["status"] == "ok"
            and diff["max_abs_diff"] is not None
            and diff["mean_abs_diff"] is not None
            and float(diff["max_abs_diff"]) <= max_abs_threshold
            and float(diff["mean_abs_diff"]) <= mean_abs_threshold
        )

        table_rows.append(
            {
                "client_chunk_size": chunk,
                "jax": {
                    "status": jax_row.get("status"),
                    "success_rate": jax_row.get("success_rate"),
                    "p95_total_e2e_ms": jax_row.get("p95_total_e2e_ms"),
                    "throughput_items_sec": jax_row.get("throughput_items_sec"),
                },
                "pytorch": {
                    "status": pyt_row.get("status"),
                    "success_rate": pyt_row.get("success_rate"),
                    "p95_total_e2e_ms": pyt_row.get("p95_total_e2e_ms"),
                    "throughput_items_sec": pyt_row.get("throughput_items_sec"),
                },
                "correctness": {
                    "status": diff["status"],
                    "max_abs_diff": diff["max_abs_diff"],
                    "mean_abs_diff": diff["mean_abs_diff"],
                    "num_compared": diff["num_compared"],
                    "pass": correctness_pass,
                    "detail": diff["detail"],
                },
            }
        )

        for backend, row in [("jax", jax_row), ("pytorch", pyt_row)]:
            candidates.append(
                {
                    "backend": backend,
                    "client_chunk_size": chunk,
                    "success_rate": row.get("success_rate"),
                    "p95_total_e2e_ms": row.get("p95_total_e2e_ms"),
                    "throughput_items_sec": row.get("throughput_items_sec"),
                    "correctness_pass": correctness_pass,
                }
            )

    overall_winner = select_winner(candidates, guardrail_ratio)
    jax_winner = select_winner([c for c in candidates if c["backend"] == "jax"], guardrail_ratio)
    pytorch_winner = select_winner(
        [c for c in candidates if c["backend"] == "pytorch"], guardrail_ratio
    )

    return {
        "guardrail_ratio": guardrail_ratio,
        "rows": table_rows,
        "backend_best": {
            "jax": jax_winner,
            "pytorch": pytorch_winner,
        },
        "winner": overall_winner,
    }


def build_best_native_view(
    jax_doc: Dict[str, Any],
    pytorch_doc: Dict[str, Any],
    guardrail_ratio: float,
    max_abs_threshold: float,
    mean_abs_threshold: float,
) -> Dict[str, Any]:
    jax_row = extract_recommended_row(jax_doc)
    pyt_row = extract_recommended_row(pytorch_doc)

    diff = compute_score_diff(
        (jax_row or {}).get("reference_scores"),
        (pyt_row or {}).get("reference_scores"),
    )

    correctness_pass = (
        diff["status"] == "ok"
        and diff["max_abs_diff"] is not None
        and diff["mean_abs_diff"] is not None
        and float(diff["max_abs_diff"]) <= max_abs_threshold
        and float(diff["mean_abs_diff"]) <= mean_abs_threshold
    )

    candidates: List[Dict[str, Any]] = []
    if jax_row:
        candidates.append(
            {
                "backend": "jax",
                "client_chunk_size": jax_row.get("client_chunk_size"),
                "success_rate": jax_row.get("success_rate"),
                "p95_total_e2e_ms": jax_row.get("p95_total_e2e_ms"),
                "throughput_items_sec": jax_row.get("throughput_items_sec"),
                "correctness_pass": correctness_pass,
            }
        )
    if pyt_row:
        candidates.append(
            {
                "backend": "pytorch",
                "client_chunk_size": pyt_row.get("client_chunk_size"),
                "success_rate": pyt_row.get("success_rate"),
                "p95_total_e2e_ms": pyt_row.get("p95_total_e2e_ms"),
                "throughput_items_sec": pyt_row.get("throughput_items_sec"),
                "correctness_pass": correctness_pass,
            }
        )

    winner = select_winner(candidates, guardrail_ratio)

    return {
        "guardrail_ratio": guardrail_ratio,
        "jax_selected": jax_row,
        "pytorch_selected": pyt_row,
        "correctness": {
            "status": diff["status"],
            "max_abs_diff": diff["max_abs_diff"],
            "mean_abs_diff": diff["mean_abs_diff"],
            "num_compared": diff["num_compared"],
            "pass": correctness_pass,
            "detail": diff["detail"],
        },
        "winner": winner,
    }


def describe_winner(winner_block: Dict[str, Any], label: str) -> str:
    winner = winner_block.get("winner")
    if not winner:
        return f"{label}: no winner (no eligible candidates)."
    winner_obj = winner.get("winner")
    if not winner_obj:
        return f"{label}: no winner (guardrail filtering removed all candidates)."
    return (
        f"{label}: {winner_obj['backend']} wins at chunk {winner_obj['client_chunk_size']} "
        f"with throughput {winner_obj['throughput_items_sec']:.2f} items/s and "
        f"p95 {winner_obj['p95_total_e2e_ms']:.2f} ms."
    )


def describe_overall(
    portable_view: Dict[str, Any],
    best_native_view: Dict[str, Any],
) -> str:
    best_native_winner = ((best_native_view.get("winner") or {}).get("winner")) or None
    if best_native_winner:
        return (
            "Overall: use best-native winner "
            f"({best_native_winner['backend']} at chunk {best_native_winner['client_chunk_size']})."
        )

    portable_winner = ((portable_view.get("winner") or {}).get("winner")) or None
    if portable_winner:
        return (
            "Overall: best-native had no eligible winner, fallback to portable view winner "
            f"({portable_winner['backend']} at chunk {portable_winner['client_chunk_size']})."
        )

    return "Overall: no eligible winner in either view."


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# JAX vs PyTorch Multi-Item Comparison")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at_utc']}")
    lines.append("")

    portable = payload["portable_view"]
    lines.append("## Portable View")
    lines.append("")
    lines.append("| Chunk | JAX p95 (ms) | JAX thr (items/s) | PyTorch p95 (ms) | PyTorch thr (items/s) | Max abs diff | Mean abs diff | Correctness pass |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in portable["rows"]:
        j = row["jax"]
        p = row["pytorch"]
        c = row["correctness"]
        lines.append(
            "| {chunk} | {jp95} | {jthr} | {pp95} | {pthr} | {mx} | {mn} | {ok} |".format(
                chunk=row["client_chunk_size"],
                jp95=(f"{j['p95_total_e2e_ms']:.2f}" if j["p95_total_e2e_ms"] is not None else "-"),
                jthr=(f"{j['throughput_items_sec']:.2f}" if j["throughput_items_sec"] is not None else "-"),
                pp95=(f"{p['p95_total_e2e_ms']:.2f}" if p["p95_total_e2e_ms"] is not None else "-"),
                pthr=(f"{p['throughput_items_sec']:.2f}" if p["throughput_items_sec"] is not None else "-"),
                mx=(f"{c['max_abs_diff']:.6f}" if c["max_abs_diff"] is not None else "-"),
                mn=(f"{c['mean_abs_diff']:.6f}" if c["mean_abs_diff"] is not None else "-"),
                ok="yes" if c["pass"] else "no",
            )
        )

    lines.append("")
    lines.append(f"- {payload['recommendations']['portable']}")
    lines.append("")

    native = payload["best_native_view"]
    lines.append("## Best-Native View")
    lines.append("")
    lines.append("| Backend | Chunk | p95 (ms) | Throughput (items/s) | Success rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for backend_key in ["jax_selected", "pytorch_selected"]:
        row = native.get(backend_key)
        if not row:
            lines.append(f"| {backend_key.replace('_selected', '')} | - | - | - | - |")
            continue
        lines.append(
            "| {backend} | {chunk} | {p95} | {thr} | {success:.2f} |".format(
                backend=backend_key.replace("_selected", ""),
                chunk=row.get("client_chunk_size", "-"),
                p95=(f"{row['p95_total_e2e_ms']:.2f}" if row.get("p95_total_e2e_ms") is not None else "-"),
                thr=(f"{row['throughput_items_sec']:.2f}" if row.get("throughput_items_sec") is not None else "-"),
                success=float(row.get("success_rate", 0.0)),
            )
        )

    c = native["correctness"]
    lines.append("")
    lines.append(
        "- Best-native correctness: status={status}, max_abs_diff={max_abs}, mean_abs_diff={mean_abs}, pass={pass_flag}".format(
            status=c["status"],
            max_abs=(f"{c['max_abs_diff']:.6f}" if c["max_abs_diff"] is not None else "-"),
            mean_abs=(f"{c['mean_abs_diff']:.6f}" if c["mean_abs_diff"] is not None else "-"),
            pass_flag="yes" if c["pass"] else "no",
        )
    )
    lines.append(f"- {payload['recommendations']['best_native']}")

    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- {payload['recommendations']['overall']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-portable-json", required=True)
    parser.add_argument("--pytorch-portable-json", required=True)
    parser.add_argument("--jax-best-native-json", required=True)
    parser.add_argument("--pytorch-best-native-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--guardrail-ratio", type=float, default=1.25)
    parser.add_argument("--correctness-threshold-max-abs", type=float, default=0.02)
    parser.add_argument("--correctness-threshold-mean-abs", type=float, default=0.01)
    args = parser.parse_args()

    jax_portable = load_json(Path(args.jax_portable_json))
    pyt_portable = load_json(Path(args.pytorch_portable_json))
    jax_native = load_json(Path(args.jax_best_native_json))
    pyt_native = load_json(Path(args.pytorch_best_native_json))

    portable_view = build_portable_view(
        jax_doc=jax_portable,
        pytorch_doc=pyt_portable,
        guardrail_ratio=args.guardrail_ratio,
        max_abs_threshold=args.correctness_threshold_max_abs,
        mean_abs_threshold=args.correctness_threshold_mean_abs,
    )

    best_native_view = build_best_native_view(
        jax_doc=jax_native,
        pytorch_doc=pyt_native,
        guardrail_ratio=args.guardrail_ratio,
        max_abs_threshold=args.correctness_threshold_max_abs,
        mean_abs_threshold=args.correctness_threshold_mean_abs,
    )

    recommendations = {
        "portable": describe_winner(portable_view["winner"], "Portable view"),
        "best_native": describe_winner(best_native_view["winner"], "Best-native view"),
        "overall": describe_overall(portable_view, best_native_view),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "portable_view": portable_view,
        "best_native_view": best_native_view,
        "correctness": {
            "thresholds": {
                "max_abs_diff": args.correctness_threshold_max_abs,
                "mean_abs_diff": args.correctness_threshold_mean_abs,
            }
        },
        "recommendations": recommendations,
        "notes": [
            "PyTorch baseline is treated as frozen reference implementation for this comparison.",
            "Rows are eligible only with 100% success and correctness gate pass.",
        ],
        "inputs": {
            "jax_portable_json": args.jax_portable_json,
            "pytorch_portable_json": args.pytorch_portable_json,
            "jax_best_native_json": args.jax_best_native_json,
            "pytorch_best_native_json": args.pytorch_best_native_json,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {output_json}")

    if args.output_markdown:
        output_md = Path(args.output_markdown)
        write_markdown(output_md, payload)
        print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
