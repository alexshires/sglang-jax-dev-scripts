#!/usr/bin/env python3
"""Run client chunk-size matrix benchmarks against a PyTorch /v1/score endpoint."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests
except ImportError:
    requests = None


SCHEMA_VERSION = "score_matrix_v1"
DEFAULT_BASE_URL = "http://127.0.0.1:30000"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_int_csv(value: str) -> List[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * (p / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return vals[lower]
    frac = rank - lower
    return vals[lower] + (vals[upper] - vals[lower]) * frac


def parse_chunk_sizes(value: str) -> List[int]:
    sizes = parse_int_csv(value)
    if any(x <= 0 for x in sizes):
        raise ValueError("All chunk sizes must be > 0")
    return sorted(set(sizes))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    return output.strip() or None


def detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    nvidia = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        info["nvidia_smi"] = [line.strip() for line in nvidia.splitlines() if line.strip()]

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        info["cuda_visible_devices"] = cuda_visible_devices

    return info


def chunked(items: List[List[int]], chunk_size: int) -> Iterable[List[List[int]]]:
    if chunk_size >= len(items):
        yield items
        return
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def score_once(
    *,
    base_url: str,
    model: str,
    query_token_ids: List[int],
    items_token_ids: List[List[int]],
    label_token_ids: List[int],
    apply_softmax: bool,
    item_first: bool,
    client_chunk_size: int,
    timeout_sec: float,
) -> Dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/v1/score"

    all_scores: List[List[float]] = []
    call_latencies_ms: List[float] = []

    start = time.perf_counter()

    for chunk in chunked(items_token_ids, client_chunk_size):
        payload = {
            "model": model,
            "query": query_token_ids,
            "items": chunk,
            "label_token_ids": label_token_ids,
            "apply_softmax": apply_softmax,
            "item_first": item_first,
        }

        call_start = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout_sec)
        except requests.Timeout:
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "timeout",
                "failure_detail": f"timeout at chunk size {len(chunk)}",
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }
        except requests.RequestException as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "http_error",
                "failure_detail": str(e),
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }

        call_latencies_ms.append((time.perf_counter() - call_start) * 1000.0)

        if resp.status_code != 200:
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "http_error",
                "failure_detail": f"status={resp.status_code} body={resp.text[:500]}",
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }

        try:
            body = resp.json()
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "bad_json",
                "failure_detail": str(e),
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }

        chunk_scores = body.get("scores")
        if not isinstance(chunk_scores, list):
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "bad_response",
                "failure_detail": f"Missing scores field: {body}",
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }

        if len(chunk_scores) != len(chunk):
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "failure_reason": "bad_response",
                "failure_detail": (
                    f"Score length mismatch: got {len(chunk_scores)} expected {len(chunk)}"
                ),
                "total_e2e_ms": elapsed,
                "call_latencies_ms": call_latencies_ms,
            }

        all_scores.extend(chunk_scores)

    elapsed = (time.perf_counter() - start) * 1000.0

    if len(all_scores) != len(items_token_ids):
        return {
            "ok": False,
            "failure_reason": "bad_response",
            "failure_detail": (
                f"Aggregated score length mismatch: got {len(all_scores)} "
                f"expected {len(items_token_ids)}"
            ),
            "total_e2e_ms": elapsed,
            "call_latencies_ms": call_latencies_ms,
        }

    return {
        "ok": True,
        "failure_reason": None,
        "failure_detail": None,
        "total_e2e_ms": elapsed,
        "call_latencies_ms": call_latencies_ms,
        "scores": all_scores,
    }


def summarize_runs(
    *,
    chunk_size: int,
    num_items: int,
    timed_runs: int,
    run_records: List[Dict[str, Any]],
    reference_scores: Optional[List[List[float]]],
    cost_per_hour: Optional[float],
) -> Dict[str, Any]:
    success_records = [r for r in run_records if r["ok"]]
    total_ms_values = [float(r["total_e2e_ms"]) for r in success_records]

    flattened_call_ms: List[float] = []
    for r in success_records:
        flattened_call_ms.extend(float(x) for x in r["call_latencies_ms"])

    mean_total_ms = statistics.mean(total_ms_values) if total_ms_values else None
    p50_total_ms = percentile(total_ms_values, 50.0)
    p95_total_ms = percentile(total_ms_values, 95.0)

    mean_chunk_call_ms = (
        statistics.mean(flattened_call_ms) if flattened_call_ms else None
    )
    p95_chunk_call_ms = percentile(flattened_call_ms, 95.0)

    throughput_items_sec = None
    if mean_total_ms and mean_total_ms > 0:
        throughput_items_sec = num_items / (mean_total_ms / 1000.0)

    first_run_penalty_ratio = None
    if len(total_ms_values) >= 2:
        steady_state = total_ms_values[1:]
        steady_mean = statistics.mean(steady_state)
        if steady_mean > 0:
            first_run_penalty_ratio = total_ms_values[0] / steady_mean

    failure_reasons = Counter(
        r["failure_reason"] for r in run_records if not r["ok"] and r["failure_reason"]
    )

    success_rate = len(success_records) / timed_runs if timed_runs > 0 else 0.0
    if len(success_records) == timed_runs:
        status = "ok"
    elif len(success_records) == 0:
        status = "failed"
    else:
        status = "partial_failure"

    usd_per_run = None
    items_per_usd = None
    if cost_per_hour is not None and mean_total_ms is not None:
        usd_per_run = (mean_total_ms / 1000.0 / 3600.0) * cost_per_hour
        if usd_per_run > 0:
            items_per_usd = num_items / usd_per_run

    return {
        "client_chunk_size": chunk_size,
        "num_chunks": int(math.ceil(num_items / chunk_size)),
        "timed_runs": timed_runs,
        "successful_runs": len(success_records),
        "success_rate": success_rate,
        "status": status,
        "p50_total_e2e_ms": p50_total_ms,
        "p95_total_e2e_ms": p95_total_ms,
        "mean_total_e2e_ms": mean_total_ms,
        "mean_chunk_call_ms": mean_chunk_call_ms,
        "p95_chunk_call_ms": p95_chunk_call_ms,
        "throughput_items_sec": throughput_items_sec,
        "first_run_penalty_ratio": first_run_penalty_ratio,
        "failure_reason_counts": dict(failure_reasons),
        "usd_per_run": usd_per_run,
        "items_per_usd": items_per_usd,
        "reference_scores": reference_scores,
        "run_records": run_records,
    }


def choose_best_result(rows: List[Dict[str, Any]], guardrail_ratio: float) -> Dict[str, Any]:
    eligible = [
        r
        for r in rows
        if r.get("success_rate") == 1.0
        and r.get("p95_total_e2e_ms") is not None
        and r.get("throughput_items_sec") is not None
    ]

    if not eligible:
        return {
            "eligible_count": 0,
            "guardrail_ratio": guardrail_ratio,
            "guardrail_p95_ms": None,
            "selected": None,
            "notes": ["No eligible rows with 100% success and valid latency/throughput."],
        }

    min_p95 = min(float(r["p95_total_e2e_ms"]) for r in eligible)
    guardrail_p95 = min_p95 * guardrail_ratio
    within_guardrail = [
        r for r in eligible if float(r["p95_total_e2e_ms"]) <= guardrail_p95
    ]

    selected = sorted(
        within_guardrail,
        key=lambda r: (-float(r["throughput_items_sec"]), int(r["client_chunk_size"])),
    )[0]

    return {
        "eligible_count": len(eligible),
        "guardrail_ratio": guardrail_ratio,
        "guardrail_p95_ms": guardrail_p95,
        "selected": {
            "client_chunk_size": selected["client_chunk_size"],
            "throughput_items_sec": selected["throughput_items_sec"],
            "p95_total_e2e_ms": selected["p95_total_e2e_ms"],
            "success_rate": selected["success_rate"],
        },
        "notes": [],
    }


def pick_confirm_chunk_sizes(rows: List[Dict[str, Any]], sweep_choice: Dict[str, Any]) -> List[int]:
    successful_sizes = sorted(
        int(r["client_chunk_size"]) for r in rows if r.get("success_rate") == 1.0
    )
    if not successful_sizes:
        return [1]

    picks = {1}
    selected = (sweep_choice.get("selected") or {}).get("client_chunk_size")
    if selected is None:
        picks.add(successful_sizes[0])
        return sorted(picks)

    selected = int(selected)
    picks.add(selected)

    idx = successful_sizes.index(selected)
    if idx > 0:
        picks.add(successful_sizes[idx - 1])
    if idx < len(successful_sizes) - 1:
        picks.add(successful_sizes[idx + 1])

    return sorted(picks)


def run_phase(
    *,
    phase_name: str,
    base_url: str,
    model: str,
    query_token_ids: List[int],
    items_token_ids: List[List[int]],
    label_token_ids: List[int],
    apply_softmax: bool,
    item_first: bool,
    chunk_sizes: List[int],
    warmup_runs: int,
    timed_runs: int,
    timeout_sec: float,
    cost_per_hour: Optional[float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    num_items = len(items_token_ids)

    for chunk_size in chunk_sizes:
        print(f"[{phase_name}] chunk={chunk_size}: warmup={warmup_runs}, timed={timed_runs}")

        for _ in range(warmup_runs):
            _ = score_once(
                base_url=base_url,
                model=model,
                query_token_ids=query_token_ids,
                items_token_ids=items_token_ids,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                client_chunk_size=chunk_size,
                timeout_sec=timeout_sec,
            )

        run_records: List[Dict[str, Any]] = []
        reference_scores: Optional[List[List[float]]] = None

        for run_idx in range(timed_runs):
            response = score_once(
                base_url=base_url,
                model=model,
                query_token_ids=query_token_ids,
                items_token_ids=items_token_ids,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                client_chunk_size=chunk_size,
                timeout_sec=timeout_sec,
            )

            run_record = {
                "run_index": run_idx,
                "ok": response["ok"],
                "total_e2e_ms": response["total_e2e_ms"],
                "call_latencies_ms": response.get("call_latencies_ms", []),
                "failure_reason": response.get("failure_reason"),
                "failure_detail": response.get("failure_detail"),
            }
            run_records.append(run_record)

            if response["ok"] and reference_scores is None:
                reference_scores = response.get("scores")

        summary = summarize_runs(
            chunk_size=chunk_size,
            num_items=num_items,
            timed_runs=timed_runs,
            run_records=run_records,
            reference_scores=reference_scores,
            cost_per_hour=cost_per_hour,
        )
        results.append(summary)

    return results


def write_markdown_summary(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# PyTorch Score Matrix Results")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at_utc']}")
    lines.append(f"- Backend: {payload['backend']}")
    lines.append(f"- View: {payload['evaluation_view']}")
    lines.append(f"- Workload: {payload['workload']['model']}, {payload['workload']['num_items']} items")
    lines.append("")

    lines.append("## Sweep Results")
    lines.append("")
    lines.append("| Chunk | Status | Success | P95 E2E (ms) | Mean E2E (ms) | Throughput (items/s) |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in payload["results"]:
        lines.append(
            "| {chunk} | {status} | {success:.2f} | {p95} | {mean} | {thr} |".format(
                chunk=row["client_chunk_size"],
                status=row["status"],
                success=row["success_rate"],
                p95=(f"{row['p95_total_e2e_ms']:.2f}" if row["p95_total_e2e_ms"] is not None else "-"),
                mean=(f"{row['mean_total_e2e_ms']:.2f}" if row["mean_total_e2e_ms"] is not None else "-"),
                thr=(f"{row['throughput_items_sec']:.2f}" if row["throughput_items_sec"] is not None else "-"),
            )
        )

    lines.append("")
    lines.append("## Final Recommendation")
    lines.append("")
    final_selected = payload["summary"].get("final_recommendation", {}).get("selected")
    if final_selected:
        lines.append(
            "- Selected chunk size: `{}` (throughput `{:.2f}` items/s, p95 `{:.2f}` ms)".format(
                final_selected["client_chunk_size"],
                final_selected["throughput_items_sec"],
                final_selected["p95_total_e2e_ms"],
            )
        )
    else:
        lines.append("- No eligible chunk size found.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_workload(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    requests_field = obj.get("requests")
    if not isinstance(requests_field, list) or not requests_field:
        raise ValueError("Workload JSON must include non-empty requests[]")
    req = requests_field[0]

    for key in ["query_token_ids", "items_token_ids", "label_token_ids"]:
        if key not in req:
            raise ValueError(f"Missing request field: {key}")

    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--client-chunk-sizes", default="1,2,4,8,16,32,64,128,256,500")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--timed-runs-confirm", type=int, default=7)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--evaluation-view", choices=["portable", "best_native"], default="portable")
    parser.add_argument("--guardrail-ratio", type=float, default=1.25)
    parser.add_argument("--cost-per-hour", type=float)
    parser.add_argument("--server-config-note", default="")
    parser.add_argument("--delimiter-token-id", type=int, default=151643)
    parser.add_argument("--correctness-threshold-max-abs", type=float, default=0.02)
    parser.add_argument("--correctness-threshold-mean-abs", type=float, default=0.01)
    args = parser.parse_args()

    if requests is None:
        print(
            "Missing dependency: requests. Install with `pip install requests` "
            "in the environment used for benchmark execution.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    chunk_sizes = parse_chunk_sizes(args.client_chunk_sizes)
    workload_path = Path(args.workload_json)
    workload = load_workload(workload_path)
    req = workload["requests"][0]

    query_token_ids = req["query_token_ids"]
    items_token_ids = req["items_token_ids"]
    label_token_ids = req["label_token_ids"]
    model = workload["model"]
    apply_softmax = bool(req.get("apply_softmax", True))
    item_first = bool(req.get("item_first", False))

    sweep_results = run_phase(
        phase_name="sweep",
        base_url=args.base_url,
        model=model,
        query_token_ids=query_token_ids,
        items_token_ids=items_token_ids,
        label_token_ids=label_token_ids,
        apply_softmax=apply_softmax,
        item_first=item_first,
        chunk_sizes=chunk_sizes,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        timeout_sec=args.timeout_sec,
        cost_per_hour=args.cost_per_hour,
    )

    sweep_recommendation = choose_best_result(sweep_results, args.guardrail_ratio)
    confirm_chunk_sizes = pick_confirm_chunk_sizes(sweep_results, sweep_recommendation)

    confirm_results = run_phase(
        phase_name="confirm",
        base_url=args.base_url,
        model=model,
        query_token_ids=query_token_ids,
        items_token_ids=items_token_ids,
        label_token_ids=label_token_ids,
        apply_softmax=apply_softmax,
        item_first=item_first,
        chunk_sizes=confirm_chunk_sizes,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs_confirm,
        timeout_sec=args.timeout_sec,
        cost_per_hour=args.cost_per_hour,
    )

    confirm_recommendation = choose_best_result(confirm_results, args.guardrail_ratio)
    final_recommendation = (
        confirm_recommendation
        if confirm_recommendation.get("selected") is not None
        else sweep_recommendation
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "backend": "pytorch",
        "evaluation_view": args.evaluation_view,
        "hardware": detect_hardware(),
        "server_config": {
            "multi_item_scoring_delimiter": args.delimiter_token_id,
            "chunked_prefill_size": -1,
            "disable_radix_cache": True,
            "note": args.server_config_note,
        },
        "workload_ref": {
            "path": str(workload_path),
            "sha256": sha256_file(workload_path),
            "request_id": req.get("request_id", "unknown"),
        },
        "workload": {
            "model": model,
            "query_tokens": len(query_token_ids),
            "num_items": len(items_token_ids),
            "item_tokens": len(items_token_ids[0]) if items_token_ids else 0,
            "label_token_ids": label_token_ids,
        },
        "results": sweep_results,
        "confirm_results": confirm_results,
        "summary": {
            "guardrail_ratio": args.guardrail_ratio,
            "sweep_recommendation": sweep_recommendation,
            "confirm_chunk_sizes": confirm_chunk_sizes,
            "confirm_recommendation": confirm_recommendation,
            "final_recommendation": final_recommendation,
            "correctness_thresholds": {
                "max_abs_diff": args.correctness_threshold_max_abs,
                "mean_abs_diff": args.correctness_threshold_mean_abs,
            },
        },
        "notes": [
            "Correctness gate is applied in compare_score_matrix_results.py against PyTorch baseline.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {output_json}")

    if args.output_markdown:
        output_md = Path(args.output_markdown)
        write_markdown_summary(output_md, payload)
        print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
