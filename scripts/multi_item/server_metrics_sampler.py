#!/usr/bin/env python3
"""Sample server process and HTTP metrics over time for soak analysis."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


def unix_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_duration(text: str) -> float:
    raw = text.strip().lower()
    if not raw:
        raise ValueError("Duration cannot be empty.")

    unit = raw[-1]
    if unit in {"s", "m", "h"}:
        value = float(raw[:-1])
    else:
        value = float(raw)
        unit = "s"

    if value <= 0:
        raise ValueError("Duration must be positive.")

    factor = {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return value * factor


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return numerator / denominator


def parse_prometheus_text(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        value_raw = parts[-1]

        # Keep only non-labeled numeric series to keep output compact and stable.
        if "{" in name:
            continue
        try:
            value = float(value_raw)
        except ValueError:
            continue
        metrics[name] = value
    return metrics


def http_get(url: str, timeout_s: float) -> tuple[int | None, str, str]:
    req = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            status = int(response.status)
            body = response.read().decode("utf-8", errors="replace")
            return status, body, ""
    except urllib_error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace"), ""
    except urllib_error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.timeout):
            return None, "", "timeout"
        return None, "", str(reason)


def resolve_pid(args: argparse.Namespace) -> int | None:
    if args.pid is not None:
        return int(args.pid)

    if args.pid_file is not None:
        pid_file = Path(args.pid_file)
        if pid_file.exists():
            content = pid_file.read_text(encoding="utf-8").strip()
            if content:
                return int(content)

    if args.pid_pattern:
        try:
            output = subprocess.check_output(
                ["pgrep", "-f", args.pid_pattern],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            return None

        if not output:
            return None

        pids = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if not pids:
            return None
        return max(pids)

    return None


def get_process_mem(pid: int) -> tuple[float | None, float | None, bool]:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-o", "vsz=", "-p", str(pid)],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None, None, False

    if not output:
        return None, None, False

    parts = output.split()
    if len(parts) < 2:
        return None, None, False

    try:
        rss_kb = float(parts[0])
        vsz_kb = float(parts[1])
    except ValueError:
        return None, None, False

    return rss_kb / 1024.0, vsz_kb / 1024.0, True


def collect_server_info(base_url: str, timeout_s: float) -> dict[str, Any]:
    info = {
        "server_info_ok": False,
        "server_info_status": None,
        "waiting_queue_size": None,
        "running_batch_size": None,
        "prefill_decode_size": None,
        "req_to_token_pool_used": None,
        "available_kv_tokens": None,
        "memory_kvcache": None,
        "internal_state_count": None,
        "server_info_error": "",
    }

    try:
        status, body_text, err = http_get(f"{base_url}/get_server_info", timeout_s=timeout_s)
        info["server_info_status"] = status
        if status != 200:
            info["server_info_error"] = err or body_text[:300]
            return info

        body = json.loads(body_text)
        states = body.get("internal_states")
        if not isinstance(states, list):
            info["server_info_error"] = "missing internal_states"
            return info

        waiting_total = 0
        running_total = 0
        prefill_decode_total = 0
        req_pool_used_total = 0
        available_kv_tokens_total = 0
        memory_kvcache_total = 0.0

        for state in states:
            if not isinstance(state, dict):
                continue
            waiting_total += int(state.get("waiting_queue_size") or 0)
            running_total += int(state.get("running_batch_size") or 0)
            prefill_decode_total += int(state.get("prefill_decode_size") or 0)
            req_pool_used_total += int(state.get("req_to_token_pool_used") or 0)
            available_kv_tokens_total += int(state.get("available_kv_tokens") or 0)

            memory_usage = state.get("memory_usage") or {}
            if isinstance(memory_usage, dict):
                memory_kvcache_total += float(memory_usage.get("kvcache") or 0.0)

        info.update(
            {
                "server_info_ok": True,
                "waiting_queue_size": waiting_total,
                "running_batch_size": running_total,
                "prefill_decode_size": prefill_decode_total,
                "req_to_token_pool_used": req_pool_used_total,
                "available_kv_tokens": available_kv_tokens_total,
                "memory_kvcache": memory_kvcache_total,
                "internal_state_count": len(states),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for long-running sampling
        info["server_info_error"] = str(exc)[:300]

    return info


def collect_metrics_endpoint(base_url: str, timeout_s: float) -> dict[str, Any]:
    out = {
        "metrics_ok": False,
        "metrics_status": None,
        "metrics_error": "",
        "metrics_sample": {},
    }

    try:
        status, body_text, err = http_get(f"{base_url}/metrics", timeout_s=timeout_s)
        out["metrics_status"] = status
        if status != 200:
            out["metrics_error"] = err or body_text[:300]
            return out

        out["metrics_ok"] = True
        out["metrics_sample"] = parse_prometheus_text(body_text)
    except Exception as exc:  # pragma: no cover
        out["metrics_error"] = str(exc)[:300]

    return out


@dataclass
class Sample:
    timestamp_unix: float
    elapsed_sec: float
    pid: int | None
    pid_alive: bool | None
    rss_mb: float | None
    vsz_mb: float | None
    waiting_queue_size: int | None
    running_batch_size: int | None
    prefill_decode_size: int | None
    req_to_token_pool_used: int | None
    available_kv_tokens: int | None
    memory_kvcache: float | None
    internal_state_count: int | None
    server_info_ok: bool
    server_info_status: int | None
    server_info_error: str
    metrics_ok: bool
    metrics_status: int | None
    metrics_error: str
    metrics_sample: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        return {
            "timestamp_unix": round(self.timestamp_unix, 6),
            "timestamp_iso": unix_to_iso(self.timestamp_unix),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "pid": self.pid,
            "pid_alive": self.pid_alive,
            "rss_mb": None if self.rss_mb is None else round(self.rss_mb, 3),
            "vsz_mb": None if self.vsz_mb is None else round(self.vsz_mb, 3),
            "waiting_queue_size": self.waiting_queue_size,
            "running_batch_size": self.running_batch_size,
            "prefill_decode_size": self.prefill_decode_size,
            "req_to_token_pool_used": self.req_to_token_pool_used,
            "available_kv_tokens": self.available_kv_tokens,
            "memory_kvcache": self.memory_kvcache,
            "internal_state_count": self.internal_state_count,
            "server_info_ok": self.server_info_ok,
            "server_info_status": self.server_info_status,
            "server_info_error": self.server_info_error,
            "metrics_ok": self.metrics_ok,
            "metrics_status": self.metrics_status,
            "metrics_error": self.metrics_error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--duration", required=True, help="Examples: 10m, 2h, 14400s")
    parser.add_argument("--timeout", type=float, default=5.0)

    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--pid-pattern", default="python -m sgl_jax.launch_server")
    parser.add_argument(
        "--refresh-pid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-resolve pid each sample when current pid is missing/dead.",
    )
    parser.add_argument(
        "--stop-on-dead-pid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--sample-server-info",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sample-metrics-endpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--warmup-minutes", type=float, default=15.0)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default=None)

    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.warmup_minutes < 0:
        raise ValueError("--warmup-minutes must be >= 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")

    args.duration_seconds = parse_duration(args.duration)
    return args


def summarize(samples: list[Sample], args: argparse.Namespace) -> dict[str, Any]:
    rss_points = [(s.elapsed_sec, s.rss_mb) for s in samples if s.rss_mb is not None]
    vsz_points = [(s.elapsed_sec, s.vsz_mb) for s in samples if s.vsz_mb is not None]

    def summarize_series(points: list[tuple[float, float]], warmup_sec: float) -> dict[str, Any]:
        if not points:
            return {
                "samples": 0,
                "start_mb": None,
                "end_mb": None,
                "growth_mb": None,
                "growth_pct": None,
                "growth_mb_per_hour": None,
                "trend_mb_per_hour_after_warmup": None,
            }

        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        elapsed = max(1e-9, end_x - start_x)
        growth_mb = end_y - start_y
        growth_pct = (growth_mb / start_y * 100.0) if start_y > 0 else None
        growth_mb_per_hour = growth_mb / elapsed * 3600.0

        post_warmup = [(x, y) for x, y in points if x >= warmup_sec]
        if len(post_warmup) < 2:
            trend_mb_per_hour = None
        else:
            xs = [x for x, _ in post_warmup]
            ys = [y for _, y in post_warmup]
            slope = linear_slope(xs, ys)
            trend_mb_per_hour = None if slope is None else slope * 3600.0

        return {
            "samples": len(points),
            "start_mb": start_y,
            "end_mb": end_y,
            "growth_mb": growth_mb,
            "growth_pct": growth_pct,
            "growth_mb_per_hour": growth_mb_per_hour,
            "trend_mb_per_hour_after_warmup": trend_mb_per_hour,
        }

    warmup_sec = float(args.warmup_minutes) * 60.0

    server_info_ok_count = sum(1 for s in samples if s.server_info_ok)
    metrics_ok_count = sum(1 for s in samples if s.metrics_ok)

    waiting_values = [s.waiting_queue_size for s in samples if s.waiting_queue_size is not None]
    running_values = [s.running_batch_size for s in samples if s.running_batch_size is not None]

    unique_metric_names: set[str] = set()
    for sample in samples:
        unique_metric_names.update(sample.metrics_sample.keys())

    return {
        "config": {
            "base_url": args.base_url,
            "duration": args.duration,
            "duration_seconds": args.duration_seconds,
            "interval_seconds": args.interval,
            "timeout_seconds": args.timeout,
            "warmup_minutes": args.warmup_minutes,
            "sample_server_info": args.sample_server_info,
            "sample_metrics_endpoint": args.sample_metrics_endpoint,
            "stop_on_dead_pid": args.stop_on_dead_pid,
            "pid_pattern": args.pid_pattern,
        },
        "counts": {
            "samples": len(samples),
            "server_info_ok_samples": server_info_ok_count,
            "metrics_ok_samples": metrics_ok_count,
        },
        "memory": {
            "rss": summarize_series(rss_points, warmup_sec),
            "vsz": summarize_series(vsz_points, warmup_sec),
        },
        "queue": {
            "waiting_queue_max": max(waiting_values) if waiting_values else None,
            "waiting_queue_mean": (
                sum(waiting_values) / len(waiting_values) if waiting_values else None
            ),
            "running_batch_max": max(running_values) if running_values else None,
            "running_batch_mean": (
                sum(running_values) / len(running_values) if running_values else None
            ),
        },
        "metrics_endpoint": {
            "unique_metric_names": sorted(unique_metric_names),
        },
    }


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.output_prefix or f"server_metrics_{run_id}"

    csv_path = output_dir / f"{prefix}_samples.csv"
    jsonl_path = output_dir / f"{prefix}_samples.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"

    pid = resolve_pid(args)

    samples: list[Sample] = []
    started_unix = time.time()
    started_perf = time.perf_counter()

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp_unix",
                "timestamp_iso",
                "elapsed_sec",
                "pid",
                "pid_alive",
                "rss_mb",
                "vsz_mb",
                "waiting_queue_size",
                "running_batch_size",
                "prefill_decode_size",
                "req_to_token_pool_used",
                "available_kv_tokens",
                "memory_kvcache",
                "internal_state_count",
                "server_info_ok",
                "server_info_status",
                "server_info_error",
                "metrics_ok",
                "metrics_status",
                "metrics_error",
            ],
        )
        writer.writeheader()

        sample_idx = 0
        next_target = 0.0

        while True:
            now_perf = time.perf_counter()
            elapsed = now_perf - started_perf
            if elapsed >= args.duration_seconds:
                break
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break

            if (pid is None or pid <= 0) and args.refresh_pid:
                pid = resolve_pid(args)

            rss_mb = None
            vsz_mb = None
            pid_alive = None
            if pid is not None:
                rss_mb, vsz_mb, pid_alive = get_process_mem(pid)
                if not pid_alive and args.refresh_pid:
                    pid = resolve_pid(args)
                    if pid is not None:
                        rss_mb, vsz_mb, pid_alive = get_process_mem(pid)

            server_info = {
                "server_info_ok": False,
                "server_info_status": None,
                "waiting_queue_size": None,
                "running_batch_size": None,
                "prefill_decode_size": None,
                "req_to_token_pool_used": None,
                "available_kv_tokens": None,
                "memory_kvcache": None,
                "internal_state_count": None,
                "server_info_error": "",
            }
            if args.sample_server_info:
                server_info = collect_server_info(args.base_url, timeout_s=float(args.timeout))

            metrics = {
                "metrics_ok": False,
                "metrics_status": None,
                "metrics_error": "",
                "metrics_sample": {},
            }
            if args.sample_metrics_endpoint:
                metrics = collect_metrics_endpoint(args.base_url, timeout_s=float(args.timeout))

            ts_unix = time.time()
            sample = Sample(
                timestamp_unix=ts_unix,
                elapsed_sec=elapsed,
                pid=pid,
                pid_alive=pid_alive,
                rss_mb=rss_mb,
                vsz_mb=vsz_mb,
                waiting_queue_size=server_info["waiting_queue_size"],
                running_batch_size=server_info["running_batch_size"],
                prefill_decode_size=server_info["prefill_decode_size"],
                req_to_token_pool_used=server_info["req_to_token_pool_used"],
                available_kv_tokens=server_info["available_kv_tokens"],
                memory_kvcache=server_info["memory_kvcache"],
                internal_state_count=server_info["internal_state_count"],
                server_info_ok=bool(server_info["server_info_ok"]),
                server_info_status=server_info["server_info_status"],
                server_info_error=server_info["server_info_error"],
                metrics_ok=bool(metrics["metrics_ok"]),
                metrics_status=metrics["metrics_status"],
                metrics_error=metrics["metrics_error"],
                metrics_sample=metrics["metrics_sample"],
            )
            samples.append(sample)

            row = sample.to_row()
            writer.writerow(row)
            csv_file.flush()
            jsonl_file.write(
                json.dumps(
                    {
                        **row,
                        "metrics_sample": sample.metrics_sample,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            jsonl_file.flush()

            sample_idx += 1

            if args.stop_on_dead_pid and pid is not None and pid_alive is False:
                break

            next_target += args.interval
            sleep_s = next_target - (time.perf_counter() - started_perf)
            if sleep_s > 0:
                time.sleep(sleep_s)

    summary = summarize(samples, args)
    ended_unix = time.time()
    summary["run"] = {
        "started_at_unix": started_unix,
        "ended_at_unix": ended_unix,
        "started_at_iso": unix_to_iso(started_unix),
        "ended_at_iso": unix_to_iso(ended_unix),
        "actual_runtime_seconds": max(0.0, ended_unix - started_unix),
        "resolved_pid": pid,
    }
    summary["artifacts"] = {
        "samples_csv": str(csv_path),
        "samples_jsonl": str(jsonl_path),
        "summary_json": str(summary_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "samples_csv": str(csv_path),
                "samples_jsonl": str(jsonl_path),
                "sample_count": len(samples),
                "resolved_pid": pid,
                "rss_growth_pct": summary["memory"]["rss"]["growth_pct"],
                "rss_trend_mb_per_hour": summary["memory"]["rss"][
                    "trend_mb_per_hour_after_warmup"
                ],
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
