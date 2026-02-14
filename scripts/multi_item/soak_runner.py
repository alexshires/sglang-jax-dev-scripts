#!/usr/bin/env python3
"""Run sustained /v1/score load with weighted multi-item request mixes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import socket
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_LABEL_TOKEN_IDS = "9834,902"


@dataclass(frozen=True)
class Scenario:
    name: str
    item_count: int
    weight: float


@dataclass
class RequestResult:
    request_id: int
    scenario: str
    expected_item_count: int
    status: str
    error_reason: str
    latency_ms: float
    started_at_unix: float
    ended_at_unix: float
    http_status: int | None
    response_item_count: int | None
    response_label_count: int | None
    timeout_s: float
    error_excerpt: str

    def to_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "expected_item_count": self.expected_item_count,
            "status": self.status,
            "error_reason": self.error_reason,
            "latency_ms": round(self.latency_ms, 3),
            "started_at_unix": round(self.started_at_unix, 6),
            "ended_at_unix": round(self.ended_at_unix, 6),
            "started_at_iso": unix_to_iso(self.started_at_unix),
            "ended_at_iso": unix_to_iso(self.ended_at_unix),
            "http_status": self.http_status,
            "response_item_count": self.response_item_count,
            "response_label_count": self.response_label_count,
            "timeout_s": self.timeout_s,
            "error_excerpt": self.error_excerpt,
        }


def unix_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def parse_duration(text: str) -> float:
    raw = text.strip().lower()
    if not raw:
        raise ValueError("Duration cannot be empty.")

    unit = raw[-1]
    if unit in {"s", "m", "h"}:
        num = float(raw[:-1])
    else:
        num = float(raw)
        unit = "s"

    if num <= 0:
        raise ValueError("Duration must be positive.")

    multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return num * multiplier


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_mix(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid mix token '{part}'. Use format: small=0.2,medium=0.5,large=0.3"
            )
        key, value = part.split("=", 1)
        name = key.strip().lower()
        out[name] = float(value.strip())

    if not out:
        raise ValueError("Mix must include at least one scenario weight.")
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    w = rank - low
    return sorted_values[low] * (1.0 - w) + sorted_values[high] * w


def build_query_tokens(num_tokens: int, seed: int) -> list[int]:
    # Deterministic synthetic query IDs with enough variation to avoid all-constant inputs.
    return [100 + ((seed + (idx * 17)) % 4000) for idx in range(num_tokens)]


def build_item_tokens(item_count: int, tokens_per_item: int, seed: int) -> list[list[int]]:
    items: list[list[int]] = []
    for item_idx in range(item_count):
        base = 500 + ((seed + item_idx * 31) % 5000)
        item = [base + ((tok_idx * 13) % 101) for tok_idx in range(tokens_per_item)]
        items.append(item)
    return items


def first_label_count(scores: Any) -> int | None:
    if not isinstance(scores, list) or not scores:
        return None
    first = scores[0]
    if isinstance(first, list):
        return len(first)
    return None


def http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except urllib_error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.timeout):
            raise TimeoutError("HTTP request timed out") from exc
        raise ConnectionError(str(reason)) from exc


class SoakRunner:
    def __init__(self, args: argparse.Namespace, scenarios: list[Scenario]):
        self.args = args
        self.scenarios = scenarios
        self.seed = int(args.seed)
        self.rng = random.Random(self.seed)
        self.results: list[RequestResult] = []
        self._result_lock = asyncio.Lock()

        self.payload_by_scenario: dict[str, dict[str, Any]] = {}
        query_tokens = build_query_tokens(args.query_tokens, self.seed)
        for idx, scenario in enumerate(scenarios):
            items = build_item_tokens(
                item_count=scenario.item_count,
                tokens_per_item=args.tokens_per_item,
                seed=self.seed + idx * 1000,
            )
            self.payload_by_scenario[scenario.name] = {
                "model": args.model,
                "query": query_tokens,
                "items": items,
                "label_token_ids": args.label_token_ids,
                "apply_softmax": args.apply_softmax,
                "item_first": False,
            }

        self.total_duration_s = parse_duration(args.duration)
        self.arrival_rate = (
            float(args.arrival_rate)
            if args.arrival_rate is not None
            else max(0.01, float(args.concurrency))
        )
        self.max_requests = args.max_requests

        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_prefix = args.output_prefix or f"soak_{run_id}"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.requests_csv_path = output_dir / f"{output_prefix}_requests.csv"
        self.requests_jsonl_path = output_dir / f"{output_prefix}_requests.jsonl"
        self.summary_json_path = output_dir / f"{output_prefix}_summary.json"

        self._csv_file = self.requests_csv_path.open("w", encoding="utf-8", newline="")
        self._jsonl_file = self.requests_jsonl_path.open("w", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "request_id",
                "scenario",
                "expected_item_count",
                "status",
                "error_reason",
                "latency_ms",
                "started_at_unix",
                "ended_at_unix",
                "started_at_iso",
                "ended_at_iso",
                "http_status",
                "response_item_count",
                "response_label_count",
                "timeout_s",
                "error_excerpt",
            ],
        )
        self._csv_writer.writeheader()

    def close(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        finally:
            self._jsonl_file.flush()
            self._jsonl_file.close()

    def pick_scenario(self) -> Scenario:
        names = [s.name for s in self.scenarios]
        weights = [s.weight for s in self.scenarios]
        selected_name = self.rng.choices(names, weights=weights, k=1)[0]
        for scenario in self.scenarios:
            if scenario.name == selected_name:
                return scenario
        raise RuntimeError(f"Scenario selection failed for '{selected_name}'.")

    def _sync_issue_request(self, request_id: int, scenario: Scenario) -> RequestResult:
        payload = self.payload_by_scenario[scenario.name]
        start_unix = time.time()
        t0 = time.perf_counter()

        status = "error"
        error_reason = "unknown"
        error_excerpt = ""
        http_status: int | None = None
        response_item_count: int | None = None
        response_label_count: int | None = None

        try:
            http_status, response_text = http_post_json(
                self.args.url,
                payload=payload,
                timeout_s=float(self.args.timeout),
            )

            if http_status != 200:
                error_reason = "http_error"
                error_excerpt = response_text[:500]
            else:
                try:
                    body = json.loads(response_text)
                except ValueError:
                    error_reason = "invalid_json"
                    error_excerpt = response_text[:500]
                else:
                    scores = body.get("scores")
                    if not isinstance(scores, list):
                        error_reason = "missing_scores"
                        error_excerpt = json.dumps(body, ensure_ascii=True)[:500]
                    else:
                        response_item_count = len(scores)
                        response_label_count = first_label_count(scores)

                        if response_item_count != scenario.item_count:
                            error_reason = "response_count_mismatch"
                            error_excerpt = (
                                f"expected {scenario.item_count}, got {response_item_count}"
                            )
                        elif self.args.validate_score_vector_length and response_label_count != len(
                            self.args.label_token_ids
                        ):
                            error_reason = "label_count_mismatch"
                            error_excerpt = (
                                f"expected label length {len(self.args.label_token_ids)}, "
                                f"got {response_label_count}"
                            )
                        else:
                            status = "ok"
                            error_reason = ""
                            error_excerpt = ""
        except TimeoutError:
            error_reason = "timeout"
        except ConnectionError as exc:
            error_reason = "connection_error"
            error_excerpt = str(exc)[:500]
        except Exception as exc:  # pragma: no cover - broad capture for soak robustness
            error_reason = "exception"
            error_excerpt = str(exc)[:500]

        end_unix = time.time()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RequestResult(
            request_id=request_id,
            scenario=scenario.name,
            expected_item_count=scenario.item_count,
            status=status,
            error_reason=error_reason,
            latency_ms=latency_ms,
            started_at_unix=start_unix,
            ended_at_unix=end_unix,
            http_status=http_status,
            response_item_count=response_item_count,
            response_label_count=response_label_count,
            timeout_s=float(self.args.timeout),
            error_excerpt=error_excerpt,
        )

    async def _record_result(self, result: RequestResult) -> None:
        row = result.to_row()
        async with self._result_lock:
            self.results.append(result)
            self._csv_writer.writerow(row)
            self._csv_file.flush()
            self._jsonl_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._jsonl_file.flush()

    async def _run_one(self, request_id: int, scenario: Scenario) -> None:
        result = await asyncio.to_thread(self._sync_issue_request, request_id, scenario)
        await self._record_result(result)

    async def run(self) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(self.args.concurrency)
        tasks: set[asyncio.Task[None]] = set()

        started_perf = time.perf_counter()
        scheduled_requests = 0
        target_elapsed = 0.0

        async def launch_one(req_id: int, scenario: Scenario) -> None:
            await semaphore.acquire()

            async def wrapped() -> None:
                try:
                    await self._run_one(req_id, scenario)
                finally:
                    semaphore.release()

            task = asyncio.create_task(wrapped())
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        while True:
            now_elapsed = time.perf_counter() - started_perf
            if now_elapsed >= self.total_duration_s:
                break
            if self.max_requests is not None and scheduled_requests >= self.max_requests:
                break

            if scheduled_requests > 0:
                target_elapsed += self.rng.expovariate(self.arrival_rate)
                sleep_s = target_elapsed - now_elapsed
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            scenario = self.pick_scenario()
            await launch_one(scheduled_requests, scenario)
            scheduled_requests += 1

        if tasks:
            await asyncio.gather(*tasks)

        ended_perf = time.perf_counter()
        summary = self.build_summary(
            run_start_unix=time.time() - (ended_perf - started_perf),
            run_end_unix=time.time(),
            actual_runtime_s=ended_perf - started_perf,
            scheduled_requests=scheduled_requests,
        )

        with self.summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def build_summary(
        self,
        *,
        run_start_unix: float,
        run_end_unix: float,
        actual_runtime_s: float,
        scheduled_requests: int,
    ) -> dict[str, Any]:
        lat_all = [r.latency_ms for r in self.results]
        ok_results = [r for r in self.results if r.status == "ok"]
        lat_ok = [r.latency_ms for r in ok_results]

        total = len(self.results)
        ok = len(ok_results)
        err = total - ok

        ok_items = sum(r.expected_item_count for r in ok_results)
        throughput_items_per_sec = ok_items / actual_runtime_s if actual_runtime_s > 0 else 0.0
        throughput_reqs_per_sec = ok / actual_runtime_s if actual_runtime_s > 0 else 0.0

        reason_counts: dict[str, int] = {}
        for result in self.results:
            if result.status == "ok":
                continue
            reason_counts[result.error_reason] = reason_counts.get(result.error_reason, 0) + 1

        by_scenario: dict[str, dict[str, Any]] = {}
        for scenario in self.scenarios:
            subset = [r for r in self.results if r.scenario == scenario.name]
            subset_ok = [r for r in subset if r.status == "ok"]
            subset_lat_ok = [r.latency_ms for r in subset_ok]
            scenario_items = sum(r.expected_item_count for r in subset_ok)
            by_scenario[scenario.name] = {
                "configured_item_count": scenario.item_count,
                "configured_weight": scenario.weight,
                "requests_total": len(subset),
                "requests_ok": len(subset_ok),
                "requests_error": len(subset) - len(subset_ok),
                "error_rate": (
                    (len(subset) - len(subset_ok)) / len(subset) if subset else 0.0
                ),
                "items_ok": scenario_items,
                "latency_ms": {
                    "p50": percentile(subset_lat_ok, 50),
                    "p95": percentile(subset_lat_ok, 95),
                    "p99": percentile(subset_lat_ok, 99),
                    "mean": statistics.mean(subset_lat_ok) if subset_lat_ok else 0.0,
                    "min": min(subset_lat_ok) if subset_lat_ok else 0.0,
                    "max": max(subset_lat_ok) if subset_lat_ok else 0.0,
                },
            }

        completion_order_ok_lat = [r.latency_ms for r in self.results if r.status == "ok"]
        warmup_n = min(20, len(completion_order_ok_lat))
        steady_n = min(20, len(completion_order_ok_lat))
        first_run_penalty = None
        if warmup_n > 0 and steady_n > 0:
            first_mean = statistics.mean(completion_order_ok_lat[:warmup_n])
            steady_mean = statistics.mean(completion_order_ok_lat[-steady_n:])
            first_run_penalty = {
                "first_window_requests": warmup_n,
                "steady_window_requests": steady_n,
                "first_mean_latency_ms": first_mean,
                "steady_mean_latency_ms": steady_mean,
                "ratio_first_over_steady": first_mean / steady_mean if steady_mean > 0 else None,
            }

        return {
            "config": {
                "url": self.args.url,
                "model": self.args.model,
                "duration": self.args.duration,
                "duration_seconds": self.total_duration_s,
                "concurrency": self.args.concurrency,
                "arrival_rate_requests_per_sec": self.arrival_rate,
                "seed": self.seed,
                "query_tokens": self.args.query_tokens,
                "tokens_per_item": self.args.tokens_per_item,
                "label_token_ids": self.args.label_token_ids,
                "apply_softmax": self.args.apply_softmax,
                "validate_score_vector_length": self.args.validate_score_vector_length,
                "mix": {scenario.name: scenario.weight for scenario in self.scenarios},
                "max_requests": self.max_requests,
            },
            "artifacts": {
                "requests_csv": str(self.requests_csv_path),
                "requests_jsonl": str(self.requests_jsonl_path),
                "summary_json": str(self.summary_json_path),
            },
            "run": {
                "started_at_unix": run_start_unix,
                "ended_at_unix": run_end_unix,
                "started_at_iso": unix_to_iso(run_start_unix),
                "ended_at_iso": unix_to_iso(run_end_unix),
                "actual_runtime_seconds": actual_runtime_s,
                "scheduled_requests": scheduled_requests,
                "completed_requests": total,
            },
            "results": {
                "requests_total": total,
                "requests_ok": ok,
                "requests_error": err,
                "error_rate": (err / total) if total else 0.0,
                "ok_items_total": ok_items,
                "throughput_items_per_sec": throughput_items_per_sec,
                "throughput_requests_per_sec": throughput_reqs_per_sec,
                "latency_ms_all": {
                    "p50": percentile(lat_all, 50),
                    "p95": percentile(lat_all, 95),
                    "p99": percentile(lat_all, 99),
                    "mean": statistics.mean(lat_all) if lat_all else 0.0,
                    "min": min(lat_all) if lat_all else 0.0,
                    "max": max(lat_all) if lat_all else 0.0,
                },
                "latency_ms_ok_only": {
                    "p50": percentile(lat_ok, 50),
                    "p95": percentile(lat_ok, 95),
                    "p99": percentile(lat_ok, 99),
                    "mean": statistics.mean(lat_ok) if lat_ok else 0.0,
                    "min": min(lat_ok) if lat_ok else 0.0,
                    "max": max(lat_ok) if lat_ok else 0.0,
                },
                "error_reason_counts": reason_counts,
                "by_scenario": by_scenario,
                "first_run_penalty": first_run_penalty,
            },
        }


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    mix = parse_mix(args.mix)

    allowed = {
        "small": int(args.small_items),
        "medium": int(args.medium_items),
        "large": int(args.large_items),
    }

    unknown = sorted(set(mix) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown mix scenario(s): {', '.join(unknown)}")

    scenarios: list[Scenario] = []
    total_weight = 0.0
    for name, item_count in allowed.items():
        weight = float(mix.get(name, 0.0))
        if item_count <= 0:
            raise ValueError(f"Scenario '{name}' item count must be positive.")
        if weight < 0:
            raise ValueError(f"Scenario '{name}' weight must be >= 0.")
        if weight > 0:
            scenarios.append(Scenario(name=name, item_count=item_count, weight=weight))
            total_weight += weight

    if not scenarios:
        raise ValueError("At least one scenario must have non-zero weight.")

    normalized: list[Scenario] = []
    for scenario in scenarios:
        normalized.append(
            Scenario(
                name=scenario.name,
                item_count=scenario.item_count,
                weight=scenario.weight / total_weight,
            )
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/score")
    parser.add_argument("--model", default="/models/Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--duration",
        required=True,
        help="Target runtime. Examples: 600s, 10m, 2h",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson arrival rate in requests/sec. Defaults to --concurrency.",
    )
    parser.add_argument(
        "--mix",
        default="small=0.4,medium=0.4,large=0.2",
        help="Weighted mix over small/medium/large.",
    )
    parser.add_argument("--small-items", type=int, default=3)
    parser.add_argument("--medium-items", type=int, default=30)
    parser.add_argument("--large-items", type=int, default=500)
    parser.add_argument("--query-tokens", type=int, default=2000)
    parser.add_argument("--tokens-per-item", type=int, default=20)
    parser.add_argument("--label-token-ids", default=DEFAULT_LABEL_TOKEN_IDS)
    parser.add_argument(
        "--apply-softmax",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-score-vector-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260213)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-requests", type=int, default=None)

    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.arrival_rate is not None and args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be > 0")
    if args.query_tokens <= 0:
        raise ValueError("--query-tokens must be > 0")
    if args.tokens_per_item <= 0:
        raise ValueError("--tokens-per-item must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    args.label_token_ids = parse_int_list(args.label_token_ids)
    return args


async def async_main() -> int:
    args = parse_args()
    scenarios = build_scenarios(args)
    runner = SoakRunner(args, scenarios)

    try:
        summary = await runner.run()
    finally:
        runner.close()

    print(
        json.dumps(
            {
                "summary_json": str(runner.summary_json_path),
                "requests_csv": str(runner.requests_csv_path),
                "requests_jsonl": str(runner.requests_jsonl_path),
                "error_rate": summary["results"]["error_rate"],
                "throughput_items_per_sec": summary["results"]["throughput_items_per_sec"],
                "p99_latency_ms": summary["results"]["latency_ms_ok_only"]["p99"],
            }
        )
    )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
