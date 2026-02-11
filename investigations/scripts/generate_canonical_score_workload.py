#!/usr/bin/env python3
"""Generate a deterministic canonical workload for /v1/score multi-item benchmarking."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = "canonical_score_workload_v1"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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
        raise ValueError("Expected at least one integer value")
    return out


def get_safe_token_pool(tokenizer: Any, delimiter_token_id: int, min_pool_size: int = 16) -> List[int]:
    special_ids = set(tokenizer.all_special_ids or [])

    candidate_texts = [
        " yes",
        " no",
        " maybe",
        " true",
        " false",
        " apple",
        " banana",
        " cat",
        " dog",
        " red",
        " blue",
        " alpha",
        " beta",
        " gamma",
        " delta",
        " one",
        " two",
        " three",
        " north",
        " south",
        " east",
        " west",
    ]

    pool: List[int] = []
    seen = set()

    for text in candidate_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = token_ids[0]
        if token_id == delimiter_token_id or token_id in special_ids:
            continue
        if token_id in seen:
            continue
        seen.add(token_id)
        pool.append(token_id)

    if len(pool) < min_pool_size:
        vocab_size = int(tokenizer.vocab_size)
        for token_id in range(1, vocab_size):
            if token_id == delimiter_token_id or token_id in special_ids:
                continue
            if token_id in seen:
                continue
            seen.add(token_id)
            pool.append(token_id)
            if len(pool) >= min_pool_size:
                break

    if len(pool) < 2:
        raise RuntimeError(
            "Unable to find enough safe token IDs. Need at least 2 non-special, non-delimiter IDs."
        )

    return pool


def build_request(
    query_tokens: int,
    num_items: int,
    item_tokens: int,
    label_token_ids: List[int],
    token_pool: List[int],
) -> Dict[str, Any]:
    query_token_id = token_pool[0]
    item_pool = token_pool[1:] if len(token_pool) > 1 else token_pool

    query_token_ids = [query_token_id for _ in range(query_tokens)]

    items_token_ids: List[List[int]] = []
    for item_idx in range(num_items):
        token_a = item_pool[item_idx % len(item_pool)]
        token_b = item_pool[(item_idx + 3) % len(item_pool)]
        item = [token_a if j % 2 == 0 else token_b for j in range(item_tokens)]
        items_token_ids.append(item)

    return {
        "request_id": "canonical_score_request_000",
        "query_token_ids": query_token_ids,
        "items_token_ids": items_token_ids,
        "label_token_ids": label_token_ids,
        "apply_softmax": True,
        "item_first": False,
    }


def compute_request_checksum(request: Dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query-tokens", type=int, default=2000)
    parser.add_argument("--num-items", type=int, default=500)
    parser.add_argument("--item-tokens", type=int, default=20)
    parser.add_argument("--delimiter-token-id", type=int, default=151643)
    parser.add_argument("--label-token-ids", default="9454,2753")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    label_token_ids = parse_int_csv(args.label_token_ids)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "Missing dependency: transformers. Install with `pip install transformers` "
            "in the environment used for workload generation.",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_pool = get_safe_token_pool(tokenizer, args.delimiter_token_id)

    request = build_request(
        query_tokens=args.query_tokens,
        num_items=args.num_items,
        item_tokens=args.item_tokens,
        label_token_ids=label_token_ids,
        token_pool=token_pool,
    )

    workload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "model": args.model,
        "query_tokens": args.query_tokens,
        "num_items": args.num_items,
        "item_tokens": args.item_tokens,
        "delimiter_token_id": args.delimiter_token_id,
        "label_token_ids": label_token_ids,
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", args.model),
            "vocab_size": int(tokenizer.vocab_size),
            "special_token_ids": list(tokenizer.all_special_ids or []),
        },
        "safe_token_pool": token_pool,
        "requests": [request],
        "checksums": {
            "canonical_score_request_000_sha256": compute_request_checksum(request),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote workload: {output_path}")


if __name__ == "__main__":
    main()
