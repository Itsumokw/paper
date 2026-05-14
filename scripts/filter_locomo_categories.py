#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


def parse_categories(value: str) -> set[str]:
    categories = {item.strip() for item in value.split(",") if item.strip()}
    if not categories:
        raise argparse.ArgumentTypeError("at least one category is required")
    return categories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a LoCoMo dataset copy containing only selected QA categories."
    )
    parser.add_argument("--input", required=True, help="Source LoCoMo JSON file.")
    parser.add_argument("--output", required=True, help="Filtered LoCoMo JSON file.")
    parser.add_argument(
        "--categories",
        default="1,2,3,4",
        type=parse_categories,
        help="Comma-separated category IDs to keep. Default: 1,2,3,4.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"expected dataset list in {src}, got {type(data).__name__}")

    filtered = []
    counts: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for sample in data:
        item = deepcopy(sample)
        kept_qa = []
        for qa in item.get("qa", []):
            category = str(qa.get("category"))
            if category in args.categories:
                kept_qa.append(qa)
                counts[category] += 1
            else:
                dropped[category] += 1
        item["qa"] = kept_qa
        filtered.append(item)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(src),
                "output": str(dst),
                "samples": len(filtered),
                "kept_qa": sum(counts.values()),
                "kept_categories": dict(sorted(counts.items())),
                "dropped_categories": dict(sorted(dropped.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
