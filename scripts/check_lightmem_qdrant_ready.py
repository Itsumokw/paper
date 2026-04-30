#!/usr/bin/env python3
"""Validate that LightMem pre-update Qdrant collections contain points."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def load_sample_ids(dataset: Path) -> list[str]:
    with dataset.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [item["sample_id"] for item in data]


def storage_candidates(qdrant_dir: Path, collection_name: str) -> list[Path]:
    return [
        qdrant_dir / collection_name / "collection" / collection_name / "storage.sqlite",
        qdrant_dir / "collection" / collection_name / "storage.sqlite",
    ]


def count_points(storage_sqlite: Path) -> int:
    conn = sqlite3.connect(storage_sqlite)
    try:
        row = conn.execute("SELECT count(*) FROM points").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def collection_point_count(qdrant_dir: Path, collection_name: str) -> tuple[int, str]:
    for candidate in storage_candidates(qdrant_dir, collection_name):
        if candidate.exists():
            try:
                return count_points(candidate), str(candidate)
            except sqlite3.Error as exc:
                return -1, f"{candidate} ({exc})"
    return 0, "missing storage.sqlite"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/home/stu0032/paper/baseline/MAGMA/data/locomo10.json",
        help="LoCoMo dataset JSON used by LightMem",
    )
    parser.add_argument(
        "--qdrant-dir",
        required=True,
        help="LightMem qdrant_pre_update directory",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Minimum required points per sample collection",
    )
    parser.add_argument(
        "--require-summaries",
        action="store_true",
        help="Also require non-empty LightMem summary collections",
    )
    parser.add_argument(
        "--summary-suffix",
        default="_summary",
        help="Suffix used for LightMem summary collections",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = Path(args.dataset)
    qdrant_dir = Path(args.qdrant_dir)

    if not dataset.exists():
        print(f"Missing dataset: {dataset}", file=sys.stderr)
        return 2
    if not qdrant_dir.exists():
        print(f"Missing LightMem Qdrant directory: {qdrant_dir}", file=sys.stderr)
        return 2

    sample_ids = load_sample_ids(dataset)
    bad: list[tuple[str, int, str]] = []
    for sample_id in sample_ids:
        points, source = collection_point_count(qdrant_dir, sample_id)
        if points < args.min_points:
            bad.append((sample_id, points, source))
        if args.require_summaries:
            summary_name = f"{sample_id}{args.summary_suffix}"
            summary_points, summary_source = collection_point_count(qdrant_dir, summary_name)
            if summary_points < args.min_points:
                bad.append((summary_name, summary_points, summary_source))

    if bad:
        print(
            f"LightMem Qdrant not ready: {len(bad)} collection(s) failed validation "
            f"for {len(sample_ids)} sample(s). "
            "Required collections are missing or empty.",
            file=sys.stderr,
        )
        for sample_id, points, source in bad[:20]:
            print(f"  - {sample_id}: points={points}, source={source}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... {len(bad) - 20} more", file=sys.stderr)
        return 1

    collection_label = "sample and summary collections" if args.require_summaries else "sample collections"
    print(
        f"LightMem Qdrant ready: {len(sample_ids)}/{len(sample_ids)} "
        f"{collection_label} have at least {args.min_points} point(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
