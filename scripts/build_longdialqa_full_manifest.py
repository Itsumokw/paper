#!/usr/bin/env python3
"""Build a manifest-controlled full LongDialQA/DialSim evaluation file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 20260517


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--longdialqa-dir", type=Path, default=Path("datasets/DialSim/longdialqa_normalized_v1.1_seed0"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/subsets"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.longdialqa_dir / "selected_qa.jsonl"
    sessions_path = args.longdialqa_dir / "sessions.jsonl"
    normalized_manifest_path = args.longdialqa_dir / "manifest.json"
    rows = read_jsonl(source_path)

    out_path = args.output_dir / f"longdialqa_full_seed{args.seed}.json"
    manifest_path = args.output_dir / f"longdialqa_full_seed{args.seed}_manifest.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    show_counts = Counter(row["show_name"] for row in rows)
    source_counts = Counter(str(row.get("question_source", "unknown")) for row in rows)
    type_counts = Counter(str(row.get("question_type", "unknown")) for row in rows)
    answerability_counts = Counter("answerable" if row.get("answerable") else "unanswerable" for row in rows)
    hop_counts = Counter(str(row.get("hop_type", "unknown")) for row in rows)
    selected_scene_ids = {row.get("scene_id") for row in rows if row.get("scene_id")}
    evidence_scene_ids = {
        scene_id
        for row in rows
        for scene_id in (row.get("evidence_scene_ids") or [])
        if scene_id
    }

    manifest = {
        "dataset": "LongDialQA/DialSim",
        "split_label": "full",
        "fraction": 1.0,
        "seed": args.seed,
        "strategy": "manifest_controlled_full_reproduction",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "sessions_path": str(sessions_path),
        "sessions_sha256": sha256_file(sessions_path),
        "normalized_manifest_path": str(normalized_manifest_path),
        "normalized_manifest_sha256": sha256_file(normalized_manifest_path),
        "subset_path": str(out_path),
        "subset_sha256": sha256_file(out_path),
        "original_question_count": len(rows),
        "subset_question_count": len(rows),
        "per_show_counts": dict(sorted(show_counts.items())),
        "per_source_counts": dict(sorted(source_counts.items())),
        "per_type_counts": dict(sorted(type_counts.items())),
        "per_answerability_counts": dict(sorted(answerability_counts.items())),
        "per_hop_counts": dict(sorted(hop_counts.items())),
        "selected_scene_count": len(selected_scene_ids),
        "evidence_scene_count": len(evidence_scene_ids),
        "selected_scene_ids_sample": sorted(selected_scene_ids)[:50],
        "evidence_scene_ids_sample": sorted(evidence_scene_ids)[:50],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"subset": str(out_path), "manifest": str(manifest_path), "questions": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
