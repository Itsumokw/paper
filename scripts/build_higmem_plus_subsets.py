#!/usr/bin/env python3
"""Build deterministic stratified subsets for HiGMemPlus experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


SEED = 20260517


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--locomo-path", type=Path, default=Path("datasets/locomo/data/locomo10.json"))
    parser.add_argument("--longdialqa-dir", type=Path, default=Path("datasets/DialSim/longdialqa_normalized_v1.1_seed0"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/subsets"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.05])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct_label(fraction: float) -> str:
    value = int(round(fraction * 100))
    return f"{value}pct"


def target_count(n: int, fraction: float) -> int:
    if n <= 0:
        return 0
    if fraction >= 0.5:
        return max(1, int(round(n * fraction)))
    return max(1, int(round(n * fraction)))


def stratified_indices(groups: dict[Any, list[int]], fraction: float, rng: random.Random) -> set[int]:
    selected: set[int] = set()
    for key in sorted(groups, key=lambda item: str(item)):
        idxs = list(groups[key])
        rng.shuffle(idxs)
        selected.update(idxs[: target_count(len(idxs), fraction)])
    return selected


def build_locomo(args: argparse.Namespace, fraction: float) -> tuple[Path, Path]:
    data = json.loads(args.locomo_path.read_text(encoding="utf-8"))
    flat: list[tuple[int, int, dict[str, Any]]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    per_category = Counter()
    for sample_idx, sample in enumerate(data):
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            flat_idx = len(flat)
            category = str(qa.get("category", "unknown"))
            flat.append((sample_idx, qa_idx, qa))
            groups[category].append(flat_idx)
            per_category[category] += 1

    rng = random.Random(args.seed + int(fraction * 10000) + 11)
    selected = stratified_indices(groups, fraction, rng)
    selected_by_sample: dict[int, set[int]] = defaultdict(set)
    for flat_idx in selected:
        sample_idx, qa_idx, _qa = flat[flat_idx]
        selected_by_sample[sample_idx].add(qa_idx)

    filtered = []
    for sample_idx, sample in enumerate(data):
        qa_indices = selected_by_sample.get(sample_idx, set())
        if not qa_indices:
            continue
        new_sample = deepcopy(sample)
        new_sample["qa"] = [qa for idx, qa in enumerate(sample.get("qa", [])) if idx in qa_indices]
        filtered.append(new_sample)

    label = pct_label(fraction)
    out_path = args.output_dir / f"locomo10_{label}_seed{args.seed}.json"
    manifest_path = args.output_dir / f"locomo10_{label}_seed{args.seed}_manifest.json"
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    subset_category = Counter(str(qa.get("category", "unknown")) for sample in filtered for qa in sample.get("qa", []))
    manifest = {
        "dataset": "LoCoMo10",
        "fraction": fraction,
        "seed": args.seed,
        "strategy": "stratified_by_category",
        "source_path": str(args.locomo_path),
        "source_sha256": sha256_file(args.locomo_path),
        "subset_path": str(out_path),
        "subset_sha256": sha256_file(out_path),
        "original_sample_count": len(data),
        "subset_sample_count": len(filtered),
        "original_qa_count": len(flat),
        "subset_qa_count": sum(len(sample.get("qa", [])) for sample in filtered),
        "per_category_original_counts": dict(sorted(per_category.items())),
        "per_category_subset_counts": dict(sorted(subset_category.items())),
        "sample_ids_included": [sample.get("sample_id", str(i)) for i, sample in enumerate(filtered)],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out_path, manifest_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def build_longdialqa(args: argparse.Namespace, fraction: float) -> tuple[Path, Path]:
    source_path = args.longdialqa_dir / "selected_qa.jsonl"
    sessions_path = args.longdialqa_dir / "sessions.jsonl"
    rows = read_jsonl(source_path)
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    show_counts = Counter()
    type_counts = Counter()
    answerability_counts = Counter()
    for idx, row in enumerate(rows):
        answerability = "answerable" if row.get("answerable") else "unanswerable"
        key = (row["show"], row.get("question_source", "unknown"), row.get("question_type", "unknown"), answerability)
        groups[key].append(idx)
        show_counts[row["show_name"]] += 1
        type_counts[str(row.get("question_type", "unknown"))] += 1
        answerability_counts[answerability] += 1

    rng = random.Random(args.seed + int(fraction * 10000) + 29)
    selected_indices = stratified_indices(groups, fraction, rng)
    filtered = [row for idx, row in enumerate(rows) if idx in selected_indices]

    selected_scene_ids = set()
    evidence_scene_ids = set()
    for row in filtered:
        selected_scene_ids.add(row.get("scene_id"))
        for scene_id in row.get("evidence_scene_ids") or []:
            evidence_scene_ids.add(scene_id)

    label = pct_label(fraction)
    out_path = args.output_dir / f"longdialqa_{label}_seed{args.seed}.json"
    manifest_path = args.output_dir / f"longdialqa_{label}_seed{args.seed}_manifest.json"
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    subset_show_counts = Counter(row["show_name"] for row in filtered)
    subset_type_counts = Counter(str(row.get("question_type", "unknown")) for row in filtered)
    subset_source_counts = Counter(str(row.get("question_source", "unknown")) for row in filtered)
    subset_answerability_counts = Counter("answerable" if row.get("answerable") else "unanswerable" for row in filtered)
    subset_hop_counts = Counter(str(row.get("hop_type", "unknown")) for row in filtered)
    manifest = {
        "dataset": "LongDialQA/DialSim",
        "fraction": fraction,
        "seed": args.seed,
        "strategy": "stratified_by_show_question_source_type_answerability",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "sessions_path": str(sessions_path),
        "sessions_sha256": sha256_file(sessions_path),
        "normalized_manifest_path": str(args.longdialqa_dir / "manifest.json"),
        "normalized_manifest_sha256": sha256_file(args.longdialqa_dir / "manifest.json"),
        "subset_path": str(out_path),
        "subset_sha256": sha256_file(out_path),
        "original_question_count": len(rows),
        "subset_question_count": len(filtered),
        "per_show_original_counts": dict(sorted(show_counts.items())),
        "per_show_subset_counts": dict(sorted(subset_show_counts.items())),
        "per_type_original_counts": dict(sorted(type_counts.items())),
        "per_type_subset_counts": dict(sorted(subset_type_counts.items())),
        "per_source_subset_counts": dict(sorted(subset_source_counts.items())),
        "per_answerability_original_counts": dict(sorted(answerability_counts.items())),
        "per_answerability_subset_counts": dict(sorted(subset_answerability_counts.items())),
        "per_hop_subset_counts": dict(sorted(subset_hop_counts.items())),
        "selected_scene_count": len(selected_scene_ids),
        "evidence_scene_count": len(evidence_scene_ids),
        "scene_session_coverage": {
            "selected_scene_ids_sample": sorted(scene for scene in selected_scene_ids if scene)[:50],
            "evidence_scene_ids_sample": sorted(scene for scene in evidence_scene_ids if scene)[:50],
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out_path, manifest_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fraction in args.fractions:
        outputs.append(build_locomo(args, fraction))
        outputs.append(build_longdialqa(args, fraction))
    print(json.dumps({"outputs": [[str(path), str(manifest)] for path, manifest in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
