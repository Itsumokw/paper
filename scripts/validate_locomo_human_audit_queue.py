#!/usr/bin/env python3
"""Validate human-audit queue coverage against the project minimum rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audit_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("source_dataset")), str(row.get("sample_id")), int(row.get("qa_idx", 0)))


def load_qa_audit_rows(sidecar_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in SOURCE_TO_ARTIFACT.values():
        path = sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        for row in iter_jsonl(path):
            rows.append(row)
    return rows


def qa_audit_paths(sidecar_root: Path) -> list[Path]:
    return [
        sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        for artifact in SOURCE_TO_ARTIFACT.values()
    ]


def source_origins(row: dict[str, Any]) -> set[str]:
    return {
        str(detail.get("source_origin"))
        for detail in row.get("evidence_detail", [])
        if isinstance(detail, dict) and detail.get("source_origin")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-json", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    samples = load_json(args.primary_json)
    qa_rows = load_qa_audit_rows(args.sidecar_root)
    all_by_key = {audit_key(row): row for row in qa_rows}
    queue_rows = list(iter_jsonl(args.queue_jsonl))
    queue_keys = [audit_key(row) for row in queue_rows]
    selected = set(queue_keys)

    errors: list[str] = []
    duplicate_count = len(queue_keys) - len(selected)
    if duplicate_count:
        errors.append(f"queue has {duplicate_count} duplicate QA keys")

    missing_from_audit = sorted(key for key in selected if key not in all_by_key)
    if missing_from_audit:
        errors.append(f"queue contains {len(missing_from_audit)} rows not found in qa_audit sidecars")

    qa_by_sample: dict[tuple[str, str], set[tuple[str, str, int]]] = defaultdict(set)
    selected_by_sample: dict[tuple[str, str], set[tuple[str, str, int]]] = defaultdict(set)
    for row in qa_rows:
        key = audit_key(row)
        qa_by_sample[(key[0], key[1])].add(key)
    for key in selected:
        selected_by_sample[(key[0], key[1])].add(key)

    full_samples_by_source: Counter[str] = Counter()
    for sample in samples:
        source = str(sample.get("source_dataset"))
        sample_id = str(sample.get("sample_id"))
        all_keys = qa_by_sample[(source, sample_id)]
        selected_keys = selected_by_sample[(source, sample_id)]
        if all_keys and all_keys <= selected_keys:
            full_samples_by_source[source] += 1

    for source in sorted(SOURCE_TO_ARTIFACT):
        if full_samples_by_source[source] < 2:
            errors.append(f"{source}: full-sample audit coverage={full_samples_by_source[source]} expected>=2")

    category_coverage: dict[str, dict[str, int]] = {}
    for category in (2, 4, 5):
        all_category = {audit_key(row) for row in qa_rows if int(row.get("category") or 0) == category}
        selected_category = all_category & selected
        required = math.ceil(len(all_category) * 0.30)
        category_coverage[str(category)] = {
            "total": len(all_category),
            "selected": len(selected_category),
            "required": required,
        }
        if len(selected_category) < required:
            errors.append(f"category {category}: selected={len(selected_category)} expected>={required}")

    synthetic_origins = {"synthetic_bridge_turn", "synthetic_continuation_turn"}
    synthetic_adjacent = {
        audit_key(row)
        for row in qa_rows
        if source_origins(row) & synthetic_origins
    }
    missing_synthetic = sorted(synthetic_adjacent - selected)
    if missing_synthetic:
        errors.append(f"synthetic-adjacent QA missing from audit queue: {len(missing_synthetic)}")

    perltqa_memory_anchor = {
        audit_key(row)
        for row in qa_rows
        if row.get("source_dataset") == "PerLTQA"
        and source_origins(row) & {"memory_anchor_turn", "synthetic_bridge_turn", "synthetic_continuation_turn"}
    }
    selected_perltqa_memory_anchor = perltqa_memory_anchor & selected
    required_perltqa_memory_anchor = math.ceil(len(perltqa_memory_anchor) * 0.50)
    if len(selected_perltqa_memory_anchor) < required_perltqa_memory_anchor:
        errors.append(
            "PerLTQA memory-anchor QA: "
            f"selected={len(selected_perltqa_memory_anchor)} expected>={required_perltqa_memory_anchor}"
        )

    selected_by_reason: Counter[str] = Counter()
    selected_by_source: Counter[str] = Counter()
    for row in queue_rows:
        selected_by_source[str(row.get("source_dataset"))] += 1
        for reason in row.get("audit_reasons", []):
            selected_by_reason[str(reason)] += 1

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(args.primary_json),
            "queue_jsonl": file_record(args.queue_jsonl),
            "qa_audit_files": [file_record(path) for path in qa_audit_paths(args.sidecar_root)],
        },
        "primary_json": str(args.primary_json),
        "sidecar_root": str(args.sidecar_root),
        "queue_jsonl": str(args.queue_jsonl),
        "total_qa_audit_rows": len(qa_rows),
        "selected_qa": len(queue_rows),
        "duplicate_count": duplicate_count,
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "selected_by_reason": dict(sorted(selected_by_reason.items())),
        "full_samples_by_source": dict(sorted(full_samples_by_source.items())),
        "category_coverage": category_coverage,
        "synthetic_adjacent": {
            "total": len(synthetic_adjacent),
            "selected": len(synthetic_adjacent & selected),
        },
        "perltqa_memory_anchor": {
            "total": len(perltqa_memory_anchor),
            "selected": len(selected_perltqa_memory_anchor),
            "required": required_perltqa_memory_anchor,
        },
        "errors": errors,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
