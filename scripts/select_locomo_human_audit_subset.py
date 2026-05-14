#!/usr/bin/env python3
"""Create a deterministic human-audit queue for LoCoMo-style eval artifacts."""

from __future__ import annotations

import argparse
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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qa_audit_rows(sidecar_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, artifact in SOURCE_TO_ARTIFACT.items():
        path = sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        for row in iter_jsonl(path):
            row = dict(row)
            row["audit_source_file"] = str(path)
            rows.append(row)
    return rows


def audit_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("source_dataset")), str(row.get("sample_id")), int(row.get("qa_idx", 0)))


def add_reason(selected: dict[tuple[str, str, int], dict[str, Any]], row: dict[str, Any], reason: str) -> None:
    key = audit_key(row)
    if key not in selected:
        selected[key] = {
            "source_dataset": row.get("source_dataset"),
            "sample_id": row.get("sample_id"),
            "qa_idx": row.get("qa_idx"),
            "category": row.get("category"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "whether_cross_session": row.get("whether_cross_session"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "evidence": row.get("evidence", []),
            "negative_evidence": row.get("negative_evidence", []),
            "answer_facts": row.get("answer_facts", []),
            "evidence_detail": row.get("evidence_detail", []),
            "audit_reasons": [],
            "human_decision": "todo",
            "human_notes": "",
        }
    if reason not in selected[key]["audit_reasons"]:
        selected[key]["audit_reasons"].append(reason)


def select_rows(primary_json: Path, sidecar_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_json(primary_json)
    qa_rows = load_qa_audit_rows(sidecar_root)
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}

    by_source_samples: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        source = str(sample.get("source_dataset"))
        sample_id = str(sample.get("sample_id"))
        if sample_id not in by_source_samples[source]:
            by_source_samples[source].append(sample_id)

    for source, sample_ids in by_source_samples.items():
        for sample_id in sorted(sample_ids)[:2]:
            for row in qa_rows:
                if row.get("source_dataset") == source and row.get("sample_id") == sample_id:
                    add_reason(selected, row, "full_sample_minimum_2_per_source")

    for category in (2, 4, 5):
        category_rows = [row for row in qa_rows if int(row.get("category") or 0) == category]
        target = math.ceil(len(category_rows) * 0.30)
        for row in sorted(category_rows, key=audit_key)[:target]:
            add_reason(selected, row, f"category_{category}_30_percent_minimum")

    synthetic_adjacent_origins = {"memory_anchor_turn", "synthetic_bridge_turn", "synthetic_continuation_turn"}
    perltqa_memory_anchor_rows: list[dict[str, Any]] = []
    for row in qa_rows:
        origins = {
            detail.get("source_origin")
            for detail in row.get("evidence_detail", [])
            if isinstance(detail, dict)
        }
        if origins & {"synthetic_bridge_turn", "synthetic_continuation_turn"}:
            add_reason(selected, row, "all_synthetic_adjacent_qa")
        if row.get("source_dataset") == "PerLTQA" and origins & synthetic_adjacent_origins:
            perltqa_memory_anchor_rows.append(row)

    target = math.ceil(len(perltqa_memory_anchor_rows) * 0.50)
    for row in sorted(perltqa_memory_anchor_rows, key=audit_key)[:target]:
        add_reason(selected, row, "perltqa_memory_anchor_50_percent_minimum")

    output_rows = sorted(selected.values(), key=lambda row: (str(row["source_dataset"]), str(row["sample_id"]), int(row["qa_idx"])))
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for row in output_rows:
        source_counts[str(row["source_dataset"])] += 1
        category_counts[str(row["category"])] += 1
        for reason in row["audit_reasons"]:
            reason_counts[reason] += 1

    summary = {
        "primary_json": str(primary_json),
        "sidecar_root": str(sidecar_root),
        "selected_qa": len(output_rows),
        "selected_by_source": dict(sorted(source_counts.items())),
        "selected_by_category": dict(sorted(category_counts.items())),
        "selected_by_reason": dict(sorted(reason_counts.items())),
        "status": "queue_created_human_review_not_completed",
    }
    return output_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-json", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = select_rows(args.primary_json, args.sidecar_root)
    write_jsonl(args.output_jsonl, rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
