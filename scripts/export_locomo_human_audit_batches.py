#!/usr/bin/env python3
"""Export human-audit packet rows into smaller editable CSV batches."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from export_locomo_human_audit_csv import FIELDNAMES, compact_evidence_text, json_cell


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def packet_row_to_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": row.get("source_dataset", ""),
        "sample_id": row.get("sample_id", ""),
        "qa_idx": row.get("qa_idx", ""),
        "category": row.get("category", ""),
        "question_type": row.get("question_type", ""),
        "difficulty": row.get("difficulty", ""),
        "whether_cross_session": row.get("whether_cross_session", ""),
        "audit_reasons": "; ".join(row.get("audit_reasons", [])),
        "human_decision": row.get("human_decision", "todo"),
        "human_notes": row.get("human_notes", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "adversarial_answer": row.get("adversarial_answer", ""),
        "evidence": json_cell(row.get("evidence", [])),
        "negative_evidence": json_cell(row.get("negative_evidence", [])),
        "evidence_detail": json_cell(row.get("evidence_detail", [])),
        "evidence_text": compact_evidence_text(row.get("evidence_text", [])),
        "negative_evidence_text": compact_evidence_text(row.get("negative_evidence_text", [])),
        "answer_facts": json_cell(row.get("answer_facts", [])),
        "corrected_question": row.get("corrected_question", ""),
        "corrected_answer": row.get("corrected_answer", ""),
        "corrected_category": row.get("corrected_category", ""),
        "corrected_evidence": json_cell(row.get("corrected_evidence", [])) if row.get("corrected_evidence") else "",
        "corrected_answer_facts": (
            json_cell(row.get("corrected_answer_facts", []))
            if row.get("corrected_answer_facts")
            else ""
        ),
        "corrected_evidence_detail": (
            json_cell(row.get("corrected_evidence_detail", []))
            if row.get("corrected_evidence_detail")
            else ""
        ),
        "corrected_adversarial_answer": row.get("corrected_adversarial_answer", ""),
        "corrected_negative_evidence": (
            json_cell(row.get("corrected_negative_evidence", []))
            if row.get("corrected_negative_evidence")
            else ""
        ),
        "corrected_adversarial_reason": row.get("corrected_adversarial_reason", ""),
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "batch"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet_row_to_csv_row(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args()

    if args.max_rows < 1:
        raise SystemExit("--max-rows must be positive")

    rows = list(iter_jsonl(args.input_jsonl))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_dataset", "unknown"))].append(row)

    batches: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in args.output_dir.glob("batch_*.csv"):
        old_csv.unlink()

    batch_idx = 1
    for source in sorted(by_source):
        source_rows = sorted(
            by_source[source],
            key=lambda row: (str(row.get("sample_id")), int(row.get("qa_idx", 0))),
        )
        for start in range(0, len(source_rows), args.max_rows):
            chunk = source_rows[start : start + args.max_rows]
            filename = f"batch_{batch_idx:03d}_{safe_name(source)}_{start + 1:03d}-{start + len(chunk):03d}.csv"
            path = args.output_dir / filename
            write_csv(path, chunk)
            batches.append(
                {
                    "batch_id": batch_idx,
                    "source_dataset": source,
                    "path": str(path),
                    "rows": len(chunk),
                    "first_key": {
                        "source_dataset": chunk[0].get("source_dataset"),
                        "sample_id": chunk[0].get("sample_id"),
                        "qa_idx": chunk[0].get("qa_idx"),
                    },
                    "last_key": {
                        "source_dataset": chunk[-1].get("source_dataset"),
                        "sample_id": chunk[-1].get("sample_id"),
                        "qa_idx": chunk[-1].get("qa_idx"),
                    },
                }
            )
            batch_idx += 1

    summary = {
        "status": "exported",
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "max_rows": args.max_rows,
        "batches": batches,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
