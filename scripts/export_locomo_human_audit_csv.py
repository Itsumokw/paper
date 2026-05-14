#!/usr/bin/env python3
"""Export the enriched human-audit packet JSONL to an editable CSV sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "source_dataset",
    "sample_id",
    "qa_idx",
    "category",
    "question_type",
    "difficulty",
    "whether_cross_session",
    "audit_reasons",
    "human_decision",
    "human_notes",
    "question",
    "answer",
    "adversarial_answer",
    "evidence",
    "negative_evidence",
    "evidence_detail",
    "evidence_text",
    "negative_evidence_text",
    "answer_facts",
    "corrected_question",
    "corrected_answer",
    "corrected_category",
    "corrected_evidence",
    "corrected_answer_facts",
    "corrected_evidence_detail",
    "corrected_adversarial_answer",
    "corrected_negative_evidence",
    "corrected_adversarial_reason",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_evidence_text(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows or []:
        if row.get("missing"):
            parts.append(f"{row.get('dia_id')}: <missing>")
            continue
        text = " ".join(str(row.get("text", "")).split())
        parts.append(f"{row.get('dia_id')} {row.get('session_id', '')} {row.get('speaker', '')}: {text}")
    return "\n".join(parts)


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for row in iter_jsonl(args.input_jsonl):
        rows.append(
            {
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
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "exported",
        "input_jsonl": str(args.input_jsonl),
        "output_csv": str(args.output_csv),
        "rows": len(rows),
        "editable_fields": [
            "human_decision",
            "human_notes",
            "corrected_question",
            "corrected_answer",
            "corrected_category",
            "corrected_evidence",
            "corrected_answer_facts",
            "corrected_evidence_detail",
            "corrected_adversarial_answer",
            "corrected_negative_evidence",
            "corrected_adversarial_reason",
        ],
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
