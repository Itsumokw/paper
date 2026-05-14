#!/usr/bin/env python3
"""Self-test reviewer todo index export."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from export_locomo_human_audit_reviewer_todos import export_reviewer_todos, sha256_file


FIELDS = [
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
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_row(idx: int, decision: str) -> dict[str, str]:
    question = f"Question {idx}?"
    if idx == 0:
        question = "What concrete information did the fixture reveal?"
    return {
        "source_dataset": "Fixture",
        "sample_id": "fixture_0",
        "qa_idx": str(idx),
        "category": "1",
        "question_type": "single-hop",
        "difficulty": "easy",
        "whether_cross_session": "False",
        "audit_reasons": "fixture",
        "human_decision": decision,
        "human_notes": "",
        "question": question,
        "answer": f"Answer {idx}",
        "adversarial_answer": "",
        "evidence": "[\"D1:1\"]",
        "negative_evidence": "[]",
        "evidence_detail": "[]",
        "evidence_text": f"D1:1 evidence {idx}",
        "negative_evidence_text": "",
        "answer_facts": json.dumps([{"fact": f"Answer fact {idx}", "supported_by": ["D1:1"]}]),
    }


def packet_row(idx: int, decision: str) -> dict[str, Any]:
    csv_row = base_row(idx, decision)
    return {
        "source_dataset": csv_row["source_dataset"],
        "sample_id": csv_row["sample_id"],
        "qa_idx": idx,
        "category": 1,
        "question_type": "single-hop",
        "difficulty": "easy",
        "whether_cross_session": False,
        "audit_reasons": ["fixture"],
        "question": csv_row["question"],
        "answer": csv_row["answer"],
        "adversarial_answer": "",
        "evidence": ["D1:1"],
        "negative_evidence": [],
        "evidence_detail": [
            {
                "dia_id": "D1:1",
                "session_id": "session_1",
                "turn_id": 1,
                "source_origin": "original_turn",
            }
        ],
        "answer_facts": [{"fact": f"Answer fact {idx}", "supported_by": ["D1:1"]}],
    }


def make_assignment_summary(path: Path, batch_csv: Path, status: str = "completed") -> None:
    summary = {
        "status": status,
        "input_files": {
            "batch_csvs": {
                batch_csv.name: {
                    "path": str(batch_csv),
                    "sha256": sha256_file(batch_csv),
                }
            }
        },
        "reviewers": {
            "A": {
                "batches": [batch_csv.name],
                "primary_risk": "fixture risk",
                "rows": 3,
                "completed": 1,
                "remaining": 2,
            }
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case(name: str, expect_success: bool, work: Path, bad_status: bool = False) -> dict[str, object]:
    batch_csv = work / name / "human_audit_batches" / "batch_001_Fixture_001-003.csv"
    write_csv(batch_csv, [base_row(0, "todo"), base_row(1, "pass"), base_row(2, "")])
    audit_packet = work / name / "human_audit_packet.jsonl"
    write_jsonl(audit_packet, [packet_row(0, "todo"), packet_row(1, "pass"), packet_row(2, "")])
    summary_path = work / name / "summary.json"
    make_assignment_summary(summary_path, batch_csv, status="failed" if bad_status else "completed")
    output_dir = work / name / "todos"
    manifest_path = output_dir / "manifest.json"
    manifest = export_reviewer_todos(
        summary_path,
        output_dir,
        work / name / "human_audit_batch_reviews",
        manifest_path,
        audit_packet,
    )
    ok = manifest["status"] == "completed"
    errors: list[str] = []
    if ok != expect_success:
        errors.append(f"expected success={expect_success} got status={manifest['status']}")
    if expect_success:
        csv_path = output_dir / "reviewer_A_todo_index.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2:
            errors.append(f"expected 2 todo rows, got {len(rows)}")
        if [row["qa_idx"] for row in rows] != ["0", "2"]:
            errors.append(f"unexpected qa_idx export order: {[row['qa_idx'] for row in rows]}")
        if "flag_score" not in rows[0] or "reviewer_flags" not in rows[0]:
            errors.append("flag_score/reviewer_flags columns were not exported")
        elif "template_like_question" not in rows[0]["reviewer_flags"]:
            errors.append(f"expected template_like_question flag, got {rows[0]['reviewer_flags']!r}")
        if manifest.get("total_todo_rows") != 2:
            errors.append(f"unexpected total_todo_rows={manifest.get('total_todo_rows')}")
    return {
        "name": name,
        "status": "passed" if not errors else "failed",
        "expect_success": expect_success,
        "manifest_status": manifest["status"],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos_selftest.json"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="locomo_reviewer_todos_selftest_") as tmp:
        root = Path(tmp)
        cases = [
            run_case("valid_reviewer_todos_are_exported", True, root),
            run_case("failed_assignment_summary_is_rejected", False, root, bad_status=True),
        ]
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
