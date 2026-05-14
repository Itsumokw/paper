#!/usr/bin/env python3
"""Self-test reviewer todo freshness checker."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from check_locomo_human_audit_reviewer_todos import check_reviewer_todos
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


def make_fixture(root: Path) -> tuple[Path, Path]:
    (root / "human_audit_batch_reviews").mkdir(parents=True, exist_ok=True)
    batch_csv = root / "human_audit_batches" / "batch_001_Fixture_001-003.csv"
    write_csv(batch_csv, [base_row(0, "todo"), base_row(1, "pass"), base_row(2, "")])
    audit_packet = root / "human_audit_packet.jsonl"
    write_jsonl(audit_packet, [packet_row(0, "todo"), packet_row(1, "pass"), packet_row(2, "")])
    assignment_summary = {
        "status": "completed",
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
    summary_path = root / "human_audit_assignment_risk_summary.json"
    summary_path.write_text(json.dumps(assignment_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = root / "human_audit_reviewer_todos" / "manifest.json"
    export_reviewer_todos(
        summary_path,
        root / "human_audit_reviewer_todos",
        root / "human_audit_batch_reviews",
        manifest_path,
        audit_packet,
    )
    return summary_path, manifest_path


def case_result(
    name: str,
    root: Path,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, object]:
    summary_path, manifest_path = make_fixture(root / name)
    report = check_reviewer_todos(
        manifest_path,
        assignment_summary_path=summary_path,
        output_dir=root / name / "human_audit_reviewer_todos",
        review_md_dir=root / name / "human_audit_batch_reviews",
    )
    errors = [str(item) for item in report.get("errors", [])]
    if expect_success:
        passed = report["status"] == "passed"
    else:
        passed = report["status"] == "failed" and any((expected_error_fragment or "") in item for item in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "expected_error_fragment": expected_error_fragment,
        "report_status": report["status"],
        "errors": errors,
    }


def stale_batch_case(root: Path) -> dict[str, object]:
    name = "stale_batch_csv_is_rejected"
    summary_path, manifest_path = make_fixture(root / name)
    batch_csv = root / name / "human_audit_batches" / "batch_001_Fixture_001-003.csv"
    write_csv(batch_csv, [base_row(0, "pass"), base_row(1, "pass"), base_row(2, "")])
    report = check_reviewer_todos(
        manifest_path,
        assignment_summary_path=summary_path,
        output_dir=root / name / "human_audit_reviewer_todos",
        review_md_dir=root / name / "human_audit_batch_reviews",
    )
    errors = [str(item) for item in report.get("errors", [])]
    passed = report["status"] == "failed" and any("sha256 mismatch" in item for item in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": False,
        "expected_error_fragment": "sha256 mismatch",
        "report_status": report["status"],
        "errors": errors,
    }


def stale_index_case(root: Path) -> dict[str, object]:
    name = "stale_reviewer_index_is_rejected"
    summary_path, manifest_path = make_fixture(root / name)
    index_csv = root / name / "human_audit_reviewer_todos" / "reviewer_A_todo_index.csv"
    text = index_csv.read_text(encoding="utf-8-sig")
    index_csv.write_text(
        text.replace("What concrete information did the fixture reveal?", "Changed question?"),
        encoding="utf-8",
    )
    report = check_reviewer_todos(
        manifest_path,
        assignment_summary_path=summary_path,
        output_dir=root / name / "human_audit_reviewer_todos",
        review_md_dir=root / name / "human_audit_batch_reviews",
    )
    errors = [str(item) for item in report.get("errors", [])]
    passed = report["status"] == "failed" and any("todo CSV" in item for item in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": False,
        "expected_error_fragment": "todo CSV",
        "report_status": report["status"],
        "errors": errors,
    }


def stale_audit_packet_case(root: Path) -> dict[str, object]:
    name = "stale_audit_packet_is_rejected"
    summary_path, manifest_path = make_fixture(root / name)
    audit_packet = root / name / "human_audit_packet.jsonl"
    changed_row = packet_row(0, "todo")
    changed_row["question"] = "Changed packet question?"
    write_jsonl(audit_packet, [changed_row, packet_row(1, "pass"), packet_row(2, "")])
    report = check_reviewer_todos(
        manifest_path,
        assignment_summary_path=summary_path,
        output_dir=root / name / "human_audit_reviewer_todos",
        review_md_dir=root / name / "human_audit_batch_reviews",
    )
    errors = [str(item) for item in report.get("errors", [])]
    passed = report["status"] == "failed" and any("audit_packet.sha256 mismatch" in item for item in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": False,
        "expected_error_fragment": "audit_packet.sha256 mismatch",
        "report_status": report["status"],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos_check_selftest.json"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="locomo_reviewer_todos_check_selftest_") as tmp:
        root = Path(tmp)
        cases = [
            case_result("fresh_reviewer_todos_are_accepted", root, True),
            stale_batch_case(root),
            stale_audit_packet_case(root),
            stale_index_case(root),
        ]
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "checker": str(Path(__file__).with_name("check_locomo_human_audit_reviewer_todos.py")),
        "checker_sha256": sha256_file(Path(__file__).with_name("check_locomo_human_audit_reviewer_todos.py")),
        "selftest": str(Path(__file__)),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
