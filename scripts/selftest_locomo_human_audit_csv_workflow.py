#!/usr/bin/env python3
"""Self-test human-audit CSV import and batch-merge workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from export_locomo_human_audit_csv import FIELDNAMES


SCRIPT_NAMES = [
    "export_locomo_human_audit_csv.py",
    "import_locomo_human_audit_csv.py",
    "finalize_locomo_human_audit_csv.py",
    "export_locomo_human_audit_batches.py",
    "merge_locomo_human_audit_batches.py",
    "summarize_locomo_human_audit_batches.py",
    "check_locomo_human_audit_batch_edits.py",
    "finalize_locomo_human_audit_batches.py",
]
SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_trace_sidecars(root: Path) -> None:
    for artifact in SOURCE_TO_ARTIFACT.values():
        artifact_root = root / artifact
        artifact_root.mkdir(parents=True, exist_ok=True)
        fact_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        if artifact == "JLongChat-LoCoMo-style-eval":
            fact_rows = [
                {"fact_id": "f0", "source_type": "original_turn", "source_text": "old answer"},
                {"fact_id": "f1", "source_type": "original_turn", "source_text": "old second answer"},
            ]
            provenance_rows = [
                {"sample_id": "fixture_sample_0", "dia_id": "D1:1", "source_origin": "original_turn"},
                {"sample_id": "fixture_sample_0", "dia_id": "D1:2", "source_origin": "original_turn"},
            ]
        write_jsonl(artifact_root / f"{artifact}_fact_ledger.jsonl", fact_rows)
        write_jsonl(artifact_root / f"{artifact}_provenance.jsonl", provenance_rows)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def packet_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "JLongChat",
            "sample_id": "fixture_sample_0",
            "qa_idx": 0,
            "category": 1,
            "question_type": "single-hop",
            "difficulty": "easy",
            "whether_cross_session": False,
            "audit_reasons": ["selftest"],
            "human_decision": "todo",
            "human_notes": "",
            "question": "What concrete information did A mention in session 1?",
            "answer": "old answer...",
            "evidence": ["D1:1"],
            "negative_evidence": [],
            "evidence_text": [{"dia_id": "D1:1", "text": "old answer"}],
            "negative_evidence_text": [],
            "answer_facts": [{"fact": "old answer...", "supported_by": ["D1:1"], "source_fact_id": "f0"}],
            "evidence_detail": [
                {"dia_id": "D1:1", "source_origin": "original_turn", "supports_answer_fact": ["f0"]}
            ],
        },
        {
            "source_dataset": "JLongChat",
            "sample_id": "fixture_sample_0",
            "qa_idx": 1,
            "category": 1,
            "question_type": "single-hop",
            "difficulty": "easy",
            "whether_cross_session": False,
            "audit_reasons": ["selftest"],
            "human_decision": "todo",
            "human_notes": "",
            "question": "What did B say?",
            "answer": "old second answer",
            "evidence": ["D1:2"],
            "negative_evidence": [],
            "evidence_text": [{"dia_id": "D1:2", "text": "old second answer"}],
            "negative_evidence_text": [],
            "answer_facts": [{"fact": "old second answer", "supported_by": ["D1:2"], "source_fact_id": "f1"}],
            "evidence_detail": [
                {"dia_id": "D1:2", "source_origin": "original_turn", "supports_answer_fact": ["f1"]}
            ],
        },
    ]


def run_json(command: list[str], tempdir: Path) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    parsed["_tempdir"] = str(tempdir)
    if completed.stderr.strip() and completed.stdout.strip():
        parsed["stderr"] = completed.stderr
    return completed.returncode, parsed


def normalize_errors(result: dict[str, Any]) -> list[str]:
    tempdir = str(result.get("_tempdir", ""))
    return [str(item).replace(tempdir + "/", "<tmp>/") for item in result.get("errors", [])]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def edit_full_csv(path: Path) -> None:
    rows = read_csv(path)
    rows[0]["human_decision"] = "pass"
    rows[0]["human_notes"] = "accepted"
    rows[1]["human_decision"] = "fix"
    rows[1]["human_notes"] = "corrected answer/evidence"
    rows[1]["corrected_answer"] = "corrected second answer"
    rows[1]["corrected_evidence"] = '["D1:2"]'
    rows[1]["corrected_answer_facts"] = json.dumps(
        [{"fact": "corrected second answer", "supported_by": ["D1:2"], "source_fact_id": "f1"}],
        ensure_ascii=False,
    )
    rows[1]["corrected_evidence_detail"] = json.dumps(
        [{"dia_id": "D1:2", "source_origin": "original_turn", "supports_answer_fact": ["f1"]}],
        ensure_ascii=False,
    )
    write_csv(path, rows)


def edit_batch_csvs(input_dir: Path) -> None:
    paths = sorted(input_dir.glob("batch_*.csv"))
    for idx, path in enumerate(paths):
        rows = read_csv(path)
        rows[0]["human_decision"] = "pass" if idx == 0 else "delete"
        rows[0]["human_notes"] = "batch accepted" if idx == 0 else "batch delete"
        write_csv(path, rows)


def case(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", **(details or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_csv_workflow_selftest.json"),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    scripts = {name: script_dir / name for name in SCRIPT_NAMES}
    cases: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_csv_workflow_selftest_") as tmp:
        tempdir = Path(tmp)
        base_jsonl = tempdir / "packet.jsonl"
        write_jsonl(base_jsonl, packet_rows())
        sidecar_root = tempdir / "sidecars"
        write_trace_sidecars(sidecar_root)

        csv_path = tempdir / "audit.csv"
        export_summary = tempdir / "export_summary.json"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["export_locomo_human_audit_csv.py"]),
                "--input-jsonl",
                str(base_jsonl),
                "--output-csv",
                str(csv_path),
                "--summary-json",
                str(export_summary),
            ],
            tempdir,
        )
        exported_rows = read_csv(csv_path) if csv_path.exists() else []
        cases.append(
            case(
                "csv_export_writes_editable_rows",
                rc == 0
                and result.get("status") == "exported"
                and result.get("rows") == 2
                and len(exported_rows) == 2
                and "corrected_answer_facts" in exported_rows[0]
                and "evidence_detail" in exported_rows[0]
                and "source_origin" in exported_rows[0]["evidence_detail"],
                {"errors": normalize_errors(result)},
            )
        )

        rc, result = run_json(
            [
                sys.executable,
                str(scripts["import_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(csv_path),
                "--output-jsonl",
                str(tempdir / "should_not_write.jsonl"),
                "--summary-json",
                str(tempdir / "import_incomplete_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        cases.append(
            case(
                "csv_import_require_complete_rejects_todo",
                rc != 0 and any("still have human_decision=todo" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        full_csv_incomplete_target = tempdir / "full_csv_incomplete_should_not_write.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(csv_path),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(full_csv_incomplete_target),
                "--summary-json",
                str(tempdir / "full_csv_finalize_incomplete_summary.json"),
                "--import-summary",
                str(tempdir / "full_csv_finalize_incomplete_import.json"),
                "--validation-summary",
                str(tempdir / "full_csv_finalize_incomplete_validation.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "full_csv_finalize_rejects_incomplete_without_writing",
                rc != 0
                and result.get("status") == "failed"
                and result.get("committed") is False
                and not full_csv_incomplete_target.exists()
                and any("still have human_decision=todo" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        read_only_csv = tempdir / "audit_read_only_tampered.csv"
        read_only_rows = read_csv(csv_path)
        read_only_rows[0]["human_decision"] = "pass"
        read_only_rows[0]["human_notes"] = "accepted"
        read_only_rows[0]["answer"] = "edited in read-only answer column"
        read_only_rows[1]["human_decision"] = "delete"
        read_only_rows[1]["human_notes"] = "delete second row"
        write_csv(read_only_csv, read_only_rows)
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["import_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(read_only_csv),
                "--output-jsonl",
                str(tempdir / "read_only_tamper_should_not_write.jsonl"),
                "--summary-json",
                str(tempdir / "read_only_tamper_import_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        cases.append(
            case(
                "csv_import_rejects_read_only_field_edits",
                rc != 0
                and result.get("status") == "failed"
                and any("read-only field 'answer' was modified" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        edit_full_csv(csv_path)
        imported_jsonl = tempdir / "packet_imported.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["import_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(csv_path),
                "--output-jsonl",
                str(imported_jsonl),
                "--summary-json",
                str(tempdir / "import_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        imported_rows = list(iter_jsonl(imported_jsonl)) if imported_jsonl.exists() else []
        cases.append(
            case(
                "csv_import_complete_parses_corrections",
                rc == 0
                and result.get("status") == "imported"
                and result.get("decision_counts") == {"fix": 1, "pass": 1}
                and imported_rows[1]["corrected_answer"] == "corrected second answer"
                and imported_rows[1]["corrected_evidence"] == ["D1:2"]
                and isinstance(imported_rows[1]["corrected_answer_facts"], list)
                and isinstance(imported_rows[1]["corrected_evidence_detail"], list),
                {"errors": normalize_errors(result)},
            )
        )

        full_csv_dry_run_target = tempdir / "full_csv_dry_run_should_not_write.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(csv_path),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(full_csv_dry_run_target),
                "--summary-json",
                str(tempdir / "full_csv_finalize_dry_run_summary.json"),
                "--import-summary",
                str(tempdir / "full_csv_finalize_dry_run_import.json"),
                "--validation-summary",
                str(tempdir / "full_csv_finalize_dry_run_validation.json"),
                "--dry-run",
            ],
            tempdir,
        )
        cases.append(
            case(
                "full_csv_finalize_dry_run_validates_without_writing",
                rc == 0
                and result.get("status") == "dry_run_valid"
                and result.get("committed") is False
                and result.get("import", {}).get("status") == "imported"
                and result.get("validation", {}).get("status") == "completed"
                and not full_csv_dry_run_target.exists()
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        full_csv_commit_target = tempdir / "full_csv_committed_packet.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_csv.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--decisions-csv",
                str(csv_path),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(full_csv_commit_target),
                "--summary-json",
                str(tempdir / "full_csv_finalize_commit_summary.json"),
                "--import-summary",
                str(tempdir / "full_csv_finalize_commit_import.json"),
                "--validation-summary",
                str(tempdir / "full_csv_finalize_commit_validation.json"),
            ],
            tempdir,
        )
        full_csv_committed_rows = list(iter_jsonl(full_csv_commit_target)) if full_csv_commit_target.exists() else []
        cases.append(
            case(
                "full_csv_finalize_commits_completed_packet",
                rc == 0
                and result.get("status") == "committed"
                and result.get("committed") is True
                and result.get("validation", {}).get("status") == "completed"
                and [row["human_decision"] for row in full_csv_committed_rows] == ["pass", "fix"]
                and full_csv_committed_rows[1].get("corrected_answer") == "corrected second answer"
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        batch_dir = tempdir / "batches"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["export_locomo_human_audit_batches.py"]),
                "--input-jsonl",
                str(base_jsonl),
                "--output-dir",
                str(batch_dir),
                "--summary-json",
                str(tempdir / "batch_export_summary.json"),
                "--max-rows",
                "1",
            ],
            tempdir,
        )
        batch_paths = sorted(batch_dir.glob("batch_*.csv"))
        cases.append(
            case(
                "batch_export_splits_rows",
                rc == 0 and result.get("status") == "exported" and len(batch_paths) == 2,
                {"errors": normalize_errors(result)},
            )
        )

        read_only_batch_dir = tempdir / "batches_read_only_tampered"
        rc, _ = run_json(
            [
                sys.executable,
                str(scripts["export_locomo_human_audit_batches.py"]),
                "--input-jsonl",
                str(base_jsonl),
                "--output-dir",
                str(read_only_batch_dir),
                "--summary-json",
                str(tempdir / "batch_read_only_export_summary.json"),
                "--max-rows",
                "1",
            ],
            tempdir,
        )
        edit_batch_csvs(read_only_batch_dir)
        read_only_batch_paths = sorted(read_only_batch_dir.glob("batch_*.csv"))
        read_only_batch_rows = read_csv(read_only_batch_paths[0])
        read_only_batch_rows[0]["question"] = "edited in read-only question column"
        write_csv(read_only_batch_paths[0], read_only_batch_rows)
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["merge_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(read_only_batch_dir),
                "--output-jsonl",
                str(tempdir / "batch_read_only_tamper_should_not_write.jsonl"),
                "--summary-json",
                str(tempdir / "batch_read_only_tamper_merge_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_merge_rejects_read_only_field_edits",
                rc != 0
                and result.get("status") == "failed"
                and any("read-only field 'question' was modified" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        rc, result = run_json(
            [
                sys.executable,
                str(scripts["summarize_locomo_human_audit_batches.py"]),
                "--input-dir",
                str(batch_dir),
                "--base-jsonl",
                str(base_jsonl),
                "--output-json",
                str(tempdir / "batch_progress_incomplete.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_progress_incomplete_is_successful_status",
                rc == 0
                and result.get("status") == "incomplete"
                and result.get("remaining") == 2
                and result.get("by_audit_reason", {}).get("selftest", {}).get("total") == 2
                and result.get("by_evidence_provenance", {}).get("original_turn", {}).get("total") == 2
                and result.get("batches", [{}])[0].get("by_category", {}).get("1", {}).get("total") == 1
                and result.get("batches", [{}])[0].get("by_evidence_provenance", {}).get("original_turn", {}).get("total") == 1
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        batch_review_dir = tempdir / "batch_review_md"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["summarize_locomo_human_audit_batches.py"]),
                "--input-dir",
                str(batch_dir),
                "--base-jsonl",
                str(base_jsonl),
                "--output-json",
                str(tempdir / "batch_progress_with_review_md.json"),
                "--output-review-md-dir",
                str(batch_review_dir),
                "--sidecar-root",
                str(sidecar_root),
            ],
            tempdir,
        )
        first_review = batch_review_dir / "batch_001_JLongChat_001-001.md"
        first_review_text = first_review.read_text(encoding="utf-8") if first_review.exists() else ""
        cases.append(
            case(
                "batch_review_markdown_includes_reviewer_flags",
                rc == 0
                and result.get("status") == "incomplete"
                and "literal_ascii_ellipsis_in_answer_or_fact" in first_review_text
                and "template_like_question" in first_review_text
                and "Source fact ledger support" in first_review_text
                and '"source_fact_id": "f0"' in first_review_text
                and '"source_text": "old answer"' in first_review_text
                and "Use this file for reading evidence/context" in first_review_text,
                {"errors": normalize_errors(result)},
            )
        )

        rc, result = run_json(
            [
                sys.executable,
                str(scripts["check_locomo_human_audit_batch_edits.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--sidecar-root",
                str(sidecar_root),
                "--output-summary",
                str(tempdir / "batch_edits_partial_summary.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_edit_check_allows_incomplete_valid_rows",
                rc == 0
                and result.get("status") == "partial_valid"
                and result.get("satisfies_release_gate") is False
                and result.get("merged_jsonl") == "temporary_discarded"
                and result.get("validation", {}).get("incomplete_count") == 2
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        incomplete_commit_target = tempdir / "incomplete_commit_should_not_write.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(incomplete_commit_target),
                "--summary-json",
                str(tempdir / "batch_finalize_incomplete_summary.json"),
                "--validation-summary",
                str(tempdir / "batch_finalize_incomplete_validation.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_finalize_rejects_incomplete_without_writing",
                rc != 0
                and result.get("status") == "failed"
                and result.get("committed") is False
                and not incomplete_commit_target.exists()
                and any("still have human_decision=todo" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        edit_batch_csvs(batch_dir)
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["summarize_locomo_human_audit_batches.py"]),
                "--input-dir",
                str(batch_dir),
                "--base-jsonl",
                str(base_jsonl),
                "--output-json",
                str(tempdir / "batch_progress_completed.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_progress_completed_is_successful_status",
                rc == 0
                and result.get("status") == "completed"
                and result.get("remaining") == 0
                and result.get("by_audit_reason", {}).get("selftest", {}).get("completed") == 2
                and result.get("by_evidence_provenance", {}).get("original_turn", {}).get("completed") == 2
                and result.get("batches", [{}])[0].get("by_audit_reason", {}).get("selftest", {}).get("completed") == 1
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        rc, result = run_json(
            [
                sys.executable,
                str(scripts["check_locomo_human_audit_batch_edits.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--sidecar-root",
                str(sidecar_root),
                "--output-summary",
                str(tempdir / "batch_edits_completed_summary.json"),
                "--merged-jsonl",
                str(tempdir / "batch_edits_completed_packet.jsonl"),
                "--validation-summary",
                str(tempdir / "batch_edits_completed_validation.json"),
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_edit_check_completed_rows",
                rc == 0
                and result.get("status") == "completed"
                and result.get("satisfies_release_gate") is False
                and result.get("merged_jsonl", "").endswith("batch_edits_completed_packet.jsonl")
                and result.get("validation", {}).get("incomplete_count") == 0
                and result.get("validation", {}).get("decision_counts") == {"delete": 1, "pass": 1}
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        dry_run_target = tempdir / "dry_run_should_not_write.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(dry_run_target),
                "--summary-json",
                str(tempdir / "batch_finalize_dry_run_summary.json"),
                "--validation-summary",
                str(tempdir / "batch_finalize_dry_run_validation.json"),
                "--dry-run",
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_finalize_dry_run_validates_without_writing",
                rc == 0
                and result.get("status") == "dry_run_valid"
                and result.get("committed") is False
                and result.get("validation", {}).get("status") == "completed"
                and not dry_run_target.exists()
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        commit_target = tempdir / "committed_packet.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["finalize_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--sidecar-root",
                str(sidecar_root),
                "--output-jsonl",
                str(commit_target),
                "--summary-json",
                str(tempdir / "batch_finalize_commit_summary.json"),
                "--validation-summary",
                str(tempdir / "batch_finalize_commit_validation.json"),
            ],
            tempdir,
        )
        committed_rows = list(iter_jsonl(commit_target)) if commit_target.exists() else []
        cases.append(
            case(
                "batch_finalize_commits_completed_packet",
                rc == 0
                and result.get("status") == "committed"
                and result.get("committed") is True
                and result.get("validation", {}).get("status") == "completed"
                and [row["human_decision"] for row in committed_rows] == ["pass", "delete"]
                and result.get("errors") == [],
                {"errors": normalize_errors(result)},
            )
        )

        merged_jsonl = tempdir / "packet_merged.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["merge_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(batch_dir),
                "--output-jsonl",
                str(merged_jsonl),
                "--summary-json",
                str(tempdir / "batch_merge_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        merged_rows = list(iter_jsonl(merged_jsonl)) if merged_jsonl.exists() else []
        cases.append(
            case(
                "batch_merge_complete_rows",
                rc == 0
                and result.get("status") == "merged"
                and result.get("decision_counts") == {"delete": 1, "pass": 1}
                and [row["human_decision"] for row in merged_rows] == ["pass", "delete"],
                {"errors": normalize_errors(result)},
            )
        )

        duplicate_dir = tempdir / "batches_duplicate"
        rc, _ = run_json(
            [
                sys.executable,
                str(scripts["export_locomo_human_audit_batches.py"]),
                "--input-jsonl",
                str(base_jsonl),
                "--output-dir",
                str(duplicate_dir),
                "--summary-json",
                str(tempdir / "batch_duplicate_export_summary.json"),
                "--max-rows",
                "1",
            ],
            tempdir,
        )
        edit_batch_csvs(duplicate_dir)
        duplicate_paths = sorted(duplicate_dir.glob("batch_*.csv"))
        first_rows = read_csv(duplicate_paths[0])
        second_rows = read_csv(duplicate_paths[1])
        second_rows.append(first_rows[0])
        write_csv(duplicate_paths[1], second_rows)
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["merge_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(duplicate_dir),
                "--output-jsonl",
                str(tempdir / "duplicate_should_not_write.jsonl"),
                "--summary-json",
                str(tempdir / "duplicate_merge_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_merge_rejects_duplicate_key",
                rc != 0 and any("duplicate key" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

        missing_dir = tempdir / "batches_missing"
        rc, _ = run_json(
            [
                sys.executable,
                str(scripts["export_locomo_human_audit_batches.py"]),
                "--input-jsonl",
                str(base_jsonl),
                "--output-dir",
                str(missing_dir),
                "--summary-json",
                str(tempdir / "batch_missing_export_summary.json"),
                "--max-rows",
                "1",
            ],
            tempdir,
        )
        edit_batch_csvs(missing_dir)
        sorted(missing_dir.glob("batch_*.csv"))[-1].unlink()
        rc, result = run_json(
            [
                sys.executable,
                str(scripts["merge_locomo_human_audit_batches.py"]),
                "--base-jsonl",
                str(base_jsonl),
                "--input-dir",
                str(missing_dir),
                "--output-jsonl",
                str(tempdir / "missing_should_not_write.jsonl"),
                "--summary-json",
                str(tempdir / "missing_merge_summary.json"),
                "--require-complete",
            ],
            tempdir,
        )
        cases.append(
            case(
                "batch_merge_rejects_missing_row",
                rc != 0 and any("missing 1 audit rows" in error for error in normalize_errors(result)),
                {"errors": normalize_errors(result)},
            )
        )

    report = {
        "status": "passed" if all(item["status"] == "passed" for item in cases) else "failed",
        "script_hashes": {name: sha256_file(path) for name, path in sorted(scripts.items())},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
