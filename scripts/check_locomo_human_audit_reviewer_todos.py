#!/usr/bin/env python3
"""Validate read-only per-reviewer human-audit todo indexes are fresh."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_locomo_human_audit_reviewer_todos import CSV_FIELDS, load_flag_index, read_todo_rows, sha256_file


PURPOSE = "read_only_reviewer_routing_indexes_not_audit_decisions"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_index_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def first_row_mismatch(expected: list[dict[str, str]], actual: list[dict[str, str]]) -> str:
    if len(expected) != len(actual):
        return f"row_count expected={len(expected)} actual={len(actual)}"
    for idx, (expected_row, actual_row) in enumerate(zip(expected, actual), start=1):
        for field in CSV_FIELDS:
            if str(expected_row.get(field, "")) != str(actual_row.get(field, "")):
                return (
                    f"row {idx} field {field}: "
                    f"expected={expected_row.get(field, '')!r} actual={actual_row.get(field, '')!r}"
                )
    return ""


def to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_reviewer_todos(
    manifest_path: Path,
    assignment_summary_path: Path | None = None,
    output_dir: Path | None = None,
    review_md_dir: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return {
            "status": "failed",
            "manifest": str(manifest_path),
            "errors": [f"manifest missing: {manifest_path}"],
        }

    manifest = load_json(manifest_path)
    if manifest.get("status") != "completed":
        errors.append(f"manifest.status={manifest.get('status')!r} expected='completed'")
    if manifest.get("purpose") != PURPOSE:
        errors.append(f"manifest.purpose={manifest.get('purpose')!r} expected={PURPOSE!r}")

    manifest_assignment = manifest.get("assignment_summary") if isinstance(manifest, dict) else None
    manifest_assignment_path = Path(str((manifest_assignment or {}).get("path", "")))
    assignment_path = assignment_summary_path or manifest_assignment_path
    if not assignment_path.is_file():
        errors.append(f"assignment summary missing: {assignment_path}")
        assignment_summary: dict[str, Any] = {}
    else:
        assignment_summary = load_json(assignment_path)
        expected_assignment_hash = str((manifest_assignment or {}).get("sha256", ""))
        actual_assignment_hash = sha256_file(assignment_path)
        if expected_assignment_hash and expected_assignment_hash != actual_assignment_hash:
            errors.append("manifest assignment_summary.sha256 mismatch")

    if assignment_summary.get("status") != "completed":
        errors.append(f"assignment summary status={assignment_summary.get('status')!r} expected='completed'")

    output_path = output_dir or Path(str(manifest.get("output_dir", "")))
    review_path = review_md_dir or Path(str(manifest.get("review_md_dir", "")))
    if not output_path.is_dir():
        errors.append(f"output_dir missing: {output_path}")
    if not review_path.is_dir():
        errors.append(f"review_md_dir missing: {review_path}")

    audit_packet_row = manifest.get("audit_packet") if isinstance(manifest, dict) else None
    audit_packet_path = Path(str((audit_packet_row or {}).get("path", "")))
    if not audit_packet_path.is_file():
        errors.append(f"audit packet missing: {audit_packet_path}")
        flags_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    else:
        if str((audit_packet_row or {}).get("sha256", "")) != sha256_file(audit_packet_path):
            errors.append("manifest audit_packet.sha256 mismatch")
        flags_by_key = load_flag_index(audit_packet_path)

    reviewers = assignment_summary.get("reviewers") if isinstance(assignment_summary, dict) else {}
    manifest_reviewers = manifest.get("reviewers") if isinstance(manifest, dict) else {}
    if not isinstance(reviewers, dict):
        errors.append("assignment summary reviewers missing")
        reviewers = {}
    if not isinstance(manifest_reviewers, dict):
        errors.append("manifest reviewers missing")
        manifest_reviewers = {}

    missing_manifest_reviewers = sorted(set(reviewers) - set(manifest_reviewers))
    extra_manifest_reviewers = sorted(set(manifest_reviewers) - set(reviewers))
    if missing_manifest_reviewers:
        errors.append(f"manifest missing reviewers={missing_manifest_reviewers}")
    if extra_manifest_reviewers:
        errors.append(f"manifest has extra reviewers={extra_manifest_reviewers}")

    batch_csvs = ((assignment_summary.get("input_files") or {}).get("batch_csvs") or {})
    if not isinstance(batch_csvs, dict):
        errors.append("assignment summary input_files.batch_csvs missing")
        batch_csvs = {}

    expected_total_todo = 0
    reviewer_reports: dict[str, Any] = {}
    for reviewer, reviewer_summary in sorted(reviewers.items()):
        expected_rows: list[dict[str, str]] = []
        for batch_name in reviewer_summary.get("batches", []):
            batch_meta = batch_csvs.get(batch_name)
            if not isinstance(batch_meta, dict):
                errors.append(f"reviewer {reviewer} missing batch metadata: {batch_name}")
                continue
            batch_path = Path(str(batch_meta.get("path", "")))
            if not batch_path.is_file():
                errors.append(f"reviewer {reviewer} batch CSV missing: {batch_path}")
                continue
            if str(batch_meta.get("sha256", "")) != sha256_file(batch_path):
                errors.append(f"reviewer {reviewer} batch {batch_name} sha256 mismatch")
            expected_rows.extend(
                read_todo_rows(
                    str(reviewer),
                    str(batch_name),
                    batch_path,
                    review_path,
                    start_rank=len(expected_rows) + 1,
                    flags_by_key=flags_by_key,
                )
            )

        expected_total_todo += len(expected_rows)
        manifest_row = manifest_reviewers.get(str(reviewer), {})
        if not isinstance(manifest_row, dict):
            errors.append(f"manifest reviewer {reviewer} row is not object")
            manifest_row = {}
        if to_int(manifest_row.get("rows")) != to_int(reviewer_summary.get("rows"), 0):
            errors.append(f"reviewer {reviewer} manifest rows mismatch")
        if to_int(manifest_row.get("completed")) != to_int(reviewer_summary.get("completed"), 0):
            errors.append(f"reviewer {reviewer} manifest completed mismatch")
        if to_int(manifest_row.get("remaining_from_assignment")) != to_int(reviewer_summary.get("remaining"), 0):
            errors.append(f"reviewer {reviewer} manifest remaining mismatch")
        if to_int(manifest_row.get("todo_rows_exported")) != len(expected_rows):
            errors.append(
                f"reviewer {reviewer} todo_rows_exported={manifest_row.get('todo_rows_exported')!r} "
                f"expected={len(expected_rows)}"
            )

        csv_meta = manifest_row.get("csv") if isinstance(manifest_row, dict) else {}
        csv_path = Path(str((csv_meta or {}).get("path", "")))
        actual_rows: list[dict[str, str]] = []
        if not csv_path.is_file():
            errors.append(f"reviewer {reviewer} todo CSV missing: {csv_path}")
        else:
            if str((csv_meta or {}).get("sha256", "")) != sha256_file(csv_path):
                errors.append(f"reviewer {reviewer} todo CSV sha256 mismatch")
            fieldnames, actual_rows = read_index_csv(csv_path)
            if fieldnames != CSV_FIELDS:
                errors.append(f"reviewer {reviewer} todo CSV fields mismatch")
            mismatch = first_row_mismatch(expected_rows, actual_rows)
            if mismatch:
                errors.append(f"reviewer {reviewer} todo CSV stale: {mismatch}")

        md_meta = manifest_row.get("markdown") if isinstance(manifest_row, dict) else {}
        md_path = Path(str((md_meta or {}).get("path", "")))
        if not md_path.is_file():
            errors.append(f"reviewer {reviewer} todo markdown missing: {md_path}")
        elif str((md_meta or {}).get("sha256", "")) != sha256_file(md_path):
            errors.append(f"reviewer {reviewer} todo markdown sha256 mismatch")

        reviewer_reports[str(reviewer)] = {
            "expected_todo_rows": len(expected_rows),
            "actual_todo_rows": len(actual_rows),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    assigned_remaining = sum(to_int(row.get("remaining"), 0) for row in reviewers.values())
    if to_int(manifest.get("total_todo_rows")) != expected_total_todo:
        errors.append(f"manifest.total_todo_rows={manifest.get('total_todo_rows')!r} expected={expected_total_todo}")
    if to_int(manifest.get("assigned_remaining")) != assigned_remaining:
        errors.append(f"manifest.assigned_remaining={manifest.get('assigned_remaining')!r} expected={assigned_remaining}")
    if expected_total_todo != assigned_remaining:
        errors.append(f"expected_total_todo={expected_total_todo} assigned_remaining={assigned_remaining}")
    if manifest.get("errors"):
        errors.append(f"manifest.errors={manifest.get('errors')!r}")

    return {
        "status": "passed" if not errors else "failed",
        "purpose": "validate_read_only_reviewer_todo_indexes_are_fresh",
        "checker": str(Path(__file__)),
        "checker_sha256": sha256_file(Path(__file__)),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "assignment_summary": {
            "path": str(assignment_path),
            "sha256": sha256_file(assignment_path) if assignment_path.is_file() else "",
        },
        "audit_packet": {
            "path": str(audit_packet_path),
            "sha256": sha256_file(audit_packet_path) if audit_packet_path.is_file() else "",
        },
        "output_dir": str(output_path),
        "review_md_dir": str(review_path),
        "total_todo_rows": expected_total_todo,
        "assigned_remaining": assigned_remaining,
        "reviewers": reviewer_reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos/manifest.json"),
    )
    parser.add_argument("--assignment-summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--review-md-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos_check.json"),
    )
    args = parser.parse_args()

    report = check_reviewer_todos(
        args.manifest,
        assignment_summary_path=args.assignment_summary_json,
        output_dir=args.output_dir,
        review_md_dir=args.review_md_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
