#!/usr/bin/env python3
"""Export read-only per-reviewer human-audit todo indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_locomo_human_audit_flags import flag_score, iter_jsonl, row_flags


CSV_FIELDS = [
    "do_not_edit",
    "reviewer",
    "todo_rank",
    "flag_score",
    "reviewer_flags",
    "batch",
    "csv_path",
    "csv_line",
    "review_md_path",
    "source_dataset",
    "sample_id",
    "qa_idx",
    "category",
    "question_type",
    "difficulty",
    "whether_cross_session",
    "audit_reasons",
    "question_preview",
    "answer_preview",
    "evidence_preview",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shorten(text: str, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def markdown_escape(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def row_identity(row: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (str(row.get("source_dataset", "")), str(row.get("sample_id", "")), int(row.get("qa_idx", -1)))
    except (TypeError, ValueError):
        return None


def load_flag_index(audit_packet_jsonl: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = list(iter_jsonl(audit_packet_jsonl))
    question_counts = Counter(str(row.get("question", "")) for row in rows)
    flags_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = row_identity(row)
        if key is None:
            continue
        flags = sorted(row_flags(row, question_counts))
        flags_by_key[key] = {
            "flag_score": str(flag_score(flags)),
            "reviewer_flags": "; ".join(flags),
        }
    return flags_by_key


def read_todo_rows(
    reviewer: str,
    batch_name: str,
    batch_path: Path,
    review_md_dir: Path,
    start_rank: int,
    flags_by_key: dict[tuple[str, str, int], dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    review_md_path = review_md_dir / f"{batch_path.stem}.md"
    flags_by_key = flags_by_key or {}
    with batch_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            decision = str(row.get("human_decision", "")).strip().lower()
            if decision and decision != "todo":
                continue
            answer_text = row.get("answer") or row.get("adversarial_answer") or ""
            key = row_identity(row)
            flag_row = flags_by_key.get(key or ("", "", -1), {"flag_score": "0", "reviewer_flags": ""})
            rows.append(
                {
                    "do_not_edit": "read_only_index_edit_assigned_batch_csv_instead",
                    "reviewer": reviewer,
                    "todo_rank": str(start_rank + len(rows)),
                    "flag_score": flag_row["flag_score"],
                    "reviewer_flags": flag_row["reviewer_flags"],
                    "batch": batch_name,
                    "csv_path": str(batch_path),
                    "csv_line": str(line_number),
                    "review_md_path": str(review_md_path),
                    "source_dataset": str(row.get("source_dataset", "")),
                    "sample_id": str(row.get("sample_id", "")),
                    "qa_idx": str(row.get("qa_idx", "")),
                    "category": str(row.get("category", "")),
                    "question_type": str(row.get("question_type", "")),
                    "difficulty": str(row.get("difficulty", "")),
                    "whether_cross_session": str(row.get("whether_cross_session", "")),
                    "audit_reasons": str(row.get("audit_reasons", "")),
                    "question_preview": shorten(str(row.get("question", ""))),
                    "answer_preview": shorten(str(answer_text)),
                    "evidence_preview": shorten(str(row.get("evidence_text", "") or row.get("negative_evidence_text", ""))),
                }
            )
    return rows


def write_reviewer_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_reviewer_md(path: Path, reviewer: str, reviewer_summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        f"# Reviewer {reviewer} Todo Index",
        "",
        "Read-only routing file. Record audit decisions in the assigned batch CSV files, not here.",
        "",
        f"- Primary risk: {reviewer_summary.get('primary_risk', '')}",
        f"- Assigned rows: {reviewer_summary.get('rows', 0)}",
        f"- Remaining rows exported: {len(rows)}",
        f"- Batches: {', '.join(str(batch) for batch in reviewer_summary.get('batches', []))}",
        "",
        "| Rank | Score | Flags | Batch | CSV line | Source | Sample | QA | Cat | Reason | Question preview |",
        "| ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row["todo_rank"]),
                    markdown_escape(row["flag_score"]),
                    markdown_escape(row["reviewer_flags"]),
                    markdown_escape(row["batch"]),
                    markdown_escape(row["csv_line"]),
                    markdown_escape(row["source_dataset"]),
                    markdown_escape(row["sample_id"]),
                    markdown_escape(row["qa_idx"]),
                    markdown_escape(row["category"]),
                    markdown_escape(row["audit_reasons"]),
                    markdown_escape(row["question_preview"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def export_reviewer_todos(
    assignment_summary_path: Path,
    output_dir: Path,
    review_md_dir: Path,
    manifest_path: Path,
    audit_packet_jsonl: Path,
) -> dict[str, Any]:
    summary = load_json(assignment_summary_path)
    errors: list[str] = []
    if summary.get("status") != "completed":
        errors.append(f"assignment summary status={summary.get('status')!r} expected 'completed'")
    if audit_packet_jsonl.is_file():
        flags_by_key = load_flag_index(audit_packet_jsonl)
    else:
        errors.append(f"audit packet missing: {audit_packet_jsonl}")
        flags_by_key = {}

    batch_csvs = ((summary.get("input_files") or {}).get("batch_csvs") or {})
    reviewers = summary.get("reviewers") or {}
    reviewer_outputs: dict[str, Any] = {}
    total_todo_rows = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for reviewer, reviewer_summary in sorted(reviewers.items()):
        todo_rows: list[dict[str, str]] = []
        for batch_name in reviewer_summary.get("batches", []):
            batch_meta = batch_csvs.get(batch_name)
            if not batch_meta:
                errors.append(f"reviewer {reviewer} references batch without hash metadata: {batch_name}")
                continue
            batch_path = Path(str(batch_meta.get("path", "")))
            if not batch_path.is_file():
                errors.append(f"reviewer {reviewer} batch CSV missing: {batch_path}")
                continue
            expected_hash = str(batch_meta.get("sha256", ""))
            actual_hash = sha256_file(batch_path)
            if actual_hash != expected_hash:
                errors.append(f"reviewer {reviewer} batch {batch_name} sha256 mismatch")
                continue
            todo_rows.extend(
                read_todo_rows(
                    str(reviewer),
                    str(batch_name),
                    batch_path,
                    review_md_dir,
                    start_rank=len(todo_rows) + 1,
                    flags_by_key=flags_by_key,
                )
            )

        csv_path = output_dir / f"reviewer_{reviewer}_todo_index.csv"
        md_path = output_dir / f"reviewer_{reviewer}_todo_index.md"
        write_reviewer_csv(csv_path, todo_rows)
        write_reviewer_md(md_path, str(reviewer), dict(reviewer_summary), todo_rows)
        total_todo_rows += len(todo_rows)
        reviewer_outputs[str(reviewer)] = {
            "rows": int(reviewer_summary.get("rows", 0) or 0),
            "completed": int(reviewer_summary.get("completed", 0) or 0),
            "remaining_from_assignment": int(reviewer_summary.get("remaining", 0) or 0),
            "todo_rows_exported": len(todo_rows),
            "csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
            "markdown": {"path": str(md_path), "sha256": sha256_file(md_path)},
        }

    assigned_remaining = sum(int(row.get("remaining", 0) or 0) for row in reviewers.values())
    if total_todo_rows != assigned_remaining:
        errors.append(f"exported todo rows={total_todo_rows} does not match assignment remaining={assigned_remaining}")

    manifest = {
        "status": "completed" if not errors else "failed",
        "purpose": "read_only_reviewer_routing_indexes_not_audit_decisions",
        "assignment_summary": {
            "path": str(assignment_summary_path),
            "sha256": sha256_file(assignment_summary_path),
        },
        "audit_packet": {
            "path": str(audit_packet_jsonl),
            "sha256": sha256_file(audit_packet_jsonl) if audit_packet_jsonl.is_file() else "",
        },
        "review_md_dir": str(review_md_dir),
        "output_dir": str(output_dir),
        "reviewers": reviewer_outputs,
        "total_todo_rows": total_todo_rows,
        "assigned_remaining": assigned_remaining,
        "errors": errors,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignment-summary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_assignment_risk_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos"),
    )
    parser.add_argument(
        "--review-md-dir",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_batch_reviews"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_reviewer_todos/manifest.json"),
    )
    parser.add_argument(
        "--audit-packet-jsonl",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_packet.jsonl"),
    )
    args = parser.parse_args()

    manifest = export_reviewer_todos(
        args.assignment_summary_json,
        args.output_dir,
        args.review_md_dir,
        args.manifest,
        args.audit_packet_jsonl,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
