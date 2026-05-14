#!/usr/bin/env python3
"""Summarize human-audit reviewer assignments with current risk counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "A": {
        "batches": ["batch_006_PerLTQA_001-050.csv", "batch_007_PerLTQA_051-100.csv"],
        "primary_risk": "memory-anchor evidence; original fact-ledger support",
    },
    "B": {
        "batches": ["batch_008_PerLTQA_101-150.csv", "batch_009_PerLTQA_151-162.csv"],
        "primary_risk": "memory-anchor cat2/cat3/cat4 rows",
    },
    "C": {
        "batches": ["batch_002_JLongChat_051-100.csv", "batch_003_JLongChat_101-116.csv"],
        "primary_risk": "cat2/cat4/cat5; multi-hop and adversarial checks",
    },
    "D": {
        "batches": ["batch_010_deL1L2IM_001-044.csv", "batch_001_JLongChat_001-050.csv"],
        "primary_risk": "template-like questions; original-turn evidence",
    },
    "E": {
        "batches": ["batch_004_OPELA_001-050.csv", "batch_005_OPELA_051-051.csv"],
        "primary_risk": "persona/emotion evidence; source-backed cat2 row",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_nested_counts(target: Counter[str], source: dict[str, Any] | None) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if isinstance(value, dict):
            target[str(key)] += int(value.get("total", 0) or 0)


def shorten(text: str, limit: int = 96) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def next_todo_rows(batch_names: list[str], batches: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_name in batch_names:
        batch = batches.get(batch_name)
        if not batch:
            continue
        batch_path = Path(str(batch.get("path", "")))
        if not batch_path.is_file():
            continue
        with batch_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=2):
                decision = str(row.get("human_decision", "")).strip().lower()
                if decision and decision != "todo":
                    continue
                rows.append(
                    {
                        "batch": batch_name,
                        "path": str(batch_path),
                        "line": line_number,
                        "source_dataset": row.get("source_dataset", ""),
                        "sample_id": row.get("sample_id", ""),
                        "qa_idx": row.get("qa_idx", ""),
                        "category": row.get("category", ""),
                        "question_type": row.get("question_type", ""),
                        "audit_reasons": row.get("audit_reasons", ""),
                        "question_preview": shorten(str(row.get("question", ""))),
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human Audit Assignment Risk Summary",
        "",
        "Generated reviewer-routing aid. This does not make audit decisions and does not satisfy the release gate.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: {summary['rows']}",
        f"- Assigned rows: {summary['assigned_rows']}",
        f"- Completed assigned rows: {summary['completed_assigned_rows']}",
        "",
        "| Reviewer | Rows | Completed | Flagged | Cat2 | Cat4 | Cat5 | Memory-anchor | Template-like | Literal `...` | Batches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for reviewer, row in summary["reviewers"].items():
        risk = row["risk_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    reviewer,
                    str(row["rows"]),
                    str(row["completed"]),
                    str(row["flagged_rows"]),
                    str(risk["cat2"]),
                    str(risk["cat4"]),
                    str(risk["cat5"]),
                    str(risk["memory_anchor"]),
                    str(risk["template_like"]),
                    str(risk["literal_ascii_ellipsis"]),
                    ", ".join(row["batches"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next Todo Rows",
            "",
            "These rows are routing shortcuts for reviewers. Edit the CSV files, not this summary.",
            "",
            "| Reviewer | Batch | CSV line | Source | Sample | QA | Category | Reason | Question preview |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for reviewer, row in summary["reviewers"].items():
        for todo in row.get("next_todo_rows", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        reviewer,
                        str(todo["batch"]),
                        str(todo["line"]),
                        str(todo["source_dataset"]),
                        str(todo["sample_id"]),
                        str(todo["qa_idx"]),
                        str(todo["category"]),
                        str(todo["audit_reasons"]),
                        str(todo["question_preview"]).replace("|", "\\|"),
                    ]
                )
                + " |"
            )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize(progress_path: Path, flags_path: Path, next_todo_limit: int) -> dict[str, Any]:
    progress = load_json(progress_path)
    flags = load_json(flags_path)
    batches = {
        Path(str(row.get("path", ""))).name: row
        for row in progress.get("batches", [])
        if isinstance(row, dict)
    }
    flagged_by_batch = flags.get("flagged_by_batch", {}) if isinstance(flags, dict) else {}
    errors: list[str] = []
    assigned_batch_names: list[str] = []
    reviewers: dict[str, Any] = {}
    batch_file_hashes: dict[str, dict[str, str]] = {}
    for batch_name, batch in batches.items():
        batch_path = Path(str(batch.get("path", "")))
        if batch_path.is_file():
            batch_file_hashes[batch_name] = {"path": str(batch_path), "sha256": sha256_file(batch_path)}

    for reviewer, assignment in ASSIGNMENTS.items():
        rows = 0
        completed = 0
        flagged_rows = 0
        by_category: Counter[str] = Counter()
        by_provenance: Counter[str] = Counter()
        flag_counts: Counter[str] = Counter()
        for batch_name in assignment["batches"]:
            assigned_batch_names.append(batch_name)
            batch = batches.get(batch_name)
            if batch is None:
                errors.append(f"assignment {reviewer} references missing batch {batch_name}")
                continue
            rows += int(batch.get("rows", 0) or 0)
            completed += int(batch.get("completed", 0) or 0)
            add_nested_counts(by_category, batch.get("by_category"))
            add_nested_counts(by_provenance, batch.get("by_evidence_provenance"))
            flag_row = flagged_by_batch.get(batch_name, {}) if isinstance(flagged_by_batch, dict) else {}
            flagged_rows += int(flag_row.get("flagged_rows", 0) or 0)
            for flag, count in (flag_row.get("flag_counts", {}) or {}).items():
                flag_counts[str(flag)] += int(count or 0)

        reviewers[reviewer] = {
            "batches": assignment["batches"],
            "primary_risk": assignment["primary_risk"],
            "rows": rows,
            "completed": completed,
            "remaining": rows - completed,
            "flagged_rows": flagged_rows,
            "risk_counts": {
                "cat2": by_category["2"],
                "cat4": by_category["4"],
                "cat5": by_category["5"],
                "memory_anchor": by_provenance["memory_anchor_turn"],
                "template_like": flag_counts["template_like_question"],
                "literal_ascii_ellipsis": flag_counts["literal_ascii_ellipsis_in_answer_or_fact"],
                "duplicate_question_text": flag_counts["duplicate_question_text"],
            },
            "by_category": dict(sorted(by_category.items())),
            "by_evidence_provenance": dict(sorted(by_provenance.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "next_todo_rows": next_todo_rows(assignment["batches"], batches, next_todo_limit),
        }

    duplicate_batches = sorted(
        batch for batch, count in Counter(assigned_batch_names).items() if count > 1
    )
    if duplicate_batches:
        errors.append(f"duplicate assigned batches: {duplicate_batches}")
    unassigned_batches = sorted(set(batches) - set(assigned_batch_names))
    if unassigned_batches:
        errors.append(f"unassigned batches: {unassigned_batches}")
    unknown_batches = sorted(set(assigned_batch_names) - set(batches))
    if unknown_batches:
        errors.append(f"assigned batches missing from progress: {unknown_batches}")

    rows = int(progress.get("rows", 0) or 0)
    assigned_rows = sum(int(row["rows"]) for row in reviewers.values())
    if assigned_rows != rows:
        errors.append(f"assigned_rows={assigned_rows} does not match progress rows={rows}")

    return {
        "status": "completed" if not errors else "failed",
        "purpose": "reviewer_routing_aid_only_not_audit_decision",
        "input_files": {
            "batch_progress": {"path": str(progress_path), "sha256": sha256_file(progress_path)},
            "flags": {"path": str(flags_path), "sha256": sha256_file(flags_path)},
            "batch_csvs": batch_file_hashes,
        },
        "assignment_source": "scripts/summarize_locomo_human_audit_assignments.py",
        "summary_script_sha256": sha256_file(Path(__file__)),
        "rows": rows,
        "assigned_rows": assigned_rows,
        "completed_assigned_rows": sum(int(row["completed"]) for row in reviewers.values()),
        "assigned_batches": assigned_batch_names,
        "next_todo_limit": next_todo_limit,
        "reviewers": reviewers,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-progress-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_batches_progress.json"),
    )
    parser.add_argument(
        "--flags-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_flags.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_assignment_risk_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_assignment_risk_summary.md"),
    )
    parser.add_argument("--next-todo-limit", type=int, default=5)
    args = parser.parse_args()

    summary = summarize(args.batch_progress_json, args.flags_json, args.next_todo_limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
