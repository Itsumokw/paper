#!/usr/bin/env python3
"""Summarize human-audit batch CSV completion without merging them."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from import_locomo_human_audit_csv import row_key


VALID_DECISIONS = {"todo", "pass", "fail", "fix", "delete"}
COMPLETE_DECISIONS = {"pass", "fail", "fix", "delete"}
GENERIC_PATTERNS = [
    re.compile(r"^What concrete information did\b", re.IGNORECASE),
    re.compile(r"^What two details must be combined\b", re.IGNORECASE),
    re.compile(r"^What general conversational act is shown\b", re.IGNORECASE),
    re.compile(r"^Which cited detail\b", re.IGNORECASE),
]
SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(done: int, total: int) -> float:
    return round(done / total, 6) if total else 0.0


def empty_group() -> dict[str, Any]:
    return {"total": 0, "completed": 0, "remaining": 0, "completion_ratio": 0.0}


def add_group(groups: dict[str, dict[str, Any]], key: str, completed: bool) -> None:
    group = groups.setdefault(key, empty_group())
    group["total"] += 1
    if completed:
        group["completed"] += 1
    else:
        group["remaining"] += 1
    group["completion_ratio"] = pct(group["completed"], group["total"])


def parse_audit_reasons(value: str | None) -> list[str]:
    reasons = [item.strip() for item in str(value or "").split(";") if item.strip()]
    return reasons or ["unspecified"]


def parse_json_list_cell(value: str | None) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def evidence_provenance_bucket(row: dict[str, str]) -> str:
    details = parse_json_list_cell(row.get("evidence_detail"))
    origins = sorted(
        {
            str(detail.get("source_origin", "missing"))
            for detail in details
            if isinstance(detail, dict)
        }
    )
    if origins:
        return "+".join(origins)
    if parse_json_list_cell(row.get("negative_evidence")):
        return "negative_only"
    return "none"


def dia_sessions(ids: list[Any]) -> set[str]:
    sessions = set()
    for dia_id in ids:
        match = re.match(r"^D(\d+):", str(dia_id))
        if match:
            sessions.add(match.group(1))
    return sessions


def is_truthy_cell(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def is_generic_question(question: str) -> bool:
    return any(pattern.search(question.strip()) for pattern in GENERIC_PATTERNS)


def has_literal_ascii_ellipsis(row: dict[str, str]) -> bool:
    fields: list[str] = [
        str(row.get("answer", "")),
        str(row.get("adversarial_answer", "")),
    ]
    for fact in parse_json_list_cell(row.get("answer_facts")):
        if isinstance(fact, dict):
            fields.append(str(fact.get("fact", "")))
    return any("..." in field for field in fields)


def row_reviewer_flags(row: dict[str, str], question_counts: Counter[str]) -> list[str]:
    flags: list[str] = []
    try:
        category = int(row.get("category", "0"))
    except ValueError:
        category = 0
    question = str(row.get("question", ""))
    provenance = evidence_provenance_bucket(row)
    evidence = parse_json_list_cell(row.get("evidence"))
    negative_evidence = parse_json_list_cell(row.get("negative_evidence"))
    answer_facts = parse_json_list_cell(row.get("answer_facts"))

    if category == 5:
        flags.append("cat5_adversarial")
        if not negative_evidence:
            flags.append("cat5_missing_negative_evidence")
    if category == 2:
        flags.append("cat2_multi_hop")
    if category == 4:
        flags.append("cat4_reasoning")
    if "memory_anchor_turn" in provenance:
        flags.append("memory_anchor_evidence")
    if "synthetic" in provenance:
        flags.append("synthetic_adjacent_evidence")
    if question_counts[question] > 1:
        flags.append("duplicate_question_text")
    if is_generic_question(question):
        flags.append("template_like_question")
    if has_literal_ascii_ellipsis(row):
        flags.append("literal_ascii_ellipsis_in_answer_or_fact")
    if category != 5 and not evidence:
        flags.append("answerable_missing_evidence")
    if category == 5 and evidence:
        flags.append("cat5_has_positive_evidence")
    if is_truthy_cell(row.get("whether_cross_session")) and len(dia_sessions(evidence)) < 2:
        flags.append("cross_session_label_single_evidence_session")
    if category != 5 and not answer_facts:
        flags.append("answerable_missing_answer_facts")
    return sorted(flags)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def render_json_list(value: str | None) -> str:
    parsed = parse_json_list_cell(value)
    if parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return markdown_escape(value)


def load_fact_ledger_index(sidecar_root: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if sidecar_root is None:
        return {}
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_dataset, artifact in SOURCE_TO_ARTIFACT.items():
        ledger_path = sidecar_root / artifact / f"{artifact}_fact_ledger.jsonl"
        if not ledger_path.exists():
            continue
        for row in iter_jsonl(ledger_path):
            fact_id = str(row.get("fact_id") or row.get("source_fact_id") or "")
            if not fact_id:
                continue
            sample_id = str(row.get("sample_id") or "")
            index[(source_dataset, sample_id, fact_id)] = row
            index[(source_dataset, "", fact_id)] = row
    return index


def fact_ledger_support(
    row: dict[str, str],
    fact_ledger_index: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not fact_ledger_index:
        return []
    source_dataset = str(row.get("source_dataset") or "")
    sample_id = str(row.get("sample_id") or "")
    support_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer_fact in parse_json_list_cell(row.get("answer_facts")):
        if not isinstance(answer_fact, dict):
            continue
        fact_id = str(answer_fact.get("source_fact_id") or "")
        if not fact_id or fact_id in seen:
            continue
        seen.add(fact_id)
        ledger_row = fact_ledger_index.get((source_dataset, sample_id, fact_id)) or fact_ledger_index.get(
            (source_dataset, "", fact_id)
        )
        if ledger_row:
            support_rows.append(
                {
                    "source_fact_id": fact_id,
                    "lookup_status": "found",
                    "source_type": ledger_row.get("source_type"),
                    "source_id": ledger_row.get("source_id"),
                    "field": ledger_row.get("field"),
                    "source_text": ledger_row.get("source_text"),
                }
            )
        else:
            support_rows.append(
                {
                    "source_fact_id": fact_id,
                    "lookup_status": "missing_from_fact_ledger_index",
                }
            )
    return support_rows


def write_batch_review_markdown(
    path: Path,
    batch_path: Path,
    rows: list[dict[str, str]],
    question_counts: Counter[str],
    fact_ledger_index: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> None:
    lines = [
        f"# Human Audit Review: {batch_path.name}",
        "",
        "Use this file for reading evidence/context. Record decisions in the matching CSV file, not here.",
        "",
        "Decision values: `pass`, `fix`, `delete`, `fail`.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        flags = row_reviewer_flags(row, question_counts)
        lines.extend(
            [
                f"## {idx}. {row.get('source_dataset', '')} / {row.get('sample_id', '')} / QA {row.get('qa_idx', '')}",
                "",
                f"- Category: {row.get('category', '')} ({row.get('question_type', '')})",
                f"- Difficulty: {row.get('difficulty', '')}",
                f"- Cross-session: {row.get('whether_cross_session', '')}",
                f"- Audit reasons: {markdown_escape(row.get('audit_reasons', ''))}",
                f"- Reviewer flags: {', '.join(flags) if flags else 'none'}",
                f"- Current decision: `{markdown_escape(row.get('human_decision', 'todo'))}`",
                "",
                f"Question: {markdown_escape(row.get('question', ''))}",
                "",
            ]
        )
        if str(row.get("category", "")) == "5":
            lines.append(f"Adversarial unsupported answer: {markdown_escape(row.get('adversarial_answer', ''))}")
        else:
            lines.append(f"Answer: {markdown_escape(row.get('answer', ''))}")
        lines.extend(["", "Answer facts:", "", f"```json\n{render_json_list(row.get('answer_facts'))}\n```", ""])
        source_support = fact_ledger_support(row, fact_ledger_index or {})
        if source_support:
            lines.extend(
                [
                    "Source fact ledger support:",
                    "",
                    f"```json\n{json.dumps(source_support, ensure_ascii=False)}\n```",
                    "",
                ]
            )
        lines.extend(["Evidence detail:", "", f"```json\n{render_json_list(row.get('evidence_detail'))}\n```", ""])
        if row.get("evidence_text"):
            lines.extend(["Evidence text:", "", markdown_escape(row.get("evidence_text")), ""])
        if row.get("negative_evidence_text"):
            lines.extend(["Negative evidence text:", "", markdown_escape(row.get("negative_evidence_text")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human Audit Batch Progress",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: {summary['rows']}",
        f"- Completed: {summary['completed']}",
        f"- Remaining: {summary['remaining']}",
        f"- Completion ratio: {summary['completion_ratio']:.2%}",
    ]
    if summary.get("review_md_dir"):
        lines.append(f"- Review Markdown dir: `{summary['review_md_dir']}`")
    lines.extend(["", "## Decisions", ""])
    for decision, count in summary["decision_counts"].items():
        lines.append(f"- `{decision}`: {count}")
    lines.extend(["", "## Batches", ""])
    for batch in summary["batches"]:
        lines.append(
            f"- `{Path(batch['path']).name}`: {batch['completed']}/{batch['rows']} "
            f"({batch['completion_ratio']:.2%})"
        )
        category_counts = ", ".join(
            f"cat{category}={stats['total']}"
            for category, stats in batch.get("by_category", {}).items()
        )
        provenance_counts = ", ".join(
            f"{provenance}={stats['total']}"
            for provenance, stats in batch.get("by_evidence_provenance", {}).items()
        )
        if category_counts:
            lines.append(f"  - Categories: {category_counts}")
        if provenance_counts:
            lines.append(f"  - Evidence provenance: {provenance_counts}")
        if summary.get("review_md_dir"):
            review_path = Path(summary["review_md_dir"]) / f"{Path(batch['path']).stem}.md"
            lines.append(f"  - Review Markdown: `{review_path}`")
    audit_reason_rows = [
        [
            reason,
            stats["total"],
            stats["completed"],
            stats["remaining"],
            f"{stats['completion_ratio']:.2%}",
        ]
        for reason, stats in summary["by_audit_reason"].items()
    ]
    provenance_rows = [
        [
            provenance,
            stats["total"],
            stats["completed"],
            stats["remaining"],
            f"{stats['completion_ratio']:.2%}",
        ]
        for provenance, stats in summary["by_evidence_provenance"].items()
    ]
    lines.extend(
        [
            "",
            "## By Audit Reason",
            "",
            markdown_table(["Audit Reason", "Total", "Completed", "Remaining", "Done"], audit_reason_rows),
            "",
            "## By Evidence Provenance",
            "",
            markdown_table(["Evidence Provenance", "Total", "Completed", "Remaining", "Done"], provenance_rows),
        ]
    )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in summary["errors"][:50]:
            lines.append(f"- {error}")
    if summary["remaining_examples"]:
        lines.extend(["", "## Remaining Examples", ""])
        for row in summary["remaining_examples"]:
            lines.append(
                f"- {row['path']}:{row['line']}: {row['source_dataset']} / "
                f"{row['sample_id']} / QA {row['qa_idx']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--output-review-md-dir",
        type=Path,
        default=None,
        help="Optional directory for per-batch Markdown review files generated from existing batch CSVs.",
    )
    parser.add_argument("--base-jsonl", type=Path, default=None)
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=None,
        help="Optional sidecar root used to add fact-ledger source_text to generated batch review Markdown.",
    )
    parser.add_argument("--pattern", default="batch_*.csv")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.pattern))
    errors: list[str] = []
    if not paths:
        errors.append(f"No batch CSV files matched {args.input_dir / args.pattern}")

    expected_keys = set()
    if args.base_jsonl:
        expected_keys = {row_key(row) for row in iter_jsonl(args.base_jsonl)}

    question_counts: Counter[str] = Counter()
    fact_ledger_index = load_fact_ledger_index(args.sidecar_root) if args.output_review_md_dir else {}
    if args.output_review_md_dir:
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                question_counts.update(str(row.get("question", "")) for row in reader)

    seen_keys: set[tuple[str, str, int]] = set()
    decision_counts: Counter[str] = Counter()
    by_source: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    by_audit_reason: dict[str, dict[str, Any]] = {}
    by_evidence_provenance: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    remaining_examples: list[dict[str, Any]] = []
    invalid_examples: list[str] = []

    rows = 0
    completed = 0
    for path in paths:
        batch_rows = 0
        batch_completed = 0
        batch_decision_counts: Counter[str] = Counter()
        batch_by_category: dict[str, dict[str, Any]] = {}
        batch_by_audit_reason: dict[str, dict[str, Any]] = {}
        batch_by_evidence_provenance: dict[str, dict[str, Any]] = {}
        batch_review_rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_idx, row in enumerate(reader, start=2):
                batch_review_rows.append(row)
                rows += 1
                batch_rows += 1
                try:
                    key = row_key(row)
                except (TypeError, ValueError) as exc:
                    errors.append(f"{path}:{line_idx}: invalid key: {exc}")
                    continue
                if key in seen_keys:
                    errors.append(f"{path}:{line_idx}: duplicate key={key}")
                seen_keys.add(key)

                decision = str(row.get("human_decision", "todo")).strip().lower()
                if decision not in VALID_DECISIONS:
                    errors.append(f"{path}:{line_idx}: invalid human_decision={decision!r}")
                    if len(invalid_examples) < 20:
                        invalid_examples.append(f"{path}:{line_idx}: {decision!r}")
                is_completed = decision in COMPLETE_DECISIONS
                if is_completed:
                    completed += 1
                    batch_completed += 1
                elif len(remaining_examples) < 20:
                    remaining_examples.append(
                        {
                            "path": str(path),
                            "line": line_idx,
                            "source_dataset": row.get("source_dataset", ""),
                            "sample_id": row.get("sample_id", ""),
                            "qa_idx": row.get("qa_idx", ""),
                            "human_decision": decision,
                        }
                    )

                decision_counts[decision] += 1
                batch_decision_counts[decision] += 1
                add_group(by_source, str(row.get("source_dataset", "")), is_completed)
                add_group(by_category, str(row.get("category", "")), is_completed)
                add_group(batch_by_category, str(row.get("category", "")), is_completed)
                provenance = evidence_provenance_bucket(row)
                add_group(by_evidence_provenance, provenance, is_completed)
                add_group(batch_by_evidence_provenance, provenance, is_completed)
                for reason in parse_audit_reasons(row.get("audit_reasons")):
                    add_group(by_audit_reason, reason, is_completed)
                    add_group(batch_by_audit_reason, reason, is_completed)
        batches.append(
            {
                "path": str(path),
                "rows": batch_rows,
                "completed": batch_completed,
                "remaining": batch_rows - batch_completed,
                "completion_ratio": pct(batch_completed, batch_rows),
                "decision_counts": dict(sorted(batch_decision_counts.items())),
                "by_category": dict(sorted(batch_by_category.items())),
                "by_audit_reason": dict(sorted(batch_by_audit_reason.items())),
                "by_evidence_provenance": dict(sorted(batch_by_evidence_provenance.items())),
            }
        )
        if args.output_review_md_dir:
            write_batch_review_markdown(
                args.output_review_md_dir / f"{path.stem}.md",
                path,
                batch_review_rows,
                question_counts,
                fact_ledger_index,
            )

    if expected_keys:
        missing = sorted(expected_keys - seen_keys)
        extra = sorted(seen_keys - expected_keys)
        if missing:
            errors.append(f"Batch CSVs are missing {len(missing)} audit rows from base JSONL")
        if extra:
            errors.append(f"Batch CSVs have {len(extra)} rows not present in base JSONL")

    remaining = rows - completed
    if errors or rows == 0:
        status = "failed"
    elif remaining == 0:
        status = "completed"
    else:
        status = "incomplete"
    summary = {
        "status": status,
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "base_jsonl": str(args.base_jsonl) if args.base_jsonl else None,
        "rows": rows,
        "completed": completed,
        "remaining": remaining,
        "completion_ratio": pct(completed, rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_audit_reason": dict(sorted(by_audit_reason.items())),
        "by_evidence_provenance": dict(sorted(by_evidence_provenance.items())),
        "batches": batches,
        "review_md_dir": str(args.output_review_md_dir) if args.output_review_md_dir else None,
        "review_md_fact_ledger_support": bool(args.output_review_md_dir and fact_ledger_index),
        "remaining_examples": remaining_examples,
        "invalid_examples": invalid_examples,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status in {"completed", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
