#!/usr/bin/env python3
"""Create reviewer-facing heuristic flags for human-audit rows.

This script does not make or modify human-audit decisions. It only highlights
rows that may deserve earlier manual attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from import_locomo_human_audit_csv import row_key


GENERIC_PATTERNS = [
    re.compile(r"^What concrete information did\b", re.IGNORECASE),
    re.compile(r"^What two details must be combined\b", re.IGNORECASE),
    re.compile(r"^What general conversational act is shown\b", re.IGNORECASE),
    re.compile(r"^Which cited detail\b", re.IGNORECASE),
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_batch_index(input_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(input_dir.glob("batch_*.csv")):
        review_md = path.parent.parent / "human_audit_batch_reviews" / f"{path.stem}.md"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_no, row in enumerate(reader, start=2):
                try:
                    key = row_key(row)
                except (TypeError, ValueError):
                    continue
                index[key] = {
                    "csv": str(path),
                    "line": line_no,
                    "review_md": str(review_md),
                }
    return index


def dia_sessions(ids: list[Any]) -> set[str]:
    sessions = set()
    for dia_id in ids:
        match = re.match(r"^D(\d+):", str(dia_id))
        if match:
            sessions.add(match.group(1))
    return sessions


def provenance_bucket(row: dict[str, Any]) -> str:
    origins = sorted(
        {
            str(detail.get("source_origin", "missing"))
            for detail in row.get("evidence_detail", []) or []
            if isinstance(detail, dict)
        }
    )
    if origins:
        return "+".join(origins)
    if row.get("negative_evidence"):
        return "negative_only"
    return "none"


def is_generic_question(question: str) -> bool:
    return any(pattern.search(question.strip()) for pattern in GENERIC_PATTERNS)


def has_literal_ascii_ellipsis(row: dict[str, Any]) -> bool:
    """Flag literal ASCII ellipses in generated answer-facing fields.

    The flag is intentionally conservative and reviewer-facing only: `...` may
    be source punctuation, but in this artifact it can also indicate a
    shortened answer/fact that needs manual review.
    """

    fields: list[Any] = [
        row.get("answer", ""),
        row.get("adversarial_answer", ""),
    ]
    fields.extend(fact.get("fact", "") for fact in row.get("answer_facts", []) or [] if isinstance(fact, dict))
    return any("..." in str(value) for value in fields)


def row_flags(row: dict[str, Any], question_counts: Counter[str]) -> list[str]:
    flags: list[str] = []
    category = int(row.get("category", 0))
    question = str(row.get("question", ""))
    provenance = provenance_bucket(row)
    evidence = row.get("evidence", []) or []
    negative_evidence = row.get("negative_evidence", []) or []

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
    if category != 5 and not evidence:
        flags.append("answerable_missing_evidence")
    if category == 5 and evidence:
        flags.append("cat5_has_positive_evidence")
    if has_literal_ascii_ellipsis(row):
        flags.append("literal_ascii_ellipsis_in_answer_or_fact")
    if bool(row.get("whether_cross_session")) and len(dia_sessions(evidence)) < 2:
        flags.append("cross_session_label_single_evidence_session")
    if category != 5 and not row.get("answer_facts"):
        flags.append("answerable_missing_answer_facts")
    return flags


def flag_score(flags: list[str]) -> int:
    weights = {
        "answerable_missing_evidence": 100,
        "answerable_missing_answer_facts": 100,
        "cat5_missing_negative_evidence": 100,
        "cat5_has_positive_evidence": 100,
        "cross_session_label_single_evidence_session": 60,
        "memory_anchor_evidence": 35,
        "synthetic_adjacent_evidence": 35,
        "cat5_adversarial": 30,
        "cat2_multi_hop": 20,
        "cat4_reasoning": 20,
        "duplicate_question_text": 15,
        "literal_ascii_ellipsis_in_answer_or_fact": 12,
        "template_like_question": 10,
    }
    return sum(weights.get(flag, 1) for flag in flags)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human Audit Heuristic Flags",
        "",
        "This is a reviewer aid only. It does not make audit decisions and must not be used to auto-pass rows.",
        "",
        f"- Rows: {summary['rows']}",
        f"- Flagged rows: {summary['flagged_rows']}",
        f"- Unflagged rows: {summary['unflagged_rows']}",
        "",
        "## Flag Counts",
        "",
        markdown_table(["Flag", "Rows"], [[k, v] for k, v in summary["flag_counts"].items()]),
        "",
        "## By Source",
        "",
        markdown_table(
            ["Source", "Flagged rows"],
            [[source, count] for source, count in summary["flagged_by_source"].items()],
        ),
        "",
        "## By Batch",
        "",
        markdown_table(
            ["Batch", "Flagged rows", "Top flags"],
            [
                [
                    batch,
                    stats["flagged_rows"],
                    ", ".join(f"{flag}={count}" for flag, count in stats["flag_counts"].items()),
                ]
                for batch, stats in summary["flagged_by_batch"].items()
            ],
        ),
        "",
        "## Top Flagged Rows",
        "",
    ]
    top_rows = []
    for row in summary["top_flagged_rows"]:
        top_rows.append(
            [
                row["score"],
                row["source_dataset"],
                row["sample_id"],
                row["qa_idx"],
                row["category"],
                ", ".join(row["flags"]),
                row.get("csv", ""),
                row.get("review_md", ""),
                row["question"],
            ]
        )
    lines.append(
        markdown_table(
            ["Score", "Source", "Sample", "QA", "Cat", "Flags", "CSV", "Review MD", "Question"],
            top_rows,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input_jsonl))
    question_counts = Counter(str(row.get("question", "")) for row in rows)
    batch_index = load_batch_index(args.batch_dir)

    flagged_rows: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    batch_flag_counts: dict[str, Counter[str]] = defaultdict(Counter)
    batch_flagged_rows: Counter[str] = Counter()
    for row in rows:
        flags = row_flags(row, question_counts)
        if not flags:
            continue
        key = row_key(row)
        location = batch_index.get(key, {})
        batch_name = Path(str(location.get("csv", "unknown_batch"))).name
        score = flag_score(flags)
        flag_counts.update(flags)
        by_source[str(row.get("source_dataset"))] += 1
        batch_flag_counts[batch_name].update(flags)
        batch_flagged_rows[batch_name] += 1
        flagged_rows.append(
            {
                "score": score,
                "flags": sorted(flags),
                "source_dataset": row.get("source_dataset"),
                "sample_id": row.get("sample_id"),
                "qa_idx": row.get("qa_idx"),
                "category": row.get("category"),
                "question": row.get("question"),
                "csv": location.get("csv"),
                "csv_line": location.get("line"),
                "batch": batch_name,
                "review_md": location.get("review_md"),
            }
        )

    flagged_rows.sort(key=lambda item: (-int(item["score"]), str(item["source_dataset"]), str(item["sample_id"]), int(item["qa_idx"])))
    summary = {
        "status": "completed",
        "purpose": "reviewer_aid_only_not_audit_decision",
        "input_jsonl": str(args.input_jsonl),
        "batch_dir": str(args.batch_dir),
        "rows": len(rows),
        "flagged_rows": len(flagged_rows),
        "unflagged_rows": len(rows) - len(flagged_rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "flagged_by_source": dict(sorted(by_source.items())),
        "flagged_by_batch": {
            batch: {
                "flagged_rows": batch_flagged_rows[batch],
                "flag_counts": dict(sorted(batch_flag_counts[batch].items())),
            }
            for batch in sorted(batch_flagged_rows)
        },
        "top_flagged_rows": flagged_rows[: args.top_n],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
