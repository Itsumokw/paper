#!/usr/bin/env python3
"""Summarize human-audit completion progress without modifying audit rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMPLETE_DECISIONS = {"pass", "fail", "fix", "delete"}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(done: int, total: int) -> float:
    return round(done / total, 4) if total else 0.0


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def evidence_provenance_bucket(row: dict[str, Any]) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input_jsonl))
    decision_counts: Counter[str] = Counter()
    source_totals: Counter[str] = Counter()
    source_done: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_done: Counter[str] = Counter()
    source_category_totals: dict[str, Counter[str]] = defaultdict(Counter)
    source_category_done: dict[str, Counter[str]] = defaultdict(Counter)
    audit_reason_totals: Counter[str] = Counter()
    audit_reason_done: Counter[str] = Counter()
    evidence_provenance_totals: Counter[str] = Counter()
    evidence_provenance_done: Counter[str] = Counter()
    remaining_examples: list[dict[str, Any]] = []

    for row in rows:
        decision = str(row.get("human_decision", "todo")).strip().lower()
        source = str(row.get("source_dataset", ""))
        category = str(row.get("category", ""))
        evidence_provenance = evidence_provenance_bucket(row)
        audit_reasons = [str(reason) for reason in row.get("audit_reasons", [])] or ["unspecified"]
        done = decision in COMPLETE_DECISIONS

        decision_counts[decision] += 1
        source_totals[source] += 1
        category_totals[category] += 1
        source_category_totals[source][category] += 1
        evidence_provenance_totals[evidence_provenance] += 1
        for reason in audit_reasons:
            audit_reason_totals[reason] += 1
        if done:
            source_done[source] += 1
            category_done[category] += 1
            source_category_done[source][category] += 1
            evidence_provenance_done[evidence_provenance] += 1
            for reason in audit_reasons:
                audit_reason_done[reason] += 1
        elif len(remaining_examples) < 20:
            remaining_examples.append(
                {
                    "source_dataset": source,
                    "sample_id": row.get("sample_id"),
                    "qa_idx": row.get("qa_idx"),
                    "category": row.get("category"),
                    "audit_reasons": row.get("audit_reasons", []),
                    "evidence_provenance": evidence_provenance,
                    "question": row.get("question"),
                }
            )

    total = len(rows)
    completed = sum(count for decision, count in decision_counts.items() if decision in COMPLETE_DECISIONS)
    remaining = total - completed

    by_source = {
        source: {
            "total": source_totals[source],
            "completed": source_done[source],
            "remaining": source_totals[source] - source_done[source],
            "completion_ratio": pct(source_done[source], source_totals[source]),
        }
        for source in sorted(source_totals)
    }
    by_category = {
        category: {
            "total": category_totals[category],
            "completed": category_done[category],
            "remaining": category_totals[category] - category_done[category],
            "completion_ratio": pct(category_done[category], category_totals[category]),
        }
        for category in sorted(category_totals, key=lambda item: int(item) if item.isdigit() else item)
    }
    by_source_category = {
        source: {
            category: {
                "total": source_category_totals[source][category],
                "completed": source_category_done[source][category],
                "remaining": source_category_totals[source][category] - source_category_done[source][category],
                "completion_ratio": pct(
                    source_category_done[source][category],
                    source_category_totals[source][category],
                ),
            }
            for category in sorted(
                source_category_totals[source],
                key=lambda item: int(item) if item.isdigit() else item,
            )
        }
        for source in sorted(source_category_totals)
    }
    by_audit_reason = {
        reason: {
            "total": audit_reason_totals[reason],
            "completed": audit_reason_done[reason],
            "remaining": audit_reason_totals[reason] - audit_reason_done[reason],
            "completion_ratio": pct(audit_reason_done[reason], audit_reason_totals[reason]),
        }
        for reason in sorted(audit_reason_totals)
    }
    by_evidence_provenance = {
        provenance: {
            "total": evidence_provenance_totals[provenance],
            "completed": evidence_provenance_done[provenance],
            "remaining": evidence_provenance_totals[provenance] - evidence_provenance_done[provenance],
            "completion_ratio": pct(evidence_provenance_done[provenance], evidence_provenance_totals[provenance]),
        }
        for provenance in sorted(evidence_provenance_totals)
    }

    report = {
        "status": "completed" if total and remaining == 0 else "incomplete",
        "input_jsonl": str(args.input_jsonl),
        "rows": total,
        "completed": completed,
        "remaining": remaining,
        "completion_ratio": pct(completed, total),
        "decision_counts": dict(sorted(decision_counts.items())),
        "by_source": by_source,
        "by_category": by_category,
        "by_source_category": by_source_category,
        "by_audit_reason": by_audit_reason,
        "by_evidence_provenance": by_evidence_provenance,
        "remaining_examples": remaining_examples,
        "next_action": (
            "run validate/apply audit scripts"
            if remaining == 0
            else "fill human_decision for remaining rows in the audit packet or CSV"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.output_md:
        source_rows = [
            [
                source,
                stats["total"],
                stats["completed"],
                stats["remaining"],
                f"{stats['completion_ratio']:.1%}",
            ]
            for source, stats in by_source.items()
        ]
        category_rows = [
            [
                category,
                stats["total"],
                stats["completed"],
                stats["remaining"],
                f"{stats['completion_ratio']:.1%}",
            ]
            for category, stats in by_category.items()
        ]
        audit_reason_rows = [
            [
                reason,
                stats["total"],
                stats["completed"],
                stats["remaining"],
                f"{stats['completion_ratio']:.1%}",
            ]
            for reason, stats in by_audit_reason.items()
        ]
        provenance_rows = [
            [
                provenance,
                stats["total"],
                stats["completed"],
                stats["remaining"],
                f"{stats['completion_ratio']:.1%}",
            ]
            for provenance, stats in by_evidence_provenance.items()
        ]
        md = [
            "# Human Audit Progress",
            "",
            f"- Input: `{args.input_jsonl}`",
            f"- Status: `{report['status']}`",
            f"- Rows: {total}",
            f"- Completed: {completed}",
            f"- Remaining: {remaining}",
            f"- Completion ratio: {report['completion_ratio']:.1%}",
            f"- Decision counts: `{json.dumps(report['decision_counts'], ensure_ascii=False)}`",
            "",
            "## By Source",
            "",
            markdown_table(["Source", "Total", "Completed", "Remaining", "Done"], source_rows),
            "",
            "## By Category",
            "",
            markdown_table(["Category", "Total", "Completed", "Remaining", "Done"], category_rows),
            "",
            "## By Audit Reason",
            "",
            markdown_table(["Audit Reason", "Total", "Completed", "Remaining", "Done"], audit_reason_rows),
            "",
            "## By Evidence Provenance",
            "",
            markdown_table(["Evidence Provenance", "Total", "Completed", "Remaining", "Done"], provenance_rows),
            "",
            "## Next Action",
            "",
            report["next_action"],
            "",
        ]
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
