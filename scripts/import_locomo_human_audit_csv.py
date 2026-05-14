#!/usr/bin/env python3
"""Import edited human-audit CSV decisions back into the packet JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


EDITABLE_FIELDS = [
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
]
COMPLETE_DECISIONS = {"todo", "pass", "fail", "fix", "delete"}
READ_ONLY_CSV_FIELDS = [
    "source_dataset",
    "sample_id",
    "qa_idx",
    "category",
    "question_type",
    "difficulty",
    "whether_cross_session",
    "audit_reasons",
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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("source_dataset")), str(row.get("sample_id")), int(row.get("qa_idx")))


def parse_json_or_csv_list(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def parse_json_list(value: str) -> list[Any]:
    text = value.strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("value must be a JSON list")
    return parsed


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def compact_evidence_text(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows or []:
        if row.get("missing"):
            parts.append(f"{row.get('dia_id')}: <missing>")
            continue
        text = " ".join(str(row.get("text", "")).split())
        parts.append(f"{row.get('dia_id')} {row.get('session_id', '')} {row.get('speaker', '')}: {text}")
    return "\n".join(parts)


def expected_read_only_cells(row: dict[str, Any]) -> dict[str, str]:
    return {
        "source_dataset": str(row.get("source_dataset", "")),
        "sample_id": str(row.get("sample_id", "")),
        "qa_idx": str(row.get("qa_idx", "")),
        "category": str(row.get("category", "")),
        "question_type": str(row.get("question_type", "")),
        "difficulty": str(row.get("difficulty", "")),
        "whether_cross_session": str(row.get("whether_cross_session", "")),
        "audit_reasons": "; ".join(row.get("audit_reasons", [])),
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "adversarial_answer": str(row.get("adversarial_answer") or ""),
        "evidence": json_cell(row.get("evidence", [])),
        "negative_evidence": json_cell(row.get("negative_evidence", [])),
        "evidence_detail": json_cell(row.get("evidence_detail", [])),
        "evidence_text": compact_evidence_text(row.get("evidence_text", [])),
        "negative_evidence_text": compact_evidence_text(row.get("negative_evidence_text", [])),
        "answer_facts": json_cell(row.get("answer_facts", [])),
    }


def normalize_cell(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def read_only_field_errors(
    base_row: dict[str, Any],
    csv_row: dict[str, str],
    source_label: str,
    key: tuple[str, str, int],
) -> list[str]:
    expected = expected_read_only_cells(base_row)
    errors: list[str] = []
    for field in READ_ONLY_CSV_FIELDS:
        if field not in csv_row:
            continue
        observed = normalize_cell(csv_row.get(field, ""))
        if observed != expected[field]:
            errors.append(
                f"{source_label}: read-only field {field!r} was modified for key={key}; "
                "use the matching corrected_* field instead"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any imported row still has human_decision=todo.",
    )
    args = parser.parse_args()

    base_rows = list(iter_jsonl(args.base_jsonl))
    base_by_key = {row_key(row): row for row in base_rows}

    csv_rows: dict[tuple[str, str, int], dict[str, str]] = {}
    errors: list[str] = []
    with args.decisions_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_idx, row in enumerate(reader, start=2):
            try:
                key = row_key(row)
            except (TypeError, ValueError) as exc:
                errors.append(f"{args.decisions_csv}:{line_idx}: invalid key: {exc}")
                continue
            if key in csv_rows:
                errors.append(f"{args.decisions_csv}:{line_idx}: duplicate key={key}")
            csv_rows[key] = row

    missing_in_csv = sorted(set(base_by_key) - set(csv_rows))
    extra_in_csv = sorted(set(csv_rows) - set(base_by_key))
    if missing_in_csv:
        errors.append(f"CSV missing {len(missing_in_csv)} audit rows from base JSONL")
    if extra_in_csv:
        errors.append(f"CSV has {len(extra_in_csv)} rows not present in base JSONL")

    output_rows = []
    decision_counts: Counter[str] = Counter()
    changed_rows = 0
    for base_row in base_rows:
        key = row_key(base_row)
        csv_row = csv_rows.get(key, {})
        output = dict(base_row)
        before = {field: output.get(field) for field in EDITABLE_FIELDS}
        errors.extend(read_only_field_errors(base_row, csv_row, str(args.decisions_csv), key))

        decision = str(csv_row.get("human_decision", output.get("human_decision", "todo"))).strip().lower()
        if decision not in COMPLETE_DECISIONS:
            errors.append(f"{args.decisions_csv}: invalid human_decision={decision!r} for key={key}")
        output["human_decision"] = decision

        for field in EDITABLE_FIELDS:
            if field == "human_decision":
                continue
            if field not in csv_row:
                continue
            value = str(csv_row.get(field, "")).strip()
            if not value:
                output.pop(field, None) if field.startswith("corrected_") else output.__setitem__(field, "")
                continue
            if field in {"corrected_evidence", "corrected_negative_evidence"}:
                output[field] = parse_json_or_csv_list(value)
            elif field in {"corrected_answer_facts", "corrected_evidence_detail"}:
                try:
                    output[field] = parse_json_list(value)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{args.decisions_csv}: {field} must be a JSON list for key={key}: {exc}")
                    output[field] = value
            elif field == "corrected_category":
                try:
                    output[field] = int(value)
                except ValueError:
                    errors.append(f"{args.decisions_csv}: corrected_category={value!r} is not int for key={key}")
                    output[field] = value
            else:
                output[field] = value

        if decision == "fix" and not any(output.get(field) for field in EDITABLE_FIELDS if field.startswith("corrected_")):
            errors.append(f"{args.decisions_csv}: fix decision requires at least one corrected_* field for key={key}")

        after = {field: output.get(field) for field in EDITABLE_FIELDS}
        if before != after:
            changed_rows += 1
        decision_counts[decision] += 1
        output_rows.append(output)

    if args.require_complete and decision_counts.get("todo", 0):
        errors.append(f"{decision_counts['todo']} audit rows still have human_decision=todo")

    status = "imported" if not errors else "failed"
    if status == "imported":
        write_jsonl(args.output_jsonl, output_rows)

    summary = {
        "status": status,
        "base_jsonl": str(args.base_jsonl),
        "decisions_csv": str(args.decisions_csv),
        "require_complete": args.require_complete,
        "output_jsonl": str(args.output_jsonl) if status == "imported" else None,
        "rows": len(base_rows),
        "changed_rows": changed_rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "errors": errors,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "imported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
