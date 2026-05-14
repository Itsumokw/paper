#!/usr/bin/env python3
"""Merge edited human-audit batch CSVs back into the packet JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from import_locomo_human_audit_csv import (
    COMPLETE_DECISIONS,
    EDITABLE_FIELDS,
    iter_jsonl,
    parse_json_or_csv_list,
    read_only_field_errors,
    row_key,
    write_jsonl,
)


def load_batch_rows(input_dir: Path, pattern: str) -> tuple[dict[tuple[str, str, int], dict[str, str]], list[dict[str, Any]], list[str]]:
    rows_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    batches: list[dict[str, Any]] = []
    errors: list[str] = []
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        errors.append(f"No batch CSV files matched {input_dir / pattern}")
        return rows_by_key, batches, errors

    for path in paths:
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_idx, row in enumerate(reader, start=2):
                row_count += 1
                try:
                    key = row_key(row)
                except (TypeError, ValueError) as exc:
                    errors.append(f"{path}:{line_idx}: invalid key: {exc}")
                    continue
                if key in rows_by_key:
                    errors.append(f"{path}:{line_idx}: duplicate key={key}")
                    continue
                rows_by_key[key] = row
        batches.append({"path": str(path), "rows": row_count})
    return rows_by_key, batches, errors


def merge_rows(
    base_rows: list[dict[str, Any]],
    csv_rows: dict[tuple[str, str, int], dict[str, str]],
    source_label: str,
) -> tuple[list[dict[str, Any]], Counter[str], int, list[str]]:
    errors: list[str] = []
    base_by_key = {row_key(row): row for row in base_rows}

    missing_in_csv = sorted(set(base_by_key) - set(csv_rows))
    extra_in_csv = sorted(set(csv_rows) - set(base_by_key))
    if missing_in_csv:
        errors.append(f"Batch CSVs are missing {len(missing_in_csv)} audit rows from base JSONL")
    if extra_in_csv:
        errors.append(f"Batch CSVs have {len(extra_in_csv)} rows not present in base JSONL")

    output_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    changed_rows = 0
    for base_row in base_rows:
        key = row_key(base_row)
        csv_row = csv_rows.get(key, {})
        output = dict(base_row)
        before = {field: output.get(field) for field in EDITABLE_FIELDS}
        errors.extend(read_only_field_errors(base_row, csv_row, source_label, key))

        decision = str(csv_row.get("human_decision", output.get("human_decision", "todo"))).strip().lower()
        if decision not in COMPLETE_DECISIONS:
            errors.append(f"{source_label}: invalid human_decision={decision!r} for key={key}")
        output["human_decision"] = decision

        for field in EDITABLE_FIELDS:
            if field == "human_decision" or field not in csv_row:
                continue
            value = str(csv_row.get(field, "")).strip()
            if not value:
                if field.startswith("corrected_"):
                    output.pop(field, None)
                else:
                    output[field] = ""
                continue
            if field in {"corrected_evidence", "corrected_negative_evidence"}:
                output[field] = parse_json_or_csv_list(value)
            elif field in {"corrected_answer_facts", "corrected_evidence_detail"}:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    errors.append(f"{source_label}: {field} must be a JSON list for key={key}: {exc}")
                    output[field] = value
                    continue
                if not isinstance(parsed, list):
                    errors.append(f"{source_label}: {field} must be a JSON list for key={key}")
                    output[field] = value
                else:
                    output[field] = parsed
            elif field == "corrected_category":
                try:
                    output[field] = int(value)
                except ValueError:
                    errors.append(f"{source_label}: corrected_category={value!r} is not int for key={key}")
                    output[field] = value
            else:
                output[field] = value

        if decision == "fix" and not any(output.get(field) for field in EDITABLE_FIELDS if field.startswith("corrected_")):
            errors.append(f"{source_label}: fix decision requires at least one corrected_* field for key={key}")

        after = {field: output.get(field) for field in EDITABLE_FIELDS}
        if before != after:
            changed_rows += 1
        decision_counts[decision] += 1
        output_rows.append(output)

    return output_rows, decision_counts, changed_rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--pattern", default="batch_*.csv")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any merged row still has human_decision=todo.",
    )
    args = parser.parse_args()

    base_rows = list(iter_jsonl(args.base_jsonl))
    csv_rows, batches, load_errors = load_batch_rows(args.input_dir, args.pattern)
    output_rows, decision_counts, changed_rows, merge_errors = merge_rows(
        base_rows,
        csv_rows,
        f"{args.input_dir}/{args.pattern}",
    )
    errors = load_errors + merge_errors
    if args.require_complete and decision_counts.get("todo", 0):
        errors.append(f"{decision_counts['todo']} audit rows still have human_decision=todo")

    status = "merged" if not errors else "failed"
    if status == "merged":
        write_jsonl(args.output_jsonl, output_rows)

    summary = {
        "status": status,
        "base_jsonl": str(args.base_jsonl),
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "require_complete": args.require_complete,
        "output_jsonl": str(args.output_jsonl) if status == "merged" else None,
        "base_rows": len(base_rows),
        "batch_rows": len(csv_rows),
        "batches": batches,
        "changed_rows": changed_rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "errors": errors,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "merged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
