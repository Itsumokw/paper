#!/usr/bin/env python3
"""Validate completed human-audit decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


COMPLETE_DECISIONS = {"pass", "fail", "fix", "delete"}
CORRECTED_FIELDS = [
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
SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}
ORIGINAL_SOURCE_TYPES = {"original_event", "original_memory", "original_persona", "original_turn"}
ALLOWED_ADVERSARIAL_REASONS = {"unsupported_fact", "time_swap", "entity_swap"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_trace_files_sha256(sidecar_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        {
            *sidecar_root.glob("*/*_fact_ledger.jsonl"),
            *sidecar_root.glob("*/*_provenance.jsonl"),
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def corrected_values(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in CORRECTED_FIELDS if has_value(row.get(field))}


def parse_category(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        category = int(value)
    except (TypeError, ValueError):
        return None
    return category if 1 <= category <= 5 else None


def parse_json_list(value: Any) -> list[Any] | None:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def normalize_id_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = parse_json_list(value)
        if parsed is not None:
            return [str(item) for item in parsed]
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def validate_adversarial_reason(path: Path, lineno: int, value: Any, field: str = "adversarial_reason") -> list[str]:
    if not has_value(value):
        return []
    reason = str(value)
    if reason in ALLOWED_ADVERSARIAL_REASONS:
        return []
    return [
        f"{path}:{lineno}: {field}={reason!r} must be one of {sorted(ALLOWED_ADVERSARIAL_REASONS)}"
    ]


def iter_sidecar_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_trace_indexes(sidecar_root: Path | None) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if sidecar_root is None:
        return {}, {}
    facts: dict[str, dict[str, Any]] = {}
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in SOURCE_TO_ARTIFACT.values():
        fact_path = sidecar_root / artifact / f"{artifact}_fact_ledger.jsonl"
        provenance_path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        for row in iter_sidecar_jsonl(fact_path):
            facts[str(row.get("fact_id"))] = row
        for row in iter_sidecar_jsonl(provenance_path):
            provenance[(str(row.get("sample_id")), str(row.get("dia_id")))] = row
    return facts, provenance


def validate_corrected_trace(
    path: Path,
    lineno: int,
    row: dict[str, Any],
    facts: dict[str, dict[str, Any]],
    provenance: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    sample_id = str(row.get("sample_id"))
    answer_facts = parse_json_list(row.get("corrected_answer_facts"))
    evidence_detail = parse_json_list(row.get("corrected_evidence_detail"))
    if answer_facts is None:
        errors.append(f"{path}:{lineno}: corrected_answer_facts must be a JSON list")
        answer_facts = []
    if evidence_detail is None:
        errors.append(f"{path}:{lineno}: corrected_evidence_detail must be a JSON list")
        evidence_detail = []
    if not answer_facts:
        errors.append(f"{path}:{lineno}: answer/evidence/category fix requires corrected_answer_facts")
    if not evidence_detail:
        errors.append(f"{path}:{lineno}: answer/evidence/category fix requires corrected_evidence_detail")

    corrected_evidence = row.get("corrected_evidence") if has_value(row.get("corrected_evidence")) else row.get("evidence", [])
    corrected_evidence_ids = set(normalize_id_list(corrected_evidence))
    detail_evidence_ids: set[str] = set()
    supported_fact_ids: set[str] = set()
    for detail_idx, detail in enumerate(evidence_detail or []):
        if not isinstance(detail, dict):
            errors.append(f"{path}:{lineno}: corrected_evidence_detail[{detail_idx}] must be object")
            continue
        dia_id = str(detail.get("dia_id"))
        detail_evidence_ids.add(dia_id)
        provenance_row = provenance.get((sample_id, dia_id)) if provenance else None
        if provenance and provenance_row is None:
            errors.append(f"{path}:{lineno}: corrected_evidence_detail dia_id={dia_id!r} missing provenance")
        elif provenance_row is not None and detail.get("source_origin") != provenance_row.get("source_origin"):
            errors.append(
                f"{path}:{lineno}: corrected_evidence_detail dia_id={dia_id!r} "
                f"source_origin={detail.get('source_origin')!r} expected={provenance_row.get('source_origin')!r}"
            )
        for fact_id in detail.get("supports_answer_fact", []) or []:
            supported_fact_ids.add(str(fact_id))

    if corrected_evidence_ids and detail_evidence_ids != corrected_evidence_ids:
        errors.append(
            f"{path}:{lineno}: corrected_evidence_detail dia_ids={sorted(detail_evidence_ids)} "
            f"must match evidence={sorted(corrected_evidence_ids)}"
        )

    for fact_idx, fact in enumerate(answer_facts or []):
        if not isinstance(fact, dict):
            errors.append(f"{path}:{lineno}: corrected_answer_facts[{fact_idx}] must be object")
            continue
        source_fact_id = str(fact.get("source_fact_id", ""))
        if not source_fact_id:
            errors.append(f"{path}:{lineno}: corrected_answer_facts[{fact_idx}] missing source_fact_id")
            continue
        fact_row = facts.get(source_fact_id) if facts else None
        if facts and fact_row is None:
            errors.append(f"{path}:{lineno}: source_fact_id={source_fact_id!r} missing fact ledger")
        elif fact_row is not None and fact_row.get("source_type") not in ORIGINAL_SOURCE_TYPES:
            errors.append(
                f"{path}:{lineno}: source_fact_id={source_fact_id!r} "
                f"source_type={fact_row.get('source_type')!r} is not original-backed"
            )
        if supported_fact_ids and source_fact_id not in supported_fact_ids:
            errors.append(
                f"{path}:{lineno}: source_fact_id={source_fact_id!r} not listed in corrected_evidence_detail supports"
            )
        supported_by = {str(item) for item in fact.get("supported_by", []) or []}
        if corrected_evidence_ids and not supported_by <= corrected_evidence_ids:
            errors.append(
                f"{path}:{lineno}: corrected_answer_facts[{fact_idx}].supported_by={sorted(supported_by)} "
                f"not subset of evidence={sorted(corrected_evidence_ids)}"
            )
    return errors


def validate_fix(
    path: Path,
    lineno: int,
    row: dict[str, Any],
    facts: dict[str, dict[str, Any]],
    provenance: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    corrections = corrected_values(row)
    if not str(row.get("human_notes", "")).strip():
        errors.append(f"{path}:{lineno}: fix decision requires human_notes")
    if not corrections:
        errors.append(f"{path}:{lineno}: fix decision requires at least one corrected_* field")

    corrected_category = row.get("corrected_category")
    target_category = parse_category(corrected_category) if has_value(corrected_category) else parse_category(row.get("category"))
    if target_category is None:
        errors.append(f"{path}:{lineno}: fix decision has invalid target category")
        return errors

    if "corrected_category" in corrections and parse_category(row.get("corrected_category")) is None:
        errors.append(f"{path}:{lineno}: corrected_category must be an integer from 1 to 5")

    if target_category == 5:
        if has_value(row.get("corrected_evidence")):
            errors.append(f"{path}:{lineno}: category 5 fix must not use corrected_evidence; use corrected_negative_evidence")
        if has_value(row.get("corrected_answer_facts")):
            errors.append(f"{path}:{lineno}: category 5 fix must not use corrected_answer_facts")
        if has_value(row.get("corrected_evidence_detail")):
            errors.append(f"{path}:{lineno}: category 5 fix must not use corrected_evidence_detail")
        if not has_value(row.get("corrected_adversarial_answer")) and not has_value(row.get("adversarial_answer")):
            errors.append(f"{path}:{lineno}: category 5 fix requires adversarial_answer or corrected_adversarial_answer")
        if not has_value(row.get("corrected_negative_evidence")) and not has_value(row.get("negative_evidence")):
            errors.append(f"{path}:{lineno}: category 5 fix requires negative_evidence or corrected_negative_evidence")
        if not has_value(row.get("corrected_adversarial_reason")) and not has_value(row.get("adversarial_reason")):
            errors.append(f"{path}:{lineno}: category 5 fix requires adversarial_reason or corrected_adversarial_reason")
        reason_value = (
            row.get("corrected_adversarial_reason")
            if has_value(row.get("corrected_adversarial_reason"))
            else row.get("adversarial_reason")
        )
        errors.extend(validate_adversarial_reason(path, lineno, reason_value, "corrected_adversarial_reason"))
    else:
        if has_value(row.get("corrected_adversarial_answer")):
            errors.append(f"{path}:{lineno}: answerable fix must not include corrected_adversarial_answer")
        if has_value(row.get("corrected_negative_evidence")):
            errors.append(f"{path}:{lineno}: answerable fix must not include corrected_negative_evidence")
        if has_value(row.get("corrected_adversarial_reason")):
            errors.append(f"{path}:{lineno}: answerable fix must not include corrected_adversarial_reason")
        if not has_value(row.get("corrected_answer")) and not has_value(row.get("answer")):
            errors.append(f"{path}:{lineno}: answerable fix requires answer or corrected_answer")
        if not has_value(row.get("corrected_evidence")) and not has_value(row.get("evidence")):
            errors.append(f"{path}:{lineno}: answerable fix requires evidence or corrected_evidence")
        support_changing_fields = {"corrected_answer", "corrected_evidence", "corrected_category"}
        if support_changing_fields & set(corrections):
            errors.extend(validate_corrected_trace(path, lineno, row, facts, provenance))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Validate completed rows while allowing todo rows to remain. "
            "This is for in-progress human-audit checks only; release gates "
            "still require status=completed."
        ),
    )
    parser.add_argument("--sidecar-root", type=Path, default=None)
    args = parser.parse_args()

    facts, provenance = load_trace_indexes(args.sidecar_root)
    rows = 0
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    errors: list[str] = []
    incomplete_examples: list[str] = []
    incomplete_count = 0
    fix_count = 0
    failed_or_delete: list[dict[str, Any]] = []
    for lineno, row in iter_jsonl(args.input_jsonl):
        rows += 1
        decision = str(row.get("human_decision", "todo")).strip().lower()
        decision_counts[decision] += 1
        source_counts[str(row.get("source_dataset"))] += 1
        category_counts[str(row.get("category"))] += 1
        if decision not in COMPLETE_DECISIONS:
            incomplete_count += 1
            if len(incomplete_examples) < 20:
                incomplete_examples.append(f"{args.input_jsonl}:{lineno}: incomplete human_decision={decision!r}")
            continue

        if parse_category(row.get("category")) == 5:
            errors.extend(validate_adversarial_reason(args.input_jsonl, lineno, row.get("adversarial_reason")))

        corrections = corrected_values(row)
        if decision != "fix" and corrections:
            errors.append(
                f"{args.input_jsonl}:{lineno}: corrected_* fields are only applied when human_decision='fix'"
            )

        if decision in {"fail", "delete"}:
            failed_or_delete.append(
                {
                    "source_dataset": row.get("source_dataset"),
                    "sample_id": row.get("sample_id"),
                    "qa_idx": row.get("qa_idx"),
                    "category": row.get("category"),
                    "human_decision": decision,
                    "human_notes": row.get("human_notes", ""),
                }
            )
        if decision == "fix":
            fix_count += 1
            errors.extend(validate_fix(args.input_jsonl, lineno, row, facts, provenance))

    if failed_or_delete and not args.allow_failures:
        errors.append("failed/delete decisions exist; apply fixes or rerun with --allow-failures for summary only")
    if incomplete_count and not args.allow_incomplete:
        errors.append(f"{incomplete_count} audit rows are incomplete")
        errors.extend(incomplete_examples)

    if rows > 0 and not errors and incomplete_count == 0:
        status = "completed"
    elif rows > 0 and not errors and args.allow_incomplete:
        status = "partial_valid"
    else:
        status = "incomplete_or_failed"
    summary = {
        "status": status,
        "input_jsonl": str(args.input_jsonl),
        "input_jsonl_sha256": sha256_file(args.input_jsonl),
        "validator": str(Path(__file__)),
        "validator_sha256": sha256_file(Path(__file__)),
        "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
        "sidecar_trace_files_sha256": (
            sidecar_trace_files_sha256(args.sidecar_root) if args.sidecar_root else None
        ),
        "allow_incomplete": args.allow_incomplete,
        "rows": rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "selected_by_source": dict(sorted(source_counts.items())),
        "selected_by_category": dict(sorted(category_counts.items())),
        "failed_or_delete": failed_or_delete,
        "fix_count": fix_count,
        "incomplete_count": incomplete_count,
        "errors": errors,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status in {"completed", "partial_valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
