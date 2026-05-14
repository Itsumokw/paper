#!/usr/bin/env python3
"""Apply completed human-audit decisions to a LoCoMo-style primary JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}

COMPLETE_DECISIONS = {"pass", "fail", "fix", "delete"}
DELETE_DECISIONS = {"fail", "delete"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def normalize_evidence(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def audit_key(sample_id: str, qa_idx: int) -> tuple[str, int]:
    return (sample_id, qa_idx)


def load_audit_decisions(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    incomplete_count = 0
    incomplete_examples: list[str] = []
    for lineno, row in iter_jsonl(path):
        decision = str(row.get("human_decision", "todo")).strip().lower()
        if decision not in COMPLETE_DECISIONS:
            incomplete_count += 1
            if len(incomplete_examples) < 20:
                incomplete_examples.append(f"{path}:{lineno}: incomplete or invalid human_decision={decision!r}")
            continue
        sample_id = str(row.get("sample_id"))
        try:
            qa_idx = int(row.get("qa_idx"))
        except (TypeError, ValueError):
            errors.append(f"{path}:{lineno}: invalid qa_idx={row.get('qa_idx')!r}")
            continue
        key = audit_key(sample_id, qa_idx)
        if key in decisions:
            errors.append(f"{path}:{lineno}: duplicate audit decision for {sample_id} qa_idx={qa_idx}")
        decisions[key] = row
    if incomplete_count:
        errors.append(f"{incomplete_count} audit rows are incomplete or invalid")
        errors.extend(incomplete_examples)
    return decisions, errors


def apply_fix(qa: dict[str, Any], audit_row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    fixed = deepcopy(qa)
    replacements = {
        "corrected_question": "question",
        "corrected_answer": "answer",
        "corrected_category": "category",
        "corrected_evidence": "evidence",
        "corrected_adversarial_answer": "adversarial_answer",
        "corrected_negative_evidence": "negative_evidence",
        "corrected_adversarial_reason": "adversarial_reason",
    }
    applied = False
    for audit_key_name, qa_key in replacements.items():
        if audit_key_name in audit_row and audit_row[audit_key_name] not in (None, ""):
            value = audit_row[audit_key_name]
            if qa_key in {"evidence", "negative_evidence"}:
                value = normalize_evidence(value)
            fixed[qa_key] = value
            applied = True
    if not applied:
        errors.append(
            f"{audit_row.get('sample_id')} qa_idx={audit_row.get('qa_idx')}: fix decision requires corrected_* fields"
        )
        return fixed, errors

    try:
        category = int(fixed.get("category"))
    except (TypeError, ValueError):
        errors.append(
            f"{audit_row.get('sample_id')} qa_idx={audit_row.get('qa_idx')}: invalid corrected category={fixed.get('category')!r}"
        )
        return fixed, errors

    if category == 5:
        fixed["category"] = 5
        fixed["evidence"] = []
        fixed.pop("answer", None)
        fixed.pop("negative_evidence", None)
        fixed.pop("adversarial_reason", None)
        if not fixed.get("adversarial_answer"):
            errors.append(
                f"{audit_row.get('sample_id')} qa_idx={audit_row.get('qa_idx')}: cat5 fix requires adversarial_answer"
            )
    else:
        fixed["category"] = category
        fixed.pop("adversarial_answer", None)
        fixed.pop("negative_evidence", None)
        fixed.pop("adversarial_reason", None)
        if not fixed.get("answer"):
            errors.append(
                f"{audit_row.get('sample_id')} qa_idx={audit_row.get('qa_idx')}: answerable fix requires answer"
            )
        if not fixed.get("evidence"):
            errors.append(
                f"{audit_row.get('sample_id')} qa_idx={audit_row.get('qa_idx')}: answerable fix requires evidence"
            )
    return fixed, errors


def apply_audit(primary: list[dict[str, Any]], decisions: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    errors: list[str] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    removed: list[dict[str, Any]] = []
    fixed_count = 0
    seen_decisions: set[tuple[str, int]] = set()

    for sample in primary:
        item = deepcopy(sample)
        sample_id = str(item.get("sample_id"))
        source = str(item.get("source_dataset"))
        next_qas = []
        for qa_idx, qa in enumerate(item.get("qa", [])):
            key = audit_key(sample_id, qa_idx)
            audit_row = decisions.get(key)
            if not audit_row:
                next_qas.append(qa)
                continue
            seen_decisions.add(key)
            decision = str(audit_row.get("human_decision")).strip().lower()
            decision_counts[decision] += 1
            source_counts[source] += 1
            if decision in DELETE_DECISIONS:
                removed.append(
                    {
                        "sample_id": sample_id,
                        "source_dataset": source,
                        "qa_idx": qa_idx,
                        "category": qa.get("category"),
                        "human_decision": decision,
                        "human_notes": audit_row.get("human_notes", ""),
                    }
                )
                continue
            if decision == "fix":
                fixed, fix_errors = apply_fix(qa, audit_row)
                errors.extend(fix_errors)
                next_qas.append(fixed)
                fixed_count += 1
                continue
            next_qas.append(qa)
        item["qa"] = next_qas
        output.append(item)

    unused = set(decisions) - seen_decisions
    for sample_id, qa_idx in sorted(unused):
        errors.append(f"audit decision references missing QA: {sample_id} qa_idx={qa_idx}")

    return output, {
        "status": "failed" if errors else "applied",
        "input_samples": len(primary),
        "output_samples": len(output),
        "input_qa": sum(len(sample.get("qa", [])) for sample in primary),
        "output_qa": sum(len(sample.get("qa", [])) for sample in output),
        "decision_counts": dict(sorted(decision_counts.items())),
        "audited_by_source": dict(sorted(source_counts.items())),
        "removed_count": len(removed),
        "fixed_count": fixed_count,
        "removed": removed,
        "errors": errors,
    }


def write_source_files(output_source_dir: Path, samples: list[dict[str, Any]]) -> dict[str, str]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_source.setdefault(str(sample.get("source_dataset")), []).append(sample)
    written: dict[str, str] = {}
    for source, rows in sorted(by_source.items()):
        artifact = SOURCE_TO_ARTIFACT.get(source, f"{source}-LoCoMo-style-eval")
        path = output_source_dir / f"{artifact}.json"
        write_json(path, rows)
        written[source] = str(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-json", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-source-dir", type=Path, default=None)
    args = parser.parse_args()

    primary = load_json(args.primary_json)
    if not isinstance(primary, list):
        raise TypeError("primary JSON must be a list")
    decisions, decision_errors = load_audit_decisions(args.audit_jsonl)
    if decision_errors:
        report = {
            "status": "failed",
            "input": str(args.primary_json),
            "audit_jsonl": str(args.audit_jsonl),
            "errors": decision_errors,
        }
        write_json(args.output_report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    output, report = apply_audit(primary, decisions)
    report["input"] = str(args.primary_json)
    report["audit_jsonl"] = str(args.audit_jsonl)
    report["output_json"] = str(args.output_json)
    if report["status"] == "applied":
        write_json(args.output_json, output)
        if args.output_source_dir:
            report["output_source_files"] = write_source_files(args.output_source_dir, output)
    write_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
