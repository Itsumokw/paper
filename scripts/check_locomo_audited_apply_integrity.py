#!/usr/bin/env python3
"""Check audited primary JSON is exactly derived from original primary + human audit decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from apply_locomo_human_audit_results import DELETE_DECISIONS, apply_fix, audit_key


COMPLETE_DECISIONS = {"pass", "fail", "fix", "delete"}
SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}
ORIGINAL_SOURCE_TYPES = {"original_event", "original_memory", "original_persona", "original_turn"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            *sidecar_root.glob("*/*_qa_audit.jsonl"),
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def iter_sidecar_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def parse_json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"expected JSON list, got {value!r}")


def normalize_id_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            return [str(item) for item in parse_json_list(value)]
        except (json.JSONDecodeError, ValueError):
            return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def load_trace_indexes(
    sidecar_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    facts: dict[str, dict[str, Any]] = {}
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    qa_audit: dict[tuple[str, str, int], dict[str, Any]] = {}
    for source, artifact in SOURCE_TO_ARTIFACT.items():
        fact_path = sidecar_root / artifact / f"{artifact}_fact_ledger.jsonl"
        provenance_path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        qa_audit_path = sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        for row in iter_sidecar_jsonl(fact_path):
            facts[str(row.get("fact_id"))] = row
        for row in iter_sidecar_jsonl(provenance_path):
            provenance[(str(row.get("sample_id")), str(row.get("dia_id")))] = row
        for row in iter_sidecar_jsonl(qa_audit_path):
            qa_audit[(source, str(row.get("sample_id")), int(row.get("qa_idx", -1)))] = row
    return facts, provenance, qa_audit


def load_completed_audit(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str], Counter[str]]:
    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    counts: Counter[str] = Counter()
    for lineno, row in iter_jsonl(path):
        decision = str(row.get("human_decision", "todo")).strip().lower()
        counts[decision] += 1
        if decision not in COMPLETE_DECISIONS:
            errors.append(f"{path}:{lineno}: incomplete or invalid human_decision={decision!r}")
            continue
        try:
            key = audit_key(str(row.get("sample_id")), int(row.get("qa_idx")))
        except (TypeError, ValueError) as exc:
            errors.append(f"{path}:{lineno}: invalid audit key: {exc}")
            continue
        if key in decisions:
            errors.append(f"{path}:{lineno}: duplicate audit key={key}")
        decisions[key] = row
    return decisions, errors, counts


def support_changing_fix(row: dict[str, Any]) -> bool:
    return str(row.get("human_decision", "")).strip().lower() == "fix" and any(
        has_value(row.get(field))
        for field in ("corrected_answer", "corrected_evidence", "corrected_category")
    )


def trace_payload(qa_audit_row: dict[str, Any], audit_row: dict[str, Any] | None, final_qa: dict[str, Any]) -> dict[str, Any]:
    category = int(final_qa.get("category"))
    if category == 5:
        return {
            "negative_evidence": normalize_id_list(
                audit_row.get("corrected_negative_evidence")
                if audit_row and has_value(audit_row.get("corrected_negative_evidence"))
                else qa_audit_row.get("negative_evidence", [])
            ),
            "adversarial_reason": (
                audit_row.get("corrected_adversarial_reason")
                if audit_row and has_value(audit_row.get("corrected_adversarial_reason"))
                else qa_audit_row.get("adversarial_reason")
            ),
        }
    if audit_row and support_changing_fix(audit_row):
        return {
            "answer_facts": parse_json_list(audit_row.get("corrected_answer_facts")),
            "evidence_detail": parse_json_list(audit_row.get("corrected_evidence_detail")),
        }
    return {
        "answer_facts": qa_audit_row.get("answer_facts", []),
        "evidence_detail": qa_audit_row.get("evidence_detail", []),
    }


def validate_answerable_trace(
    sample_id: str,
    qa_idx: int,
    final_qa: dict[str, Any],
    payload: dict[str, Any],
    facts: dict[str, dict[str, Any]],
    provenance: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{sample_id} original_qa_idx={qa_idx}"
    evidence_ids = {str(item) for item in final_qa.get("evidence", [])}
    answer_facts = payload.get("answer_facts", [])
    evidence_detail = payload.get("evidence_detail", [])
    if not answer_facts:
        errors.append(f"{prefix}: answerable QA missing trace answer_facts")
    if not evidence_detail:
        errors.append(f"{prefix}: answerable QA missing trace evidence_detail")

    detail_ids: set[str] = set()
    supported_fact_ids: set[str] = set()
    for detail_idx, detail in enumerate(evidence_detail):
        if not isinstance(detail, dict):
            errors.append(f"{prefix}: evidence_detail[{detail_idx}] must be object")
            continue
        dia_id = str(detail.get("dia_id"))
        detail_ids.add(dia_id)
        provenance_row = provenance.get((sample_id, dia_id))
        if provenance_row is None:
            errors.append(f"{prefix}: evidence dia_id={dia_id!r} missing provenance")
        elif detail.get("source_origin") != provenance_row.get("source_origin"):
            errors.append(
                f"{prefix}: evidence dia_id={dia_id!r} source_origin={detail.get('source_origin')!r} "
                f"expected={provenance_row.get('source_origin')!r}"
            )
        for fact_id in detail.get("supports_answer_fact", []) or []:
            supported_fact_ids.add(str(fact_id))

    if detail_ids != evidence_ids:
        errors.append(f"{prefix}: trace evidence_detail dia_ids={sorted(detail_ids)} must match evidence={sorted(evidence_ids)}")

    for fact_idx, fact in enumerate(answer_facts):
        if not isinstance(fact, dict):
            errors.append(f"{prefix}: answer_facts[{fact_idx}] must be object")
            continue
        source_fact_id = str(fact.get("source_fact_id", ""))
        if not source_fact_id:
            errors.append(f"{prefix}: answer_facts[{fact_idx}] missing source_fact_id")
            continue
        fact_row = facts.get(source_fact_id)
        if fact_row is None:
            errors.append(f"{prefix}: source_fact_id={source_fact_id!r} missing fact ledger")
        elif fact_row.get("source_type") not in ORIGINAL_SOURCE_TYPES:
            errors.append(
                f"{prefix}: source_fact_id={source_fact_id!r} "
                f"source_type={fact_row.get('source_type')!r} is not original-backed"
            )
        if supported_fact_ids and source_fact_id not in supported_fact_ids:
            errors.append(f"{prefix}: source_fact_id={source_fact_id!r} not listed in evidence_detail supports")
        supported_by = {str(item) for item in fact.get("supported_by", []) or []}
        if not supported_by <= evidence_ids:
            errors.append(
                f"{prefix}: answer_facts[{fact_idx}].supported_by={sorted(supported_by)} "
                f"not subset of evidence={sorted(evidence_ids)}"
            )
    return errors


def validate_cat5_trace(
    sample_id: str,
    qa_idx: int,
    final_qa: dict[str, Any],
    payload: dict[str, Any],
    provenance: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{sample_id} original_qa_idx={qa_idx}"
    if final_qa.get("evidence") != []:
        errors.append(f"{prefix}: cat5 primary evidence must be []")
    if "answer" in final_qa:
        errors.append(f"{prefix}: cat5 primary must omit ordinary answer")
    if not final_qa.get("adversarial_answer"):
        errors.append(f"{prefix}: cat5 primary missing adversarial_answer")
    if not payload.get("negative_evidence"):
        errors.append(f"{prefix}: cat5 trace missing negative_evidence")
    else:
        for dia_id in payload.get("negative_evidence", []):
            if provenance.get((sample_id, str(dia_id))) is None:
                errors.append(f"{prefix}: cat5 negative_evidence dia_id={dia_id!r} missing provenance")
    if not payload.get("adversarial_reason"):
        errors.append(f"{prefix}: cat5 trace missing adversarial_reason")
    return errors


def audited_trace_errors(
    original: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    decisions: dict[tuple[str, int], dict[str, Any]],
    sidecar_root: Path,
) -> tuple[list[str], int]:
    facts, provenance, qa_audit = load_trace_indexes(sidecar_root)
    errors: list[str] = []
    checked = 0
    expected_by_sample = {str(sample.get("sample_id")): sample for sample in expected}
    for sample in original:
        source = str(sample.get("source_dataset"))
        sample_id = str(sample.get("sample_id"))
        expected_sample = expected_by_sample.get(sample_id)
        if expected_sample is None:
            errors.append(f"{sample_id}: sample missing from audited expected output")
            continue
        final_qas = expected_sample.get("qa", [])
        final_idx = 0
        for qa_idx, original_qa in enumerate(sample.get("qa", [])):
            audit_row = decisions.get(audit_key(sample_id, qa_idx))
            decision = str(audit_row.get("human_decision")).strip().lower() if audit_row else "pass"
            if decision in DELETE_DECISIONS:
                continue
            if final_idx >= len(final_qas):
                errors.append(f"{sample_id} original_qa_idx={qa_idx}: audited QA missing at final_idx={final_idx}")
                continue
            final_qa = final_qas[final_idx]
            final_idx += 1
            qa_audit_row = qa_audit.get((source, sample_id, qa_idx))
            if qa_audit_row is None:
                errors.append(f"{sample_id} original_qa_idx={qa_idx}: missing qa_audit sidecar row")
                continue
            try:
                payload = trace_payload(qa_audit_row, audit_row, final_qa)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{sample_id} original_qa_idx={qa_idx}: invalid trace payload: {exc}")
                continue
            checked += 1
            if int(final_qa.get("category")) == 5:
                errors.extend(validate_cat5_trace(sample_id, qa_idx, final_qa, payload, provenance))
            else:
                errors.extend(validate_answerable_trace(sample_id, qa_idx, final_qa, payload, facts, provenance))
    return errors, checked


def expected_audited_primary(
    original: list[dict[str, Any]],
    decisions: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    used: set[tuple[str, int]] = set()
    fixed_count = 0
    removed_count = 0
    audited_count = 0

    for sample in original:
        item = deepcopy(sample)
        sample_id = str(item.get("sample_id"))
        next_qas = []
        for qa_idx, qa in enumerate(item.get("qa", [])):
            key = audit_key(sample_id, qa_idx)
            row = decisions.get(key)
            if row is None:
                next_qas.append(qa)
                continue
            used.add(key)
            audited_count += 1
            decision = str(row.get("human_decision")).strip().lower()
            if decision in DELETE_DECISIONS:
                removed_count += 1
                continue
            if decision == "fix":
                fixed, fix_errors = apply_fix(qa, row)
                errors.extend(fix_errors)
                next_qas.append(fixed)
                fixed_count += 1
                continue
            next_qas.append(qa)
        item["qa"] = next_qas
        output.append(item)

    unused = sorted(set(decisions) - used)
    if unused:
        errors.append(f"{len(unused)} audit decisions reference missing original QA; first={unused[:10]}")
    return output, {
        "audited_decisions_used": audited_count,
        "fixed_count": fixed_count,
        "removed_count": removed_count,
        "errors": errors,
    }


def first_mismatch(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):  # noqa: E721 - exact JSON type mismatch is useful here.
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in sorted(expected):
            mismatch = first_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: list length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = first_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-primary", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--audited-primary", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    if not args.audited_primary.is_file():
        errors.append(f"audited primary missing: {args.audited_primary}")
        report = {
            "status": "failed",
            "original_primary": str(args.original_primary),
            "audit_jsonl": str(args.audit_jsonl),
            "audited_primary": str(args.audited_primary),
            "errors": errors,
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    original = load_json(args.original_primary)
    audited = load_json(args.audited_primary)
    decisions, decision_errors, decision_counts = load_completed_audit(args.audit_jsonl)
    errors.extend(decision_errors)
    expected, replay = expected_audited_primary(original, decisions)
    errors.extend(replay["errors"])
    mismatch = first_mismatch(expected, audited)
    if mismatch:
        errors.append(f"audited primary differs from replayed expected output: {mismatch}")
    trace_checked_qa = 0
    if args.sidecar_root:
        if not args.sidecar_root.is_dir():
            errors.append(f"sidecar root missing: {args.sidecar_root}")
        else:
            trace_errors, trace_checked_qa = audited_trace_errors(original, expected, decisions, args.sidecar_root)
            errors.extend(trace_errors)

    expected_qa = sum(len(sample.get("qa", [])) for sample in expected)
    audited_qa = sum(len(sample.get("qa", [])) for sample in audited if isinstance(sample, dict))
    report = {
        "status": "passed" if not errors else "failed",
        "original_primary": str(args.original_primary),
        "audit_jsonl": str(args.audit_jsonl),
        "audited_primary": str(args.audited_primary),
        "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
        "sidecar_trace_files_sha256": (
            sidecar_trace_files_sha256(args.sidecar_root)
            if args.sidecar_root and args.sidecar_root.is_dir()
            else None
        ),
        "original_samples": len(original),
        "audited_samples": len(audited) if isinstance(audited, list) else None,
        "expected_qa": expected_qa,
        "audited_qa": audited_qa,
        "trace_checked_qa": trace_checked_qa,
        "decision_counts": dict(sorted(decision_counts.items())),
        **replay,
        "errors": errors,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
