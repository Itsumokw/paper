#!/usr/bin/env python3
"""Check OPELA QA evidence policy for the LoCoMo-style eval artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, str | None]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def normalize_category(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def primary_qa_keys(primary_rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for sample in primary_rows:
        if sample.get("source_dataset") != "OPELA":
            continue
        sample_id = str(sample.get("sample_id"))
        qa_rows = sample.get("qa")
        if not isinstance(qa_rows, list):
            continue
        for idx, _qa in enumerate(qa_rows):
            keys.add((sample_id, idx))
    return keys


def validate(primary_json: Path, fact_ledger: Path, qa_audit: Path) -> dict[str, Any]:
    errors: list[str] = []
    primary_rows = read_json(primary_json)
    if not isinstance(primary_rows, list):
        primary_rows = []
        errors.append("primary_json is not a list")
    fact_rows = iter_jsonl(fact_ledger)
    audit_rows = iter_jsonl(qa_audit)

    fact_source_types: dict[str, str] = {}
    for row in fact_rows:
        if row.get("source_dataset") != "OPELA":
            continue
        fact_id = row.get("fact_id")
        source_type = row.get("source_type")
        if isinstance(fact_id, str) and isinstance(source_type, str):
            fact_source_types[fact_id] = source_type

    expected_keys = primary_qa_keys(primary_rows)
    seen_keys: set[tuple[str, int]] = set()
    answerable_count = 0
    cat5_count = 0
    evidence_origin_counts: dict[str, int] = {}
    answer_fact_source_type_counts: dict[str, int] = {}

    for line_no, row in enumerate(audit_rows, start=1):
        if row.get("source_dataset") != "OPELA":
            continue
        sample_id = str(row.get("sample_id"))
        qa_idx_raw = row.get("qa_idx")
        if not isinstance(qa_idx_raw, int):
            errors.append(f"qa_audit:{line_no}: qa_idx is not int")
            continue
        key = (sample_id, qa_idx_raw)
        if key in seen_keys:
            errors.append(f"qa_audit:{line_no}: duplicate OPELA QA key {key}")
        seen_keys.add(key)

        category = normalize_category(row.get("category"))
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: evidence is not a list")
        evidence_detail = row.get("evidence_detail")
        if not isinstance(evidence_detail, list):
            evidence_detail = []
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: evidence_detail is not a list")
        answer_facts = row.get("answer_facts")
        if not isinstance(answer_facts, list):
            answer_facts = []
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: answer_facts is not a list")

        if category == 5:
            cat5_count += 1
            if evidence:
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: category 5 has ordinary evidence")
            if evidence_detail:
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: category 5 has evidence_detail")
            if answer_facts:
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: category 5 has answer_facts")
            if not row.get("negative_evidence"):
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: category 5 missing negative_evidence")
            if not row.get("adversarial_reason"):
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: category 5 missing adversarial_reason")
            continue

        answerable_count += 1
        if not evidence:
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: answerable QA has empty evidence")
        detail_origins = []
        for detail in evidence_detail:
            if not isinstance(detail, dict):
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: evidence_detail row is not object")
                continue
            origin = str(detail.get("source_origin"))
            detail_origins.append(origin)
            evidence_origin_counts[origin] = evidence_origin_counts.get(origin, 0) + 1
            if origin in {"original_memory", "llm_summary"}:
                errors.append(
                    f"{sample_id} qa_idx={qa_idx_raw}: answerable evidence uses {origin}, "
                    "which cannot be sole OPELA answer evidence"
                )
        if "original_turn" not in detail_origins:
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: no original_turn evidence_detail")

        if not answer_facts:
            errors.append(f"{sample_id} qa_idx={qa_idx_raw}: answerable QA missing answer_facts")
        for fact in answer_facts:
            if not isinstance(fact, dict):
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: answer_fact is not object")
                continue
            fact_id = fact.get("source_fact_id")
            if not isinstance(fact_id, str):
                errors.append(f"{sample_id} qa_idx={qa_idx_raw}: answer_fact missing source_fact_id")
                continue
            source_type = fact_source_types.get(fact_id)
            answer_fact_source_type_counts[str(source_type)] = (
                answer_fact_source_type_counts.get(str(source_type), 0) + 1
            )
            if source_type != "original_turn":
                errors.append(
                    f"{sample_id} qa_idx={qa_idx_raw}: answer_fact {fact_id} has "
                    f"source_type={source_type!r}, expected 'original_turn'"
                )

    missing = sorted(expected_keys - seen_keys)
    extra = sorted(seen_keys - expected_keys)
    if missing:
        errors.append(f"qa_audit missing primary OPELA QA keys: {missing[:10]}")
    if extra:
        errors.append(f"qa_audit has unexpected OPELA QA keys: {extra[:10]}")

    return {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_info(primary_json),
            "fact_ledger": file_info(fact_ledger),
            "qa_audit": file_info(qa_audit),
        },
        "opela_primary_qa_count": len(expected_keys),
        "opela_qa_audit_count": len(seen_keys),
        "answerable_qa_count": answerable_count,
        "cat5_qa_count": cat5_count,
        "evidence_origin_counts": dict(sorted(evidence_origin_counts.items())),
        "answer_fact_source_type_counts": dict(sorted(answer_fact_source_type_counts.items())),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary/OPELA-LoCoMo-style-eval.json"),
    )
    parser.add_argument(
        "--fact-ledger",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/OPELA-LoCoMo-style-eval/"
            "OPELA-LoCoMo-style-eval_fact_ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--qa-audit",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/OPELA-LoCoMo-style-eval/"
            "OPELA-LoCoMo-style-eval_qa_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/opela_evidence_policy_report.json"),
    )
    args = parser.parse_args()

    report = validate(args.primary_json, args.fact_ledger, args.qa_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
