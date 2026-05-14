#!/usr/bin/env python3
"""Check deL1L2IM source-policy constraints for the LoCoMo-style eval artifact."""

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
        if sample.get("source_dataset") != "deL1L2IM":
            continue
        sample_id = str(sample.get("sample_id"))
        qa_rows = sample.get("qa")
        if not isinstance(qa_rows, list):
            continue
        for idx, _qa in enumerate(qa_rows):
            keys.add((sample_id, idx))
    return keys


def validate(primary_json: Path, provenance: Path, fact_ledger: Path, qa_audit: Path) -> dict[str, Any]:
    errors: list[str] = []
    primary_rows = read_json(primary_json)
    if not isinstance(primary_rows, list):
        primary_rows = []
        errors.append("primary_json is not a list")
    provenance_rows = iter_jsonl(provenance)
    fact_rows = iter_jsonl(fact_ledger)
    audit_rows = iter_jsonl(qa_audit)

    sample_files: dict[str, set[str]] = {}
    sample_records: dict[str, set[str]] = {}
    file_samples: dict[str, set[str]] = {}
    record_samples: dict[str, set[str]] = {}
    provenance_origin_counts: dict[str, int] = {}
    source_speakers_by_sample: dict[str, set[str]] = {}

    for line_no, row in enumerate(provenance_rows, start=1):
        if row.get("source_dataset") != "deL1L2IM":
            continue
        sample_id = str(row.get("sample_id"))
        source_file = row.get("source_file")
        source_record_id = row.get("source_record_id")
        source_origin = row.get("source_origin")
        order_policy = row.get("order_policy")
        source_speaker = row.get("source_speaker")

        provenance_origin_counts[str(source_origin)] = provenance_origin_counts.get(str(source_origin), 0) + 1
        if source_origin != "original_turn":
            errors.append(f"provenance:{line_no}: source_origin={source_origin!r}, expected original_turn")
        if order_policy != "source_order":
            errors.append(f"provenance:{line_no}: order_policy={order_policy!r}, expected source_order")
        if not isinstance(source_file, str) or not source_file.endswith(".xml"):
            errors.append(f"provenance:{line_no}: source_file is not TEI XML path: {source_file!r}")
        else:
            sample_files.setdefault(sample_id, set()).add(source_file)
            file_samples.setdefault(source_file, set()).add(sample_id)
        if not isinstance(source_record_id, str) or not source_record_id.startswith("Chat-"):
            errors.append(f"provenance:{line_no}: bad source_record_id={source_record_id!r}")
        else:
            sample_records.setdefault(sample_id, set()).add(source_record_id)
            record_samples.setdefault(source_record_id, set()).add(sample_id)
        if isinstance(source_speaker, str):
            source_speakers_by_sample.setdefault(sample_id, set()).add(source_speaker)

    for sample_id, files in sorted(sample_files.items()):
        if len(files) != 1:
            errors.append(f"{sample_id}: expected exactly one TEI XML source file, got {sorted(files)}")
    for sample_id, records in sorted(sample_records.items()):
        if len(records) != 1:
            errors.append(f"{sample_id}: expected exactly one source_record_id, got {sorted(records)}")
    for source_file, samples in sorted(file_samples.items()):
        if len(samples) != 1:
            errors.append(f"{source_file}: appears in multiple samples {sorted(samples)}")
    for source_record_id, samples in sorted(record_samples.items()):
        if len(samples) != 1:
            errors.append(f"{source_record_id}: appears in multiple samples {sorted(samples)}")
    for sample_id, speakers in sorted(source_speakers_by_sample.items()):
        if not any(speaker.startswith("L") for speaker in speakers):
            errors.append(f"{sample_id}: missing learner speaker in provenance source_speaker")
        if not any(speaker.startswith("N") for speaker in speakers):
            errors.append(f"{sample_id}: missing native speaker in provenance source_speaker")

    fact_source_types: dict[str, str] = {}
    for line_no, row in enumerate(fact_rows, start=1):
        if row.get("source_dataset") != "deL1L2IM":
            continue
        fact_id = row.get("fact_id")
        source_type = row.get("source_type")
        if isinstance(fact_id, str):
            fact_source_types[fact_id] = str(source_type)
        if source_type != "original_turn":
            errors.append(f"fact_ledger:{line_no}: source_type={source_type!r}, expected original_turn")

    expected_keys = primary_qa_keys(primary_rows)
    seen_keys: set[tuple[str, int]] = set()
    answerable_count = 0
    cat5_count = 0
    evidence_origin_counts: dict[str, int] = {}
    answer_fact_source_type_counts: dict[str, int] = {}

    for line_no, row in enumerate(audit_rows, start=1):
        if row.get("source_dataset") != "deL1L2IM":
            continue
        sample_id = str(row.get("sample_id"))
        qa_idx = row.get("qa_idx")
        if not isinstance(qa_idx, int):
            errors.append(f"qa_audit:{line_no}: qa_idx is not int")
            continue
        key = (sample_id, qa_idx)
        if key in seen_keys:
            errors.append(f"qa_audit:{line_no}: duplicate deL1L2IM QA key {key}")
        seen_keys.add(key)

        category = normalize_category(row.get("category"))
        evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
        evidence_detail = row.get("evidence_detail") if isinstance(row.get("evidence_detail"), list) else []
        answer_facts = row.get("answer_facts") if isinstance(row.get("answer_facts"), list) else []

        if category == 5:
            cat5_count += 1
            if evidence or evidence_detail or answer_facts:
                errors.append(f"{sample_id} qa_idx={qa_idx}: category 5 has ordinary evidence or answer facts")
            if not row.get("negative_evidence"):
                errors.append(f"{sample_id} qa_idx={qa_idx}: category 5 missing negative_evidence")
            if not row.get("adversarial_reason"):
                errors.append(f"{sample_id} qa_idx={qa_idx}: category 5 missing adversarial_reason")
            continue

        answerable_count += 1
        if not evidence:
            errors.append(f"{sample_id} qa_idx={qa_idx}: answerable QA has empty evidence")
        detail_origins = []
        for detail in evidence_detail:
            if not isinstance(detail, dict):
                errors.append(f"{sample_id} qa_idx={qa_idx}: evidence_detail row is not object")
                continue
            origin = str(detail.get("source_origin"))
            detail_origins.append(origin)
            evidence_origin_counts[origin] = evidence_origin_counts.get(origin, 0) + 1
        if set(detail_origins) != {"original_turn"}:
            errors.append(f"{sample_id} qa_idx={qa_idx}: evidence origins={sorted(set(detail_origins))}")
        if not answer_facts:
            errors.append(f"{sample_id} qa_idx={qa_idx}: answerable QA missing answer_facts")
        for fact in answer_facts:
            if not isinstance(fact, dict):
                errors.append(f"{sample_id} qa_idx={qa_idx}: answer_fact is not object")
                continue
            fact_id = fact.get("source_fact_id")
            if not isinstance(fact_id, str):
                errors.append(f"{sample_id} qa_idx={qa_idx}: answer_fact missing source_fact_id")
                continue
            source_type = fact_source_types.get(fact_id)
            answer_fact_source_type_counts[str(source_type)] = (
                answer_fact_source_type_counts.get(str(source_type), 0) + 1
            )
            if source_type != "original_turn":
                errors.append(
                    f"{sample_id} qa_idx={qa_idx}: answer_fact {fact_id} "
                    f"source_type={source_type!r}, expected original_turn"
                )

    missing = sorted(expected_keys - seen_keys)
    extra = sorted(seen_keys - expected_keys)
    if missing:
        errors.append(f"qa_audit missing primary deL1L2IM QA keys: {missing[:10]}")
    if extra:
        errors.append(f"qa_audit has unexpected deL1L2IM QA keys: {extra[:10]}")

    return {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_info(primary_json),
            "provenance": file_info(provenance),
            "fact_ledger": file_info(fact_ledger),
            "qa_audit": file_info(qa_audit),
        },
        "sample_count": len(sample_files),
        "source_xml_count": len(file_samples),
        "source_record_count": len(record_samples),
        "provenance_origin_counts": dict(sorted(provenance_origin_counts.items())),
        "primary_qa_count": len(expected_keys),
        "qa_audit_count": len(seen_keys),
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
        default=Path("datasets/locomo_style_eval/primary/deL1L2IM-LoCoMo-style-eval.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/deL1L2IM-LoCoMo-style-eval/"
            "deL1L2IM-LoCoMo-style-eval_provenance.jsonl"
        ),
    )
    parser.add_argument(
        "--fact-ledger",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/deL1L2IM-LoCoMo-style-eval/"
            "deL1L2IM-LoCoMo-style-eval_fact_ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--qa-audit",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/deL1L2IM-LoCoMo-style-eval/"
            "deL1L2IM-LoCoMo-style-eval_qa_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/del1l2im_source_policy_report.json"),
    )
    args = parser.parse_args()

    report = validate(args.primary_json, args.provenance, args.fact_ledger, args.qa_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
