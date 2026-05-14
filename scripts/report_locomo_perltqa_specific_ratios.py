#!/usr/bin/env python3
"""Report PerLTQA-specific source-fidelity ratios for the LoCoMo-style eval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE = "PerLTQA"
ARTIFACT = "PerLTQA-LoCoMo-style-eval"
ORIGINAL_FACT_PREFIX = "original_"
REQUIRED_RATIO_FIELDS = {
    "original_turn_evidence_ratio",
    "memory_anchor_evidence_ratio",
    "synthetic_bridge_turn_ratio",
    "answer_fact_original_backed_ratio",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def input_file(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def perltqa_samples(primary_json: Path) -> list[dict[str, Any]]:
    rows = load_json(primary_json)
    return [row for row in rows if row.get("source_dataset") == SOURCE]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/locomo_style_eval"),
        help="Root directory for the LoCoMo-style eval artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/perltqa_specific_ratios.json"),
    )
    args = parser.parse_args()

    root = args.root
    primary_json = root / "primary" / "multilingual_locomo_style_eval.json"
    sidecar_root = root / "sidecars" / ARTIFACT
    provenance_path = sidecar_root / f"{ARTIFACT}_provenance.jsonl"
    fact_ledger_path = sidecar_root / f"{ARTIFACT}_fact_ledger.jsonl"
    qa_audit_path = sidecar_root / f"{ARTIFACT}_qa_audit.jsonl"

    input_paths = {
        "primary_json": primary_json,
        "provenance": provenance_path,
        "fact_ledger": fact_ledger_path,
        "qa_audit": qa_audit_path,
    }
    errors: list[str] = []
    for label, path in input_paths.items():
        if not path.is_file():
            errors.append(f"{label} missing: {path}")

    if errors:
        report = {
            "status": "failed",
            "source_dataset": SOURCE,
            "artifact": ARTIFACT,
            "input_files": {
                label: {"path": str(path), "sha256": None}
                for label, path in input_paths.items()
            },
            "errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    provenance_by_dia: dict[tuple[str, str], dict[str, Any]] = {}
    turn_origin_counts: Counter[str] = Counter()
    per_sample_turn_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in iter_jsonl(provenance_path):
        sample_id = str(row.get("sample_id"))
        dia_id = str(row.get("dia_id"))
        source_origin = str(row.get("source_origin", ""))
        provenance_by_dia[(sample_id, dia_id)] = row
        turn_origin_counts[source_origin] += 1
        per_sample_turn_counts[sample_id][source_origin] += 1

    fact_source_types: dict[str, str] = {}
    for row in iter_jsonl(fact_ledger_path):
        fact_id = str(row.get("fact_id"))
        fact_source_types[fact_id] = str(row.get("source_type", ""))

    samples = perltqa_samples(primary_json)
    sample_ids = {str(sample.get("sample_id")) for sample in samples}
    if not samples:
        errors.append("no PerLTQA samples found in primary JSON")

    qa_rows = [row for row in iter_jsonl(qa_audit_path) if row.get("source_dataset") == SOURCE]
    answerable_rows = [row for row in qa_rows if int(row.get("category", 0)) != 5]
    if not answerable_rows:
        errors.append("no answerable PerLTQA QA rows found")

    original_turn_evidence_qa = 0
    memory_anchor_evidence_qa = 0
    answer_fact_original_backed_qa = 0
    qa_with_missing_trace: list[dict[str, Any]] = []
    evidence_origin_histogram: Counter[str] = Counter()

    for row in answerable_rows:
        sample_id = str(row.get("sample_id"))
        evidence_detail = row.get("evidence_detail")
        origins: set[str] = set()
        if isinstance(evidence_detail, list):
            for detail in evidence_detail:
                if isinstance(detail, dict) and detail.get("source_origin"):
                    origins.add(str(detail["source_origin"]))
        if not origins:
            for dia_id in row.get("evidence", []):
                provenance = provenance_by_dia.get((sample_id, str(dia_id)))
                if provenance and provenance.get("source_origin"):
                    origins.add(str(provenance["source_origin"]))
        if not origins:
            qa_with_missing_trace.append(
                {
                    "sample_id": sample_id,
                    "qa_idx": row.get("qa_idx"),
                    "reason": "missing evidence provenance",
                }
            )
        if "original_turn" in origins:
            original_turn_evidence_qa += 1
        if "memory_anchor_turn" in origins:
            memory_anchor_evidence_qa += 1
        for origin in sorted(origins):
            evidence_origin_histogram[origin] += 1

        answer_facts = row.get("answer_facts")
        fact_ids: list[str] = []
        if isinstance(answer_facts, list):
            for fact in answer_facts:
                if isinstance(fact, dict) and fact.get("source_fact_id"):
                    fact_ids.append(str(fact["source_fact_id"]))
        missing_fact_ids = [fact_id for fact_id in fact_ids if fact_id not in fact_source_types]
        non_original_fact_ids = [
            fact_id
            for fact_id in fact_ids
            if not fact_source_types.get(fact_id, "").startswith(ORIGINAL_FACT_PREFIX)
        ]
        if fact_ids and not missing_fact_ids and not non_original_fact_ids:
            answer_fact_original_backed_qa += 1
        else:
            qa_with_missing_trace.append(
                {
                    "sample_id": sample_id,
                    "qa_idx": row.get("qa_idx"),
                    "reason": "answer facts are not fully original-backed",
                    "fact_ids": fact_ids,
                    "missing_fact_ids": missing_fact_ids,
                    "non_original_fact_ids": non_original_fact_ids,
                }
            )

    total_turns = sum(turn_origin_counts.values())
    ratios = {
        "original_turn_evidence_ratio": safe_ratio(original_turn_evidence_qa, len(answerable_rows)),
        "memory_anchor_evidence_ratio": safe_ratio(memory_anchor_evidence_qa, len(answerable_rows)),
        "synthetic_bridge_turn_ratio": safe_ratio(turn_origin_counts["synthetic_bridge_turn"], total_turns),
        "answer_fact_original_backed_ratio": safe_ratio(answer_fact_original_backed_qa, len(answerable_rows)),
    }
    missing_ratio_fields = sorted(REQUIRED_RATIO_FIELDS - set(ratios))
    if missing_ratio_fields:
        errors.append(f"missing ratio fields: {missing_ratio_fields}")
    if ratios["answer_fact_original_backed_ratio"] != 1.0:
        errors.append(
            "answer_fact_original_backed_ratio must be 1.0 for PerLTQA PlanMode D; "
            f"observed={ratios['answer_fact_original_backed_ratio']}"
        )
    if qa_with_missing_trace:
        errors.append(f"QA with missing or non-original answer trace: {qa_with_missing_trace[:20]}")
    if sample_ids != set(per_sample_turn_counts):
        errors.append(
            "PerLTQA primary/provenance sample_id mismatch: "
            f"primary_only={sorted(sample_ids - set(per_sample_turn_counts))[:10]} "
            f"provenance_only={sorted(set(per_sample_turn_counts) - sample_ids)[:10]}"
        )

    report = {
        "status": "passed" if not errors else "failed",
        "source_dataset": SOURCE,
        "artifact": ARTIFACT,
        "input_files": {label: input_file(path) for label, path in input_paths.items()},
        "counts": {
            "samples": len(samples),
            "qa_total": len(qa_rows),
            "answerable_qa": len(answerable_rows),
            "answer_fact_original_backed_qa": answer_fact_original_backed_qa,
            "original_turn_evidence_qa": original_turn_evidence_qa,
            "memory_anchor_evidence_qa": memory_anchor_evidence_qa,
            "turns_total": total_turns,
            "turn_origin_counts": dict(sorted(turn_origin_counts.items())),
            "evidence_origin_histogram": dict(sorted(evidence_origin_histogram.items())),
        },
        "ratios": ratios,
        "notes": [
            "Evidence ratios use answerable PerLTQA QA as denominator.",
            "synthetic_bridge_turn_ratio uses PerLTQA provenance turns as denominator.",
            "answer_fact_original_backed_ratio requires every answer_facts.source_fact_id to exist in the fact ledger and have a source_type starting with original_.",
        ],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
