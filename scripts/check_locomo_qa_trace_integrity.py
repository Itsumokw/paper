#!/usr/bin/env python3
"""Check QA, QA-audit, evidence, provenance, and fact-ledger consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_ARTIFACTS = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}

FINAL_QA_SET = "locomo_style_main"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def session_keys(conversation: dict[str, Any]) -> list[str]:
    def key_num(key: str) -> int:
        suffix = key.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    return sorted(
        [
            key
            for key, value in conversation.items()
            if key.startswith("session_")
            and not key.endswith("_date_time")
            and isinstance(value, list)
        ],
        key=key_num,
    )


def dia_session(dia_id: str) -> str:
    return str(dia_id).split(":", 1)[0]


def expected_question_type(category: int) -> str:
    return {
        1: "single-hop",
        2: "multi-hop",
        3: "temporal",
        4: "commonsense",
        5: "adversarial",
    }.get(category, "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=Path("datasets/locomo_style_eval/primary"))
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/qa_trace_integrity_report.json"))
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    per_artifact: dict[str, Any] = {}
    primary_files: list[Path] = []
    provenance_files: list[Path] = []
    fact_ledger_files: list[Path] = []
    qa_audit_files: list[Path] = []

    for source, artifact in SOURCE_ARTIFACTS.items():
        primary_path = args.primary_root / f"{artifact}.json"
        sidecar_dir = args.sidecar_root / artifact
        provenance_path = sidecar_dir / f"{artifact}_provenance.jsonl"
        fact_path = sidecar_dir / f"{artifact}_fact_ledger.jsonl"
        qa_audit_path = sidecar_dir / f"{artifact}_qa_audit.jsonl"
        primary_files.append(primary_path)
        provenance_files.append(provenance_path)
        fact_ledger_files.append(fact_path)
        qa_audit_files.append(qa_audit_path)

        samples = load_json(primary_path)
        samples_by_id = {str(sample.get("sample_id")): sample for sample in samples}
        primary_qa_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        dia_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for sample in samples:
            sample_id = str(sample.get("sample_id"))
            for qa_idx, qa in enumerate(sample.get("qa", [])):
                primary_qa_by_key[(sample_id, qa_idx)] = qa
            conversation = sample.get("conversation", {})
            for session_key in session_keys(conversation):
                for turn in conversation.get(session_key, []):
                    dia_id = str(turn.get("dia_id"))
                    dia_by_sample[sample_id][dia_id] = {
                        "session_id": session_key,
                        "turn": turn,
                    }

        provenance_by_key = {
            (str(row.get("sample_id")), str(row.get("dia_id"))): row
            for row in iter_jsonl(provenance_path)
        }
        fact_sources = {
            str(row.get("fact_id")): str(row.get("source_type"))
            for row in iter_jsonl(fact_path)
        }
        qa_audit_rows = list(iter_jsonl(qa_audit_path))
        qa_audit_keys = [(str(row.get("sample_id")), int(row.get("qa_idx"))) for row in qa_audit_rows]
        duplicate_audit_keys = len(qa_audit_keys) - len(set(qa_audit_keys))
        category_counts: Counter[str] = Counter()
        answer_fact_count = 0
        cat5_count = 0
        cross_session_count = 0
        evidence_detail_mismatch = 0
        fact_link_failures = 0

        if duplicate_audit_keys:
            errors.append(f"{artifact}: duplicate qa_audit keys={duplicate_audit_keys}")
        missing_audit = sorted(set(primary_qa_by_key) - set(qa_audit_keys))
        extra_audit = sorted(set(qa_audit_keys) - set(primary_qa_by_key))
        if missing_audit:
            errors.append(f"{artifact}: missing qa_audit rows for {len(missing_audit)} primary QA")
        if extra_audit:
            errors.append(f"{artifact}: qa_audit has {len(extra_audit)} rows not in primary QA")

        for row_idx, audit in enumerate(qa_audit_rows, start=1):
            sample_id = str(audit.get("sample_id"))
            qa_idx = int(audit.get("qa_idx"))
            prefix = f"{artifact} qa_audit[{row_idx}] {sample_id}#{qa_idx}"
            sample = samples_by_id.get(sample_id)
            if sample is None:
                errors.append(f"{prefix}: sample_id not found in primary")
                continue
            primary_qa = primary_qa_by_key.get((sample_id, qa_idx))
            if primary_qa is None:
                continue

            if audit.get("qa_set") != FINAL_QA_SET:
                errors.append(f"{prefix}: qa_set={audit.get('qa_set')!r} expected {FINAL_QA_SET!r}")
            for key in ("question", "category", "evidence"):
                if audit.get(key) != primary_qa.get(key):
                    errors.append(f"{prefix}: audit {key} does not match primary QA")
            category = int(primary_qa.get("category") or 0)
            category_counts[str(category)] += 1
            if str(audit.get("question_type")) != expected_question_type(category):
                errors.append(f"{prefix}: question_type={audit.get('question_type')!r} inconsistent with category={category}")

            evidence = list(primary_qa.get("evidence", []))
            evidence_sessions = {dia_session(ev) for ev in evidence}
            expected_cross_session = len(evidence_sessions) > 1
            if audit.get("whether_cross_session") is not expected_cross_session:
                errors.append(
                    f"{prefix}: whether_cross_session={audit.get('whether_cross_session')!r} "
                    f"expected={expected_cross_session}"
                )
            if expected_cross_session:
                cross_session_count += 1

            evidence_detail = audit.get("evidence_detail", [])
            detail_ids = [str(detail.get("dia_id")) for detail in evidence_detail if isinstance(detail, dict)]
            if category == 5:
                cat5_count += 1
                if evidence:
                    errors.append(f"{prefix}: cat5 primary evidence must be []")
                if audit.get("answer_facts"):
                    errors.append(f"{prefix}: cat5 audit answer_facts must be []")
                if "answer" in primary_qa:
                    errors.append(f"{prefix}: cat5 primary must omit ordinary answer")
                if audit.get("answer") not in (None, ""):
                    errors.append(f"{prefix}: cat5 audit should not carry ordinary answer")
                if audit.get("adversarial_answer") not in (None, primary_qa.get("adversarial_answer")):
                    errors.append(f"{prefix}: audit adversarial_answer differs from primary")
                negative_evidence = list(audit.get("negative_evidence") or [])
                if not negative_evidence:
                    errors.append(f"{prefix}: cat5 missing negative_evidence")
                for dia_id in negative_evidence:
                    if dia_id not in dia_by_sample[sample_id]:
                        errors.append(f"{prefix}: negative_evidence dia_id missing from primary: {dia_id}")
                    if (sample_id, str(dia_id)) not in provenance_by_key:
                        errors.append(f"{prefix}: negative_evidence dia_id missing provenance: {dia_id}")
                if not audit.get("adversarial_reason"):
                    errors.append(f"{prefix}: cat5 missing adversarial_reason")
                continue

            if audit.get("answer") != primary_qa.get("answer"):
                errors.append(f"{prefix}: audit answer does not match primary QA")
            if not evidence:
                errors.append(f"{prefix}: answerable QA missing evidence")
            if sorted(detail_ids) != sorted(str(ev) for ev in evidence):
                evidence_detail_mismatch += 1
                if evidence_detail_mismatch <= 10:
                    errors.append(f"{prefix}: evidence_detail dia_ids do not match evidence")
            for dia_id in evidence:
                if dia_id not in dia_by_sample[sample_id]:
                    errors.append(f"{prefix}: evidence dia_id missing from primary: {dia_id}")
                if (sample_id, str(dia_id)) not in provenance_by_key:
                    errors.append(f"{prefix}: evidence dia_id missing provenance: {dia_id}")

            answer_facts = list(audit.get("answer_facts") or [])
            if not answer_facts:
                errors.append(f"{prefix}: missing answer_facts")
            answer_fact_ids = {str(fact.get("source_fact_id")) for fact in answer_facts if fact.get("source_fact_id")}
            detail_supported_ids = {
                str(fact_id)
                for detail in evidence_detail
                if isinstance(detail, dict)
                for fact_id in detail.get("supports_answer_fact", [])
            }
            for fact_idx, fact in enumerate(answer_facts):
                answer_fact_count += 1
                fact_prefix = f"{prefix} answer_fact[{fact_idx}]"
                source_fact_id = str(fact.get("source_fact_id") or "")
                if not source_fact_id:
                    errors.append(f"{fact_prefix}: missing source_fact_id")
                    continue
                source_type = fact_sources.get(source_fact_id)
                if source_type is None:
                    errors.append(f"{fact_prefix}: source_fact_id not in fact ledger: {source_fact_id}")
                elif not source_type.startswith("original_"):
                    errors.append(f"{fact_prefix}: source_type={source_type!r} is not original-backed")
                supported_by = list(fact.get("supported_by") or [])
                if not supported_by:
                    errors.append(f"{fact_prefix}: missing supported_by")
                if not set(str(dia_id) for dia_id in supported_by) <= set(str(ev) for ev in evidence):
                    errors.append(f"{fact_prefix}: supported_by is not a subset of QA evidence")
                if source_fact_id not in detail_supported_ids:
                    fact_link_failures += 1
                    if fact_link_failures <= 10:
                        errors.append(f"{fact_prefix}: source_fact_id missing from evidence_detail supports_answer_fact")
                provenance_supports_fact = False
                for dia_id in supported_by:
                    provenance = provenance_by_key.get((sample_id, str(dia_id)))
                    if not provenance:
                        continue
                    provenance_fact_ids = {str(item) for item in provenance.get("source_fact_ids", [])}
                    provenance_fact_ids |= {str(item) for item in provenance.get("grounded_in_fact_ids", [])}
                    if source_fact_id in provenance_fact_ids:
                        provenance_supports_fact = True
                        break
                if not provenance_supports_fact:
                    fact_link_failures += 1
                    if fact_link_failures <= 10:
                        errors.append(f"{fact_prefix}: no supported evidence provenance links source_fact_id")

            unsupported_detail_ids = detail_supported_ids - answer_fact_ids
            if unsupported_detail_ids:
                warnings.append(f"{prefix}: evidence_detail references non-answer fact ids {sorted(unsupported_detail_ids)[:5]}")

        if evidence_detail_mismatch > 10:
            errors.append(f"{artifact}: total evidence_detail/evidence mismatches={evidence_detail_mismatch}")
        if fact_link_failures > 10:
            errors.append(f"{artifact}: total answer fact link failures={fact_link_failures}")

        per_artifact[artifact] = {
            "source_dataset": source,
            "primary_samples": len(samples),
            "primary_qa": len(primary_qa_by_key),
            "qa_audit_rows": len(qa_audit_rows),
            "duplicate_audit_keys": duplicate_audit_keys,
            "missing_audit_rows": len(missing_audit),
            "extra_audit_rows": len(extra_audit),
            "categories": dict(sorted(category_counts.items())),
            "answer_facts": answer_fact_count,
            "cat5_qa": cat5_count,
            "cross_session_qa": cross_session_count,
            "evidence_detail_mismatch_rows": evidence_detail_mismatch,
            "fact_link_failures": fact_link_failures,
        }

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_files": [file_record(path) for path in primary_files],
            "provenance_files": [file_record(path) for path in provenance_files],
            "fact_ledger_files": [file_record(path) for path in fact_ledger_files],
            "qa_audit_files": [file_record(path) for path in qa_audit_files],
        },
        "primary_root": str(args.primary_root),
        "sidecar_root": str(args.sidecar_root),
        "per_artifact": per_artifact,
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
