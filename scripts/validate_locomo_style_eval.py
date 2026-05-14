#!/usr/bin/env python3
"""Validate LoCoMo-style eval primary JSON and optional sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DIA_ID_RE = re.compile(r"^D(\d+):(\d+)$")
REQUIRED_SAMPLE_KEYS = {
    "sample_id",
    "source_dataset",
    "language",
    "split",
    "conversation",
    "observation",
    "session_summary",
    "event_summary",
    "qa",
}
REQUIRED_QA_KEYS = {"question", "category", "evidence"}
ALLOWED_SAMPLE_KEYS = REQUIRED_SAMPLE_KEYS
ALLOWED_TURN_KEYS = {"speaker", "dia_id", "text"}
ANSWERABLE_QA_KEYS = {"question", "answer", "category", "evidence"}
CAT5_QA_KEYS = {"question", "category", "evidence", "adversarial_answer"}
ALLOWED_PROVENANCE_LABELS = {
    "original_turn",
    "original_memory",
    "memory_anchor_turn",
    "synthetic_bridge_turn",
    "synthetic_continuation_turn",
    "llm_summary",
}
FINAL_QA_SET = "locomo_style_main"
ALLOWED_ADVERSARIAL_REASONS = {"unsupported_fact", "time_swap", "entity_swap"}
REQUIRED_PROVENANCE_KEYS = {
    "source_dataset",
    "sample_id",
    "dia_id",
    "session_id",
    "turn_index",
    "source_origin",
    "source_file",
    "source_record_id",
    "source_turn_id",
    "source_fact_id",
    "raw_text_hash",
    "text",
}
REQUIRED_FACT_LEDGER_KEYS = {
    "source_dataset",
    "sample_id",
    "fact_id",
    "source_type",
    "source_text",
    "source_id",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def session_keys(conv: dict[str, Any]) -> list[str]:
    def key_num(key: str) -> int:
        suffix = key.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    return sorted(
        [key for key, value in conv.items() if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)],
        key=key_num,
    )


def validate_primary(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    data = load_json(path)
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list")

    counts: Counter[str] = Counter()
    total_turns = 0
    total_sessions = 0
    total_qa = 0

    for sample_idx, sample in enumerate(data):
        prefix = f"sample[{sample_idx}] {sample.get('sample_id', '<missing>')}"
        missing = REQUIRED_SAMPLE_KEYS - set(sample)
        if missing:
            errors.append(f"{prefix}: missing sample keys {sorted(missing)}")
            continue
        extra_sample_keys = set(sample) - ALLOWED_SAMPLE_KEYS
        if extra_sample_keys:
            errors.append(f"{prefix}: primary sample has non-loader extra keys {sorted(extra_sample_keys)}")
        if sample.get("split") != "eval":
            errors.append(f"{prefix}: split must be 'eval'")
        for metadata_key in ("observation", "session_summary", "event_summary"):
            if not isinstance(sample.get(metadata_key), dict):
                errors.append(f"{prefix}: {metadata_key} must be object")
        conv = sample.get("conversation")
        if not isinstance(conv, dict):
            errors.append(f"{prefix}: conversation must be object")
            continue
        if not conv.get("speaker_a") or not conv.get("speaker_b"):
            errors.append(f"{prefix}: missing speaker_a/speaker_b")
        allowed_speakers = {str(conv.get("speaker_a", "")), str(conv.get("speaker_b", ""))}
        allowed_conversation_keys = {"speaker_a", "speaker_b"}
        for session_key in session_keys(conv):
            allowed_conversation_keys.add(session_key)
            allowed_conversation_keys.add(f"{session_key}_date_time")
        extra_conversation_keys = set(conv) - allowed_conversation_keys
        if extra_conversation_keys:
            errors.append(f"{prefix}: conversation has non-loader extra keys {sorted(extra_conversation_keys)}")

        dia_ids: set[str] = set()
        sessions = session_keys(conv)
        total_sessions += len(sessions)
        if not sessions:
            errors.append(f"{prefix}: no session_i lists")
        for session_key in sessions:
            if f"{session_key}_date_time" not in conv:
                errors.append(f"{prefix}: missing {session_key}_date_time")
            session_num = int(session_key.rsplit("_", 1)[-1])
            turns = conv[session_key]
            for turn_idx, turn in enumerate(turns, start=1):
                total_turns += 1
                if not isinstance(turn, dict):
                    errors.append(f"{prefix}: {session_key}[{turn_idx}] is not object")
                    continue
                extra_turn_keys = set(turn) - ALLOWED_TURN_KEYS
                if extra_turn_keys:
                    errors.append(f"{prefix}: {session_key}[{turn_idx}] has non-loader extra keys {sorted(extra_turn_keys)}")
                for key in ("speaker", "dia_id", "text"):
                    if key not in turn:
                        errors.append(f"{prefix}: {session_key}[{turn_idx}] missing {key}")
                speaker = str(turn.get("speaker", ""))
                if speaker not in allowed_speakers:
                    errors.append(
                        f"{prefix}: {session_key}[{turn_idx}] speaker {speaker!r} "
                        f"not in speaker_a/speaker_b {sorted(allowed_speakers)}"
                    )
                dia_id = str(turn.get("dia_id", ""))
                match = DIA_ID_RE.match(dia_id)
                if not match:
                    errors.append(f"{prefix}: invalid dia_id {dia_id!r}")
                elif int(match.group(1)) != session_num:
                    errors.append(f"{prefix}: dia_id {dia_id} does not match {session_key}")
                if dia_id in dia_ids:
                    errors.append(f"{prefix}: duplicate dia_id {dia_id}")
                dia_ids.add(dia_id)
                if not str(turn.get("text", "")).strip():
                    warnings.append(f"{prefix}: empty text at {dia_id}")

        qas = sample.get("qa")
        if not isinstance(qas, list) or not qas:
            errors.append(f"{prefix}: qa must be a non-empty list")
            continue
        total_qa += len(qas)
        for qa_idx, qa in enumerate(qas):
            qprefix = f"{prefix} qa[{qa_idx}]"
            missing_qa = REQUIRED_QA_KEYS - set(qa)
            if missing_qa:
                errors.append(f"{qprefix}: missing QA keys {sorted(missing_qa)}")
                continue
            category = str(qa.get("category"))
            counts[category] += 1
            evidence = qa.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"{qprefix}: evidence must be list")
                continue
            if category == "5":
                extra_qa_keys = set(qa) - CAT5_QA_KEYS
                if extra_qa_keys:
                    errors.append(f"{qprefix}: cat5 has non-loader extra keys {sorted(extra_qa_keys)}")
                missing_cat5_keys = CAT5_QA_KEYS - set(qa)
                if missing_cat5_keys:
                    errors.append(f"{qprefix}: cat5 missing keys {sorted(missing_cat5_keys)}")
                if evidence:
                    errors.append(f"{qprefix}: cat5 evidence must be []")
                if "answer" in qa:
                    errors.append(f"{qprefix}: cat5 must omit ordinary answer")
                if not qa.get("adversarial_answer"):
                    errors.append(f"{qprefix}: cat5 should include adversarial_answer")
            else:
                extra_qa_keys = set(qa) - ANSWERABLE_QA_KEYS
                if extra_qa_keys:
                    errors.append(f"{qprefix}: answerable QA has non-loader extra keys {sorted(extra_qa_keys)}")
                missing_answerable_keys = ANSWERABLE_QA_KEYS - set(qa)
                if missing_answerable_keys:
                    errors.append(f"{qprefix}: answerable QA missing keys {sorted(missing_answerable_keys)}")
                if "answer" not in qa:
                    errors.append(f"{qprefix}: answerable QA missing answer field")
                if qa.get("answer") in (None, ""):
                    errors.append(f"{qprefix}: answerable QA missing answer")
                if not evidence:
                    errors.append(f"{qprefix}: answerable QA missing evidence")
            for dia_id in evidence:
                if dia_id not in dia_ids:
                    errors.append(f"{qprefix}: evidence dia_id not found: {dia_id}")

    return {
        "path": str(path),
        "samples": len(data),
        "sessions": total_sessions,
        "turns": total_turns,
        "qa": total_qa,
        "categories": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def validate_provenance(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = 0
    origin_counts: Counter[str] = Counter()
    for lineno, row in iter_jsonl(path):
        rows += 1
        prefix = f"{path}:{lineno}"
        missing = REQUIRED_PROVENANCE_KEYS - set(row)
        if missing:
            errors.append(f"{prefix}: missing provenance keys {sorted(missing)}")
        for key in REQUIRED_PROVENANCE_KEYS:
            if key in row and row.get(key) in (None, ""):
                errors.append(f"{prefix}: empty provenance key {key}")
        origin_counts[str(row.get("source_origin", "missing"))] += 1
        origin = str(row.get("source_origin", ""))
        if origin not in ALLOWED_PROVENANCE_LABELS:
            errors.append(f"{path}:{lineno}: unsupported source_origin={origin!r}")
        text = row.get("text")
        expected = row.get("raw_text_hash")
        if text is None:
            warnings.append(f"{path}:{lineno}: provenance row has no text for hash recheck")
            continue
        if not expected:
            errors.append(f"{prefix}: missing raw_text_hash")
            continue
        actual = sha256_text(str(text))
        if actual != expected:
            errors.append(f"{path}:{lineno}: raw_text_hash mismatch for {row.get('dia_id')}")
    return {
        "path": str(path),
        "rows": rows,
        "source_origin_counts": dict(sorted(origin_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def load_fact_ids(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    fact_sources: dict[str, str] = {}
    source_types: Counter[str] = Counter()
    rows = 0
    for lineno, row in iter_jsonl(path):
        rows += 1
        prefix = f"{path}:{lineno}"
        missing = REQUIRED_FACT_LEDGER_KEYS - set(row)
        if missing:
            errors.append(f"{prefix}: missing fact-ledger keys {sorted(missing)}")
        for key in REQUIRED_FACT_LEDGER_KEYS:
            if key in row and row.get(key) in (None, ""):
                errors.append(f"{prefix}: empty fact-ledger key {key}")
        fact_id = row.get("fact_id")
        if not fact_id:
            errors.append(f"{path}:{lineno}: missing fact_id")
            continue
        fact_id = str(fact_id)
        if fact_id in fact_sources:
            errors.append(f"{path}:{lineno}: duplicate fact_id {fact_id}")
        source_type = str(row.get("source_type", "missing"))
        fact_sources[fact_id] = source_type
        source_types[source_type] += 1
        if not source_type.startswith("original_"):
            errors.append(f"{path}:{lineno}: fact_id {fact_id} has non-original source_type={source_type!r}")
        if not row.get("source_text"):
            warnings.append(f"{path}:{lineno}: empty source_text for {fact_id}")
    return fact_sources, {
        "path": str(path),
        "rows": rows,
        "source_types": dict(sorted(source_types.items())),
        "errors": errors,
        "warnings": warnings,
    }


def validate_qa_audit(path: Path, fact_sources: dict[str, str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = 0
    categories: Counter[str] = Counter()
    cross_session = 0
    for lineno, row in iter_jsonl(path):
        rows += 1
        prefix = f"{path}:{lineno}"
        category = str(row.get("category"))
        categories[category] += 1
        if row.get("qa_set") != FINAL_QA_SET:
            errors.append(f"{prefix}: qa_set must be {FINAL_QA_SET!r}, got {row.get('qa_set')!r}")
        if "whether_cross_session" not in row or not isinstance(row.get("whether_cross_session"), bool):
            errors.append(f"{prefix}: whether_cross_session must be boolean")
        elif row["whether_cross_session"]:
            cross_session += 1
        if not row.get("difficulty"):
            errors.append(f"{prefix}: missing difficulty")
        if not row.get("question_type"):
            errors.append(f"{prefix}: missing question_type")
        verifier_status = row.get("verifier_status")
        if not isinstance(verifier_status, str) or not verifier_status.strip():
            errors.append(f"{prefix}: missing verifier_status")
        if not row.get("human_audit_status"):
            errors.append(f"{prefix}: missing human_audit_status")
        for detail_idx, detail in enumerate(row.get("evidence_detail", [])):
            if not isinstance(detail, dict):
                errors.append(f"{prefix}: evidence_detail[{detail_idx}] must be object")
                continue
            origin = detail.get("source_origin")
            if origin not in ALLOWED_PROVENANCE_LABELS:
                errors.append(f"{prefix}: evidence_detail[{detail_idx}] unsupported source_origin={origin!r}")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list):
            errors.append(f"{prefix}: evidence must be list")
            evidence = []
        answer_facts = row.get("answer_facts", [])
        if category == "5":
            if evidence:
                errors.append(f"{prefix}: cat5 evidence must be []")
            if answer_facts:
                errors.append(f"{prefix}: cat5 answer_facts must be []")
            if not row.get("negative_evidence"):
                errors.append(f"{prefix}: cat5 missing negative_evidence")
            adversarial_reason = row.get("adversarial_reason")
            if not adversarial_reason:
                errors.append(f"{prefix}: cat5 missing adversarial_reason")
            elif str(adversarial_reason) not in ALLOWED_ADVERSARIAL_REASONS:
                errors.append(
                    f"{prefix}: cat5 adversarial_reason={adversarial_reason!r} "
                    f"not in {sorted(ALLOWED_ADVERSARIAL_REASONS)}"
                )
            continue
        if not answer_facts:
            errors.append(f"{prefix}: answerable QA missing answer_facts")
            continue
        for fact_idx, fact in enumerate(answer_facts):
            source_fact_id = fact.get("source_fact_id")
            if not source_fact_id:
                errors.append(f"{prefix}: answer_fact[{fact_idx}] missing source_fact_id")
            elif fact_sources is not None and str(source_fact_id) not in fact_sources:
                errors.append(f"{prefix}: answer_fact[{fact_idx}] unknown source_fact_id {source_fact_id}")
            elif fact_sources is not None and not fact_sources[str(source_fact_id)].startswith("original_"):
                errors.append(
                    f"{prefix}: answer_fact[{fact_idx}] source_fact_id {source_fact_id} "
                    f"is not backed by original source_type={fact_sources[str(source_fact_id)]!r}"
                )
            supported_by = fact.get("supported_by", [])
            if not supported_by:
                errors.append(f"{prefix}: answer_fact[{fact_idx}] missing supported_by")
    return {
        "path": str(path),
        "rows": rows,
        "categories": dict(sorted(categories.items())),
        "cross_session_qa": cross_session,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--fact-ledger", type=Path, default=None)
    parser.add_argument("--qa-audit", type=Path, default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "input_files": {
            "primary": file_record(args.primary_json),
        },
        "primary": validate_primary(args.primary_json),
    }
    fact_ids = None
    if args.provenance:
        report["input_files"]["provenance"] = file_record(args.provenance)
        report["provenance"] = validate_provenance(args.provenance)
    if args.fact_ledger:
        report["input_files"]["fact_ledger"] = file_record(args.fact_ledger)
        fact_ids, report["fact_ledger"] = load_fact_ids(args.fact_ledger)
    if args.qa_audit:
        report["input_files"]["qa_audit"] = file_record(args.qa_audit)
        report["qa_audit"] = validate_qa_audit(args.qa_audit, fact_ids)

    errors = list(report["primary"]["errors"])
    warnings = list(report["primary"]["warnings"])
    for section in ("provenance", "fact_ledger", "qa_audit"):
        if section in report:
            errors.extend(report[section]["errors"])
            warnings.extend(report[section]["warnings"])

    report["status"] = "failed" if errors or (args.fail_on_warning and warnings) else "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
