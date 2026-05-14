#!/usr/bin/env python3
"""Check that OPELA pause metadata is used only as session-gap hints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_TIMELINE_TERMS = (
    "month",
    "months",
    "monthly",
    "year",
    "years",
    "202",
    "월",
    "개월",
    "년",
    "月",
    "ヶ月",
    "年",
)
SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


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
        for line in f:
            if line.strip():
                yield json.loads(line)


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [key for key, value in conversation.items() if SESSION_KEY_RE.match(key) and isinstance(value, list)]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def load_opela_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def sample_to_source_doc(provenance_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    conflicts: list[str] = []
    for row in iter_jsonl(provenance_path):
        if row.get("source_dataset") != "OPELA":
            continue
        if row.get("source_origin") != "original_turn":
            continue
        sample_id = str(row.get("sample_id"))
        doc_id = str(row.get("source_record_id"))
        previous = mapping.setdefault(sample_id, doc_id)
        if previous != doc_id:
            conflicts.append(f"{sample_id}: source_record_id conflict {previous!r} vs {doc_id!r}")
    if conflicts:
        raise ValueError("; ".join(conflicts[:10]))
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary/OPELA-LoCoMo-style-eval.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/OPELA-LoCoMo-style-eval/"
            "OPELA-LoCoMo-style-eval_provenance.jsonl"
        ),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("datasets/OPELA/data/oplea_open_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/opela_temporal_policy_report.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    source_rows = load_opela_rows(args.source_csv)
    try:
        doc_by_sample = sample_to_source_doc(args.provenance)
    except ValueError as exc:
        doc_by_sample = {}
        errors.append(str(exc))

    samples = load_json(args.primary_json)
    checked_sessions = 0
    per_sample: dict[str, Any] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        if sample.get("source_dataset") != "OPELA":
            errors.append(f"{sample_id}: source_dataset={sample.get('source_dataset')!r} expected 'OPELA'")
            continue
        doc_id = doc_by_sample.get(sample_id)
        if not doc_id:
            errors.append(f"{sample_id}: no original-turn provenance source_record_id found")
            continue
        source_row = source_rows.get(doc_id)
        if not source_row:
            errors.append(f"{sample_id}: source doc_id not found in OPELA CSV: {doc_id}")
            continue
        pause_hour = str(source_row.get("pause_hour", ""))
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            errors.append(f"{sample_id}: conversation must be object")
            continue
        sample_errors: list[str] = []
        for session_key in session_keys(conversation):
            session_num = int(session_key.rsplit("_", 1)[1])
            key = f"{session_key}_date_time"
            value = str(conversation.get(key, ""))
            checked_sessions += 1
            expected = f"OPELA virtual session {session_num}; pause_hours={pause_hour}"
            if value != expected:
                sample_errors.append(f"{sample_id}: {key}={value!r} expected={expected!r}")
            lowered = value.lower()
            found_forbidden = [term for term in FORBIDDEN_TIMELINE_TERMS if term in lowered or term in value]
            if found_forbidden:
                sample_errors.append(f"{sample_id}: {key} contains forbidden timeline terms {found_forbidden}: {value!r}")
        if sample_errors:
            errors.extend(sample_errors[:10])
            if len(sample_errors) > 10:
                errors.append(f"{sample_id}: additional temporal-policy errors={len(sample_errors) - 10}")
        per_sample[sample_id] = {
            "source_record_id": doc_id,
            "pause_hour": pause_hour,
            "session_count": len(session_keys(conversation)),
            "errors": sample_errors[:10],
            "error_count": len(sample_errors),
        }

    if checked_sessions == 0:
        errors.append("no OPELA sessions checked")

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(args.primary_json),
            "provenance": file_record(args.provenance),
            "source_csv": file_record(args.source_csv),
        },
        "policy": {
            "session_datetime_format": "OPELA virtual session {session_num}; pause_hours={source.pause_hour}",
            "forbidden_timeline_terms": list(FORBIDDEN_TIMELINE_TERMS),
            "no_absolute_or_month_scale_timeline_claims": True,
        },
        "checked_samples": len(samples) if isinstance(samples, list) else None,
        "checked_sessions": checked_sessions,
        "per_sample": per_sample,
        "warnings": warnings,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
