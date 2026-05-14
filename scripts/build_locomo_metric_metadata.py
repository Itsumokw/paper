#!/usr/bin/env python3
"""Build per-QA grouping metadata for fixed baseline reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_TO_ARTIFACT = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def session_from_dia_id(dia_id: str) -> str | None:
    if not dia_id.startswith("D") or ":" not in dia_id:
        return None
    session_raw = dia_id[1:].split(":", 1)[0]
    return f"session_{session_raw}" if session_raw.isdigit() else None


def load_provenance(sidecar_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in SOURCE_TO_ARTIFACT.values():
        path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        for row in iter_jsonl(path):
            rows[(str(row.get("sample_id")), str(row.get("dia_id")))] = row
    return rows


def evidence_origins(
    sample_id: str,
    evidence: list[Any],
    provenance: dict[tuple[str, str], dict[str, Any]],
    errors: list[str],
    prefix: str,
) -> list[str]:
    origins: list[str] = []
    for raw_dia_id in evidence:
        dia_id = str(raw_dia_id)
        row = provenance.get((sample_id, dia_id))
        if row is None:
            errors.append(f"{prefix}: evidence dia_id={dia_id!r} missing provenance")
            continue
        origins.append(str(row.get("source_origin")))
    return sorted(set(origins))


def origin_bucket(origins: list[str], *, cat5: bool) -> str:
    if origins:
        return "+".join(origins)
    return "negative_only" if cat5 else "none"


def qa_metadata_rows(
    dataset: list[dict[str, Any]],
    provenance: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    by_source: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_cross_session: Counter[str] = Counter()
    by_evidence_provenance: Counter[str] = Counter()

    for sample in dataset:
        sample_id = str(sample.get("sample_id"))
        source = str(sample.get("source_dataset"))
        language = str(sample.get("language"))
        by_source[source] += len(sample.get("qa", []))
        by_language[language] += len(sample.get("qa", []))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            category = int(qa.get("category"))
            cat5 = category == 5
            evidence = qa.get("negative_evidence", []) if cat5 else qa.get("evidence", [])
            evidence_ids = [str(item) for item in evidence]
            sessions = sorted({session for session in (session_from_dia_id(item) for item in evidence_ids) if session})
            whether_cross_session = len(sessions) > 1
            prefix = f"{sample_id} qa_idx={qa_idx}"
            origins = evidence_origins(sample_id, evidence_ids, provenance, errors, prefix)
            bucket = origin_bucket(origins, cat5=cat5)

            by_category[str(category)] += 1
            by_cross_session[str(whether_cross_session).lower()] += 1
            by_evidence_provenance[bucket] += 1
            rows.append(
                {
                    "source_dataset": source,
                    "language": language,
                    "sample_id": sample_id,
                    "qa_idx": qa_idx,
                    "category": category,
                    "answerable": not cat5,
                    "whether_cross_session": whether_cross_session,
                    "evidence_ids": evidence_ids,
                    "evidence_sessions": sessions,
                    "evidence_origins": origins,
                    "evidence_provenance": bucket,
                }
            )

    summary = {
        "rows": len(rows),
        "by_source_dataset": dict(sorted(by_source.items())),
        "by_language": dict(sorted(by_language.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_cross_session": dict(sorted(by_cross_session.items())),
        "by_evidence_provenance": dict(sorted(by_evidence_provenance.items())),
    }
    return rows, errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-json", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_json(args.primary_json)
    if not isinstance(dataset, list):
        raise TypeError("primary JSON must be a list")
    provenance = load_provenance(args.sidecar_root)
    rows, errors, count_summary = qa_metadata_rows(dataset, provenance)
    write_jsonl(args.output_jsonl, rows)
    report = {
        "status": "passed" if not errors else "failed",
        "primary_json": str(args.primary_json),
        "primary_sha256": sha256_file(args.primary_json),
        "sidecar_root": str(args.sidecar_root),
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": sha256_file(args.output_jsonl),
        **count_summary,
        "errors": errors,
    }
    write_json(args.summary_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
