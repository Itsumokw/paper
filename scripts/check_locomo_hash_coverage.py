#!/usr/bin/env python3
"""Check that original-turn provenance rows are covered by hash_check sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACTS = [
    "PerLTQA-LoCoMo-style-eval",
    "OPELA-LoCoMo-style-eval",
    "JLongChat-LoCoMo-style-eval",
    "deL1L2IM-LoCoMo-style-eval",
]
REQUIRED_SOURCE_FAMILIES = {"JLongChat_LAC", "JLongChat_JMSC", "deL1L2IM"}


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


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("sample_id")), str(row.get("dia_id")), str(row.get("source_turn_id")))


def source_family(row: dict[str, Any]) -> str:
    source_dataset = str(row.get("source_dataset"))
    source_file = str(row.get("source_file", ""))
    if source_dataset == "JLongChat":
        if "lac-public-dialogue" in source_file:
            return "JLongChat_LAC"
        if "jmsc-public-dialogue" in source_file:
            return "JLongChat_JMSC"
    if source_dataset == "deL1L2IM":
        return "deL1L2IM"
    return source_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/hash_coverage_report.json"))
    args = parser.parse_args()

    errors: list[str] = []
    per_artifact: dict[str, Any] = {}
    provenance_files: list[Path] = []
    hash_check_files: list[Path] = []
    family_rows: dict[str, Any] = defaultdict(
        lambda: {
            "original_turns": 0,
            "hash_check_rows": 0,
            "missing_hash_check_rows": 0,
            "hash_mismatch_rows": 0,
            "bad_status_rows": 0,
            "sample_ids": set(),
            "source_files": set(),
        }
    )

    for artifact in ARTIFACTS:
        sidecar_dir = args.sidecar_root / artifact
        provenance_path = sidecar_dir / f"{artifact}_provenance.jsonl"
        hash_path = sidecar_dir / f"{artifact}_hash_check.jsonl"
        provenance_files.append(provenance_path)
        hash_check_files.append(hash_path)

        original_rows = {
            key(row): row
            for row in iter_jsonl(provenance_path)
            if row.get("source_origin") == "original_turn"
        }
        hash_rows = {key(row): row for row in iter_jsonl(hash_path)}

        missing = sorted(set(original_rows) - set(hash_rows))
        extra = sorted(set(hash_rows) - set(original_rows))
        mismatch = []
        bad_status = []
        for item_key, hash_row in hash_rows.items():
            if hash_row.get("status") != "captured_for_recheck":
                bad_status.append(item_key)
            original = original_rows.get(item_key)
            if original and original.get("raw_text_hash") != hash_row.get("raw_text_hash"):
                mismatch.append(item_key)

        for item_key, original in original_rows.items():
            family = source_family(original)
            family_rows[family]["original_turns"] += 1
            family_rows[family]["sample_ids"].add(str(original.get("sample_id")))
            family_rows[family]["source_files"].add(str(original.get("source_file")))
            hash_row = hash_rows.get(item_key)
            if hash_row is None:
                family_rows[family]["missing_hash_check_rows"] += 1
                continue
            family_rows[family]["hash_check_rows"] += 1
            if hash_row.get("status") != "captured_for_recheck":
                family_rows[family]["bad_status_rows"] += 1
            if original.get("raw_text_hash") != hash_row.get("raw_text_hash"):
                family_rows[family]["hash_mismatch_rows"] += 1

        if missing:
            errors.append(f"{artifact}: missing hash_check rows for {len(missing)} original turns")
        if extra:
            errors.append(f"{artifact}: hash_check has {len(extra)} rows not matched to original_turn provenance")
        if mismatch:
            errors.append(f"{artifact}: raw_text_hash mismatch in {len(mismatch)} hash_check rows")
        if bad_status:
            errors.append(f"{artifact}: unexpected hash_check status in {len(bad_status)} rows")

        per_artifact[artifact] = {
            "provenance_original_turns": len(original_rows),
            "hash_check_rows": len(hash_rows),
            "missing_hash_check_rows": len(missing),
            "extra_hash_check_rows": len(extra),
            "hash_mismatch_rows": len(mismatch),
            "bad_status_rows": len(bad_status),
        }

    per_source_family: dict[str, Any] = {}
    for family, row in sorted(family_rows.items()):
        per_source_family[family] = {
            "original_turns": row["original_turns"],
            "hash_check_rows": row["hash_check_rows"],
            "missing_hash_check_rows": row["missing_hash_check_rows"],
            "hash_mismatch_rows": row["hash_mismatch_rows"],
            "bad_status_rows": row["bad_status_rows"],
            "sample_count": len(row["sample_ids"]),
            "source_files": sorted(row["source_files"]),
        }
        if row["missing_hash_check_rows"]:
            errors.append(f"{family}: missing hash_check rows for {row['missing_hash_check_rows']} original turns")
        if row["hash_mismatch_rows"]:
            errors.append(f"{family}: raw_text_hash mismatch in {row['hash_mismatch_rows']} rows")
        if row["bad_status_rows"]:
            errors.append(f"{family}: unexpected hash_check status in {row['bad_status_rows']} rows")

    for family in sorted(REQUIRED_SOURCE_FAMILIES):
        if per_source_family.get(family, {}).get("original_turns", 0) <= 0:
            errors.append(f"{family}: required source family has no original_turn hash coverage rows")

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "provenance_files": [file_record(path) for path in provenance_files],
            "hash_check_files": [file_record(path) for path in hash_check_files],
        },
        "sidecar_root": str(args.sidecar_root),
        "per_artifact": per_artifact,
        "per_source_family": per_source_family,
        "required_source_families": sorted(REQUIRED_SOURCE_FAMILIES),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
