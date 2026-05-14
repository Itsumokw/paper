#!/usr/bin/env python3
"""Check LoCoMo-style session/turn ordering and provenance order alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_ARTIFACTS = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}
SYNTHETIC_TURN_ORIGINS = {"synthetic_bridge_turn", "synthetic_continuation_turn"}


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def session_index(key: str) -> int:
    suffix = key.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        [
            key
            for key, value in conversation.items()
            if key.startswith("session_")
            and not key.endswith("_date_time")
            and isinstance(value, list)
        ],
        key=session_index,
    )


def primary_turn_sequence(samples: list[dict[str, Any]], errors: list[str], artifact: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        conversation = sample.get("conversation", {})
        keys = session_keys(conversation)
        expected_keys = [f"session_{index}" for index in range(1, len(keys) + 1)]
        if keys != expected_keys:
            errors.append(f"{artifact} {sample_id}: session keys={keys[:10]} expected consecutive session_1..session_{len(keys)}")
        for key in keys:
            s_idx = session_index(key)
            date_key = f"{key}_date_time"
            if date_key not in conversation:
                errors.append(f"{artifact} {sample_id}: missing {date_key}")
            turns = conversation.get(key, [])
            if not isinstance(turns, list):
                errors.append(f"{artifact} {sample_id}: {key} is not a list")
                continue
            for turn_index, turn in enumerate(turns, start=1):
                expected_dia_id = f"D{s_idx}:{turn_index}"
                dia_id = str(turn.get("dia_id"))
                if dia_id != expected_dia_id:
                    errors.append(f"{artifact} {sample_id}: dia_id={dia_id!r} expected={expected_dia_id!r}")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "session_id": key,
                        "turn_index": turn_index,
                        "dia_id": dia_id,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=Path("datasets/locomo_style_eval/primary"))
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/session_order_report.json"))
    args = parser.parse_args()

    errors: list[str] = []
    per_artifact: dict[str, Any] = {}
    primary_files: list[Path] = []
    provenance_files: list[Path] = []
    dia_pattern = re.compile(r"^D([1-9]\d*):([1-9]\d*)$")

    for source, artifact in SOURCE_ARTIFACTS.items():
        primary_path = args.primary_root / f"{artifact}.json"
        provenance_path = args.sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        primary_files.append(primary_path)
        provenance_files.append(provenance_path)
        samples = load_json(primary_path)
        primary_rows = primary_turn_sequence(samples, errors, artifact)
        provenance_rows = list(iter_jsonl(provenance_path))

        if len(primary_rows) != len(provenance_rows):
            errors.append(
                f"{artifact}: primary turn count={len(primary_rows)} provenance rows={len(provenance_rows)}"
            )

        order_mismatches = 0
        provenance_metadata_mismatches = 0
        source_order_policy_rows = 0
        session_origin_counts: dict[tuple[str, str], set[str]] = {}
        for index, (primary, provenance) in enumerate(zip(primary_rows, provenance_rows), start=1):
            expected_key = (primary["sample_id"], primary["dia_id"])
            actual_key = (str(provenance.get("sample_id")), str(provenance.get("dia_id")))
            if actual_key != expected_key:
                order_mismatches += 1
                if order_mismatches <= 10:
                    errors.append(
                        f"{artifact}: provenance row {index} key={actual_key} expected primary order key={expected_key}"
                    )
            if str(provenance.get("session_id")) != primary["session_id"]:
                provenance_metadata_mismatches += 1
                if provenance_metadata_mismatches <= 10:
                    errors.append(
                        f"{artifact} {expected_key}: provenance session_id={provenance.get('session_id')!r} "
                        f"expected={primary['session_id']!r}"
                    )
            if int(provenance.get("turn_index", -1)) != primary["turn_index"]:
                provenance_metadata_mismatches += 1
                if provenance_metadata_mismatches <= 10:
                    errors.append(
                        f"{artifact} {expected_key}: provenance turn_index={provenance.get('turn_index')!r} "
                        f"expected={primary['turn_index']!r}"
                    )
            match = dia_pattern.fullmatch(primary["dia_id"])
            if not match:
                errors.append(f"{artifact} {expected_key}: invalid dia_id format")
            if provenance.get("order_policy") == "source_order":
                source_order_policy_rows += 1
            session_origin_counts.setdefault(
                (str(provenance.get("sample_id")), str(provenance.get("session_id"))),
                set(),
            ).add(str(provenance.get("source_origin")))

        if order_mismatches > 10:
            errors.append(f"{artifact}: total provenance order mismatches={order_mismatches}")
        if provenance_metadata_mismatches > 10:
            errors.append(f"{artifact}: total provenance metadata mismatches={provenance_metadata_mismatches}")
        mixed_synthetic_original_sessions = 0
        for (sample_id, session_id), origins in sorted(session_origin_counts.items()):
            if "original_turn" in origins and origins & SYNTHETIC_TURN_ORIGINS:
                mixed_synthetic_original_sessions += 1
                if mixed_synthetic_original_sessions <= 10:
                    errors.append(
                        f"{artifact} {sample_id} {session_id}: synthetic turns share a session with original_turn; "
                        f"origins={sorted(origins)}"
                    )
        if mixed_synthetic_original_sessions > 10:
            errors.append(
                f"{artifact}: total sessions mixing synthetic turns with original_turn={mixed_synthetic_original_sessions}"
            )

        per_artifact[artifact] = {
            "source_dataset": source,
            "primary_path": str(primary_path),
            "provenance_path": str(provenance_path),
            "primary_turns": len(primary_rows),
            "provenance_rows": len(provenance_rows),
            "provenance_order_mismatches": order_mismatches,
            "provenance_metadata_mismatches": provenance_metadata_mismatches,
            "source_order_policy_rows": source_order_policy_rows,
            "mixed_synthetic_original_sessions": mixed_synthetic_original_sessions,
        }

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_files": [file_record(path) for path in primary_files],
            "provenance_files": [file_record(path) for path in provenance_files],
        },
        "primary_root": str(args.primary_root),
        "sidecar_root": str(args.sidecar_root),
        "per_artifact": per_artifact,
        "errors": errors,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
