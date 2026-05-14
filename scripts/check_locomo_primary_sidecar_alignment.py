#!/usr/bin/env python3
"""Check primary turn records against provenance sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_ARTIFACTS = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
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


def primary_turns(samples: list[dict[str, Any]]):
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        source_dataset = str(sample.get("source_dataset"))
        conversation = sample.get("conversation", {})
        allowed_speakers = {str(conversation.get("speaker_a")), str(conversation.get("speaker_b"))}
        for session_key in session_keys(conversation):
            for turn_index, turn in enumerate(conversation.get(session_key, []), start=1):
                yield {
                    "source_dataset": source_dataset,
                    "sample_id": sample_id,
                    "session_id": session_key,
                    "turn_index": turn_index,
                    "dia_id": str(turn.get("dia_id")),
                    "speaker": str(turn.get("speaker")),
                    "text": str(turn.get("text", "")),
                    "allowed_speakers": allowed_speakers,
                }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary"),
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=Path("datasets/locomo_style_eval/sidecars"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary_sidecar_alignment_report.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    per_artifact: dict[str, Any] = {}
    primary_files: list[Path] = []
    provenance_files: list[Path] = []

    for source_dataset, artifact in SOURCE_ARTIFACTS.items():
        primary_path = args.primary_root / f"{artifact}.json"
        provenance_path = args.sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        primary_files.append(primary_path)
        provenance_files.append(provenance_path)
        samples = load_json(primary_path)
        provenance_rows = {
            (str(row.get("sample_id")), str(row.get("dia_id"))): row
            for row in iter_jsonl(provenance_path)
        }
        seen_primary: set[tuple[str, str]] = set()
        source_origin_counts: Counter[str] = Counter()
        primary_turn_count = 0
        missing_provenance = 0
        text_mismatch = 0
        hash_mismatch = 0
        speaker_mismatch = 0
        missing_source_speaker = 0

        for turn in primary_turns(samples):
            primary_turn_count += 1
            key = (turn["sample_id"], turn["dia_id"])
            seen_primary.add(key)
            if turn["source_dataset"] != source_dataset:
                errors.append(
                    f"{artifact}: primary source_dataset={turn['source_dataset']!r} "
                    f"does not match expected {source_dataset!r}"
                )
            if turn["speaker"] not in turn["allowed_speakers"]:
                speaker_mismatch += 1
                if speaker_mismatch <= 10:
                    errors.append(
                        f"{artifact} {turn['sample_id']} {turn['dia_id']}: "
                        f"speaker={turn['speaker']!r} not in {sorted(turn['allowed_speakers'])}"
                    )
            provenance = provenance_rows.get(key)
            if provenance is None:
                missing_provenance += 1
                if missing_provenance <= 10:
                    errors.append(f"{artifact} {turn['sample_id']} {turn['dia_id']}: missing provenance row")
                continue
            source_origin = str(provenance.get("source_origin"))
            source_origin_counts[source_origin] += 1
            if str(provenance.get("text", "")) != turn["text"]:
                text_mismatch += 1
                if text_mismatch <= 10:
                    errors.append(f"{artifact} {turn['sample_id']} {turn['dia_id']}: primary/provenance text mismatch")
            if provenance.get("raw_text_hash") != sha256_text(turn["text"]):
                hash_mismatch += 1
                if hash_mismatch <= 10:
                    errors.append(f"{artifact} {turn['sample_id']} {turn['dia_id']}: raw_text_hash does not match primary text")
            if source_origin == "original_turn" and not str(provenance.get("source_speaker") or "").strip():
                missing_source_speaker += 1
                if missing_source_speaker <= 10:
                    warnings.append(f"{artifact} {turn['sample_id']} {turn['dia_id']}: original_turn missing source_speaker")

        extra_provenance = sorted(set(provenance_rows) - seen_primary)
        if extra_provenance:
            errors.append(f"{artifact}: provenance has {len(extra_provenance)} rows not present in primary JSON")
        if missing_source_speaker:
            errors.append(f"{artifact}: {missing_source_speaker} original_turn rows missing source_speaker")
        if missing_provenance > 10:
            errors.append(f"{artifact}: total primary turns missing provenance={missing_provenance}")
        if text_mismatch > 10:
            errors.append(f"{artifact}: total primary/provenance text mismatches={text_mismatch}")
        if hash_mismatch > 10:
            errors.append(f"{artifact}: total primary/provenance hash mismatches={hash_mismatch}")
        if speaker_mismatch > 10:
            errors.append(f"{artifact}: total speaker mismatches={speaker_mismatch}")

        per_artifact[artifact] = {
            "primary_path": str(primary_path),
            "provenance_path": str(provenance_path),
            "primary_turns": primary_turn_count,
            "provenance_rows": len(provenance_rows),
            "source_origin_counts_aligned": dict(sorted(source_origin_counts.items())),
            "missing_provenance_rows": missing_provenance,
            "extra_provenance_rows": len(extra_provenance),
            "text_mismatch_rows": text_mismatch,
            "hash_mismatch_rows": hash_mismatch,
            "speaker_mismatch_rows": speaker_mismatch,
            "missing_source_speaker_rows": missing_source_speaker,
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
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
