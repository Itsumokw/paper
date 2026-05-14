#!/usr/bin/env python3
"""Check that primary summary/observation fields are empty loader placeholders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_ARTIFACTS = [
    "PerLTQA-LoCoMo-style-eval",
    "OPELA-LoCoMo-style-eval",
    "JLongChat-LoCoMo-style-eval",
    "deL1L2IM-LoCoMo-style-eval",
    "multilingual_locomo_style_eval",
]
SESSION_RE = re.compile(r"^session_(\d+)$")


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


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [key for key, value in conversation.items() if SESSION_RE.match(key) and isinstance(value, list)]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def check_sample(sample: dict[str, Any], artifact: str, sample_idx: int) -> list[str]:
    errors: list[str] = []
    sample_id = sample.get("sample_id", f"sample[{sample_idx}]")
    prefix = f"{artifact} {sample_id}"
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        return [f"{prefix}: conversation must be object"]
    sessions = session_keys(conversation)

    expected_observation = {f"{session}_observation" for session in sessions}
    expected_summary = {f"{session}_summary" for session in sessions}
    expected_events = {f"events_{session}" for session in sessions}
    expected = {
        "observation": expected_observation,
        "session_summary": expected_summary,
        "event_summary": expected_events,
    }
    empty_allowed = {
        "observation": [],
        "session_summary": "",
        "event_summary": [],
    }

    for field_name, expected_keys in expected.items():
        value = sample.get(field_name)
        if not isinstance(value, dict):
            errors.append(f"{prefix}: {field_name} must be object")
            continue
        actual_keys = set(value)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            errors.append(f"{prefix}: {field_name} missing keys {missing[:10]}")
        if extra:
            errors.append(f"{prefix}: {field_name} has extra keys {extra[:10]}")
        expected_empty = empty_allowed[field_name]
        for key, item in sorted(value.items()):
            if item != expected_empty:
                errors.append(
                    f"{prefix}: {field_name}.{key} must be empty placeholder "
                    f"{expected_empty!r}, got {item!r}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=Path("datasets/locomo_style_eval/primary"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/summary_placeholder_report.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    per_artifact: dict[str, Any] = {}
    primary_files: list[Path] = []
    for artifact in SOURCE_ARTIFACTS:
        path = args.primary_root / f"{artifact}.json"
        primary_files.append(path)
        if not path.is_file():
            errors.append(f"missing primary file: {path}")
            continue
        rows = load_json(path)
        artifact_errors: list[str] = []
        for sample_idx, sample in enumerate(rows):
            if not isinstance(sample, dict):
                artifact_errors.append(f"{artifact} sample[{sample_idx}] must be object")
                continue
            artifact_errors.extend(check_sample(sample, artifact, sample_idx))
        if artifact_errors:
            errors.extend(artifact_errors[:50])
            if len(artifact_errors) > 50:
                errors.append(f"{artifact}: additional summary-placeholder errors={len(artifact_errors) - 50}")
        per_artifact[artifact] = {
            "path": str(path),
            "samples": len(rows) if isinstance(rows, list) else None,
            "errors": artifact_errors[:50],
            "error_count": len(artifact_errors),
        }

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_files": [file_record(path) for path in primary_files if path.is_file()],
        },
        "primary_root": str(args.primary_root),
        "policy": {
            "summary_visible": False,
            "observation_value": [],
            "session_summary_value": "",
            "event_summary_value": [],
        },
        "per_artifact": per_artifact,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
