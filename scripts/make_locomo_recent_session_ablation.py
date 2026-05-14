#!/usr/bin/env python3
"""Create recent-session context ablation files for LoCoMo-style diagnostics.

The output keeps the same QA set but truncates the visible conversation to
the last N sessions. Evidence IDs may point to turns outside the truncated
conversation by design; these files are for model-side diagnostic runs, not
for primary schema validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qa_counts(data: list[dict[str, Any]]) -> dict[str, int]:
    total = 0
    cat5 = 0
    for sample in data:
        for qa in sample.get("qa", []):
            total += 1
            if qa.get("category") == 5:
                cat5 += 1
    return {
        "samples": len(data),
        "qa_count": total,
        "answerable_qa_count": total - cat5,
        "cat5_qa_count": cat5,
    }


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


def truncate_sample(sample: dict[str, Any], keep_last: int) -> dict[str, Any]:
    item = deepcopy(sample)
    conversation = item["conversation"]
    keep = set(session_keys(conversation)[-keep_last:])
    next_conversation: dict[str, Any] = {
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
    }
    for key in session_keys(conversation):
        if key not in keep:
            continue
        next_conversation[f"{key}_date_time"] = conversation.get(f"{key}_date_time", "")
        next_conversation[key] = conversation[key]
    item["conversation"] = next_conversation
    item["observation"] = {
        key: value
        for key, value in item.get("observation", {}).items()
        if key.removesuffix("_observation") in keep
    }
    item["session_summary"] = {
        key: value
        for key, value in item.get("session_summary", {}).items()
        if key.removesuffix("_summary") in keep
    }
    item["event_summary"] = {
        key: value
        for key, value in item.get("event_summary", {}).items()
        if key.removeprefix("events_") in keep
    }
    return item


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise TypeError("input must be a JSON list")

    full_path = args.output_root / "full_conversation.json"
    last1_path = args.output_root / "last_session_only.json"
    last3_path = args.output_root / "last_3_sessions_only.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, full_path)
    write_json(last1_path, [truncate_sample(sample, 1) for sample in data])
    write_json(last3_path, [truncate_sample(sample, 3) for sample in data])

    report = {
        "status": "created",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output_root": str(args.output_root),
        **qa_counts(data),
        "files": {
            "full_conversation": str(full_path),
            "last_session_only": str(last1_path),
            "last_3_sessions_only": str(last3_path),
        },
        "note": "QA is unchanged. Evidence may refer to hidden turns in truncated diagnostic contexts.",
    }
    report_path = args.output_root / "recent_session_ablation_manifest.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
