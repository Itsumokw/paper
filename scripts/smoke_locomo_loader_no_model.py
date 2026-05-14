#!/usr/bin/env python3
"""No-model smoke test for LoCoMo-style eval JSON files.

This script verifies that the primary eval JSON can be consumed as a
conversation-only LoCoMo-style benchmark. It intentionally performs zero
local-model, remote-model, embedding, or API calls.
"""

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


def render_conversation_only_context(sample: dict[str, Any]) -> str:
    conversation = sample["conversation"]
    lines: list[str] = []
    for session_key in session_keys(conversation):
        date_key = f"{session_key}_date_time"
        lines.append(f"{session_key} {conversation.get(date_key, '')}".strip())
        for turn in conversation[session_key]:
            lines.append(f"{turn['dia_id']} {turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


def parse_category_expectation(value: str) -> dict[str, int]:
    expected: dict[str, int] = {}
    if not value:
        return expected
    for item in value.split(","):
        key, sep, raw_count = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(
                "category expectations must use KEY=COUNT, e.g. 1=239,2=28"
            )
        key = key.strip()
        try:
            expected[key] = int(raw_count.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid count in {item!r}") from exc
    return expected


def smoke(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"expected top-level JSON list in {path}")

    errors: list[str] = []
    warnings: list[str] = []
    categories: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    total_sessions = 0
    total_turns = 0
    total_qa = 0
    rendered_context_chars = 0

    for sample_index, sample in enumerate(data):
        sample_id = str(sample.get("sample_id", f"sample[{sample_index}]"))
        prefix = f"{sample_id}"
        missing = REQUIRED_SAMPLE_KEYS - set(sample)
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}")
            continue
        if sample.get("split") != "eval":
            errors.append(f"{prefix}: split must be eval")

        source_counts[str(sample.get("source_dataset", ""))] += 1
        language_counts[str(sample.get("language", ""))] += 1
        conversation = sample["conversation"]
        if not isinstance(conversation, dict):
            errors.append(f"{prefix}: conversation must be object")
            continue
        allowed_speakers = {str(conversation.get("speaker_a", "")), str(conversation.get("speaker_b", ""))}

        dia_ids: set[str] = set()
        sessions = session_keys(conversation)
        total_sessions += len(sessions)
        if not sessions:
            errors.append(f"{prefix}: no session_i arrays")
        for session_key in sessions:
            session_num = int(session_key.rsplit("_", 1)[-1])
            if f"{session_key}_date_time" not in conversation:
                errors.append(f"{prefix}: missing {session_key}_date_time")
            for turn_index, turn in enumerate(conversation[session_key], start=1):
                total_turns += 1
                if not isinstance(turn, dict):
                    errors.append(f"{prefix}: {session_key}[{turn_index}] is not object")
                    continue
                for key in ("speaker", "dia_id", "text"):
                    if key not in turn:
                        errors.append(f"{prefix}: {session_key}[{turn_index}] missing {key}")
                speaker = str(turn.get("speaker", ""))
                if speaker not in allowed_speakers:
                    errors.append(
                        f"{prefix}: {session_key}[{turn_index}] speaker {speaker!r} "
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

        try:
            rendered_context_chars += len(render_conversation_only_context(sample))
        except KeyError as exc:
            errors.append(f"{prefix}: cannot render conversation-only context: {exc}")

        qa_items = sample.get("qa")
        if not isinstance(qa_items, list) or not qa_items:
            errors.append(f"{prefix}: qa must be a non-empty list")
            continue
        total_qa += len(qa_items)
        for qa_index, qa in enumerate(qa_items):
            qprefix = f"{prefix} qa[{qa_index}]"
            if not isinstance(qa, dict):
                errors.append(f"{qprefix}: QA is not object")
                continue
            category = str(qa.get("category"))
            categories[category] += 1
            evidence = qa.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"{qprefix}: evidence must be list")
                continue
            if category == "5":
                if evidence:
                    errors.append(f"{qprefix}: cat5 evidence must be empty")
                if "answer" in qa:
                    errors.append(f"{qprefix}: cat5 must omit ordinary answer")
                if not qa.get("adversarial_answer"):
                    warnings.append(f"{qprefix}: cat5 missing adversarial_answer")
            else:
                if not qa.get("answer"):
                    errors.append(f"{qprefix}: answerable QA missing answer")
                if not evidence:
                    errors.append(f"{qprefix}: answerable QA missing evidence")
            for dia_id in evidence:
                if dia_id not in dia_ids:
                    errors.append(f"{qprefix}: evidence dia_id not found: {dia_id}")

    return {
        "status": "failed" if errors else "passed",
        "input_files": {
            "primary_json": file_record(path),
        },
        "path": str(path),
        "model_calls": 0,
        "input_fields_rendered": ["conversation"],
        "input_fields_excluded": ["observation", "session_summary", "event_summary", "sidecars"],
        "samples": len(data),
        "source_datasets": dict(sorted(source_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "sessions": total_sessions,
        "turns": total_turns,
        "qa": total_qa,
        "categories": dict(sorted(categories.items())),
        "rendered_context_chars": rendered_context_chars,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--expected-qa", type=int, default=None)
    parser.add_argument("--expected-categories", type=parse_category_expectation, default={})
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = smoke(args.primary_json)
    errors = list(report["errors"])
    if args.expected_samples is not None and report["samples"] != args.expected_samples:
        errors.append(f"expected {args.expected_samples} samples, found {report['samples']}")
    if args.expected_qa is not None and report["qa"] != args.expected_qa:
        errors.append(f"expected {args.expected_qa} QA, found {report['qa']}")
    categories = {str(key): int(value) for key, value in report["categories"].items()}
    for category, expected in args.expected_categories.items():
        if categories.get(category, 0) != expected:
            errors.append(f"expected category {category} count {expected}, found {categories.get(category, 0)}")
    report["errors"] = errors
    report["status"] = "failed" if errors else "passed"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] != "passed" else 0


if __name__ == "__main__":
    sys.exit(main())
