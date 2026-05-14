#!/usr/bin/env python3
"""No-model compatibility checks for local LoCoMo baseline loaders."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PAPER_ROOT = Path("/home/stu0032/paper")
LOADER_FILES = [
    Path("scripts/locomo_2026_sota.py"),
    Path("baseline/A-MEM/load_dataset.py"),
    Path("baseline/SimpleMem/OmniSimpleMem/benchmarks/locomo/run_locomo.py"),
]


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


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PAPER_ROOT / path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def check_mem0_shape(samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    sessions = 0
    turns = 0
    for sample_idx, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", f"sample_{sample_idx}"))
        conversation = sample.get("conversation", {})
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        allowed = {speaker_a, speaker_b}
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            chats = conversation.get(key)
            if not isinstance(chats, list):
                continue
            date_time_key = key + "_date_time"
            if date_time_key not in conversation:
                errors.append(f"{sample_id}: mem0 missing timestamp key {date_time_key}")
                continue
            sessions += 1
            for chat_idx, chat in enumerate(chats, start=1):
                turns += 1
                speaker = chat.get("speaker")
                if speaker not in allowed:
                    errors.append(
                        f"{sample_id}: mem0 unknown speaker at {key}[{chat_idx}] "
                        f"{speaker!r}, expected one of {sorted(str(item) for item in allowed)}"
                    )
                if not str(chat.get("text", "")).strip():
                    errors.append(f"{sample_id}: mem0 empty text at {key}[{chat_idx}]")
    return {"sessions_seen": sessions, "turns_seen": turns, "errors": errors}


def check_locomo_2026_adapter(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    module = load_module("locomo_2026_sota_for_loader_smoke", repo_path(LOADER_FILES[0]))
    errors: list[str] = []
    loaded = module.load_locomo(path)
    sample_count, qa_count, categories = module.dataset_count(path)
    if sample_count != len(samples):
        errors.append(f"dataset_count samples={sample_count} != {len(samples)}")
    expected_qa = sum(len(sample.get("qa", [])) for sample in samples)
    if qa_count != expected_qa:
        errors.append(f"dataset_count qa={qa_count} != {expected_qa}")
    rendered_lines = 0
    empty_cat5_refs = 0
    for idx, sample in enumerate(loaded):
        sid = module.sample_id(sample, idx)
        if not sid:
            errors.append(f"sample[{idx}]: empty sample_id")
        nums = module.session_numbers(sample)
        if not nums:
            errors.append(f"{sid}: no session numbers")
        for session_num in nums:
            lines = module.session_lines(sample, session_num)
            if not lines:
                errors.append(f"{sid}: no rendered lines for session {session_num}")
            rendered_lines += len(lines)
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            reference = module.qa_reference(qa)
            if str(qa.get("category")) == "5" and not reference:
                empty_cat5_refs += 1
                errors.append(f"{sid}: cat5 qa[{qa_idx}] has empty locomo_2026 reference")
    return {
        "samples": sample_count,
        "qa": qa_count,
        "categories": dict(sorted((str(k), v) for k, v in categories.items())),
        "rendered_lines": rendered_lines,
        "empty_cat5_references": empty_cat5_refs,
        "errors": errors,
    }


def check_amem_loader(path: Path) -> dict[str, Any]:
    module = load_module("amem_load_dataset_for_loader_smoke", repo_path(LOADER_FILES[1]))
    errors: list[str] = []
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            samples = module.load_locomo_dataset(path)
        stats = module.get_dataset_statistics(samples)
    except Exception as exc:  # noqa: BLE001
        return {"errors": [str(exc)], "captured_stdout_chars": len(captured.getvalue())}
    cat5_final_missing = 0
    for sample_idx, sample in enumerate(samples):
        for qa_idx, qa in enumerate(sample.qa):
            if qa.category == 5 and not qa.final_answer:
                cat5_final_missing += 1
                errors.append(f"sample[{sample_idx}] qa[{qa_idx}]: A-MEM cat5 final_answer is empty")
    return {
        "samples": stats.get("num_samples"),
        "qa": stats.get("total_qa_pairs"),
        "sessions": stats.get("total_sessions"),
        "turns": stats.get("total_turns"),
        "qa_with_adversarial": stats.get("qa_with_adversarial"),
        "cat5_final_missing": cat5_final_missing,
        "captured_stdout_chars": len(captured.getvalue()),
        "errors": errors,
    }


def check_omni_parser(samples: list[dict[str, Any]]) -> dict[str, Any]:
    module = load_module(
        "omni_locomo_runner_for_loader_smoke",
        repo_path(LOADER_FILES[2]),
    )
    errors: list[str] = []
    total_dialogues = 0
    question_texts = 0
    for idx, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", f"sample_{idx}"))
        try:
            dialogues = module._parse_locomo_sample_to_dialogues(sample)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sample_id}: Omni parser failed: {exc}")
            continue
        if not dialogues:
            errors.append(f"{sample_id}: Omni parser produced zero dialogues")
        total_dialogues += len(dialogues)
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            try:
                question = module.build_question_text(qa)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sample_id}: Omni build_question_text qa[{qa_idx}] failed: {exc}")
                continue
            if not str(question).strip():
                errors.append(f"{sample_id}: Omni question text empty for qa[{qa_idx}]")
            question_texts += 1
    return {"dialogues": total_dialogues, "question_texts": question_texts, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = load_json(args.primary_json)
    if not isinstance(samples, list):
        raise SystemExit(f"{args.primary_json} must contain a JSON list")

    source_counts = Counter(str(sample.get("source_dataset")) for sample in samples)
    checks = {
        "mem0_shape": check_mem0_shape(samples),
        "locomo_2026_adapter": check_locomo_2026_adapter(args.primary_json, samples),
        "amem_load_dataset": check_amem_loader(args.primary_json),
        "omni_parser": check_omni_parser(samples),
    }
    errors = []
    for name, check in checks.items():
        for error in check.get("errors", []):
            errors.append(f"{name}: {error}")
    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(args.primary_json),
            "loader_files": [file_record(path) for path in LOADER_FILES],
        },
        "primary_json": str(args.primary_json),
        "model_calls": 0,
        "source_counts": dict(sorted(source_counts.items())),
        "checks": checks,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
