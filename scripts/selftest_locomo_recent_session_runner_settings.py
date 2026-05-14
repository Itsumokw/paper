#!/usr/bin/env python3
"""Self-test recent-session runner audited-input enforcement without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dataset(path: Path) -> None:
    write_json(
        path,
        [
            {
                "sample_id": "fixture_0",
                "source_dataset": "Fixture",
                "language": "en",
                "split": "eval",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1_date_time": "Fixture day 1",
                    "session_1": [
                        {"speaker": "A", "dia_id": "D1:1", "text": "The archive code is cobalt."}
                    ],
                },
                "observation": {"session_1_observation": []},
                "session_summary": {"session_1_summary": ""},
                "event_summary": {"events_session_1": []},
                "qa": [
                    {"question": "What is the archive code?", "answer": "cobalt", "category": 1, "evidence": ["D1:1"]},
                    {"question": "What city was mentioned?", "category": 5, "evidence": [], "adversarial_answer": "unsupported"},
                ],
            }
        ],
    )


def build_settings(path: Path, audited_primary: Path, ablation_root: Path) -> None:
    write_json(
        path,
        {
            "status": "predeclared",
            "dataset": {
                "audited_primary": str(audited_primary),
                "input_policy": "conversation_only",
                "summary_visible": False,
            },
            "model": {"served_model": "Qwen/Qwen3-8B"},
            "fixed_baselines": {"settings_source": "predeclared_fixed_eval_settings"},
            "recent_session_diagnostic": {
                "ablation_root": str(ablation_root),
                "categories": ["1", "2", "3", "4"],
                "max_context_chars": 24000,
                "max_answer_tokens": 96,
                "request_timeout_seconds": 90,
                "workers": 3,
            },
        },
    )


def build_ablation_root(path: Path, input_path: Path, input_hash: str | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    for filename in ("full_conversation.json", "last_session_only.json", "last_3_sessions_only.json"):
        write_json(path / filename, data)
    write_json(
        path / "recent_session_ablation_manifest.json",
        {
            "input": str(input_path),
            "input_sha256": input_hash if input_hash is not None else sha256_file(input_path),
            "files": {
                "full_conversation": str(path / "full_conversation.json"),
                "last_session_only": str(path / "last_session_only.json"),
                "last_3_sessions_only": str(path / "last_3_sessions_only.json"),
            },
        },
    )


def run_dry_runner(root: Path, settings: Path, ablation_root: Path, output_name: str) -> dict[str, Any]:
    runner = Path(__file__).with_name("run_locomo_recent_session_model_diagnostic.py")
    output = root / output_name
    records = root / f"{output_name}.records.jsonl"
    command = [
        sys.executable,
        str(runner),
        "--ablation-root",
        str(ablation_root),
        "--output",
        str(output),
        "--records-output",
        str(records),
        "--model",
        "Qwen/Qwen3-8B",
        "--settings-file",
        str(settings),
        "--enforce-settings",
        "--dry-run",
        "--max-context-chars",
        "24000",
        "--max-answer-tokens",
        "96",
        "--request-timeout",
        "90",
        "--workers",
        "3",
    ]
    subprocess.run(command, check=False, text=True, capture_output=True)
    return json.loads(output.read_text(encoding="utf-8"))


def case_result(name: str, observed: dict[str, Any], expect_error_fragment: str | None) -> dict[str, Any]:
    errors = [str(item) for item in observed.get("settings_errors", [])]
    passed = not errors if expect_error_fragment is None else any(expect_error_fragment in error for error in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expected_error_fragment": expect_error_fragment,
        "settings_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/recent_session_runner_settings_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_recent_runner_selftest_") as tmp:
        root = Path(tmp)
        audited = root / "primary" / "multilingual_locomo_style_eval_audited.json"
        bootstrap = root / "primary" / "multilingual_locomo_style_eval.json"
        build_dataset(audited)
        build_dataset(bootstrap)

        valid_ablation = root / "valid_ablation"
        settings = root / "fixed_eval_settings.json"
        build_ablation_root(valid_ablation, audited)
        build_settings(settings, audited, valid_ablation)
        observed = run_dry_runner(root, settings, valid_ablation, "valid.json")
        cases.append(case_result("audited_ablation_manifest_is_accepted", observed, None))

        bootstrap_ablation = root / "bootstrap_ablation"
        build_ablation_root(bootstrap_ablation, bootstrap)
        build_settings(settings, audited, bootstrap_ablation)
        observed = run_dry_runner(root, settings, bootstrap_ablation, "bootstrap.json")
        cases.append(case_result("bootstrap_ablation_input_is_rejected", observed, "ablation_manifest.input="))

        missing_audited = root / "primary" / "missing_audited.json"
        missing_ablation = root / "missing_ablation"
        build_ablation_root(missing_ablation, bootstrap)
        build_settings(settings, missing_audited, missing_ablation)
        observed = run_dry_runner(root, settings, missing_ablation, "missing.json")
        cases.append(case_result("missing_audited_primary_is_rejected", observed, "audited primary not found"))

        hash_ablation = root / "hash_ablation"
        build_ablation_root(hash_ablation, audited, input_hash="0" * 64)
        build_settings(settings, audited, hash_ablation)
        observed = run_dry_runner(root, settings, hash_ablation, "hash.json")
        cases.append(case_result("ablation_input_hash_mismatch_is_rejected", observed, "input_sha256"))

    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    runner = Path(__file__).with_name("run_locomo_recent_session_model_diagnostic.py")
    report = {
        "status": status,
        "runner_script": str(runner),
        "runner_script_sha256": sha256_file(runner),
        "selftest": str(Path(__file__)),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
