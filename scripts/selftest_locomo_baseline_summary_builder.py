#!/usr/bin/env python3
"""Self-test fixed-baseline summary builder fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_METHODS = ["Full Context", "A-MEM", "Mem0", "SimpleMem", "HiGMem"]
REQUIRED_FIXED_MODEL = "Qwen/Qwen3-8B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_fixture(tempdir: Path) -> dict[str, Any]:
    dataset = tempdir / "multilingual_locomo_style_eval_audited.json"
    metadata = tempdir / "metric_metadata.jsonl"
    settings = tempdir / "fixed_eval_settings.json"
    summary = tempdir / "summary.json"
    prediction_root = tempdir / "predictions"

    write_json(
        dataset,
        [
            {
                "sample_id": "fixture_0",
                "source_dataset": "Fixture",
                "language": "en",
                "split": "eval",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1_date_time": "2026-01-01",
                    "session_1": [
                        {"speaker": "A", "dia_id": "D1:1", "text": "Alice likes tea."},
                    ],
                    "session_2_date_time": "2026-01-02",
                    "session_2": [
                        {"speaker": "B", "dia_id": "D2:1", "text": "Bob remembered Alice's tea."},
                    ],
                },
                "observation": {},
                "session_summary": {},
                "event_summary": {},
                "qa": [
                    {
                        "question": "What does Alice like?",
                        "answer": "tea",
                        "category": 1,
                        "evidence": ["D1:1"],
                    },
                    {
                        "question": "What did Bob remember?",
                        "answer": "Alice's tea",
                        "category": 2,
                        "evidence": ["D1:1", "D2:1"],
                    },
                    {
                        "question": "Does Alice live on Mars?",
                        "category": 5,
                        "evidence": [],
                        "adversarial_answer": "unsupported",
                    },
                ],
            }
        ],
    )
    write_jsonl(
        metadata,
        [
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 0,
                "category": 1,
                "answerable": True,
                "whether_cross_session": False,
                "evidence_provenance": "original_turn",
            },
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 1,
                "category": 2,
                "answerable": True,
                "whether_cross_session": True,
                "evidence_provenance": "original_turn",
            },
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 2,
                "category": 5,
                "answerable": False,
                "whether_cross_session": False,
                "evidence_provenance": "negative_only",
            },
        ],
    )
    write_json(
        settings,
        {
            "status": "predeclared",
            "anti_tuning_contract": {
                "benchmark_tuning_allowed": False,
                "frozen_before_final_runs": True,
                "do_not_tune_on_this_benchmark": [
                    "prompt format",
                    "retrieval top-k",
                    "chunking",
                    "memory compression",
                    "context truncation",
                    "cat5 refusal rules",
                ],
                "allowed_service_check": "tiny chat readiness preflight only; response must not enter dataset or metrics",
            },
            "dataset": {
                "audited_primary": str(dataset),
                "input_policy": "conversation_only",
                "summary_visible": False,
            },
            "model": {
                "served_model": "Qwen/Qwen3-8B",
                "temperature": 0,
            },
            "fixed_baselines": {
                "summary_json": str(summary),
                "metric_metadata_jsonl": str(metadata),
                "settings_source": "predeclared_fixed_eval_settings",
                "required_methods": REQUIRED_METHODS,
                "optional_methods": {
                    "MemGAS": {
                        "required_for_release": False,
                        "include_policy": "only_if_clean_metrics_available",
                        "include_in_main_table": False,
                        "clean_metrics_required": True,
                    }
                },
                "execution_contract": {
                    "final_dataset": str(dataset),
                    "model": "Qwen/Qwen3-8B",
                    "input_policy": "conversation_only",
                    "summary_visible": False,
                    "summary_builder": "scripts/build_locomo_baseline_summary.py",
                    "legacy_locomo10_defaults_forbidden": True,
                    "prediction_jsonl_contract": {
                        "identity_fields": ["sample_id", "qa_idx"],
                        "required_row_fields": ["sample_id", "qa_idx", "model", "dataset_sha256"],
                        "required_model": REQUIRED_FIXED_MODEL,
                        "prediction_field_options": ["prediction", "answer", "response"],
                    },
                },
                "common": {
                    "input_policy": "conversation_only",
                    "summary_visible": False,
                    "cat5_metrics": ["refusal_accuracy", "unsupported_claim_rate"],
                },
                "methods": {
                    method: {"prediction_source": "fixed_qwen3_8b_prediction_jsonl"}
                    for method in REQUIRED_METHODS
                },
            },
            "reporting": {
                "group_by": [
                    "source_dataset",
                    "language",
                    "category",
                    "whether_cross_session",
                    "evidence_origin",
                ]
            },
        },
    )
    prediction_files: dict[str, Path] = {}
    dataset_sha256 = sha256_file(dataset)
    rows = [
        {
            "sample_id": "fixture_0",
            "qa_idx": 0,
            "model": REQUIRED_FIXED_MODEL,
            "dataset_sha256": dataset_sha256,
            "prediction": "tea",
        },
        {
            "sample_id": "fixture_0",
            "qa_idx": 1,
            "model": REQUIRED_FIXED_MODEL,
            "dataset_sha256": dataset_sha256,
            "prediction": "Alice's tea",
        },
        {
            "sample_id": "fixture_0",
            "qa_idx": 2,
            "model": REQUIRED_FIXED_MODEL,
            "dataset_sha256": dataset_sha256,
            "prediction": "Not enough information.",
        },
    ]
    for method in REQUIRED_METHODS:
        path = prediction_root / f"{method.replace(' ', '_')}.jsonl"
        write_jsonl(path, rows)
        prediction_files[method] = path
    return {
        "dataset": dataset,
        "metadata": metadata,
        "settings": settings,
        "summary": summary,
        "prediction_files": prediction_files,
    }


def builder_command(builder: Path, paths: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(builder),
        "--dataset",
        str(paths["dataset"]),
        "--metric-metadata",
        str(paths["metadata"]),
        "--settings-file",
        str(paths["settings"]),
        "--output",
        str(paths["summary"]),
    ]
    for method, path in paths["prediction_files"].items():
        command.extend(["--prediction-jsonl", f"{method}={path}"])
    return command


def run_json(command: list[str]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    if completed.stderr.strip() and completed.stdout.strip():
        parsed["stderr"] = completed.stderr
    return completed.returncode, parsed


def case_result(
    name: str,
    returncode: int,
    result: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = "; ".join(str(error) for error in result.get("errors", []))
    observed_success = returncode == 0 and result.get("status") in {"completed", "passed"}
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None or expected_error_fragment in errors or expected_error_fragment in str(result.get("error", ""))
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "observed_status": result.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": result.get("errors", result.get("error", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, default=Path("scripts/build_locomo_baseline_summary.py"))
    parser.add_argument("--validator", type=Path, default=Path("scripts/validate_locomo_baseline_results.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/baseline_summary_builder_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_baseline_summary_builder_selftest_") as tmp:
        tempdir = Path(tmp)
        paths = build_fixture(tempdir)
        dataset_sha256 = sha256_file(paths["dataset"])
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(case_result("valid_prediction_files_build_completed_summary", rc, result, expect_success=True))
        rc, result = run_json([sys.executable, str(args.validator), str(paths["summary"])])
        cases.append(case_result("built_summary_passes_baseline_validator", rc, result, expect_success=True))

        paths = build_fixture(tempdir)
        paths["prediction_files"].pop("Mem0")
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "missing_required_method_prediction_file_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="missing prediction files for required methods",
            )
        )

        paths = build_fixture(tempdir)
        paths["prediction_files"]["MemGAS"] = paths["prediction_files"]["Mem0"]
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "unexpected_prediction_method_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="unexpected prediction methods",
            )
        )

        paths = build_fixture(tempdir)
        mem0_path = paths["prediction_files"]["Mem0"]
        write_jsonl(
            mem0_path,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "tea again",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 2,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Not enough information.",
                },
            ],
        )
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "duplicate_and_missing_prediction_keys_are_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="duplicate prediction key",
            )
        )

        paths = build_fixture(tempdir)
        mem0_path = paths["prediction_files"]["Mem0"]
        write_jsonl(
            mem0_path,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 1,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Alice's tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 2,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Not enough information.",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 99,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "extra",
                },
            ],
        )
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "unexpected_prediction_key_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="not found in dataset",
            )
        )

        paths = build_fixture(tempdir)
        mem0_path = paths["prediction_files"]["Mem0"]
        write_jsonl(
            mem0_path,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 1,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Alice's tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 2,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Not enough information.",
                },
            ],
        )
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "missing_prediction_field_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="missing prediction/answer/response field",
            )
        )

        paths = build_fixture(tempdir)
        mem0_path = paths["prediction_files"]["Mem0"]
        write_jsonl(
            mem0_path,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": "Qwen/Qwen2.5-3B-Instruct",
                    "dataset_sha256": dataset_sha256,
                    "prediction": "tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 1,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Alice's tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 2,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Not enough information.",
                },
            ],
        )
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "wrong_prediction_model_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="expected='Qwen/Qwen3-8B'",
            )
        )

        paths = build_fixture(tempdir)
        mem0_path = paths["prediction_files"]["Mem0"]
        write_jsonl(
            mem0_path,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": "0" * 64,
                    "prediction": "tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 1,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Alice's tea",
                },
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 2,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "Not enough information.",
                },
            ],
        )
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "wrong_prediction_dataset_sha256_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="dataset_sha256",
            )
        )

        paths = build_fixture(tempdir)
        metadata_rows = [
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 0,
                "category": 5,
                "answerable": False,
                "whether_cross_session": False,
                "evidence_provenance": "negative_only",
            }
        ]
        write_jsonl(paths["metadata"], metadata_rows)
        rc, result = run_json(builder_command(args.builder, paths))
        cases.append(
            case_result(
                "metadata_dataset_alignment_error_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="metric metadata missing dataset QA keys",
            )
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "builder": str(args.builder),
        "builder_sha256": sha256_file(args.builder),
        "validator": str(args.validator),
        "validator_sha256": sha256_file(args.validator),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
