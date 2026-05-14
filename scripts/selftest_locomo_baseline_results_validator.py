#!/usr/bin/env python3
"""Self-test the fixed-baseline result validator with temporary fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
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


def build_fixture(tempdir: Path) -> dict[str, Path]:
    dataset = tempdir / "multilingual_locomo_style_eval_audited.json"
    settings = tempdir / "fixed_eval_settings.json"
    metadata = tempdir / "metric_metadata.jsonl"
    summary = tempdir / "summary.json"
    prediction_dir = tempdir / "predictions"

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
                        {"speaker": "B", "dia_id": "D1:2", "text": "Bob remembers it."},
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
    prediction_paths: dict[str, Path] = {}
    dataset_sha256 = sha256_file(dataset)
    for method in REQUIRED_METHODS:
        path = prediction_dir / f"{method.replace(' ', '_').replace('-', '_')}.jsonl"
        write_jsonl(
            path,
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
                    "prediction": "I don't know.",
                },
            ],
        )
        prediction_paths[method] = path
    return {
        "dataset": dataset,
        "settings": settings,
        "metadata": metadata,
        "summary": summary,
        **{f"prediction_{method}": path for method, path in prediction_paths.items()},
    }


def valid_summary(paths: dict[str, Path]) -> dict[str, Any]:
    result_template = {
        "qa_count": 2,
        "answerable_qa_count": 1,
        "cat5_qa_count": 1,
        "overall_answerable": {"token_f1": 1.0},
        "by_source_dataset": {"Fixture": {"token_f1": 1.0}},
        "by_language": {"en": {"token_f1": 1.0}},
        "by_category": {"1": {"token_f1": 1.0}},
        "by_cross_session": {"false": {"token_f1": 1.0}},
        "by_evidence_provenance": {"original_turn": {"token_f1": 1.0}},
        "cat5_refusal": {"accuracy": 1.0},
        "cat5_unsupported_claim": {"rate": 0.0},
    }
    return {
        "status": "completed",
        "dataset": str(paths["dataset"]),
        "model": REQUIRED_FIXED_MODEL,
        "settings_source": "predeclared_fixed_eval_settings",
        "settings_file": str(paths["settings"]),
        "settings_sha256": sha256_file(paths["settings"]),
        "metric_metadata_file": str(paths["metadata"]),
        "metric_metadata_sha256": sha256_file(paths["metadata"]),
        "input_policy": "conversation_only",
        "summary_visible": False,
        "methods": REQUIRED_METHODS,
        "prediction_files": {
            method: {
                "path": str(paths[f"prediction_{method}"]),
                "sha256": sha256_file(paths[f"prediction_{method}"]),
            }
            for method in REQUIRED_METHODS
        },
        "results": [
            {"method": method, **deepcopy(result_template)}
            for method in REQUIRED_METHODS
        ],
    }


def run_validator(validator: Path, summary: Path, tempdir: Path) -> tuple[int, dict[str, Any]]:
    command = [sys.executable, str(validator), str(summary)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    parsed["_tempdir"] = str(tempdir)
    if completed.stderr.strip() and completed.stdout.strip():
        parsed["stderr"] = completed.stderr
    return completed.returncode, parsed


def case_result(
    name: str,
    returncode: int,
    result: dict[str, Any],
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    tempdir = str(result.get("_tempdir", ""))
    error = str(result.get("error", "")).replace(tempdir + "/", "<tmp>/")
    observed_success = returncode == 0 and result.get("status") == "passed"
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None or expected_error_fragment in error
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "validator_status": result.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path("scripts/validate_locomo_baseline_results.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/baseline_results_validator_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_baseline_validator_selftest_") as tmp:
        tempdir = Path(tmp)
        paths = build_fixture(tempdir)
        dataset_sha256 = sha256_file(paths["dataset"])

        summary = valid_summary(paths)
        write_json(paths["summary"], summary)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(case_result("valid_completed_summary_is_accepted", rc, result, True))

        bad = deepcopy(summary)
        bad.pop("prediction_files")
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "missing_prediction_files_are_rejected",
                rc,
                result,
                False,
                "prediction_files must be an object",
            )
        )

        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"]["sha256"] = "0" * 64
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_hash_mismatch_is_rejected",
                rc,
                result,
                False,
                "prediction_files[A-MEM].sha256 does not match file",
            )
        )

        bad_prediction = tempdir / "bad_missing_prediction.jsonl"
        write_jsonl(
            bad_prediction,
            [
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "tea",
                }
            ],
        )
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_missing_qa_key_is_rejected",
                rc,
                result,
                False,
                "missing predictions for 1 QA",
            )
        )

        bad_prediction = tempdir / "bad_duplicate_prediction.jsonl"
        write_jsonl(
            bad_prediction,
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
                    "qa_idx": 1,
                    "model": REQUIRED_FIXED_MODEL,
                    "dataset_sha256": dataset_sha256,
                    "prediction": "I don't know.",
                },
            ],
        )
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_duplicate_qa_key_is_rejected",
                rc,
                result,
                False,
                "duplicate prediction key",
            )
        )

        bad_prediction = tempdir / "bad_unexpected_prediction.jsonl"
        write_jsonl(
            bad_prediction,
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
                    "prediction": "I don't know.",
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
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_unexpected_qa_key_is_rejected",
                rc,
                result,
                False,
                "unexpected QA keys",
            )
        )

        bad_prediction = tempdir / "bad_missing_prediction_field.jsonl"
        write_jsonl(
            bad_prediction,
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
                    "prediction": "I don't know.",
                },
            ],
        )
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_missing_prediction_field_is_rejected",
                rc,
                result,
                False,
                "missing prediction/answer/response field",
            )
        )

        bad_prediction = tempdir / "bad_wrong_model.jsonl"
        write_jsonl(
            bad_prediction,
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
                    "prediction": "I don't know.",
                },
            ],
        )
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_wrong_model_is_rejected",
                rc,
                result,
                False,
                "expected='Qwen/Qwen3-8B'",
            )
        )

        bad_prediction = tempdir / "bad_wrong_dataset_sha256.jsonl"
        write_jsonl(
            bad_prediction,
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
                    "prediction": "I don't know.",
                },
            ],
        )
        bad = deepcopy(summary)
        bad["prediction_files"]["A-MEM"] = {"path": str(bad_prediction), "sha256": sha256_file(bad_prediction)}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "prediction_file_wrong_dataset_sha256_is_rejected",
                rc,
                result,
                False,
                "dataset_sha256",
            )
        )

        good_settings = json.loads(paths["settings"].read_text(encoding="utf-8"))
        bad_settings = deepcopy(good_settings)
        bad_settings["anti_tuning_contract"]["do_not_tune_on_this_benchmark"].remove("retrieval top-k")
        write_json(paths["settings"], bad_settings)
        bad = deepcopy(summary)
        bad["settings_sha256"] = sha256_file(paths["settings"])
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "missing_anti_tuning_freeze_item_is_rejected",
                rc,
                result,
                False,
                "do_not_tune_on_this_benchmark missing",
            )
        )
        write_json(paths["settings"], good_settings)
        summary["settings_sha256"] = sha256_file(paths["settings"])

        bad_settings = deepcopy(good_settings)
        bad_settings["fixed_baselines"]["methods"]["A-MEM"]["runner"] = "scripts/run_amem_official_qwen25_3b_full.sh"
        write_json(paths["settings"], bad_settings)
        bad = deepcopy(summary)
        bad["settings_sha256"] = sha256_file(paths["settings"])
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "legacy_runner_in_fixed_settings_is_rejected",
                rc,
                result,
                False,
                "runner must not point to a legacy script",
            )
        )
        write_json(paths["settings"], good_settings)
        summary["settings_sha256"] = sha256_file(paths["settings"])

        bad_settings = deepcopy(good_settings)
        bad_settings["fixed_baselines"]["optional_methods"]["MemGAS"]["required_for_release"] = True
        write_json(paths["settings"], bad_settings)
        bad = deepcopy(summary)
        bad["settings_sha256"] = sha256_file(paths["settings"])
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "memgas_required_for_release_is_rejected",
                rc,
                result,
                False,
                "optional_methods.MemGAS.required_for_release",
            )
        )
        write_json(paths["settings"], good_settings)
        summary["settings_sha256"] = sha256_file(paths["settings"])

        bad = deepcopy(summary)
        bad["summary_visible"] = True
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "summary_visible_main_table_is_rejected",
                rc,
                result,
                False,
                "summary_visible must be false",
            )
        )

        bad = deepcopy(summary)
        bad["methods"] = ["Full Context"]
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "missing_required_methods_are_rejected",
                rc,
                result,
                False,
                "missing required methods",
            )
        )

        bad = deepcopy(summary)
        bad["metric_metadata_sha256"] = "0" * 64
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "metric_metadata_hash_mismatch_is_rejected",
                rc,
                result,
                False,
                "metric_metadata_sha256 does not match",
            )
        )

        bad_metadata_rows = [
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 0,
                "category": 5,
                "answerable": False,
                "whether_cross_session": False,
                "evidence_provenance": "negative_only",
            },
            {
                "source_dataset": "Fixture",
                "language": "en",
                "sample_id": "fixture_0",
                "qa_idx": 1,
                "category": 5,
                "answerable": False,
                "whether_cross_session": False,
                "evidence_provenance": "negative_only",
            },
        ]
        write_jsonl(paths["metadata"], bad_metadata_rows)
        bad = deepcopy(summary)
        bad["metric_metadata_sha256"] = sha256_file(paths["metadata"])
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "metric_metadata_dataset_alignment_mismatch_is_rejected",
                rc,
                result,
                False,
                "expected=1",
            )
        )
        paths = build_fixture(tempdir)
        summary = valid_summary(paths)

        bootstrap_dataset = tempdir / "multilingual_locomo_style_eval.json"
        write_json(bootstrap_dataset, json.loads(paths["dataset"].read_text(encoding="utf-8")))
        bootstrap_settings = deepcopy(json.loads(paths["settings"].read_text(encoding="utf-8")))
        bootstrap_settings["dataset"]["audited_primary"] = str(bootstrap_dataset)
        write_json(paths["settings"], bootstrap_settings)
        bad = deepcopy(summary)
        bad["dataset"] = str(bootstrap_dataset)
        bad["settings_sha256"] = sha256_file(paths["settings"])
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "bootstrap_primary_dataset_is_rejected",
                rc,
                result,
                False,
                "Final baseline results must use the human-audited eval file",
            )
        )
        restored_settings = deepcopy(bootstrap_settings)
        restored_settings["dataset"]["audited_primary"] = str(paths["dataset"])
        write_json(paths["settings"], restored_settings)
        summary["settings_sha256"] = sha256_file(paths["settings"])

        bad = deepcopy(summary)
        bad["results"][0].pop("by_evidence_provenance")
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "missing_required_result_field_is_rejected",
                rc,
                result,
                False,
                "missing fields",
            )
        )

        bad = deepcopy(summary)
        bad["results"][0]["by_source_dataset"] = {}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "empty_required_group_result_is_rejected",
                rc,
                result,
                False,
                "by_source_dataset must be non-empty",
            )
        )

        bad = deepcopy(summary)
        bad["results"][0]["by_language"] = {"en": {}}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "group_metric_without_numeric_value_is_rejected",
                rc,
                result,
                False,
                "by_language.en must be non-empty",
            )
        )

        bad = deepcopy(summary)
        bad["results"][0]["by_category"]["99"] = {"token_f1": 0.0}
        write_json(paths["summary"], bad)
        rc, result = run_validator(args.validator, paths["summary"], tempdir)
        cases.append(
            case_result(
                "unexpected_group_key_is_rejected",
                rc,
                result,
                False,
                "by_category has unexpected groups",
            )
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
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
