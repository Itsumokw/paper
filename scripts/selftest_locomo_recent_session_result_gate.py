#!/usr/bin/env python3
"""Self-test the release gate's recent-session result checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from check_locomo_style_release_gates import recent_session_result_errors


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
    write_json(
        dataset,
        [
            {
                "sample_id": "fixture_0",
                "source_dataset": "Fixture",
                "language": "en",
                "split": "eval",
                "conversation": {},
                "observation": {},
                "session_summary": {},
                "event_summary": {},
                "qa": [
                    {"question": "q1", "answer": "a1", "category": 1, "evidence": ["D1:1"]},
                    {"question": "q2", "answer": "a2", "category": 2, "evidence": ["D1:2", "D2:1"]},
                    {"question": "q5", "category": 5, "evidence": [], "adversarial_answer": "unsupported"},
                ],
            }
        ],
    )
    write_json(
        settings,
        {
            "status": "predeclared",
            "model": {"served_model": "Qwen/Qwen3-8B"},
            "fixed_baselines": {"settings_source": "predeclared_fixed_eval_settings"},
            "recent_session_diagnostic": {
                "categories": ["1", "2", "3", "4"],
                "max_context_chars": 24000,
                "max_answer_tokens": 96,
                "request_timeout_seconds": 90,
                "workers": 3,
                "long_memory_signal": {
                    "metric": "mean_token_f1",
                    "min_full_mean_token_f1_for_ratio_check": 0.05,
                    "max_last_session_ratio_of_full": 0.85,
                    "max_last_3_sessions_ratio_of_full": 0.95,
                },
            },
        },
    )
    return {"dataset": dataset, "settings": settings}


def valid_report(paths: dict[str, Path]) -> dict[str, Any]:
    runner_script = Path(__file__).with_name("run_locomo_recent_session_model_diagnostic.py")
    records_output = paths["dataset"].parent / "model_prediction_records.jsonl"
    dataset_sha256 = sha256_file(paths["dataset"])
    record_rows = []
    for context_name in ("full_conversation", "last_session_only", "last_3_sessions_only"):
        for qa_idx, category, token_f1 in ((0, 1, 0.5), (1, 2, 0.5)):
            record_rows.append(
                {
                    "context_name": context_name,
                    "sample_id": "fixture_0",
                    "source_dataset": "Fixture",
                    "qa_idx": qa_idx,
                    "category": category,
                    "model": "Qwen/Qwen3-8B",
                    "ablation_input_sha256": dataset_sha256,
                    "question": f"q{qa_idx + 1}",
                    "reference": f"a{qa_idx + 1}",
                    "prediction": f"a{qa_idx + 1}",
                    "token_f1": token_f1,
                    "latency_seconds": 0.1,
                    "token_usage": {},
                    "error": None,
                }
            )
    write_jsonl(records_output, record_rows)
    return {
        "status": "completed",
        "model": "Qwen/Qwen3-8B",
        "runner_script": str(runner_script),
        "runner_script_sha256": sha256_file(runner_script),
        "input_policy": "conversation_only",
        "summary_visible": False,
        "input_fields_rendered": ["conversation"],
        "input_fields_excluded": ["observation", "session_summary", "event_summary", "sidecars"],
        "prompt_policy": "conversation_history_only_direct_answer",
        "context_renderer": "conversation.session_i_date_time_and_turn_dia_id_speaker_text_only",
        "settings_file": str(paths["settings"]),
        "settings_sha256": sha256_file(paths["settings"]),
        "settings_source": "predeclared_fixed_eval_settings",
        "settings_errors": [],
        "ablation_input": str(paths["dataset"]),
        "ablation_input_sha256": sha256_file(paths["dataset"]),
        "categories": ["1", "2", "3", "4"],
        "limit_samples": 0,
        "limit_qa_per_sample": 0,
        "max_context_chars": 24000,
        "max_answer_tokens": 96,
        "request_timeout": 90,
        "workers": 3,
        "expected_records": 6,
        "written_records": 6,
        "context_counts": {
            "full_conversation": 2,
            "last_session_only": 2,
            "last_3_sessions_only": 2,
        },
        "records_output": str(records_output),
        "records_output_sha256": sha256_file(records_output),
        "summary": {
            "errors": 0,
            "contexts": {
                "full_conversation": {"count": 2, "mean_token_f1": 0.5},
                "last_session_only": {"count": 2, "mean_token_f1": 0.1},
                "last_3_sessions_only": {"count": 2, "mean_token_f1": 0.2},
            },
        },
    }


def case_result(
    name: str,
    report: dict[str, Any],
    paths: dict[str, Path],
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = recent_session_result_errors(report, paths["dataset"], paths["settings"])
    passed = not errors if expect_success else any((expected_error_fragment or "") in error for error in errors)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "expected_error_fragment": expected_error_fragment,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/recent_session_result_gate_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_recent_session_gate_selftest_") as tmp:
        tempdir = Path(tmp)
        paths = build_fixture(tempdir)
        good = valid_report(paths)

        cases.append(case_result("valid_completed_recent_session_result_is_accepted", good, paths, True))

        bad = deepcopy(good)
        bad["input_policy"] = "conversation_plus_summary"
        cases.append(
            case_result(
                "summary_visible_input_policy_is_rejected",
                bad,
                paths,
                False,
                "input_policy='conversation_plus_summary'",
            )
        )

        bad = deepcopy(good)
        bad["input_fields_rendered"] = ["conversation", "session_summary"]
        cases.append(
            case_result(
                "extra_rendered_input_fields_are_rejected",
                bad,
                paths,
                False,
                "input_fields_rendered",
            )
        )

        bad = deepcopy(good)
        bad["runner_script_sha256"] = "0" * 64
        cases.append(
            case_result(
                "runner_script_hash_mismatch_is_rejected",
                bad,
                paths,
                False,
                "runner_script_sha256 does not match",
            )
        )

        bad = deepcopy(good)
        bad["limit_samples"] = 1
        cases.append(
            case_result(
                "sample_limited_result_is_rejected",
                bad,
                paths,
                False,
                "limit_samples=1",
            )
        )

        bad = deepcopy(good)
        bad["settings_sha256"] = "0" * 64
        cases.append(
            case_result(
                "settings_hash_mismatch_is_rejected",
                bad,
                paths,
                False,
                "settings_sha256 does not match",
            )
        )

        bad = deepcopy(good)
        bad["ablation_input_sha256"] = "0" * 64
        cases.append(
            case_result(
                "ablation_input_hash_mismatch_is_rejected",
                bad,
                paths,
                False,
                "ablation_input_sha256 does not match",
            )
        )

        bad = deepcopy(good)
        bad["records_output_sha256"] = "0" * 64
        cases.append(
            case_result(
                "records_output_hash_mismatch_is_rejected",
                bad,
                paths,
                False,
                "records_output_sha256 does not match",
            )
        )

        bad_records = tempdir / "bad_recent_session_records.jsonl"
        dataset_sha256 = sha256_file(paths["dataset"])
        write_jsonl(
            bad_records,
            [
                {
                    "context_name": "full_conversation",
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": "Qwen/Qwen2.5-3B-Instruct",
                    "ablation_input_sha256": dataset_sha256,
                    "token_f1": 0.5,
                    "error": None,
                }
            ],
        )
        bad = deepcopy(good)
        bad["records_output"] = str(bad_records)
        bad["records_output_sha256"] = sha256_file(bad_records)
        bad["written_records"] = 1
        cases.append(
            case_result(
                "records_wrong_model_is_rejected",
                bad,
                paths,
                False,
                "model='Qwen/Qwen2.5-3B-Instruct'",
            )
        )

        bad = deepcopy(good)
        bad["summary"]["contexts"]["last_3_sessions_only"]["mean_token_f1"] = 0.49
        cases.append(
            case_result(
                "weak_long_memory_signal_is_rejected",
                bad,
                paths,
                False,
                "last_3_sessions_only mean_token_f1/full",
            )
        )

        bad = deepcopy(good)
        bad["context_counts"]["last_session_only"] = 1
        cases.append(
            case_result(
                "context_count_mismatch_is_rejected",
                bad,
                paths,
                False,
                "last_session_only count=1",
            )
        )

    gate_script = Path(__file__).with_name("check_locomo_style_release_gates.py")
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "gate_script": str(gate_script),
        "gate_script_sha256": sha256_file(gate_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
