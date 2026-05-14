#!/usr/bin/env python3
"""Self-test the LoCoMo-style experiment preflight without real services."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import preflight_locomo_style_experiment as preflight


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset(path: Path) -> None:
    write_json(
        path,
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
                    {"question": "What does Alice like?", "answer": "tea", "category": 1, "evidence": ["D1:1"]},
                    {"question": "Does Alice live on Mars?", "category": 5, "evidence": [], "adversarial_answer": "unsupported"},
                ],
            }
        ],
    )


def run_preflight_case(
    *,
    dataset: Path,
    output: Path,
    served_models: list[str] | None = None,
    gpus: list[dict[str, Any]] | None = None,
    busy_processes: list[dict[str, str]] | None = None,
    extra_args: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    argv = [
        "preflight_locomo_style_experiment.py",
        "--dataset",
        str(dataset),
        "--model",
        "Qwen/Qwen3-8B",
        "--fail-if-gpu-busy",
        "--fail-if-busy-process",
        "--output",
        str(output),
    ]
    if extra_args:
        argv.extend(extra_args)
    with (
        patch.object(sys, "argv", argv),
        patch.object(preflight, "fetch_models", return_value=(served_models or ["Qwen/Qwen3-8B"], None)),
        patch.object(preflight, "gpu_rows", return_value=(gpus or [], None)),
        patch.object(preflight, "busy_processes", return_value=busy_processes or []),
        patch.object(preflight, "chat_check", return_value=("OK", None)),
    ):
        returncode = preflight.main()
    return returncode, json.loads(output.read_text(encoding="utf-8"))


def case_result(
    name: str,
    returncode: int,
    report: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
    extra_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed_success = returncode == 0 and report.get("status") == "passed"
    errors = [str(item) for item in report.get("errors", [])]
    extra_check_errors = []
    for key, expected_value in (extra_checks or {}).items():
        if report.get(key) != expected_value:
            extra_check_errors.append(f"{key}={report.get(key)!r} expected={expected_value!r}")
    if expect_success:
        passed = observed_success and not extra_check_errors
    else:
        passed = not observed_success and (
            expected_error_fragment is None
            or any(expected_error_fragment in error for error in errors)
        ) and not extra_check_errors
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "preflight_status": report.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": errors,
        "extra_check_errors": extra_check_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/experiment_preflight_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_experiment_preflight_selftest_") as tmp:
        tempdir = Path(tmp)
        dataset = tempdir / "multilingual_locomo_style_eval_audited.json"
        output = tempdir / "experiment_preflight.json"

        rc, report = run_preflight_case(dataset=dataset, output=output)
        cases.append(
            case_result(
                "missing_audited_dataset_is_rejected",
                rc,
                report,
                expect_success=False,
                expected_error_fragment="dataset not found yet",
            )
        )

        write_dataset(dataset)
        rc, report = run_preflight_case(dataset=dataset, output=output)
        cases.append(case_result("existing_dataset_idle_service_is_accepted", rc, report, expect_success=True))

        rc, report = run_preflight_case(dataset=dataset, output=output, extra_args=["--chat-check"])
        cases.append(
            case_result(
                "chat_check_is_recorded_as_service_preflight_only",
                rc,
                report,
                expect_success=True,
                extra_checks={
                    "chat_check_requested": True,
                    "service_preflight_policy": "service_preflight_only",
                    "chat_response_used_for_dataset_or_metrics": False,
                },
            )
        )

        rc, report = run_preflight_case(dataset=dataset, output=output, served_models=["OtherModel"])
        cases.append(
            case_result(
                "missing_served_model_is_rejected",
                rc,
                report,
                expect_success=False,
                expected_error_fragment="not served",
            )
        )

        rc, report = run_preflight_case(
            dataset=dataset,
            output=output,
            gpus=[
                {
                    "index": 0,
                    "name": "Fixture GPU",
                    "memory_used_mib": 95,
                    "memory_total_mib": 100,
                    "utilization_gpu_percent": 99,
                }
            ],
        )
        cases.append(
            case_result(
                "busy_gpu_is_rejected_when_flag_set",
                rc,
                report,
                expect_success=False,
                expected_error_fragment="gpu busy above thresholds",
            )
        )

        rc, report = run_preflight_case(
            dataset=dataset,
            output=output,
            busy_processes=[{"pid": "123", "cmd": "run_mem0 fixture"}],
        )
        cases.append(
            case_result(
                "busy_process_is_rejected_when_flag_set",
                rc,
                report,
                expect_success=False,
                expected_error_fragment="busy experiment processes detected",
            )
        )

    preflight_script = Path(__file__).with_name("preflight_locomo_style_experiment.py")
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "preflight_script": str(preflight_script),
        "preflight_script_sha256": sha256_file(preflight_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
