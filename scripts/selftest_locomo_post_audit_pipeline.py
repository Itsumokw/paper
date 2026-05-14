#!/usr/bin/env python3
"""Self-test the no-model post-audit pipeline orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import run_locomo_post_audit_pipeline as pipeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def completed_process(returncode: int, stdout: str = "{}", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def run_pipeline_case(
    root: Path,
    output: Path,
    *,
    failing_step: str | None = None,
    gate_blockers: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    step_names = [
        "validate_human_audit_results",
        "apply_human_audit",
        "validate_audited_primary",
        "check_audited_apply_integrity",
        "build_metric_metadata",
        "rebuild_recent_session_ablation",
        "check_release_gates",
    ]
    step_by_script = {
        "validate_locomo_human_audit_results.py": "validate_human_audit_results",
        "apply_locomo_human_audit_results.py": "apply_human_audit",
        "validate_locomo_style_eval.py": "validate_audited_primary",
        "check_locomo_audited_apply_integrity.py": "check_audited_apply_integrity",
        "build_locomo_metric_metadata.py": "build_metric_metadata",
        "make_locomo_recent_session_ablation.py": "rebuild_recent_session_ablation",
        "check_locomo_style_release_gates.py": "check_release_gates",
    }

    def fake_run(cmd: list[str], text: bool, capture_output: bool) -> SimpleNamespace:
        del text, capture_output
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        step = step_by_script.get(script_name, "")
        if step not in step_names:
            return completed_process(2, stderr=f"unexpected command: {cmd!r}")
        if step == "check_release_gates":
            write_json(
                root / "release_gate_report.json",
                {
                    "status": "blocked" if gate_blockers else "release_ready",
                    "blocking_failed": gate_blockers or [],
                },
            )
            return completed_process(1 if gate_blockers else 0, stdout='{"status": "gate"}')
        if step == failing_step:
            return completed_process(1, stdout=f'{{"status": "failed", "step": "{step}"}}')
        return completed_process(0, stdout=f'{{"status": "passed", "step": "{step}"}}')

    argv = [
        "run_locomo_post_audit_pipeline.py",
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    with patch.object(sys, "argv", argv), patch.object(pipeline.subprocess, "run", side_effect=fake_run):
        returncode = pipeline.main()
    return returncode, load_json(output)


def case_result(
    name: str,
    returncode: int,
    report: dict[str, Any],
    *,
    expect_success: bool,
    expected_status: str,
    expected_failed_step: str | None = None,
    expected_blockers: list[str] | None = None,
) -> dict[str, Any]:
    observed_success = returncode == 0 and report.get("status") == expected_status
    if expected_failed_step is not None and report.get("failed_step") != expected_failed_step:
        observed_success = False
    if expected_blockers is not None and report.get("release_gate_blocking_failed") != expected_blockers:
        observed_success = False
    passed = observed_success if expect_success else not observed_success
    if not expect_success:
        passed = (
            returncode != 0
            and report.get("status") == expected_status
            and (expected_failed_step is None or report.get("failed_step") == expected_failed_step)
            and (expected_blockers is None or report.get("release_gate_blocking_failed") == expected_blockers)
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "observed_status": report.get("status"),
        "expected_status": expected_status,
        "observed_failed_step": report.get("failed_step"),
        "expected_failed_step": expected_failed_step,
        "observed_blockers": report.get("release_gate_blocking_failed"),
        "expected_blockers": expected_blockers,
        "steps": [step.get("name") for step in report.get("steps", [])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/post_audit_pipeline_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_post_audit_pipeline_selftest_") as tmp:
        tempdir = Path(tmp)

        root = tempdir / "incomplete"
        output = tempdir / "incomplete_report.json"
        rc, report = run_pipeline_case(root, output, failing_step="validate_human_audit_results")
        cases.append(
            case_result(
                "incomplete_audit_stops_before_apply",
                rc,
                report,
                expect_success=False,
                expected_status="failed",
                expected_failed_step="validate_human_audit_results",
            )
        )

        root = tempdir / "allowed_model_blockers"
        output = tempdir / "allowed_report.json"
        allowed = ["fixed_baseline_results_exist", "recent_session_model_results_exist"]
        rc, report = run_pipeline_case(root, output, gate_blockers=allowed)
        cases.append(
            case_result(
                "complete_audit_reaches_model_only_blockers",
                rc,
                report,
                expect_success=True,
                expected_status="post_audit_ready_for_model_runs",
                expected_blockers=allowed,
            )
        )

        root = tempdir / "unexpected_blocker"
        output = tempdir / "unexpected_report.json"
        blockers = ["metric_metadata_created", "recent_session_model_results_exist"]
        rc, report = run_pipeline_case(root, output, gate_blockers=blockers)
        cases.append(
            case_result(
                "unexpected_post_audit_blocker_is_rejected",
                rc,
                report,
                expect_success=False,
                expected_status="failed",
                expected_failed_step="check_release_gates",
                expected_blockers=blockers,
            )
        )

    pipeline_script = Path(__file__).with_name("run_locomo_post_audit_pipeline.py")
    selftest_script = Path(__file__)
    result = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "pipeline_script": str(pipeline_script),
        "pipeline_script_sha256": sha256_file(pipeline_script),
        "selftest_script": str(selftest_script),
        "selftest_script_sha256": sha256_file(selftest_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
