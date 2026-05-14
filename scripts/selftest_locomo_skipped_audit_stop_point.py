#!/usr/bin/env python3
"""Self-test skipped-audit stop-point checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from check_locomo_skipped_audit_stop_point import (
    EXPECTED_BLOCKERS,
    FINAL_OUTPUTS,
    INPUT_FILES,
    TRANSIENT_SELF_FRESHNESS_BLOCKERS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_root(base: Path) -> Path:
    root = base / "datasets" / "locomo_style_eval"
    write_json(root / "primary" / "multilingual_locomo_style_eval.json", [])
    write_json(root / "manifest.json", {"status": "bootstrap_harness_artifact_not_final_audited_release"})
    write_json(root / "release_gate_report.json", {"status": "blocked", "blocking_failed": EXPECTED_BLOCKERS})
    write_json(
        root / "human_audit_results_summary.json",
        {
            "status": "incomplete_or_failed",
            "allow_incomplete": False,
            "incomplete_count": 1,
            "decision_counts": {"todo": 1},
        },
    )
    write_json(
        root / "human_audit_batches_finalize_dry_run.json",
        {
            "status": "failed",
            "dry_run": True,
            "committed": False,
            "changed_rows": 0,
            "errors": ["1 audit rows still have human_decision=todo"],
        },
    )
    write_json(
        root / "human_audit_csv_finalize_dry_run.json",
        {
            "status": "failed",
            "dry_run": True,
            "committed": False,
            "import": {
                "changed_rows": 0,
                "errors": ["1 audit rows still have human_decision=todo"],
            },
            "errors": ["1 audit rows still have human_decision=todo"],
        },
    )
    write_json(
        root / "post_audit_pipeline_report.json",
        {"status": "failed", "failed_step": "validate_human_audit_results"},
    )
    return root


def run_checker(root: Path) -> tuple[int, dict[str, Any]]:
    output_json = root / "skipped_audit_stop_point_report.json"
    output_md = root / "skipped_audit_stop_point_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_locomo_skipped_audit_stop_point.py")),
            "--root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    report = json.loads(output_json.read_text(encoding="utf-8")) if output_json.exists() else {}
    if completed.returncode != 0 and not report:
        report = {"status": "failed", "errors": [completed.stderr.strip()]}
    return completed.returncode, report


def validate_freshness_metadata(root: Path, report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    checker_path = Path(__file__).with_name("check_locomo_skipped_audit_stop_point.py")
    if report.get("checker_sha256") != sha256_file(checker_path):
        errors.append("checker_sha256 mismatch")

    input_files = report.get("input_files") or {}
    if sorted(input_files) != sorted(INPUT_FILES):
        errors.append("input_files keys mismatch")
    for rel in INPUT_FILES:
        state = input_files.get(rel) or {}
        path = root / rel
        if state.get("exists") is not True:
            errors.append(f"{rel} exists flag mismatch")
        if state.get("sha256") != sha256_file(path):
            errors.append(f"{rel} sha256 mismatch")

    final_outputs_state = report.get("final_outputs_state") or {}
    if sorted(final_outputs_state) != sorted(FINAL_OUTPUTS):
        errors.append("final_outputs_state keys mismatch")
    for rel in FINAL_OUTPUTS:
        state = final_outputs_state.get(rel) or {}
        if state.get("exists") is not False:
            errors.append(f"{rel} absence flag mismatch")
    return errors


def case(
    name: str,
    mutate=None,
    expect_success: bool = True,
    expected_error: str | None = None,
    validate: Callable[[Path, dict[str, Any]], list[str]] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        root = make_root(Path(tmp))
        if mutate:
            mutate(root)
        returncode, report = run_checker(root)
        validation_errors = validate(root, report) if validate else []
    errors = [str(item) for item in report.get("errors", [])]
    ok = (returncode == 0) is expect_success
    if expected_error is not None:
        ok = ok and any(expected_error in error for error in errors)
    ok = ok and not validation_errors
    return {
        "name": name,
        "status": "passed" if ok else "failed",
        "expect_success": expect_success,
        "expected_error": expected_error,
        "returncode": returncode,
        "report_status": report.get("status"),
        "errors": errors,
        "validation_errors": validation_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/skipped_audit_stop_point_selftest.json"),
    )
    args = parser.parse_args()

    cases = [
        case("valid_skipped_audit_stop_point_is_accepted", validate=validate_freshness_metadata),
        case(
            "shuffled_expected_release_gate_blockers_are_accepted",
            lambda root: write_json(
                root / "release_gate_report.json",
                {"status": "blocked", "blocking_failed": list(reversed(EXPECTED_BLOCKERS))},
            ),
        ),
        case(
            "transient_self_freshness_release_gate_blocker_is_accepted",
            lambda root: write_json(
                root / "release_gate_report.json",
                {
                    "status": "blocked",
                    "blocking_failed": [
                        *EXPECTED_BLOCKERS,
                        *sorted(TRANSIENT_SELF_FRESHNESS_BLOCKERS),
                    ],
                },
            ),
        ),
        case(
            "final_audited_primary_is_rejected",
            lambda root: write_json(root / "primary" / "multilingual_locomo_style_eval_audited.json", []),
            expect_success=False,
            expected_error="final_outputs_absent",
        ),
        case(
            "final_manifest_status_is_rejected",
            lambda root: write_json(root / "manifest.json", {"status": "final_audited_release"}),
            expect_success=False,
            expected_error="manifest_not_final_release",
        ),
        case(
            "missing_manifest_is_rejected",
            lambda root: (root / "manifest.json").unlink(),
            expect_success=False,
            expected_error="manifest_not_final_release",
        ),
        case(
            "unexpected_release_gate_blockers_are_rejected",
            lambda root: write_json(root / "release_gate_report.json", {"status": "blocked", "blocking_failed": ["human_audit_completed"]}),
            expect_success=False,
            expected_error="release_gate_expected_blockers_only",
        ),
        case(
            "allow_incomplete_audit_summary_is_rejected",
            lambda root: write_json(
                root / "human_audit_results_summary.json",
                {
                    "status": "partial_valid",
                    "allow_incomplete": True,
                    "incomplete_count": 1,
                    "decision_counts": {"todo": 1},
                },
            ),
            expect_success=False,
            expected_error="strict_human_audit_incomplete",
        ),
        case(
            "committed_batch_finalizer_is_rejected",
            lambda root: write_json(
                root / "human_audit_batches_finalize_dry_run.json",
                {
                    "status": "completed",
                    "dry_run": False,
                    "committed": True,
                    "changed_rows": 1,
                    "errors": [],
                },
            ),
            expect_success=False,
            expected_error="batch_finalizer_dry_run_failed_safely",
        ),
        case(
            "committed_csv_finalizer_is_rejected",
            lambda root: write_json(
                root / "human_audit_csv_finalize_dry_run.json",
                {
                    "status": "completed",
                    "dry_run": False,
                    "committed": True,
                    "import": {"changed_rows": 1, "errors": []},
                    "errors": [],
                },
            ),
            expect_success=False,
            expected_error="csv_finalizer_dry_run_failed_safely",
        ),
        case(
            "post_audit_apply_progress_is_rejected",
            lambda root: write_json(root / "post_audit_pipeline_report.json", {"status": "passed", "failed_step": None}),
            expect_success=False,
            expected_error="post_audit_pipeline_stopped_before_apply",
        ),
    ]

    failed = [row for row in cases if row["status"] != "passed"]
    selftest_path = Path(__file__)
    checker_path = selftest_path.with_name("check_locomo_skipped_audit_stop_point.py")
    result = {
        "status": "passed" if not failed else "failed",
        "checker": str(checker_path),
        "checker_sha256": sha256_file(checker_path),
        "selftest": str(selftest_path),
        "selftest_sha256": sha256_file(selftest_path),
        "cases": cases,
        "errors": [row["name"] for row in failed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
