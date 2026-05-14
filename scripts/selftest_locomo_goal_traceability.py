#!/usr/bin/env python3
"""Self-test goal traceability matrix generation."""

from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import summarize_locomo_goal_traceability as trace


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def required_check_names() -> set[str]:
    names: set[str] = set()
    for requirement in trace.REQUIREMENTS:
        names.update(str(item) for item in requirement.get("checks", []))
    return names


def release_report(
    *,
    blocked: list[str] | None = None,
    failed_checks: list[str] | None = None,
    omit_checks: list[str] | None = None,
) -> dict[str, Any]:
    blocked = blocked or []
    failed = set(failed_checks or [])
    omitted = set(omit_checks or [])
    checks = [
        {
            "name": name,
            "status": "failed" if name in failed else "passed",
            "blocking": True,
            "evidence": f"fixture evidence for {name}",
        }
        for name in sorted(required_check_names() - omitted)
    ]
    return {
        "root": "fixture",
        "status": "blocked" if blocked or failed else "release_ready",
        "blocking_failed": blocked,
        "checks": checks,
    }


def requirement_by_id(summary: dict[str, Any], requirement_id: str) -> dict[str, Any]:
    for row in summary.get("requirements", []):
        if row.get("id") == requirement_id:
            return row
    raise AssertionError(f"requirement not found: {requirement_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/goal_traceability_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_goal_traceability_selftest_") as tmp:
        tempdir = Path(tmp)
        goal_doc = tempdir / "goal.md"
        goal_doc.write_text("# Fixture goal\n", encoding="utf-8")
        release_path = tempdir / "release.json"
        output_md = tempdir / "trace.md"

        write_json(release_path, release_report())
        summary = trace.summarize(goal_doc, release_path)
        trace.write_markdown(output_md, summary)
        cases.append(
            {
                "name": "all_requirements_pass_when_all_checks_present_and_no_blockers",
                "status": "passed"
                if summary["status"] == "passed"
                and summary["counts"] == {"passed": len(trace.REQUIREMENTS)}
                else "failed",
                "observed_status": summary["status"],
                "observed_counts": summary["counts"],
            }
        )
        cases.append(
            {
                "name": "markdown_contains_traceability_title_and_requirement_ids",
                "status": "passed"
                if "Goal Traceability Matrix" in output_md.read_text(encoding="utf-8")
                and "human_audit_minimum" in output_md.read_text(encoding="utf-8")
                and "provenance_labels_explicit" in output_md.read_text(encoding="utf-8")
                and "per_source_perltqa_pipeline" in output_md.read_text(encoding="utf-8")
                else "failed",
            }
        )
        sections = {str(row.get("section")) for row in summary.get("requirements", [])}
        cases.append(
            {
                "name": "explicit_goal_sections_have_requirement_rows",
                "status": "passed"
                if {"Provenance Labels", "Per-source Construction Requirements"}.issubset(sections)
                else "failed",
                "sections": sorted(sections),
            }
        )

        write_json(
            release_path,
            release_report(blocked=["human_audit_completed", "fixed_baseline_results_exist"]),
        )
        summary = trace.summarize(goal_doc, release_path)
        audit_row = requirement_by_id(summary, "human_audit_minimum")
        baseline_row = requirement_by_id(summary, "final_fixed_baselines")
        cases.append(
            {
                "name": "active_release_blockers_mark_requirements_blocked",
                "status": "passed"
                if summary["status"] == "blocked"
                and audit_row["status"] == "blocked"
                and audit_row["active_blockers"] == ["human_audit_completed"]
                and baseline_row["status"] == "blocked"
                and baseline_row["active_blockers"] == ["fixed_baseline_results_exist"]
                else "failed",
                "audit_row": audit_row,
                "baseline_row": baseline_row,
            }
        )

        missing_check = "qa_trace_integrity_passed"
        write_json(release_path, release_report(omit_checks=[missing_check]))
        summary = trace.summarize(goal_doc, release_path)
        qa_row = requirement_by_id(summary, "qa_generation_rules")
        cases.append(
            {
                "name": "missing_mapped_check_marks_requirement_missing_or_failed",
                "status": "passed"
                if qa_row["status"] == "missing_or_failed"
                and missing_check in qa_row["missing_or_failed_checks"]
                else "failed",
                "qa_row": qa_row,
            }
        )

        failed_check = "fixed_eval_settings_predeclared"
        write_json(release_path, release_report(failed_checks=[failed_check]))
        summary = trace.summarize(goal_doc, release_path)
        anti_tuning_row = requirement_by_id(summary, "rule_anti_tuning")
        failed_row = deepcopy(anti_tuning_row)
        cases.append(
            {
                "name": "failed_mapped_check_marks_requirement_missing_or_failed",
                "status": "passed"
                if failed_row["status"] == "missing_or_failed"
                and failed_check in failed_row["missing_or_failed_checks"]
                else "failed",
                "anti_tuning_row": failed_row,
            }
        )

    summary_script = Path(trace.__file__)
    selftest_path = Path(__file__)
    failed_cases = [case for case in cases if case["status"] != "passed"]
    result = {
        "status": "passed" if not failed_cases else "failed",
        "summary_script": str(summary_script),
        "summary_script_sha256": trace.sha256_file(summary_script),
        "selftest": str(selftest_path),
        "selftest_sha256": trace.sha256_file(selftest_path),
        "cases": cases,
        "errors": [case["name"] for case in failed_cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
