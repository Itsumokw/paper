#!/usr/bin/env python3
"""Self-test release-gate checks for the no-model construction guard."""

from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import check_locomo_no_model_construction_guard as guard
from check_locomo_style_release_gates import (
    HANDOFF_FULL_CSV_FINALIZER_DOCS,
    REQUIRED_HANDOFF_BASELINE_RESULTS_REFERENCES,
    REQUIRED_HANDOFF_FULL_CSV_FINALIZER_REFERENCES,
    REQUIRED_NO_MODEL_GUARD_SCRIPTS,
    REQUIRED_GOAL_TRACEABILITY_IDS,
    REQUIRED_HANDOFF_NEXT_STEPS_REFERENCES,
    REQUIRED_HANDOFF_PRIMARY_README_REFERENCES,
    REQUIRED_HANDOFF_RUNBOOK_SKIPPED_REFERENCES,
    assignment_risk_summary_errors,
    audit_csv_workflow_selftest_errors,
    combined_primary_partition_errors,
    construction_report_errors,
    goal_traceability_matrix_errors,
    human_audit_handoff_docs_errors,
    no_model_guard_errors,
    no_stale_final_outputs_before_audit_errors,
    perltqa_specific_ratio_errors,
    prediction_files_errors,
    primary_output_uniqueness_errors,
    REQUIRED_SKIPPED_AUDIT_BLOCKERS,
    sha256_file,
    SKIPPED_AUDIT_FINAL_OUTPUTS,
    SKIPPED_AUDIT_INPUT_FILES,
    single_qa_set_eval_split_errors,
    skipped_audit_stop_point_report_errors,
)


def valid_guard_report() -> dict[str, Any]:
    scripts = sorted(REQUIRED_NO_MODEL_GUARD_SCRIPTS)
    return {
        "status": "passed",
        "scanned_scripts": scripts,
        "script_hashes": {
            script: sha256_file(Path(script))
            for script in scripts
            if Path(script).is_file()
        },
        "missing": [],
        "findings": {},
    }


def valid_audit_csv_workflow_selftest_report() -> dict[str, Any]:
    script_names = (
        "export_locomo_human_audit_csv.py",
        "import_locomo_human_audit_csv.py",
        "finalize_locomo_human_audit_csv.py",
        "export_locomo_human_audit_batches.py",
        "merge_locomo_human_audit_batches.py",
        "summarize_locomo_human_audit_batches.py",
        "check_locomo_human_audit_batch_edits.py",
        "finalize_locomo_human_audit_batches.py",
    )
    cases = [
        {"name": "csv_import_rejects_read_only_field_edits", "status": "passed"},
        {"name": "batch_merge_rejects_read_only_field_edits", "status": "passed"},
    ]
    return {
        "status": "passed",
        "script_hashes": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in script_names
            if Path(__file__).with_name(name).is_file()
        },
        "cases": cases,
    }


def file_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        state["sha256"] = sha256_file(path)
    return state


def valid_skipped_audit_stop_point_report(root: Path) -> dict[str, Any]:
    checker_path = Path(__file__).with_name("check_locomo_skipped_audit_stop_point.py")
    return {
        "status": "passed",
        "purpose": "safe_stop_point_when_manual_human_audit_is_skipped",
        "checker": str(checker_path),
        "checker_sha256": sha256_file(checker_path),
        "expected_blockers": sorted(REQUIRED_SKIPPED_AUDIT_BLOCKERS),
        "input_files": {rel: file_state(root / rel) for rel in sorted(SKIPPED_AUDIT_INPUT_FILES)},
        "final_outputs_state": {rel: file_state(root / rel) for rel in sorted(SKIPPED_AUDIT_FINAL_OUTPUTS)},
        "checks": [],
        "errors": [],
    }


def case_result(
    name: str,
    report: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = no_model_guard_errors(report)
    observed_success = not errors
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None
            or any(expected_error_fragment in error for error in errors)
        )
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
        default=Path("datasets/locomo_style_eval/release_gate_no_model_guard_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    guard_default_scripts = {str(item) for item in guard.DEFAULT_SCRIPTS}
    required_scripts_match = guard_default_scripts == REQUIRED_NO_MODEL_GUARD_SCRIPTS
    cases.append(
        {
            "name": "release_gate_required_scripts_match_guard_defaults",
            "status": "passed" if required_scripts_match else "failed",
            "expect_success": True,
            "missing_from_gate_required": sorted(guard_default_scripts - REQUIRED_NO_MODEL_GUARD_SCRIPTS),
            "extra_in_gate_required": sorted(REQUIRED_NO_MODEL_GUARD_SCRIPTS - guard_default_scripts),
        }
    )

    good = valid_guard_report()
    cases.append(case_result("valid_no_model_guard_report_is_accepted", good, expect_success=True))

    audit_csv_good = valid_audit_csv_workflow_selftest_report()
    audit_csv_errors = audit_csv_workflow_selftest_errors(audit_csv_good)
    cases.append(
        {
            "name": "valid_audit_csv_workflow_selftest_is_accepted",
            "status": "passed" if not audit_csv_errors else "failed",
            "expect_success": True,
            "errors": audit_csv_errors,
        }
    )

    audit_csv_bad = deepcopy(audit_csv_good)
    audit_csv_bad["cases"] = [
        case
        for case in audit_csv_bad["cases"]
        if case["name"] != "csv_import_rejects_read_only_field_edits"
    ]
    audit_csv_errors = audit_csv_workflow_selftest_errors(audit_csv_bad)
    cases.append(
        {
            "name": "audit_csv_workflow_missing_read_only_case_is_rejected",
            "status": "passed"
            if any("required cases missing or not passed" in error for error in audit_csv_errors)
            else "failed",
            "expect_success": False,
            "expected_error_fragment": "required cases missing or not passed",
            "errors": audit_csv_errors,
        }
    )

    with tempfile.TemporaryDirectory(prefix="locomo_goal_traceability_gate_selftest_") as tmp:
        tempdir = Path(tmp)
        goal_doc = tempdir / "goal.md"
        goal_doc.write_text("# Goal\n", encoding="utf-8")
        release_report = tempdir / "release_gate_report.json"
        release_report.write_text(json.dumps({"status": "blocked", "checks": []}) + "\n", encoding="utf-8")
        matrix_path = tempdir / "goal_traceability_matrix.json"
        matrix = {
            "status": "blocked",
            "goal_doc": str(goal_doc),
            "goal_doc_sha256": sha256_file(goal_doc),
            "release_gate_report": str(release_report),
            "release_gate_sha256": sha256_file(release_report),
            "requirements": [{"id": requirement_id} for requirement_id in sorted(REQUIRED_GOAL_TRACEABILITY_IDS)],
        }
        matrix_path.write_text(json.dumps(matrix, ensure_ascii=False) + "\n", encoding="utf-8")
        matrix_errors = goal_traceability_matrix_errors(
            matrix,
            matrix_path=matrix_path,
            goal_doc_path=goal_doc,
            release_gate_report_path=release_report,
        )
        cases.append(
            {
                "name": "valid_goal_traceability_matrix_is_accepted",
                "status": "passed" if not matrix_errors else "failed",
                "expect_success": True,
                "errors": matrix_errors,
            }
        )

        stale_matrix = deepcopy(matrix)
        stale_matrix["release_gate_sha256"] = "0" * 64
        matrix_errors = goal_traceability_matrix_errors(
            stale_matrix,
            matrix_path=matrix_path,
            goal_doc_path=goal_doc,
            release_gate_report_path=release_report,
        )
        cases.append(
            {
                "name": "stale_goal_traceability_matrix_release_hash_is_rejected",
                "status": "passed" if any("release_gate_sha256 mismatch" in error for error in matrix_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "release_gate_sha256 mismatch",
                "errors": matrix_errors,
            }
        )

    with tempfile.TemporaryDirectory(prefix="locomo_handoff_docs_gate_selftest_") as tmp:
        root = Path(tmp) / "datasets" / "locomo_style_eval"
        doc_paths = [
            root / "NEXT_STEPS.md",
            root / "RUNBOOK.md",
            root / "HUMAN_AUDIT_HANDOFF.md",
            root / "HUMAN_AUDIT_QUICKSTART_ZH.md",
            root / "HUMAN_AUDIT_WORKPLAN.md",
            root / "HUMAN_AUDIT_ASSIGNMENTS.md",
            root / "human_audit_review_guide.md",
            root / "human_audit_batches" / "README.md",
        ]
        for path in doc_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        baseline_results_doc = root / "baseline_results" / "README.md"
        baseline_results_doc.parent.mkdir(parents=True, exist_ok=True)
        baseline_results_doc.write_text("# fixture\n", encoding="utf-8")
        primary_readme_doc = root / "primary" / "README.md"
        primary_readme_doc.parent.mkdir(parents=True, exist_ok=True)
        primary_readme_doc.write_text("# fixture\n", encoding="utf-8")
        recent_session_readme_doc = root / "recent_session_ablation" / "README.md"
        recent_session_readme_doc.parent.mkdir(parents=True, exist_ok=True)
        recent_session_readme_doc.write_text("# fixture\n", encoding="utf-8")
        review_dir = root / "human_audit_batch_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_paths = []
        for idx in range(1, 11):
            path = review_dir / f"batch_{idx:03d}_Fixture_001-001.md"
            path.write_text("# fixture\n\nSource fact ledger support:\n", encoding="utf-8")
            review_paths.append(path)
        handoff_report = {
            "status": "passed",
            "required_next_steps_references": list(REQUIRED_HANDOFF_NEXT_STEPS_REFERENCES),
            "required_runbook_skipped_references": list(REQUIRED_HANDOFF_RUNBOOK_SKIPPED_REFERENCES),
            "full_csv_finalizer_docs": sorted(HANDOFF_FULL_CSV_FINALIZER_DOCS),
            "required_full_csv_finalizer_references": list(REQUIRED_HANDOFF_FULL_CSV_FINALIZER_REFERENCES),
            "required_primary_readme_references": list(REQUIRED_HANDOFF_PRIMARY_README_REFERENCES),
            "required_baseline_results_references": list(REQUIRED_HANDOFF_BASELINE_RESULTS_REFERENCES),
            "required_docs": [{"path": str(path), "sha256": sha256_file(path)} for path in doc_paths],
            "baseline_results_doc": {
                "path": str(baseline_results_doc),
                "sha256": sha256_file(baseline_results_doc),
            },
            "primary_readme_doc": {
                "path": str(primary_readme_doc),
                "sha256": sha256_file(primary_readme_doc),
            },
            "recent_session_readme_doc": {
                "path": str(recent_session_readme_doc),
                "sha256": sha256_file(recent_session_readme_doc),
            },
            "batch_csv_count": 10,
            "batch_review_md_count": 10,
            "batch_review_md_with_fact_ledger_support": 10,
            "batch_review_docs": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "has_fact_ledger_support": True,
                }
                for path in review_paths
            ],
        }
        handoff_errors = human_audit_handoff_docs_errors(handoff_report, root)
        cases.append(
            {
                "name": "valid_handoff_docs_report_is_accepted",
                "status": "passed" if not handoff_errors else "failed",
                "expect_success": True,
                "errors": handoff_errors,
            }
        )

        stale_next_steps_report = deepcopy(handoff_report)
        (root / "NEXT_STEPS.md").write_text("# changed\n", encoding="utf-8")
        handoff_errors = human_audit_handoff_docs_errors(stale_next_steps_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_stale_next_steps_hash",
                "status": "passed" if any("NEXT_STEPS.md sha256 mismatch" in error for error in handoff_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "NEXT_STEPS.md sha256 mismatch",
                "errors": handoff_errors,
            }
        )
        (root / "NEXT_STEPS.md").write_text("# fixture\n", encoding="utf-8")

        stale_baseline_doc_report = deepcopy(handoff_report)
        baseline_results_doc.write_text("# changed\n", encoding="utf-8")
        handoff_errors = human_audit_handoff_docs_errors(stale_baseline_doc_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_stale_baseline_results_readme_hash",
                "status": "passed" if any("baseline_results/README.md sha256 mismatch" in error for error in handoff_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "baseline_results/README.md sha256 mismatch",
                "errors": handoff_errors,
            }
        )
        baseline_results_doc.write_text("# fixture\n", encoding="utf-8")

        stale_primary_readme_report = deepcopy(handoff_report)
        primary_readme_doc.write_text("# changed\n", encoding="utf-8")
        handoff_errors = human_audit_handoff_docs_errors(stale_primary_readme_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_stale_primary_readme_hash",
                "status": "passed" if any("primary/README.md sha256 mismatch" in error for error in handoff_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "primary/README.md sha256 mismatch",
                "errors": handoff_errors,
            }
        )
        primary_readme_doc.write_text("# fixture\n", encoding="utf-8")

        stale_recent_readme_report = deepcopy(handoff_report)
        recent_session_readme_doc.write_text("# changed\n", encoding="utf-8")
        handoff_errors = human_audit_handoff_docs_errors(stale_recent_readme_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_stale_recent_session_readme_hash",
                "status": "passed" if any("recent_session_ablation/README.md sha256 mismatch" in error for error in handoff_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "recent_session_ablation/README.md sha256 mismatch",
                "errors": handoff_errors,
            }
        )
        recent_session_readme_doc.write_text("# fixture\n", encoding="utf-8")

        stale_review_report = deepcopy(handoff_report)
        review_paths[0].write_text("# changed\n\nSource fact ledger support:\n", encoding="utf-8")
        handoff_errors = human_audit_handoff_docs_errors(stale_review_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_stale_review_markdown_hash",
                "status": "passed" if any("sha256 mismatch" in error and "batch_001" in error for error in handoff_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "sha256 mismatch",
                "errors": handoff_errors,
            }
        )
        review_paths[0].write_text("# fixture\n\nSource fact ledger support:\n", encoding="utf-8")

        missing_support_report = deepcopy(handoff_report)
        missing_support_report["batch_review_docs"][0]["has_fact_ledger_support"] = False
        missing_support_report["batch_review_md_with_fact_ledger_support"] = 9
        handoff_errors = human_audit_handoff_docs_errors(missing_support_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_review_fact_ledger_flag",
                "status": "passed"
                if any("batch_review_md_with_fact_ledger_support=9" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "batch_review_md_with_fact_ledger_support=9",
                "errors": handoff_errors,
            }
        )

        missing_runbook_refs_report = deepcopy(handoff_report)
        missing_runbook_refs_report.pop("required_runbook_skipped_references", None)
        handoff_errors = human_audit_handoff_docs_errors(missing_runbook_refs_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_runbook_skipped_reference_list",
                "status": "passed"
                if any("required_runbook_skipped_references missing or not list" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "required_runbook_skipped_references missing or not list",
                "errors": handoff_errors,
            }
        )

        missing_full_csv_refs_report = deepcopy(handoff_report)
        missing_full_csv_refs_report.pop("required_full_csv_finalizer_references", None)
        handoff_errors = human_audit_handoff_docs_errors(missing_full_csv_refs_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_full_csv_finalizer_reference_list",
                "status": "passed"
                if any("required_full_csv_finalizer_references missing or not list" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "required_full_csv_finalizer_references missing or not list",
                "errors": handoff_errors,
            }
        )

        missing_full_csv_docs_report = deepcopy(handoff_report)
        missing_full_csv_docs_report["full_csv_finalizer_docs"] = [
            item
            for item in missing_full_csv_docs_report.get("full_csv_finalizer_docs", [])
            if item != "NEXT_STEPS.md"
        ]
        handoff_errors = human_audit_handoff_docs_errors(missing_full_csv_docs_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_next_steps_full_csv_doc",
                "status": "passed"
                if any("full_csv_finalizer_docs missing 'NEXT_STEPS.md'" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "full_csv_finalizer_docs missing 'NEXT_STEPS.md'",
                "errors": handoff_errors,
            }
        )

        missing_primary_refs_report = deepcopy(handoff_report)
        missing_primary_refs_report["required_primary_readme_references"] = [
            item
            for item in missing_primary_refs_report.get("required_primary_readme_references", [])
            if item != "conversation-only"
        ]
        handoff_errors = human_audit_handoff_docs_errors(missing_primary_refs_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_primary_readme_input_policy_reference",
                "status": "passed"
                if any("required_primary_readme_references missing 'conversation-only'" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "required_primary_readme_references missing 'conversation-only'",
                "errors": handoff_errors,
            }
        )

        missing_baseline_refs_report = deepcopy(handoff_report)
        missing_baseline_refs_report["required_baseline_results_references"] = [
            item
            for item in missing_baseline_refs_report.get("required_baseline_results_references", [])
            if item != "MemGAS"
        ]
        handoff_errors = human_audit_handoff_docs_errors(missing_baseline_refs_report, root)
        cases.append(
            {
                "name": "handoff_docs_reject_missing_baseline_memgas_reference",
                "status": "passed"
                if any("required_baseline_results_references missing 'MemGAS'" in error for error in handoff_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "required_baseline_results_references missing 'MemGAS'",
                "errors": handoff_errors,
            }
        )

    bad = deepcopy(good)
    omitted = sorted(REQUIRED_NO_MODEL_GUARD_SCRIPTS)[0]
    bad["scanned_scripts"] = [item for item in bad["scanned_scripts"] if item != omitted]
    bad["script_hashes"].pop(omitted, None)
    cases.append(
        case_result(
            "missing_required_scanned_script_is_rejected",
            bad,
            expect_success=False,
            expected_error_fragment="scanned_scripts missing required paths",
        )
    )

    bad = deepcopy(good)
    first_script = sorted(REQUIRED_NO_MODEL_GUARD_SCRIPTS)[0]
    bad["script_hashes"][first_script] = "0" * 64
    cases.append(
        case_result(
            "script_hash_mismatch_is_rejected",
            bad,
            expect_success=False,
            expected_error_fragment=f"script_hashes[{first_script}] mismatch",
        )
    )

    bad = deepcopy(good)
    bad["findings"] = {first_script: [{"line": 1, "pattern": "forbidden_call", "text": "MODEL_CLIENT_CALL"}]}
    cases.append(
        case_result(
            "forbidden_pattern_finding_is_rejected",
            bad,
            expect_success=False,
            expected_error_fragment="findings=",
        )
    )

    with tempfile.TemporaryDirectory(prefix="locomo_construction_report_gate_selftest_") as tmp:
        tempdir = Path(tmp)
        report_path = tempdir / "PerLTQA-LoCoMo-style-eval_construction_report.md"
        valid_report = "\n".join(
            [
                "# Construction Report: PerLTQA-LoCoMo-style-eval",
                "",
                "Construction mode: no-model deterministic conversion/seed QA. Model calls: 0.",
                "",
                "## Summary",
                "",
                "This self-test report contains enough text to avoid short-file acceptance.",
                "",
                "## Notes",
                "",
                "- PerLTQA PlanMode D bootstrap uses memory_anchor_turns; original PerLTQA QA is not copied into final eval.",
            ]
        )
        report_path.write_text(valid_report + "\n", encoding="utf-8")
        report_errors = construction_report_errors(report_path, "PerLTQA-LoCoMo-style-eval")
        cases.append(
            {
                "name": "valid_construction_report_is_accepted",
                "status": "passed" if not report_errors else "failed",
                "expect_success": True,
                "errors": report_errors,
            }
        )

        bad_report_path = tempdir / "bad_construction_report.md"
        bad_report_path.write_text(valid_report.replace("Model calls: 0", "Model calls: 1") + "\n", encoding="utf-8")
        report_errors = construction_report_errors(bad_report_path, "PerLTQA-LoCoMo-style-eval")
        cases.append(
            {
                "name": "construction_report_missing_model_zero_is_rejected",
                "status": "passed" if any("Model calls: 0" in error for error in report_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "Model calls: 0",
                "errors": report_errors,
            }
        )

        bad_report_path.write_text(
            valid_report.replace("PerLTQA PlanMode D", "PerLTQA unsupported mode") + "\n",
            encoding="utf-8",
        )
        report_errors = construction_report_errors(bad_report_path, "PerLTQA-LoCoMo-style-eval")
        cases.append(
            {
                "name": "construction_report_missing_planmode_boundary_is_rejected",
                "status": "passed" if any("PerLTQA PlanMode D" in error for error in report_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "PerLTQA PlanMode D",
                "errors": report_errors,
            }
        )

        primary_root = tempdir / "primary"
        primary_root.mkdir()
        for name in (
            "PerLTQA-LoCoMo-style-eval.json",
            "OPELA-LoCoMo-style-eval.json",
            "JLongChat-LoCoMo-style-eval.json",
            "deL1L2IM-LoCoMo-style-eval.json",
            "multilingual_locomo_style_eval.json",
        ):
            (primary_root / name).write_text("[]\n", encoding="utf-8")
        uniqueness_errors = primary_output_uniqueness_errors(tempdir)
        cases.append(
            {
                "name": "valid_primary_outputs_are_accepted",
                "status": "passed" if not uniqueness_errors else "failed",
                "expect_success": True,
                "errors": uniqueness_errors,
            }
        )

        stale_errors = no_stale_final_outputs_before_audit_errors(tempdir)
        cases.append(
            {
                "name": "missing_final_outputs_before_audit_are_accepted",
                "status": "passed" if not stale_errors else "failed",
                "expect_success": True,
                "errors": stale_errors,
            }
        )

        (tempdir / "manifest.json").write_text(
            json.dumps({"status": "bootstrap_harness_artifact_not_final_audited_release"}) + "\n",
            encoding="utf-8",
        )
        (tempdir / "release_gate_report.json").write_text(
            json.dumps({"status": "blocked", "blocking_failed": sorted(REQUIRED_SKIPPED_AUDIT_BLOCKERS)}) + "\n",
            encoding="utf-8",
        )
        (tempdir / "human_audit_results_summary.json").write_text(
            json.dumps(
                {
                    "status": "incomplete_or_failed",
                    "allow_incomplete": False,
                    "incomplete_count": 1,
                    "decision_counts": {"todo": 1},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tempdir / "human_audit_batches_finalize_dry_run.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "dry_run": True,
                    "committed": False,
                    "changed_rows": 0,
                    "errors": ["1 audit rows still have human_decision=todo"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tempdir / "human_audit_csv_finalize_dry_run.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "dry_run": True,
                    "committed": False,
                    "import": {
                        "changed_rows": 0,
                        "errors": ["1 audit rows still have human_decision=todo"],
                    },
                    "errors": ["1 audit rows still have human_decision=todo"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tempdir / "post_audit_pipeline_report.json").write_text(
            json.dumps({"status": "failed", "failed_step": "validate_human_audit_results"}) + "\n",
            encoding="utf-8",
        )
        skipped_report = valid_skipped_audit_stop_point_report(tempdir)
        skipped_errors = skipped_audit_stop_point_report_errors(skipped_report, tempdir)
        cases.append(
            {
                "name": "valid_skipped_audit_stop_point_report_is_accepted",
                "status": "passed" if not skipped_errors else "failed",
                "expect_success": True,
                "errors": skipped_errors,
            }
        )

        stale_skipped_report = deepcopy(skipped_report)
        (tempdir / "manifest.json").write_text(
            json.dumps({"status": "changed_after_report"}) + "\n",
            encoding="utf-8",
        )
        skipped_errors = skipped_audit_stop_point_report_errors(stale_skipped_report, tempdir)
        cases.append(
            {
                "name": "skipped_audit_report_rejects_stale_manifest_hash",
                "status": "passed" if any("input_files[manifest.json].sha256 mismatch" in error for error in skipped_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "input_files[manifest.json].sha256 mismatch",
                "errors": skipped_errors,
            }
        )
        (tempdir / "manifest.json").write_text(
            json.dumps({"status": "bootstrap_harness_artifact_not_final_audited_release"}) + "\n",
            encoding="utf-8",
        )

        release_hash_cycle_report = valid_skipped_audit_stop_point_report(tempdir)
        (tempdir / "release_gate_report.json").write_text(
            json.dumps({"status": "blocked", "blocking_failed": ["rewritten_by_current_release_gate"]}) + "\n",
            encoding="utf-8",
        )
        skipped_errors = skipped_audit_stop_point_report_errors(release_hash_cycle_report, tempdir)
        cases.append(
            {
                "name": "skipped_audit_report_allows_release_gate_hash_cycle",
                "status": "passed" if not skipped_errors else "failed",
                "expect_success": True,
                "errors": skipped_errors,
            }
        )

        stale_final_report = valid_skipped_audit_stop_point_report(tempdir)
        audited_primary = primary_root / "multilingual_locomo_style_eval_audited.json"
        audited_primary.write_text("[]\n", encoding="utf-8")
        skipped_errors = skipped_audit_stop_point_report_errors(stale_final_report, tempdir)
        cases.append(
            {
                "name": "skipped_audit_report_rejects_new_final_output",
                "status": "passed" if any("final output exists now" in error for error in skipped_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "final output exists now",
                "errors": skipped_errors,
            }
        )
        audited_primary.unlink()

        audited_primary = primary_root / "multilingual_locomo_style_eval_audited.json"
        audited_primary.write_text("[]\n", encoding="utf-8")
        stale_errors = no_stale_final_outputs_before_audit_errors(tempdir)
        cases.append(
            {
                "name": "stale_audited_primary_before_audit_apply_is_rejected",
                "status": "passed" if any("stale final outputs" in error for error in stale_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "stale final outputs",
                "errors": stale_errors,
            }
        )

        (tempdir / "human_audit_apply_report.json").write_text(
            json.dumps({"status": "applied"}) + "\n",
            encoding="utf-8",
        )
        stale_errors = no_stale_final_outputs_before_audit_errors(tempdir)
        cases.append(
            {
                "name": "audited_primary_after_audit_apply_is_accepted",
                "status": "passed" if not stale_errors else "failed",
                "expect_success": True,
                "errors": stale_errors,
            }
        )
        audited_primary.unlink()

        predictions_dir = tempdir / "baseline_results" / "predictions"
        predictions_dir.mkdir(parents=True)
        (predictions_dir / "full_context.jsonl").write_text("{}\n", encoding="utf-8")
        (tempdir / "human_audit_apply_report.json").unlink()
        stale_errors = no_stale_final_outputs_before_audit_errors(tempdir)
        cases.append(
            {
                "name": "stale_baseline_predictions_before_audit_apply_are_rejected",
                "status": "passed"
                if any("baseline_results/predictions" in error for error in stale_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "baseline_results/predictions",
                "errors": stale_errors,
            }
        )
        (tempdir / "human_audit_apply_report.json").write_text(
            json.dumps({"status": "applied"}) + "\n",
            encoding="utf-8",
        )
        stale_errors = no_stale_final_outputs_before_audit_errors(tempdir)
        cases.append(
            {
                "name": "baseline_predictions_after_audit_apply_are_accepted",
                "status": "passed" if not stale_errors else "failed",
                "expect_success": True,
                "errors": stale_errors,
            }
        )

        sample = {
            "sample_id": "perltqa_0",
            "source_dataset": "PerLTQA",
            "language": "zh",
            "split": "eval",
            "conversation": {},
            "observation": {},
            "session_summary": {},
            "event_summary": {},
            "qa": [{"question": "What is checked?", "answer": "The QA set.", "category": 1, "evidence": []}],
        }
        (primary_root / "PerLTQA-LoCoMo-style-eval.json").write_text(
            json.dumps([sample], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (primary_root / "multilingual_locomo_style_eval.json").write_text(
            json.dumps([sample], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        partition_errors = combined_primary_partition_errors(tempdir)
        cases.append(
            {
                "name": "valid_combined_primary_partition_is_accepted",
                "status": "passed" if not partition_errors else "failed",
                "expect_success": True,
                "errors": partition_errors,
            }
        )

        for artifact in (
            "PerLTQA-LoCoMo-style-eval",
            "OPELA-LoCoMo-style-eval",
            "JLongChat-LoCoMo-style-eval",
            "deL1L2IM-LoCoMo-style-eval",
        ):
            audit_dir = tempdir / "sidecars" / artifact
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / f"{artifact}_qa_audit.jsonl"
            if artifact == "PerLTQA-LoCoMo-style-eval":
                audit_path.write_text(json.dumps({"qa_set": "locomo_style_main"}) + "\n", encoding="utf-8")
            else:
                audit_path.write_text("", encoding="utf-8")

        single_errors, _single_summary = single_qa_set_eval_split_errors(tempdir)
        cases.append(
            {
                "name": "valid_single_qa_set_eval_split_is_accepted",
                "status": "passed" if not single_errors else "failed",
                "expect_success": True,
                "errors": single_errors,
            }
        )

        bad_sample = deepcopy(sample)
        bad_sample["split"] = "train"
        (primary_root / "multilingual_locomo_style_eval.json").write_text(
            json.dumps([bad_sample], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        single_errors, _single_summary = single_qa_set_eval_split_errors(tempdir)
        cases.append(
            {
                "name": "single_qa_set_eval_split_rejects_train_split",
                "status": "passed" if any("split='train'" in error for error in single_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "split='train'",
                "errors": single_errors,
            }
        )
        (primary_root / "multilingual_locomo_style_eval.json").write_text(
            json.dumps([sample], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        perltqa_audit = (
            tempdir
            / "sidecars"
            / "PerLTQA-LoCoMo-style-eval"
            / "PerLTQA-LoCoMo-style-eval_qa_audit.jsonl"
        )
        perltqa_audit.write_text(json.dumps({"qa_set": "raw_original"}) + "\n", encoding="utf-8")
        single_errors, _single_summary = single_qa_set_eval_split_errors(tempdir)
        cases.append(
            {
                "name": "single_qa_set_eval_split_rejects_wrong_qa_set",
                "status": "passed" if any("qa_set='raw_original'" in error for error in single_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "qa_set='raw_original'",
                "errors": single_errors,
            }
        )
        perltqa_audit.write_text(json.dumps({"qa_set": "locomo_style_main"}) + "\n", encoding="utf-8")

        (primary_root / "multilingual_locomo_style_eval.json").write_text("[]\n", encoding="utf-8")
        partition_errors = combined_primary_partition_errors(tempdir)
        cases.append(
            {
                "name": "combined_primary_missing_source_sample_is_rejected",
                "status": "passed" if any("missing source sample_ids" in error for error in partition_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "missing source sample_ids",
                "errors": partition_errors,
            }
        )
        (primary_root / "multilingual_locomo_style_eval.json").write_text(
            json.dumps([sample], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        sidecar_root = tempdir / "sidecars" / "PerLTQA-LoCoMo-style-eval"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        provenance = sidecar_root / "PerLTQA-LoCoMo-style-eval_provenance.jsonl"
        fact_ledger = sidecar_root / "PerLTQA-LoCoMo-style-eval_fact_ledger.jsonl"
        qa_audit = sidecar_root / "PerLTQA-LoCoMo-style-eval_qa_audit.jsonl"
        provenance.write_text("{}\n", encoding="utf-8")
        fact_ledger.write_text("{}\n", encoding="utf-8")
        qa_audit.write_text("{}\n", encoding="utf-8")
        valid_perltqa_report = {
            "status": "passed",
            "source_dataset": "PerLTQA",
            "artifact": "PerLTQA-LoCoMo-style-eval",
            "input_files": {
                "primary_json": {
                    "path": str(primary_root / "multilingual_locomo_style_eval.json"),
                    "sha256": sha256_file(primary_root / "multilingual_locomo_style_eval.json"),
                },
                "provenance": {"path": str(provenance), "sha256": sha256_file(provenance)},
                "fact_ledger": {"path": str(fact_ledger), "sha256": sha256_file(fact_ledger)},
                "qa_audit": {"path": str(qa_audit), "sha256": sha256_file(qa_audit)},
            },
            "counts": {
                "samples": 1,
                "qa_total": 1,
                "answerable_qa": 1,
                "turns_total": 1,
            },
            "ratios": {
                "original_turn_evidence_ratio": 0.0,
                "memory_anchor_evidence_ratio": 1.0,
                "synthetic_bridge_turn_ratio": 0.0,
                "answer_fact_original_backed_ratio": 1.0,
            },
            "errors": [],
        }
        perltqa_errors = perltqa_specific_ratio_errors(valid_perltqa_report, tempdir)
        cases.append(
            {
                "name": "valid_perltqa_specific_ratio_report_is_accepted",
                "status": "passed" if not perltqa_errors else "failed",
                "expect_success": True,
                "errors": perltqa_errors,
            }
        )

        bad_perltqa_report = deepcopy(valid_perltqa_report)
        bad_perltqa_report["ratios"]["answer_fact_original_backed_ratio"] = 0.99
        perltqa_errors = perltqa_specific_ratio_errors(bad_perltqa_report, tempdir)
        cases.append(
            {
                "name": "perltqa_non_original_backed_answer_fact_ratio_is_rejected",
                "status": "passed"
                if any("answer_fact_original_backed_ratio" in error for error in perltqa_errors)
                else "failed",
                "expect_success": False,
                "expected_error_fragment": "answer_fact_original_backed_ratio",
                "errors": perltqa_errors,
            }
        )

        (primary_root / "PerLTQA-LoCoMo-style-eval-expanded.json").write_text("[]\n", encoding="utf-8")
        uniqueness_errors = primary_output_uniqueness_errors(tempdir)
        cases.append(
            {
                "name": "unexpected_extra_primary_eval_version_is_rejected",
                "status": "passed" if any("unexpected direct JSON files" in error for error in uniqueness_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "unexpected direct JSON files",
                "errors": uniqueness_errors,
            }
        )

        audited_sources = primary_root / "audited_sources"
        audited_sources.mkdir()
        (audited_sources / "PerLTQA-LoCoMo-style-eval.json").write_text("[]\n", encoding="utf-8")
        (audited_sources / "extra.json").write_text("[]\n", encoding="utf-8")
        uniqueness_errors = primary_output_uniqueness_errors(tempdir)
        cases.append(
            {
                "name": "unexpected_extra_audited_source_version_is_rejected",
                "status": "passed" if any("audited_sources has unexpected JSON files" in error for error in uniqueness_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "audited_sources has unexpected JSON files",
                "errors": uniqueness_errors,
            }
        )

    with tempfile.TemporaryDirectory(prefix="locomo_prediction_file_gate_selftest_") as tmp:
        tempdir = Path(tmp)
        dataset = tempdir / "multilingual_locomo_style_eval_audited.json"
        dataset.write_text(
            json.dumps(
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
                        ],
                    }
                ],
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        prediction_file = tempdir / "predictions.jsonl"
        prediction_file.write_text(
            json.dumps(
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": "Qwen/Qwen3-8B",
                    "dataset_sha256": sha256_file(dataset),
                    "prediction": "tea",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        report = {
            "prediction_files": {
                method: {"path": str(prediction_file), "sha256": sha256_file(prediction_file)}
                for method in ("Full Context", "A-MEM", "Mem0", "SimpleMem", "HiGMem")
            }
        }
        errors = prediction_files_errors(report, dataset)
        cases.append(
            {
                "name": "release_gate_prediction_files_accept_dataset_sha256",
                "status": "passed" if not errors else "failed",
                "expect_success": True,
                "errors": errors,
            }
        )

        prediction_file.write_text(
            json.dumps(
                {
                    "sample_id": "fixture_0",
                    "qa_idx": 0,
                    "model": "Qwen/Qwen3-8B",
                    "dataset_sha256": "0" * 64,
                    "prediction": "tea",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        report["prediction_files"]["A-MEM"]["sha256"] = sha256_file(prediction_file)
        errors = prediction_files_errors(report, dataset)
        cases.append(
            {
                "name": "release_gate_prediction_files_reject_wrong_dataset_sha256",
                "status": "passed" if any("dataset_sha256" in error for error in errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "dataset_sha256",
                "errors": errors,
            }
        )

    with tempfile.TemporaryDirectory(prefix="locomo_assignment_gate_selftest_") as tmp:
        tempdir = Path(tmp)
        batch_names = [
            "batch_006_PerLTQA_001-050.csv",
            "batch_007_PerLTQA_051-100.csv",
            "batch_008_PerLTQA_101-150.csv",
            "batch_009_PerLTQA_151-162.csv",
            "batch_002_JLongChat_051-100.csv",
            "batch_003_JLongChat_101-116.csv",
            "batch_010_deL1L2IM_001-044.csv",
            "batch_001_JLongChat_001-050.csv",
            "batch_004_OPELA_001-050.csv",
            "batch_005_OPELA_051-051.csv",
        ]
        batch_dir = tempdir / "human_audit_batches"
        batch_dir.mkdir()
        progress_batches = []
        batch_csvs = {}
        for batch_name in batch_names:
            path = batch_dir / batch_name
            path.write_text("source_dataset,sample_id,qa_idx,human_decision\nFixture,fixture_0,0,todo\n", encoding="utf-8")
            progress_batches.append({"path": str(path), "rows": 1, "completed": 0})
            batch_csvs[batch_name] = {"path": str(path), "sha256": sha256_file(path)}
        progress_path = tempdir / "human_audit_batches_progress.json"
        progress_path.write_text(
            json.dumps({"rows": 373, "batches": progress_batches}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        flags_path = tempdir / "human_audit_flags.json"
        flags_path.write_text(json.dumps({"flagged_by_batch": {}}, ensure_ascii=False) + "\n", encoding="utf-8")
        report = {
            "status": "completed",
            "purpose": "reviewer_routing_aid_only_not_audit_decision",
            "rows": 373,
            "assigned_rows": 373,
            "errors": [],
            "summary_script_sha256": sha256_file(Path(__file__).with_name("summarize_locomo_human_audit_assignments.py")),
            "input_files": {
                "batch_progress": {"path": str(progress_path), "sha256": sha256_file(progress_path)},
                "flags": {"path": str(flags_path), "sha256": sha256_file(flags_path)},
                "batch_csvs": batch_csvs,
            },
            "reviewers": {
                reviewer: {
                    "remaining": 1,
                    "next_todo_rows": [
                        {
                            "batch": batch_names[0],
                            "path": str(batch_dir / batch_names[0]),
                            "line": 2,
                            "source_dataset": "Fixture",
                            "sample_id": "fixture_0",
                            "qa_idx": "0",
                        }
                    ],
                }
                for reviewer in "ABCDE"
            },
        }
        assignment_errors = assignment_risk_summary_errors(report, tempdir)
        cases.append(
            {
                "name": "valid_assignment_risk_summary_is_accepted",
                "status": "passed" if not assignment_errors else "failed",
                "expect_success": True,
                "errors": assignment_errors,
            }
        )

        bad_report = deepcopy(report)
        bad_report["input_files"]["batch_csvs"][batch_names[0]]["sha256"] = "0" * 64
        assignment_errors = assignment_risk_summary_errors(bad_report, tempdir)
        cases.append(
            {
                "name": "assignment_risk_summary_rejects_stale_batch_csv_hash",
                "status": "passed" if any("batch_csvs" in error and "sha256 mismatch" in error for error in assignment_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "sha256 mismatch",
                "errors": assignment_errors,
            }
        )

        bad_report = deepcopy(report)
        bad_report["reviewers"]["A"]["next_todo_rows"] = []
        assignment_errors = assignment_risk_summary_errors(bad_report, tempdir)
        cases.append(
            {
                "name": "assignment_risk_summary_requires_next_todo_when_remaining",
                "status": "passed" if any("no next_todo_rows" in error for error in assignment_errors) else "failed",
                "expect_success": False,
                "expected_error_fragment": "no next_todo_rows",
                "errors": assignment_errors,
            }
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
    raise SystemExit(main())
