#!/usr/bin/env python3
"""Self-test human-audit handoff doc validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import check_locomo_human_audit_handoff_docs as checker


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def valid_doc() -> str:
    return """# Human Audit Fixture

Use `scripts/finalize_locomo_human_audit_batches.py`.
Use `scripts/finalize_locomo_human_audit_csv.py`.
Review `datasets/locomo_style_eval/human_audit_batches`.
Review `datasets/locomo_style_eval/human_audit_sheet.csv` when using the full CSV route.
Commit to `datasets/locomo_style_eval/human_audit_packet.jsonl` only through the safe finalizer.
Refresh reviewer routing with `scripts/summarize_locomo_human_audit_assignments.py`.
Export reviewer todos with `scripts/export_locomo_human_audit_reviewer_todos.py`.
Use `--audit-packet-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl` when exporting reviewer todos.
Check reviewer todos with `scripts/check_locomo_human_audit_reviewer_todos.py`.
Read `datasets/locomo_style_eval/human_audit_assignment_risk_summary.json`.
Read `datasets/locomo_style_eval/human_audit_assignment_risk_summary.md`.
Use `datasets/locomo_style_eval/human_audit_reviewer_todos` as read-only routing indexes.
Use `next_todo_rows` only as routing hints.
Refresh review Markdown with scripts/summarize_locomo_human_audit_batches.py --output-review-md-dir datasets/locomo_style_eval/human_audit_batch_reviews --sidecar-root datasets/locomo_style_eval/sidecars.
"""


def valid_next_steps_doc() -> str:
    return """# Next Steps Fixture

Use `scripts/finalize_locomo_human_audit_batches.py`.
Use `scripts/finalize_locomo_human_audit_csv.py`.
Review `datasets/locomo_style_eval/human_audit_batches`.
Review `datasets/locomo_style_eval/human_audit_sheet.csv` when using the full CSV route.
Commit to `datasets/locomo_style_eval/human_audit_packet.jsonl` only through the safe finalizer.
Refresh review Markdown with scripts/summarize_locomo_human_audit_batches.py --output-review-md-dir datasets/locomo_style_eval/human_audit_batch_reviews --sidecar-root datasets/locomo_style_eval/sidecars.

## If Human Audit Is Skipped

Run scripts/check_locomo_skipped_audit_stop_point.py and write skipped_audit_stop_point_report.json plus skipped_audit_stop_point_report.md.
It verifies human_audit_batches_finalize_dry_run.json and human_audit_csv_finalize_dry_run.json.
Stop at primary/multilingual_locomo_style_eval.json.
Do not create primary/multilingual_locomo_style_eval_audited.json.
Do not create primary/audited_sources/.
Do not create baseline_results/metric_metadata.jsonl.
Do not create baseline_results/summary.json.
Do not create baseline_results/predictions/.
Do not create baseline_results/normalized/.
Do not create baseline_results/normalization_summaries/.
Do not create recent_session_ablation/model_results_summary.json.
Do not create recent_session_ablation/model_prediction_records.jsonl.
"""


def valid_runbook_doc() -> str:
    return valid_doc() + """
## Skipped Human-audit Stop Point

Run scripts/check_locomo_skipped_audit_stop_point.py and write skipped_audit_stop_point_report.json plus skipped_audit_stop_point_report.md.
It verifies human_audit_batches_finalize_dry_run.json and human_audit_csv_finalize_dry_run.json.
"""


def valid_baseline_results_doc() -> str:
    return """# Baseline Results Fixture

Use primary/multilingual_locomo_style_eval_audited.json for final baselines.
Do not use primary/multilingual_locomo_style_eval.json.
Do not create baseline_results/metric_metadata.jsonl before audit apply.
Do not create baseline_results/summary.json before audit apply.
Do not create predictions/ before audit apply.
Do not create normalized/ before audit apply.
Do not create normalization_summaries/ before audit apply.
MemGAS may be reported only with clean audited metrics.
"""


def valid_primary_readme_doc() -> str:
    return """# Primary README Fixture

This directory contains bootstrap primary JSON files, not a final release.
Use multilingual_locomo_style_eval.json only for bootstrap inspection.
Use multilingual_locomo_style_eval_audited.json for final evaluation.
Default model input is conversation-only.
Exclude observation, session_summary, event_summary, and sidecar metadata from the main context.
"""


def valid_recent_session_readme_doc() -> str:
    return """# Recent Session README Fixture

This directory contains bootstrap diagnostic inputs, not final model-result evidence.
The bootstrap input is primary/multilingual_locomo_style_eval.json.
The final input is primary/multilingual_locomo_style_eval_audited.json.
Use fixed_eval_settings.json for final runs.
Final outputs are model_results_summary.json and model_prediction_records.jsonl.
"""


def build_root(root: Path) -> None:
    ensure_project_refs(root)
    for rel in checker.REQUIRED_DOCS:
        if rel == checker.NEXT_STEPS_DOC:
            text = valid_next_steps_doc()
        elif rel == checker.RUNBOOK_DOC:
            text = valid_runbook_doc()
        else:
            text = valid_doc()
        write(root / rel, text)
    write(root / checker.BASELINE_RESULTS_DOC, valid_baseline_results_doc())
    write(root / checker.PRIMARY_README_DOC, valid_primary_readme_doc())
    write(root / checker.RECENT_SESSION_README_DOC, valid_recent_session_readme_doc())
    for idx in range(1, 11):
        stem = f"batch_{idx:03d}_Fixture_001-001"
        write(root / "human_audit_batches" / f"{stem}.csv", "source_dataset,sample_id,qa_idx,human_decision\n")
        write(
            root / "human_audit_batch_reviews" / f"{stem}.md",
            f"# {stem}\n\nSource fact ledger support:\n\n```json\n[]\n```\n",
        )


def ensure_project_refs(root: Path) -> None:
    project_root = root.parent.parent
    write(project_root / "scripts" / "finalize_locomo_human_audit_batches.py", "# fixture\n")
    write(project_root / "scripts" / "finalize_locomo_human_audit_csv.py", "# fixture\n")
    write(project_root / "scripts" / "summarize_locomo_human_audit_assignments.py", "# fixture\n")
    write(project_root / "scripts" / "export_locomo_human_audit_reviewer_todos.py", "# fixture\n")
    write(project_root / "scripts" / "check_locomo_human_audit_reviewer_todos.py", "# fixture\n")
    write(project_root / "datasets" / "locomo_style_eval" / "human_audit_packet.jsonl", "")
    write(project_root / "datasets" / "locomo_style_eval" / "human_audit_sheet.csv", "human_decision\n")
    write(project_root / "datasets" / "locomo_style_eval" / "human_audit_assignment_risk_summary.json", "{}\n")
    write(project_root / "datasets" / "locomo_style_eval" / "human_audit_assignment_risk_summary.md", "# fixture\n")
    (project_root / "datasets" / "locomo_style_eval" / "human_audit_reviewer_todos").mkdir(
        parents=True, exist_ok=True
    )
    (project_root / "datasets" / "locomo_style_eval" / "sidecars").mkdir(parents=True, exist_ok=True)


def case_result(name: str, root: Path, expect_success: bool, expected_error_fragment: str | None = None) -> dict[str, Any]:
    report = checker.validate_docs(root)
    errors = [str(item) for item in report.get("errors", [])]
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
        default=Path("datasets/locomo_style_eval/human_audit_handoff_docs_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_handoff_docs_selftest_") as tmp:
        tempdir = Path(tmp)
        good_root = tempdir / "good" / "datasets" / "locomo_style_eval"
        build_root(good_root)
        cases.append(case_result("valid_handoff_docs_are_accepted", good_root, True))

        bad_root = tempdir / "bad_direct_merge" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / "human_audit_review_guide.md",
            valid_doc()
            + """
```bash
scripts/merge_locomo_human_audit_batches.py \\
  --output-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl
```
""",
        )
        cases.append(
            case_result(
                "direct_merge_to_real_packet_is_rejected",
                bad_root,
                False,
                "lower-level merge command writes directly",
            )
        )

        bad_root = tempdir / "bad_missing_doc" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        (bad_root / "HUMAN_AUDIT_WORKPLAN.md").unlink()
        cases.append(case_result("missing_required_doc_is_rejected", bad_root, False, "missing required doc"))

        bad_root = tempdir / "bad_missing_review" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        next((bad_root / "human_audit_batch_reviews").glob("batch_*.md")).unlink()
        cases.append(case_result("missing_batch_review_is_rejected", bad_root, False, "review markdown count"))

        bad_root = tempdir / "bad_missing_fact_ledger_support" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        first_review = next((bad_root / "human_audit_batch_reviews").glob("batch_*.md"))
        write(first_review, "# missing source support\n")
        cases.append(
            case_result(
                "missing_fact_ledger_support_in_review_is_rejected",
                bad_root,
                False,
                "review markdown fact-ledger support count",
            )
        )

        bad_root = tempdir / "bad_next_steps_skip_policy" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / "NEXT_STEPS.md",
            valid_next_steps_doc().replace("## If Human Audit Is Skipped\n\n", ""),
        )
        cases.append(
            case_result(
                "missing_next_steps_skip_policy_is_rejected",
                bad_root,
                False,
                "next-steps doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_runbook_skip_policy" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / "RUNBOOK.md",
            valid_runbook_doc().replace("## Skipped Human-audit Stop Point\n\n", ""),
        )
        cases.append(
            case_result(
                "missing_runbook_skip_policy_is_rejected",
                bad_root,
                False,
                "runbook missing skipped-audit reference",
            )
        )

        bad_root = tempdir / "bad_missing_full_csv_finalizer_ref" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        quickstart_text = (bad_root / "HUMAN_AUDIT_QUICKSTART_ZH.md").read_text(encoding="utf-8")
        write(
            bad_root / "HUMAN_AUDIT_QUICKSTART_ZH.md",
            quickstart_text.replace("Use `scripts/finalize_locomo_human_audit_csv.py`.\n", ""),
        )
        cases.append(
            case_result(
                "missing_full_csv_finalizer_reference_is_rejected",
                bad_root,
                False,
                "full-CSV finalizer doc missing reference",
            )
        )

        bad_root = tempdir / "bad_missing_next_steps_full_csv_finalizer_ref" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        next_steps_text = (bad_root / "NEXT_STEPS.md").read_text(encoding="utf-8")
        write(
            bad_root / "NEXT_STEPS.md",
            next_steps_text.replace("Use `scripts/finalize_locomo_human_audit_csv.py`.\n", ""),
        )
        cases.append(
            case_result(
                "missing_next_steps_full_csv_finalizer_reference_is_rejected",
                bad_root,
                False,
                "full-CSV finalizer doc missing reference",
            )
        )

        bad_root = tempdir / "bad_missing_workplan_full_csv_finalizer_ref" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        workplan_text = (bad_root / "HUMAN_AUDIT_WORKPLAN.md").read_text(encoding="utf-8")
        write(
            bad_root / "HUMAN_AUDIT_WORKPLAN.md",
            workplan_text.replace("Use `scripts/finalize_locomo_human_audit_csv.py`.\n", ""),
        )
        cases.append(
            case_result(
                "missing_workplan_full_csv_finalizer_reference_is_rejected",
                bad_root,
                False,
                "full-CSV finalizer doc missing reference",
            )
        )

        bad_root = tempdir / "bad_missing_baseline_boundary" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / checker.BASELINE_RESULTS_DOC,
            valid_baseline_results_doc().replace("Do not create predictions/ before audit apply.\n", ""),
        )
        cases.append(
            case_result(
                "missing_baseline_results_boundary_is_rejected",
                bad_root,
                False,
                "boundary doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_primary_boundary" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / checker.PRIMARY_README_DOC,
            valid_primary_readme_doc().replace("Use multilingual_locomo_style_eval_audited.json for final evaluation.\n", ""),
        )
        cases.append(
            case_result(
                "missing_primary_readme_boundary_is_rejected",
                bad_root,
                False,
                "boundary doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_primary_input_policy" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / checker.PRIMARY_README_DOC,
            valid_primary_readme_doc().replace("Default model input is conversation-only.\n", ""),
        )
        cases.append(
            case_result(
                "missing_primary_readme_input_policy_is_rejected",
                bad_root,
                False,
                "boundary doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_recent_session_boundary" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / checker.RECENT_SESSION_README_DOC,
            valid_recent_session_readme_doc().replace("Final outputs are model_results_summary.json and model_prediction_records.jsonl.\n", ""),
        )
        cases.append(
            case_result(
                "missing_recent_session_boundary_is_rejected",
                bad_root,
                False,
                "boundary doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_review_refresh_arg" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        write(
            bad_root / "HUMAN_AUDIT_HANDOFF.md",
            valid_doc().replace(
                "Refresh review Markdown with scripts/summarize_locomo_human_audit_batches.py --output-review-md-dir datasets/locomo_style_eval/human_audit_batch_reviews --sidecar-root datasets/locomo_style_eval/sidecars.\n",
                "",
            )
            + """
```bash
scripts/summarize_locomo_human_audit_batches.py \\
  --input-dir datasets/locomo_style_eval/human_audit_batches \\
  --base-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl \\
  --output-json datasets/locomo_style_eval/human_audit_batches_progress.json
```
""",
        )
        cases.append(
            case_result(
                "missing_review_refresh_arg_is_rejected",
                bad_root,
                False,
                "progress refresh command missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_assignment_next_todo_ref" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        assignment_text = (bad_root / "HUMAN_AUDIT_ASSIGNMENTS.md").read_text(encoding="utf-8")
        write(
            bad_root / "HUMAN_AUDIT_ASSIGNMENTS.md",
            assignment_text.replace("Use `next_todo_rows` only as routing hints.\n", ""),
        )
        cases.append(
            case_result(
                "missing_assignment_next_todo_reference_is_rejected",
                bad_root,
                False,
                "assignment doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_reviewer_todo_export_ref" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        assignment_text = (bad_root / "HUMAN_AUDIT_ASSIGNMENTS.md").read_text(encoding="utf-8")
        write(
            bad_root / "HUMAN_AUDIT_ASSIGNMENTS.md",
            assignment_text.replace(
                "Export reviewer todos with `scripts/export_locomo_human_audit_reviewer_todos.py`.\n",
                "",
            ),
        )
        cases.append(
            case_result(
                "missing_reviewer_todo_export_reference_is_rejected",
                bad_root,
                False,
                "assignment doc missing required reference",
            )
        )

        bad_root = tempdir / "bad_missing_reviewer_todo_packet_arg" / "datasets" / "locomo_style_eval"
        bad_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(good_root, bad_root)
        ensure_project_refs(bad_root)
        handoff_text = (bad_root / "HUMAN_AUDIT_HANDOFF.md").read_text(encoding="utf-8")
        write(
            bad_root / "HUMAN_AUDIT_HANDOFF.md",
            handoff_text.replace(
                "Use `--audit-packet-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl` when exporting reviewer todos.\n",
                "",
            ),
        )
        cases.append(
            case_result(
                "missing_reviewer_todo_audit_packet_arg_is_rejected",
                bad_root,
                False,
                "reviewer todo export command missing required reference",
            )
        )

    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    script = Path(__file__).with_name("check_locomo_human_audit_handoff_docs.py")
    report = {
        "status": status,
        "checker": str(script),
        "checker_sha256": sha256_file(script),
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
