#!/usr/bin/env python3
"""Static guard that construction scripts do not call model or HTTP services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SCRIPTS = [
    "scripts/build_locomo_style_eval.py",
    "scripts/validate_locomo_style_eval.py",
    "scripts/selftest_locomo_style_validator.py",
    "scripts/smoke_locomo_loader_no_model.py",
    "scripts/filter_locomo_categories.py",
    "scripts/analyze_locomo_long_memory_diagnostic.py",
    "scripts/check_locomo_planmode_provenance.py",
    "scripts/selftest_locomo_planmode_provenance.py",
    "scripts/check_locomo_perltqa_original_qa_exclusion.py",
    "scripts/selftest_locomo_perltqa_original_qa_exclusion.py",
    "scripts/check_locomo_perltqa_fact_ledger_coverage.py",
    "scripts/selftest_locomo_perltqa_fact_ledger_coverage.py",
    "scripts/check_locomo_summary_placeholders.py",
    "scripts/selftest_locomo_summary_placeholders.py",
    "scripts/check_locomo_opela_temporal_policy.py",
    "scripts/selftest_locomo_opela_temporal_policy.py",
    "scripts/check_locomo_opela_evidence_policy.py",
    "scripts/selftest_locomo_opela_evidence_policy.py",
    "scripts/check_locomo_del1l2im_source_policy.py",
    "scripts/selftest_locomo_del1l2im_source_policy.py",
    "scripts/check_locomo_hash_coverage.py",
    "scripts/selftest_locomo_hash_coverage.py",
    "scripts/check_locomo_qa_quality.py",
    "scripts/selftest_locomo_qa_quality.py",
    "scripts/select_locomo_human_audit_subset.py",
    "scripts/export_locomo_human_audit_packet.py",
    "scripts/validate_locomo_human_audit_queue.py",
    "scripts/selftest_locomo_human_audit_queue_coverage.py",
    "scripts/validate_locomo_human_audit_results.py",
    "scripts/selftest_locomo_human_audit_validator.py",
    "scripts/selftest_locomo_human_audit_results_gate.py",
    "scripts/apply_locomo_human_audit_results.py",
    "scripts/run_locomo_post_audit_pipeline.py",
    "scripts/selftest_locomo_post_audit_pipeline.py",
    "scripts/check_locomo_audited_apply_integrity.py",
    "scripts/selftest_locomo_audit_apply_integrity.py",
    "scripts/make_locomo_recent_session_ablation.py",
    "scripts/validate_locomo_baseline_results.py",
    "scripts/check_locomo_style_release_gates.py",
    "scripts/finalize_locomo_style_release.py",
    "scripts/export_locomo_human_audit_csv.py",
    "scripts/export_locomo_human_audit_batches.py",
    "scripts/import_locomo_human_audit_csv.py",
    "scripts/finalize_locomo_human_audit_csv.py",
    "scripts/merge_locomo_human_audit_batches.py",
    "scripts/check_locomo_human_audit_batch_edits.py",
    "scripts/finalize_locomo_human_audit_batches.py",
    "scripts/selftest_locomo_human_audit_csv_workflow.py",
    "scripts/check_locomo_human_audit_handoff_docs.py",
    "scripts/selftest_locomo_human_audit_handoff_docs.py",
    "scripts/summarize_locomo_human_audit_progress.py",
    "scripts/summarize_locomo_human_audit_batches.py",
    "scripts/summarize_locomo_human_audit_flags.py",
    "scripts/summarize_locomo_human_audit_assignments.py",
    "scripts/export_locomo_human_audit_reviewer_todos.py",
    "scripts/selftest_locomo_human_audit_reviewer_todos.py",
    "scripts/check_locomo_human_audit_reviewer_todos.py",
    "scripts/selftest_locomo_human_audit_reviewer_todos_check.py",
    "scripts/selftest_locomo_human_audit_flags.py",
    "scripts/summarize_locomo_release_blockers.py",
    "scripts/selftest_locomo_release_blocker_summary.py",
    "scripts/check_locomo_skipped_audit_stop_point.py",
    "scripts/selftest_locomo_skipped_audit_stop_point.py",
    "scripts/summarize_locomo_dataset_card.py",
    "scripts/selftest_locomo_dataset_card_summary.py",
    "scripts/summarize_locomo_goal_traceability.py",
    "scripts/selftest_locomo_goal_traceability.py",
    "scripts/smoke_locomo_baseline_loader_compat.py",
    "scripts/check_locomo_primary_sidecar_alignment.py",
    "scripts/selftest_locomo_primary_sidecar_alignment.py",
    "scripts/check_locomo_source_replay.py",
    "scripts/selftest_locomo_source_replay.py",
    "scripts/check_locomo_qa_trace_integrity.py",
    "scripts/selftest_locomo_qa_trace_integrity.py",
    "scripts/check_locomo_session_order.py",
    "scripts/selftest_locomo_session_order.py",
    "scripts/report_locomo_perltqa_specific_ratios.py",
    "scripts/build_locomo_metric_metadata.py",
    "scripts/selftest_locomo_metric_metadata_builder.py",
    "scripts/build_locomo_baseline_summary.py",
    "scripts/selftest_locomo_baseline_summary_builder.py",
    "scripts/selftest_locomo_baseline_results_validator.py",
    "scripts/normalize_locomo_baseline_predictions.py",
    "scripts/selftest_locomo_prediction_normalizer.py",
    "scripts/selftest_locomo_recent_session_runner_settings.py",
    "scripts/selftest_locomo_recent_session_result_gate.py",
    "scripts/selftest_locomo_experiment_preflight.py",
    "scripts/selftest_locomo_release_gate_no_model_guard.py",
    "scripts/selftest_locomo_audited_source_validation_gate.py",
]

FORBIDDEN_PATTERNS = [
    r"\bfrom\s+openai\b",
    r"\bimport\s+openai\b",
    r"\bOpenAI\s*\(",
    r"\bAsyncOpenAI\s*\(",
    r"\bfrom\s+vllm\b",
    r"\bimport\s+vllm\b",
    r"\brequests\s*\.",
    r"\bhttpx\s*\.",
    r"\baiohttp\b",
    r"\burllib\.request\b",
    r"\blitellm\b",
    r"\btransformers\.pipeline\b",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_file(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line": lineno, "pattern": pattern, "text": line.strip()})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/no_model_construction_guard.json"))
    parser.add_argument("--script", action="append", default=None, help="Script path to scan. May be repeated.")
    args = parser.parse_args()

    scripts = [Path(item) for item in (args.script or DEFAULT_SCRIPTS)]
    missing = [str(path) for path in scripts if not path.exists()]
    findings: dict[str, list[dict[str, Any]]] = {}
    for path in scripts:
        if path.exists():
            rows = scan_file(path)
            if rows:
                findings[str(path)] = rows

    report = {
        "status": "passed" if not missing and not findings else "failed",
        "scanned_scripts": [str(path) for path in scripts],
        "script_hashes": {
            str(path): sha256_file(path)
            for path in scripts
            if path.exists()
        },
        "excluded_allowed_model_diagnostic": "scripts/run_locomo_recent_session_model_diagnostic.py",
        "missing": missing,
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
