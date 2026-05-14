#!/usr/bin/env python3
"""Build a goal-to-artifact traceability matrix for the LoCoMo-style eval work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIREMENTS: list[dict[str, Any]] = [
    {
        "id": "objective_source_eval_files",
        "section": "Objective",
        "requirement": "Construct exactly one LoCoMo-style inference eval file for PerLTQA, OPELA, JLongChat, and deL1L2IM.",
        "checks": ["primary_output_uniqueness", "primary_exists_PerLTQA-LoCoMo-style-eval", "primary_exists_OPELA-LoCoMo-style-eval", "primary_exists_JLongChat-LoCoMo-style-eval", "primary_exists_deL1L2IM-LoCoMo-style-eval"],
    },
    {
        "id": "objective_merged_file",
        "section": "Objective",
        "requirement": "Optionally merge the four source-specific eval sets into multilingual_locomo_style_eval.json.",
        "checks": ["combined_primary_validation_passed", "combined_primary_partitions_source_files"],
    },
    {
        "id": "objective_non_equivalence_claim",
        "section": "Objective",
        "requirement": "Claim LoCoMo-loader compatibility without claiming LoCoMo-equivalent generation process or naturalness.",
        "checks": ["manifest_has_non_equivalence_claim"],
    },
    {
        "id": "rule_one_qa_set_eval_split",
        "section": "Non-negotiable Rules",
        "requirement": "Keep one QA set, locomo_style_main, and mark all samples split=eval without train/dev/test splits.",
        "checks": ["single_qa_set_eval_split", "combined_primary_validation_passed"],
    },
    {
        "id": "rule_anti_tuning",
        "section": "Non-negotiable Rules",
        "requirement": "Do not tune prompt, retrieval, chunking, compression, truncation, or cat5 refusal rules on this benchmark.",
        "checks": ["fixed_eval_settings_predeclared"],
    },
    {
        "id": "rule_primary_loader_only",
        "section": "Output Files",
        "requirement": "Primary eval JSON contains only loader-facing fields and is compatible with local LoCoMo-style loaders.",
        "checks": ["strict_validation_passed_PerLTQA-LoCoMo-style-eval", "strict_validation_passed_OPELA-LoCoMo-style-eval", "strict_validation_passed_JLongChat-LoCoMo-style-eval", "strict_validation_passed_deL1L2IM-LoCoMo-style-eval", "no_model_loader_smoke_passed", "baseline_loader_compat_smoke_passed"],
    },
    {
        "id": "rule_sidecars",
        "section": "Output Files",
        "requirement": "Store provenance, fact ledger, hashes, answer-fact tracing, verifier decisions, and audit details in sidecars.",
        "checks": [
            "sidecar_exists_PerLTQA-LoCoMo-style-eval_fact_ledger",
            "sidecar_exists_PerLTQA-LoCoMo-style-eval_provenance",
            "sidecar_exists_PerLTQA-LoCoMo-style-eval_qa_audit",
            "sidecar_exists_PerLTQA-LoCoMo-style-eval_hash_check",
            "sidecar_exists_PerLTQA-LoCoMo-style-eval_construction_report",
            "sidecar_exists_OPELA-LoCoMo-style-eval_fact_ledger",
            "sidecar_exists_OPELA-LoCoMo-style-eval_provenance",
            "sidecar_exists_OPELA-LoCoMo-style-eval_qa_audit",
            "sidecar_exists_OPELA-LoCoMo-style-eval_hash_check",
            "sidecar_exists_OPELA-LoCoMo-style-eval_construction_report",
            "sidecar_exists_JLongChat-LoCoMo-style-eval_fact_ledger",
            "sidecar_exists_JLongChat-LoCoMo-style-eval_provenance",
            "sidecar_exists_JLongChat-LoCoMo-style-eval_qa_audit",
            "sidecar_exists_JLongChat-LoCoMo-style-eval_hash_check",
            "sidecar_exists_JLongChat-LoCoMo-style-eval_construction_report",
            "sidecar_exists_deL1L2IM-LoCoMo-style-eval_fact_ledger",
            "sidecar_exists_deL1L2IM-LoCoMo-style-eval_provenance",
            "sidecar_exists_deL1L2IM-LoCoMo-style-eval_qa_audit",
            "sidecar_exists_deL1L2IM-LoCoMo-style-eval_hash_check",
            "sidecar_exists_deL1L2IM-LoCoMo-style-eval_construction_report",
            "qa_trace_integrity_passed",
            "locomo_style_validator_selftest_passed",
        ],
    },
    {
        "id": "provenance_labels_explicit",
        "section": "Provenance Labels",
        "requirement": "Use only explicit provenance labels and reject ambiguous labels such as source-derived.",
        "checks": ["planmode_provenance_passed", "planmode_provenance_selftest_passed", "primary_sidecar_alignment_passed"],
    },
    {
        "id": "rule_conversation_only_input",
        "section": "Model Input Policy",
        "requirement": "Default model input uses conversation turns only and excludes observation, session_summary, event_summary, and sidecar metadata.",
        "checks": ["summary_placeholders_empty", "no_model_loader_smoke_passed", "fixed_eval_settings_predeclared"],
    },
    {
        "id": "rule_source_fidelity",
        "section": "Source Fidelity Contract",
        "requirement": "Preserve source text, source order, session order, hashes, source IDs, and explicit original_turn provenance.",
        "checks": ["hash_coverage_passed", "primary_sidecar_alignment_passed", "source_replay_passed", "session_order_passed"],
    },
    {
        "id": "rule_no_hidden_model_calls",
        "section": "Generation and Preflight Policy",
        "requirement": "Dataset construction scripts must not silently call local or remote model services.",
        "checks": ["construction_has_zero_model_calls", "no_model_construction_guard_passed", "release_gate_no_model_guard_selftest_passed"],
    },
    {
        "id": "planmode_perltqa",
        "section": "PlanModes",
        "requirement": "PerLTQA uses PlanMode D memory-anchored dialogization with original-backed answer facts and original QA exclusion.",
        "checks": ["planmode_provenance_passed", "perltqa_original_qa_exclusion_passed", "perltqa_fact_ledger_coverage_passed", "perltqa_specific_ratios_reported"],
    },
    {
        "id": "planmode_opela",
        "section": "PlanModes",
        "requirement": "OPELA preserves Korean turns, treats pause metadata as gap hints, and does not use persona summaries as sole answer evidence.",
        "checks": ["opela_temporal_policy_passed", "opela_evidence_policy_passed"],
    },
    {
        "id": "planmode_jlongchat",
        "section": "PlanModes",
        "requirement": "JLongChat preserves Japanese source text and uses raw-preserving/light completion boundaries.",
        "checks": ["planmode_provenance_passed", "source_replay_passed", "hash_coverage_passed"],
    },
    {
        "id": "planmode_del1l2im",
        "section": "PlanModes",
        "requirement": "deL1L2IM remains original-only PlanMode A without expansion, XML splitting, or synthetic answer facts.",
        "checks": ["del1l2im_source_policy_passed"],
    },
    {
        "id": "per_source_perltqa_pipeline",
        "section": "Per-source Construction Requirements",
        "requirement": "PerLTQA extracts memory/profile/event/dialogue facts, excludes original QA from direct eval use, and keeps answer facts backed by the fact ledger.",
        "checks": [
            "planmode_provenance_passed",
            "perltqa_original_qa_exclusion_passed",
            "perltqa_fact_ledger_coverage_passed",
            "perltqa_specific_ratios_reported",
        ],
    },
    {
        "id": "per_source_opela_pipeline",
        "section": "Per-source Construction Requirements",
        "requirement": "OPELA preserves Korean dialogue/source turns, treats pause metadata as session-gap hints, and prevents summary-only answer evidence.",
        "checks": ["opela_temporal_policy_passed", "opela_evidence_policy_passed", "hash_coverage_passed", "source_replay_passed"],
    },
    {
        "id": "per_source_jlongchat_pipeline",
        "section": "Per-source Construction Requirements",
        "requirement": "JLongChat maps LAC/JMSC session structure without translation or polishing and keeps persona summaries out of original utterance provenance.",
        "checks": ["planmode_provenance_passed", "source_replay_passed", "hash_coverage_passed", "session_order_passed"],
    },
    {
        "id": "per_source_del1l2im_pipeline",
        "section": "Per-source Construction Requirements",
        "requirement": "deL1L2IM parses IM messages into original-only sessions/turns without expansion, XML splitting, or synthetic learning events.",
        "checks": ["del1l2im_source_policy_passed", "source_replay_passed", "session_order_passed"],
    },
    {
        "id": "qa_generation_rules",
        "section": "QA Generation Rules",
        "requirement": "Final QA follows one locomo_style_main set, 20-40 QA/sample, LoCoMo categories, evidence IDs, answer-fact tracing, and cat5 adversarial handling.",
        "checks": ["single_qa_set_eval_split", "combined_primary_validation_passed", "qa_trace_integrity_passed", "qa_quality_passed", "qa_quality_selftest_passed", "locomo_style_validator_selftest_passed"],
    },
    {
        "id": "verifier_requirements",
        "section": "Verifier Requirements",
        "requirement": "Verifier judges source entailment and rejects answer-critical facts without original-backed source_fact_id support.",
        "checks": ["qa_trace_integrity_passed", "qa_trace_integrity_selftest_passed", "locomo_style_validator_selftest_passed"],
    },
    {
        "id": "validation_requirements",
        "section": "Validation Requirements",
        "requirement": "No-model loader smoke, hash, order, evidence, trace, synthetic-only, cat5, and source-specific validations pass before release.",
        "checks": ["no_model_loader_smoke_passed", "hash_coverage_passed", "session_order_passed", "qa_trace_integrity_passed", "summary_placeholders_empty", "opela_evidence_policy_passed", "del1l2im_source_policy_passed"],
    },
    {
        "id": "human_audit_minimum",
        "section": "Validation Requirements",
        "requirement": "Complete required human audit coverage and apply resulting fixes/deletions before release.",
        "checks": [
            "hash_coverage_passed",
            "hash_coverage_selftest_passed",
            "human_audit_queue_coverage_passed",
            "human_audit_queue_coverage_selftest_passed",
            "human_audit_packet_matches_queue",
            "human_audit_validator_selftest_passed",
            "human_audit_csv_workflow_selftest_passed",
            "human_audit_handoff_docs_passed",
            "human_audit_handoff_docs_selftest_passed",
            "human_audit_flags_selftest_passed",
            "human_audit_assignment_risk_summary_passed",
            "human_audit_reviewer_todos_fresh",
            "human_audit_reviewer_todos_selftest_passed",
            "skipped_audit_stop_point_report_fresh",
            "skipped_audit_stop_point_selftest_passed",
        ],
        "blockers": ["human_audit_completed", "human_audit_applied", "audited_primary_validation_passed", "audited_source_files_validation_passed", "audited_apply_integrity_passed"],
    },
    {
        "id": "long_memory_diagnostic",
        "section": "Long-memory Diagnostic",
        "requirement": "Run recent-session diagnostics: last session, last 3 sessions, and full conversation on the audited eval.",
        "checks": [
            "construction_long_memory_diagnostic_passed",
            "recent_session_ablation_files_created",
            "recent_session_result_gate_selftest_passed",
            "recent_session_runner_settings_selftest_passed",
        ],
        "blockers": ["recent_session_model_results_exist"],
    },
    {
        "id": "final_fixed_baselines",
        "section": "Final Evaluation",
        "requirement": "Run fixed Full Context, A-MEM, Mem0, SimpleMem, and HiGMem baselines on the audited multilingual eval.",
        "checks": [
            "fixed_eval_settings_predeclared",
            "baseline_results_validator_selftest_passed",
            "baseline_summary_builder_selftest_passed",
            "prediction_normalizer_selftest_passed",
            "experiment_preflight_selftest_passed",
        ],
        "blockers": ["metric_metadata_created", "fixed_baseline_results_exist"],
    },
    {
        "id": "final_reporting",
        "section": "Final Evaluation",
        "requirement": "Report answerable and cat5 metrics by source, language, category, cross-session status, and evidence provenance.",
        "checks": [
            "fixed_eval_settings_predeclared",
            "baseline_results_validator_selftest_passed",
            "baseline_summary_builder_selftest_passed",
        ],
        "blockers": ["metric_metadata_created", "fixed_baseline_results_exist"],
    },
    {
        "id": "final_release_status",
        "section": "Execution Priority",
        "requirement": "Do not mark final release until all release gates pass.",
        "checks": [
            "manifest_not_claiming_final_release",
            "no_stale_final_outputs_before_audit_apply",
            "skipped_audit_stop_point_report_fresh",
            "skipped_audit_stop_point_selftest_passed",
            "goal_traceability_matrix_fresh",
        ],
        "blockers": ["goal_traceability_matrix_fresh", "human_audit_completed", "human_audit_applied", "audited_primary_validation_passed", "audited_source_files_validation_passed", "audited_apply_integrity_passed", "metric_metadata_created", "recent_session_model_results_exist", "fixed_baseline_results_exist"],
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_index(release_gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): row
        for row in release_gate.get("checks", [])
        if isinstance(row, dict) and row.get("name")
    }


def requirement_status(requirement: dict[str, Any], checks: dict[str, dict[str, Any]], blocking_failed: set[str]) -> tuple[str, list[str], list[str]]:
    missing: list[str] = []
    failed: list[str] = []
    active_blockers = [name for name in requirement.get("blockers", []) if name in blocking_failed]
    for name in requirement.get("checks", []):
        row = checks.get(name)
        if row is None:
            missing.append(name)
        elif row.get("status") != "passed":
            failed.append(name)
    if active_blockers:
        return "blocked", active_blockers, missing + failed
    if missing or failed:
        return "missing_or_failed", active_blockers, missing + failed
    return "passed", [], []


def compact_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + " ... [truncated]"


def summarize(goal_doc: Path, release_gate_report: Path) -> dict[str, Any]:
    release_gate = load_json(release_gate_report)
    checks = check_index(release_gate)
    blocking_failed = {str(item) for item in release_gate.get("blocking_failed", [])}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for req in REQUIREMENTS:
        status, active_blockers, missing_or_failed = requirement_status(req, checks, blocking_failed)
        counts[status] = counts.get(status, 0) + 1
        evidence = []
        for check_name in req.get("checks", []):
            check = checks.get(check_name)
            if check:
                evidence.append(
                    {
                        "check": check_name,
                        "status": check.get("status"),
                        "evidence": check.get("evidence"),
                    }
                )
            else:
                evidence.append({"check": check_name, "status": "missing", "evidence": ""})
        rows.append(
            {
                "id": req["id"],
                "section": req["section"],
                "requirement": req["requirement"],
                "status": status,
                "active_blockers": active_blockers,
                "missing_or_failed_checks": missing_or_failed,
                "evidence": evidence,
            }
        )
    status = "passed" if counts.get("blocked", 0) == 0 and counts.get("missing_or_failed", 0) == 0 else "blocked"
    return {
        "status": status,
        "goal_doc": str(goal_doc),
        "goal_doc_sha256": sha256_file(goal_doc),
        "release_gate_report": str(release_gate_report),
        "release_gate_sha256": sha256_file(release_gate_report),
        "release_gate_status": release_gate.get("status"),
        "blocking_failed": sorted(blocking_failed),
        "counts": dict(sorted(counts.items())),
        "requirements": rows,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Goal Traceability Matrix",
        "",
        f"Status: `{summary['status']}`",
        f"Release gate status: `{summary.get('release_gate_status')}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in summary.get("counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Requirements", ""])
    for row in summary.get("requirements", []):
        blockers = ", ".join(row.get("active_blockers", [])) or "none"
        missing = ", ".join(row.get("missing_or_failed_checks", [])) or "none"
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"- Section: {row['section']}",
                f"- Status: `{row['status']}`",
                f"- Active blockers: {blockers}",
                f"- Missing/failed checks: {missing}",
                f"- Requirement: {row['requirement']}",
                "- Release checks:",
            ]
        )
        for evidence in row.get("evidence", []):
            lines.append(
                f"  - `{evidence['check']}`: `{evidence['status']}`; "
                f"evidence: {compact_text(evidence.get('evidence'))}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-doc", type=Path, default=Path("docs/locomo_style_eval_goal.md"))
    parser.add_argument(
        "--release-gate-report",
        type=Path,
        default=Path("datasets/locomo_style_eval/release_gate_report.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/goal_traceability_matrix.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("datasets/locomo_style_eval/goal_traceability_matrix.md"),
    )
    args = parser.parse_args()

    summary = summarize(args.goal_doc, args.release_gate_report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
