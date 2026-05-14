#!/usr/bin/env python3
"""Check release gates for the LoCoMo-style multilingual eval artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from check_locomo_human_audit_handoff_docs import (
    FULL_CSV_FINALIZER_DOCS as HANDOFF_FULL_CSV_FINALIZER_DOCS,
    REQUIRED_BASELINE_RESULTS_REFERENCES as REQUIRED_HANDOFF_BASELINE_RESULTS_REFERENCES,
    REQUIRED_FULL_CSV_FINALIZER_REFERENCES as REQUIRED_HANDOFF_FULL_CSV_FINALIZER_REFERENCES,
    REQUIRED_NEXT_STEPS_REFERENCES as REQUIRED_HANDOFF_NEXT_STEPS_REFERENCES,
    REQUIRED_PRIMARY_README_REFERENCES as REQUIRED_HANDOFF_PRIMARY_README_REFERENCES,
    REQUIRED_RUNBOOK_SKIPPED_REFERENCES as REQUIRED_HANDOFF_RUNBOOK_SKIPPED_REFERENCES,
)


SOURCE_ARTIFACTS = [
    "PerLTQA-LoCoMo-style-eval",
    "OPELA-LoCoMo-style-eval",
    "JLongChat-LoCoMo-style-eval",
    "deL1L2IM-LoCoMo-style-eval",
]
REQUIRED_BASELINE_METHODS = {"Full Context", "A-MEM", "Mem0", "SimpleMem", "HiGMem"}
REQUIRED_QA_SET = "locomo_style_main"
REQUIRED_SPLIT = "eval"
REQUIRED_BASELINE_RESULT_FIELDS = {
    "method",
    "qa_count",
    "answerable_qa_count",
    "cat5_qa_count",
    "overall_answerable",
    "by_source_dataset",
    "by_language",
    "by_category",
    "by_cross_session",
    "by_evidence_provenance",
    "cat5_refusal",
    "cat5_unsupported_claim",
}
REQUIRED_REPORT_GROUP_BY = {
    "source_dataset",
    "language",
    "category",
    "whether_cross_session",
    "evidence_origin",
}
REQUIRED_ANTI_TUNING_ITEMS = {
    "prompt format",
    "retrieval top-k",
    "chunking",
    "memory compression",
    "context truncation",
    "cat5 refusal rules",
}
REQUIRED_AUDIT_CSV_WORKFLOW_CASES = {
    "csv_import_rejects_read_only_field_edits",
    "batch_merge_rejects_read_only_field_edits",
}
REQUIRED_GOAL_TRACEABILITY_IDS = {
    "objective_source_eval_files",
    "objective_merged_file",
    "objective_non_equivalence_claim",
    "provenance_labels_explicit",
    "per_source_perltqa_pipeline",
    "per_source_opela_pipeline",
    "per_source_jlongchat_pipeline",
    "per_source_del1l2im_pipeline",
    "human_audit_minimum",
    "long_memory_diagnostic",
    "final_fixed_baselines",
    "final_reporting",
    "final_release_status",
}
REQUIRED_FIXED_MODEL = "Qwen/Qwen3-8B"
REQUIRED_PREDICTION_IDENTITY_FIELDS = {"sample_id", "qa_idx"}
REQUIRED_PREDICTION_FIELD_OPTIONS = {"prediction", "answer", "response"}
REQUIRED_PREDICTION_ROW_FIELDS = {"sample_id", "qa_idx", "model", "dataset_sha256"}
OPTIONAL_MEMGAS_POLICY = {
    "required_for_release": False,
    "include_policy": "only_if_clean_metrics_available",
    "include_in_main_table": False,
    "clean_metrics_required": True,
}
REQUIRED_SKIPPED_AUDIT_BLOCKERS = {
    "human_audit_completed",
    "human_audit_applied",
    "audited_primary_validation_passed",
    "audited_source_files_validation_passed",
    "audited_apply_integrity_passed",
    "metric_metadata_created",
    "recent_session_model_results_exist",
    "fixed_baseline_results_exist",
}
SKIPPED_AUDIT_INPUT_FILES = {
    "primary/multilingual_locomo_style_eval.json",
    "manifest.json",
    "release_gate_report.json",
    "human_audit_results_summary.json",
    "human_audit_batches_finalize_dry_run.json",
    "human_audit_csv_finalize_dry_run.json",
    "post_audit_pipeline_report.json",
}
SKIPPED_AUDIT_NON_CIRCULAR_INPUT_FILES = SKIPPED_AUDIT_INPUT_FILES - {"release_gate_report.json"}
SKIPPED_AUDIT_FINAL_OUTPUTS = {
    "primary/multilingual_locomo_style_eval_audited.json",
    "primary/audited_sources",
    "baseline_results/metric_metadata.jsonl",
    "baseline_results/summary.json",
    "baseline_results/predictions",
    "baseline_results/normalized",
    "baseline_results/normalization_summaries",
    "recent_session_ablation/model_results_summary.json",
    "recent_session_ablation/model_prediction_records.jsonl",
}
REQUIRED_CONSTRUCTION_REPORT_FRAGMENTS = {
    "PerLTQA-LoCoMo-style-eval": [
        "PerLTQA PlanMode D",
        "memory_anchor_turns",
        "original PerLTQA QA is not copied into final eval",
    ],
    "OPELA-LoCoMo-style-eval": [
        "OPELA PlanMode C",
        "turn order is reconstructed",
    ],
    "JLongChat-LoCoMo-style-eval": [
        "JLongChat PlanMode A/B",
        "no Japanese source text was translated or polished",
    ],
    "deL1L2IM-LoCoMo-style-eval": [
        "deL1L2IM PlanMode A",
        "without synthetic turns",
    ],
}
REQUIRED_NO_MODEL_GUARD_SCRIPTS = {
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
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_trace_files_sha256(sidecar_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        {
            *sidecar_root.glob("*/*_fact_ledger.jsonl"),
            *sidecar_root.glob("*/*_provenance.jsonl"),
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sidecar_audit_trace_files_sha256(sidecar_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        {
            *sidecar_root.glob("*/*_fact_ledger.jsonl"),
            *sidecar_root.glob("*/*_provenance.jsonl"),
            *sidecar_root.glob("*/*_qa_audit.jsonl"),
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_counts(path: Path) -> dict[str, int]:
    samples = load_json(path)
    qa_count = 0
    cat5_count = 0
    for sample in samples:
        for qa in sample.get("qa", []):
            qa_count += 1
            if qa.get("category") == 5:
                cat5_count += 1
    return {
        "qa_count": qa_count,
        "answerable_qa_count": qa_count - cat5_count,
        "cat5_qa_count": cat5_count,
    }


def expected_metadata_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    samples = load_json(path)
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            category = int(qa.get("category"))
            expected[(sample_id, qa_idx)] = {
                "source_dataset": str(sample.get("source_dataset")),
                "language": str(sample.get("language")),
                "category": category,
                "answerable": category != 5,
            }
    return expected


def metadata_dataset_alignment_errors(metadata_path: Path, dataset_path: Path) -> list[str]:
    errors: list[str] = []
    expected_rows = expected_metadata_index(dataset_path)
    seen_keys: set[tuple[str, int]] = set()
    examples: list[str] = []
    for row_idx, row in enumerate(iter_jsonl(metadata_path), start=1):
        try:
            key = (str(row["sample_id"]), int(row["qa_idx"]))
        except (KeyError, TypeError, ValueError):
            if len(examples) < 10:
                examples.append(f"row {row_idx} has invalid sample_id/qa_idx")
            continue
        if key in seen_keys:
            if len(examples) < 10:
                examples.append(f"row {row_idx} duplicate key={key}")
            continue
        seen_keys.add(key)
        expected = expected_rows.get(key)
        if expected is None:
            if len(examples) < 10:
                examples.append(f"row {row_idx} key={key} not found in dataset")
            continue
        for field in ("source_dataset", "language", "category", "answerable"):
            if row.get(field) != expected[field]:
                if len(examples) < 10:
                    examples.append(
                        f"row {row_idx} key={key} {field}={row.get(field)!r} expected={expected[field]!r}"
                    )
                break
    missing_keys = set(expected_rows) - seen_keys
    if missing_keys:
        errors.append(f"metric metadata missing dataset QA keys; first={sorted(missing_keys)[:10]}")
    errors.extend(examples)
    return errors


def metadata_group_keys(metadata_path: Path) -> tuple[dict[str, set[str]], bool]:
    groups: dict[str, set[str]] = {
        "by_source_dataset": set(),
        "by_language": set(),
        "by_category": set(),
        "by_cross_session": set(),
        "by_evidence_provenance": set(),
    }
    has_cat5 = False
    for row in iter_jsonl(metadata_path):
        if row.get("answerable") is True:
            groups["by_source_dataset"].add(str(row.get("source_dataset")))
            groups["by_language"].add(str(row.get("language")))
            groups["by_category"].add(str(row.get("category")))
            groups["by_cross_session"].add(str(row.get("whether_cross_session")).lower())
            groups["by_evidence_provenance"].add(str(row.get("evidence_provenance")))
        else:
            has_cat5 = True
    return groups, has_cat5


def prediction_key(row: dict[str, Any], row_idx: int, errors: list[str], method: str) -> tuple[str, int] | None:
    try:
        sample_id = str(row["sample_id"])
        qa_idx = int(row["qa_idx"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{method}: prediction row {row_idx} missing valid sample_id/qa_idx")
        return None
    return sample_id, qa_idx


def prediction_files_errors(report: dict[str, Any], dataset_path: Path) -> list[str]:
    errors: list[str] = []
    prediction_files = report.get("prediction_files")
    if not isinstance(prediction_files, dict):
        return ["prediction_files must be an object with one entry per required method"]
    observed_methods = {str(item) for item in prediction_files}
    missing_methods = sorted(REQUIRED_BASELINE_METHODS - observed_methods)
    extra_methods = sorted(observed_methods - REQUIRED_BASELINE_METHODS)
    if missing_methods:
        errors.append(f"prediction_files missing required methods: {missing_methods}")
    if extra_methods:
        errors.append(f"prediction_files has unexpected methods: {extra_methods}")

    expected_keys = set(expected_metadata_index(dataset_path))
    expected_dataset_sha256 = sha256_file(dataset_path)
    for method in sorted(REQUIRED_BASELINE_METHODS & observed_methods):
        entry = prediction_files.get(method)
        if not isinstance(entry, dict):
            errors.append(f"prediction_files[{method}] must be an object")
            continue
        path_value = entry.get("path")
        sha_value = entry.get("sha256")
        if not path_value:
            errors.append(f"prediction_files[{method}].path is required")
            continue
        path = Path(str(path_value))
        if not path.is_file():
            errors.append(f"prediction_files[{method}].path not found: {path}")
            continue
        if sha_value != sha256_file(path):
            errors.append(f"prediction_files[{method}].sha256 does not match file")

        seen_keys: set[tuple[str, int]] = set()
        row_errors: list[str] = []
        for row_idx, row in enumerate(iter_jsonl(path), start=1):
            if not isinstance(row, dict):
                row_errors.append(f"{method}: prediction row {row_idx} is not an object")
                continue
            key = prediction_key(row, row_idx, row_errors, method)
            if key is None:
                continue
            if key in seen_keys:
                row_errors.append(f"{method}: duplicate prediction key={key}")
                continue
            seen_keys.add(key)
            if key not in expected_keys:
                row_errors.append(f"{method}: prediction key={key} not found in audited dataset")
            if not any(field in row for field in ("prediction", "answer", "response")):
                row_errors.append(f"{method}: prediction key={key} missing prediction/answer/response field")
            if row.get("model") != REQUIRED_FIXED_MODEL:
                row_errors.append(
                    f"{method}: prediction key={key} model={row.get('model')!r} "
                    f"expected={REQUIRED_FIXED_MODEL!r}"
                )
            if row.get("dataset_sha256") != expected_dataset_sha256:
                row_errors.append(
                    f"{method}: prediction key={key} dataset_sha256={row.get('dataset_sha256')!r} "
                    f"expected={expected_dataset_sha256!r}"
                )
        missing_keys = sorted(expected_keys - seen_keys)
        extra_keys = sorted(seen_keys - expected_keys)
        if missing_keys:
            row_errors.append(f"{method}: missing predictions for {len(missing_keys)} QA; first={missing_keys[:10]}")
        if extra_keys:
            row_errors.append(f"{method}: predictions contain {len(extra_keys)} unexpected QA keys; first={extra_keys[:10]}")
        errors.extend(row_errors[:20])
        if len(row_errors) > 20:
            errors.append(f"{method}: {len(row_errors) - 20} additional prediction file errors omitted")
    return errors


def has_numeric_leaf(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return any(has_numeric_leaf(item) for item in value.values())
    if isinstance(value, list):
        return any(has_numeric_leaf(item) for item in value)
    return False


def metric_object_errors(value: Any, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    if not value:
        return [f"{path} must be non-empty"]
    if not has_numeric_leaf(value):
        return [f"{path} must contain at least one numeric metric"]
    return []


def group_result_errors(value: Any, path: str, expected_keys: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    if not value:
        return [f"{path} must be non-empty"]
    observed_keys = {str(item) for item in value}
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing:
        errors.append(f"{path} missing expected groups {missing}")
    if extra:
        errors.append(f"{path} has unexpected groups {extra}")
    for key in sorted(expected_keys & observed_keys):
        errors.extend(metric_object_errors(value.get(key), f"{path}.{key}"))
    return errors


def dataset_answerable_count_for_categories(path: Path, categories: set[str]) -> int:
    samples = load_json(path)
    total = 0
    for sample in samples:
        for qa in sample.get("qa", []):
            category = str(qa.get("category"))
            if category != "5" and category in categories:
                total += 1
    return total


def recent_session_expected_record_keys(path: Path, categories: set[str]) -> set[tuple[str, str, int]]:
    samples = load_json(path)
    expected: set[tuple[str, str, int]] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            category = str(qa.get("category"))
            if category == "5" or category not in categories:
                continue
            for context_name in ("full_conversation", "last_session_only", "last_3_sessions_only"):
                expected.add((context_name, sample_id, qa_idx))
    return expected


def recent_session_prediction_record_errors(
    report: dict[str, Any],
    expected_dataset: Path | None,
    categories: set[str],
) -> list[str]:
    errors: list[str] = []
    records_output = report.get("records_output")
    if not records_output:
        return ["records_output is required"]
    records_path = Path(str(records_output))
    if not records_path.is_file():
        return [f"records_output not found: {records_path}"]
    if report.get("records_output_sha256") != sha256_file(records_path):
        errors.append("records_output_sha256 does not match records_output")

    expected_keys: set[tuple[str, str, int]] = set()
    expected_dataset_sha256: str | None = None
    if expected_dataset is not None and expected_dataset.is_file() and categories:
        expected_keys = recent_session_expected_record_keys(expected_dataset, categories)
        expected_dataset_sha256 = sha256_file(expected_dataset)

    seen_keys: set[tuple[str, str, int]] = set()
    row_errors: list[str] = []
    row_count = 0
    for row_count, row in enumerate(iter_jsonl(records_path), start=1):
        if not isinstance(row, dict):
            row_errors.append(f"record row {row_count} is not an object")
            continue
        try:
            key = (str(row["context_name"]), str(row["sample_id"]), int(row["qa_idx"]))
        except (KeyError, TypeError, ValueError):
            row_errors.append(f"record row {row_count} missing valid context_name/sample_id/qa_idx")
            continue
        if key in seen_keys:
            row_errors.append(f"duplicate recent-session record key={key}")
            continue
        seen_keys.add(key)
        if expected_keys and key not in expected_keys:
            row_errors.append(f"unexpected recent-session record key={key}")
        if row.get("model") != REQUIRED_FIXED_MODEL:
            row_errors.append(f"record key={key} model={row.get('model')!r} expected={REQUIRED_FIXED_MODEL!r}")
        if expected_dataset_sha256 is not None and row.get("ablation_input_sha256") != expected_dataset_sha256:
            row_errors.append(
                f"record key={key} ablation_input_sha256={row.get('ablation_input_sha256')!r} "
                f"expected={expected_dataset_sha256!r}"
            )
        if row.get("error") not in (None, ""):
            row_errors.append(f"record key={key} error={row.get('error')!r}")
        try:
            token_f1 = float(row.get("token_f1"))
            if not math.isfinite(token_f1):
                raise ValueError
        except (TypeError, ValueError):
            row_errors.append(f"record key={key} token_f1 is not finite numeric")

    if report.get("written_records") is not None and row_count != report.get("written_records"):
        row_errors.append(f"records_output rows={row_count} expected written_records={report.get('written_records')}")
    if expected_keys:
        missing = sorted(expected_keys - seen_keys)
        extra = sorted(seen_keys - expected_keys)
        if missing:
            row_errors.append(f"recent-session records missing {len(missing)} expected keys; first={missing[:10]}")
        if extra:
            row_errors.append(f"recent-session records contain {len(extra)} unexpected keys; first={extra[:10]}")

    errors.extend(row_errors[:20])
    if len(row_errors) > 20:
        errors.append(f"{len(row_errors) - 20} additional recent-session record errors omitted")
    return errors


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_report_source_file(source_file: str) -> Path | None:
    if not source_file:
        return None
    path = Path(source_file)
    if path.is_absolute():
        return path
    cwd_relative = Path.cwd() / source_file
    if cwd_relative.exists():
        return cwd_relative
    if path.exists():
        return path
    return path


def collect_source_replay_raw_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for artifact in SOURCE_ARTIFACTS:
        provenance_path = root / "sidecars" / artifact / f"{artifact}_provenance.jsonl"
        if not provenance_path.is_file():
            continue
        for row in iter_jsonl(provenance_path):
            source_path = resolve_report_source_file(str(row.get("source_file", "")))
            if source_path is not None:
                paths.add(source_path)
    return sorted(paths)


def validation_report_errors(
    report: dict[str, Any],
    expected_files: dict[str, Path],
    *,
    allowed_extra_keys: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    input_files = report.get("input_files")
    if not isinstance(input_files, dict):
        return [*errors, "missing input_files"]
    for key, path in expected_files.items():
        row = input_files.get(key)
        if not isinstance(row, dict):
            errors.append(f"input_files.{key} missing")
            continue
        if row.get("path") != str(path):
            errors.append(f"input_files.{key}.path={row.get('path')!r} expected={str(path)!r}")
        if not path.is_file():
            errors.append(f"{key} file missing: {path}")
        elif row.get("sha256") != sha256_file(path):
            errors.append(f"input_files.{key}.sha256 mismatch")
    allowed_extra_keys = allowed_extra_keys or set()
    extra_keys = sorted(set(input_files) - set(expected_files) - allowed_extra_keys)
    if extra_keys:
        errors.append(f"input_files has unexpected keys {extra_keys}")
    return errors


def goal_traceability_matrix_errors(
    matrix: dict[str, Any],
    *,
    matrix_path: Path,
    goal_doc_path: Path,
    release_gate_report_path: Path,
) -> list[str]:
    errors: list[str] = []
    if not matrix_path.exists():
        errors.append("matrix missing")
    if not goal_doc_path.is_file():
        errors.append(f"goal doc missing: {goal_doc_path}")
    elif matrix.get("goal_doc") != str(goal_doc_path):
        errors.append(f"goal_doc={matrix.get('goal_doc')!r} expected={str(goal_doc_path)!r}")
    elif matrix.get("goal_doc_sha256") != sha256_file(goal_doc_path):
        errors.append("goal_doc_sha256 mismatch")
    if not release_gate_report_path.is_file():
        errors.append(f"release gate report missing: {release_gate_report_path}")
    elif matrix.get("release_gate_report") != str(release_gate_report_path):
        errors.append(
            f"release_gate_report={matrix.get('release_gate_report')!r} "
            f"expected={str(release_gate_report_path)!r}"
        )
    elif matrix.get("release_gate_sha256") != sha256_file(release_gate_report_path):
        errors.append("release_gate_sha256 mismatch")
    observed_traceability_ids = {
        str(row.get("id"))
        for row in matrix.get("requirements", [])
        if isinstance(row, dict) and row.get("id")
    }
    missing_traceability_ids = sorted(REQUIRED_GOAL_TRACEABILITY_IDS - observed_traceability_ids)
    if missing_traceability_ids:
        errors.append(f"missing requirement ids={missing_traceability_ids}")
    if matrix.get("status") not in {"passed", "blocked"}:
        errors.append(f"status={matrix.get('status')!r}")
    return errors


def audit_queue_coverage_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    input_files = report.get("input_files")
    if not isinstance(input_files, dict):
        return [*errors, "missing input_files"]

    expected_scalar_files = {
        "primary_json": root / "primary" / "multilingual_locomo_style_eval.json",
        "queue_jsonl": root / "human_audit_queue.jsonl",
    }
    for key, path in expected_scalar_files.items():
        row = input_files.get(key)
        if not isinstance(row, dict):
            errors.append(f"input_files.{key} missing")
            continue
        if row.get("path") != str(path):
            errors.append(f"input_files.{key}.path={row.get('path')!r} expected={str(path)!r}")
        if not path.is_file():
            errors.append(f"{key} file missing: {path}")
        elif row.get("sha256") != sha256_file(path):
            errors.append(f"input_files.{key}.sha256 mismatch")

    expected_qa_audit_paths = [
        root / "sidecars" / artifact / f"{artifact}_qa_audit.jsonl"
        for artifact in SOURCE_ARTIFACTS
    ]
    qa_audit_rows = input_files.get("qa_audit_files")
    if not isinstance(qa_audit_rows, list):
        errors.append("input_files.qa_audit_files missing")
    else:
        seen = {str(row.get("path")): row for row in qa_audit_rows if isinstance(row, dict)}
        expected = {str(path): path for path in expected_qa_audit_paths}
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        if missing:
            errors.append(f"input_files.qa_audit_files missing paths {missing[:10]}")
        if extra:
            errors.append(f"input_files.qa_audit_files unexpected paths {extra[:10]}")
        for raw_path, path in expected.items():
            row = seen.get(raw_path)
            if row is None:
                continue
            if not path.is_file():
                errors.append(f"qa_audit file missing: {path}")
            elif row.get("sha256") != sha256_file(path):
                errors.append(f"input_files.qa_audit_files[{raw_path}].sha256 mismatch")
    return errors


def no_model_guard_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("missing"):
        errors.append(f"missing={report.get('missing')!r}")
    if report.get("findings"):
        errors.append(f"findings={report.get('findings')!r}")
    scanned = report.get("scanned_scripts")
    script_hashes = report.get("script_hashes")
    if not isinstance(scanned, list) or not scanned:
        errors.append("scanned_scripts missing")
        return errors
    if not isinstance(script_hashes, dict):
        errors.append("script_hashes missing")
        return errors
    scanned_set = {str(item) for item in scanned}
    missing_required = sorted(REQUIRED_NO_MODEL_GUARD_SCRIPTS - scanned_set)
    if missing_required:
        errors.append(f"scanned_scripts missing required paths {missing_required[:10]}")
    for raw_path in scanned:
        path = Path(str(raw_path))
        if not path.is_file():
            errors.append(f"scanned script missing now: {path}")
            continue
        if script_hashes.get(str(path)) != sha256_file(path):
            errors.append(f"script_hashes[{path}] mismatch")
    return errors


def audit_csv_workflow_selftest_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    script_hashes = report.get("script_hashes", {})
    for script_name in (
        "export_locomo_human_audit_csv.py",
        "import_locomo_human_audit_csv.py",
        "finalize_locomo_human_audit_csv.py",
        "export_locomo_human_audit_batches.py",
        "merge_locomo_human_audit_batches.py",
        "summarize_locomo_human_audit_batches.py",
        "check_locomo_human_audit_batch_edits.py",
        "finalize_locomo_human_audit_batches.py",
    ):
        script_path = Path(__file__).with_name(script_name)
        if script_path.is_file() and script_hashes.get(script_name) != sha256_file(script_path):
            errors.append(f"{script_name} sha256 mismatch")
    cases = report.get("cases")
    if not isinstance(cases, list):
        errors.append("cases missing")
        return errors
    passed_cases = {
        str(case.get("name"))
        for case in cases
        if isinstance(case, dict) and case.get("status") == "passed"
    }
    missing_cases = sorted(REQUIRED_AUDIT_CSV_WORKFLOW_CASES - passed_cases)
    if missing_cases:
        errors.append(f"required cases missing or not passed: {missing_cases}")
    return errors


def file_list_report_errors(report: dict[str, Any], expected_files: dict[str, list[Path]]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    input_files = report.get("input_files")
    if not isinstance(input_files, dict):
        return [*errors, "missing input_files"]
    for key, paths in expected_files.items():
        rows = input_files.get(key)
        if not isinstance(rows, list):
            errors.append(f"input_files.{key} missing")
            continue
        seen = {str(row.get("path")): row for row in rows if isinstance(row, dict)}
        expected = {str(path): path for path in paths}
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        if missing:
            errors.append(f"input_files.{key} missing paths {missing[:10]}")
        if extra:
            errors.append(f"input_files.{key} unexpected paths {extra[:10]}")
        for raw_path, path in expected.items():
            row = seen.get(raw_path)
            if row is None:
                continue
            if not path.is_file():
                errors.append(f"{key} file missing: {path}")
            elif row.get("sha256") != sha256_file(path):
                errors.append(f"input_files.{key}[{raw_path}].sha256 mismatch")
    return errors


def assignment_risk_summary_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "completed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("purpose") != "reviewer_routing_aid_only_not_audit_decision":
        errors.append(f"purpose={report.get('purpose')!r}")
    if report.get("rows") != 373:
        errors.append(f"rows={report.get('rows')!r} expected=373")
    if report.get("assigned_rows") != report.get("rows"):
        errors.append(f"assigned_rows={report.get('assigned_rows')!r} rows={report.get('rows')!r}")
    if report.get("errors"):
        errors.append(f"report errors={report.get('errors')!r}")

    script_path = Path(__file__).with_name("summarize_locomo_human_audit_assignments.py")
    if script_path.is_file() and report.get("summary_script_sha256") != sha256_file(script_path):
        errors.append("summary_script_sha256 mismatch")

    input_files = report.get("input_files")
    expected_files = {
        "batch_progress": root / "human_audit_batches_progress.json",
        "flags": root / "human_audit_flags.json",
    }
    if not isinstance(input_files, dict):
        errors.append("missing input_files")
    else:
        for key, path in expected_files.items():
            row = input_files.get(key)
            if not isinstance(row, dict):
                errors.append(f"input_files.{key} missing")
                continue
            if row.get("path") != str(path):
                errors.append(f"input_files.{key}.path={row.get('path')!r} expected={str(path)!r}")
            if not path.is_file():
                errors.append(f"{key} file missing: {path}")
            elif row.get("sha256") != sha256_file(path):
                errors.append(f"input_files.{key}.sha256 mismatch")
        batch_csvs = input_files.get("batch_csvs")
        if not isinstance(batch_csvs, dict):
            errors.append("input_files.batch_csvs missing")
        else:
            progress_path = root / "human_audit_batches_progress.json"
            progress = load_json(progress_path) if progress_path.exists() else {}
            expected_batches = {
                Path(str(row.get("path", ""))).name: Path(str(row.get("path", "")))
                for row in progress.get("batches", [])
                if isinstance(row, dict)
            }
            missing_batches = sorted(set(expected_batches) - set(batch_csvs))
            extra_batches = sorted(set(batch_csvs) - set(expected_batches))
            if missing_batches:
                errors.append(f"input_files.batch_csvs missing batches={missing_batches[:10]}")
            if extra_batches:
                errors.append(f"input_files.batch_csvs extra batches={extra_batches[:10]}")
            for batch_name, path in expected_batches.items():
                row = batch_csvs.get(batch_name)
                if not isinstance(row, dict):
                    continue
                if row.get("path") != str(path):
                    errors.append(f"input_files.batch_csvs[{batch_name}].path={row.get('path')!r} expected={str(path)!r}")
                if not path.is_file():
                    errors.append(f"batch CSV missing: {path}")
                elif row.get("sha256") != sha256_file(path):
                    errors.append(f"input_files.batch_csvs[{batch_name}].sha256 mismatch")

    reviewers = report.get("reviewers")
    if not isinstance(reviewers, dict):
        errors.append("reviewers missing")
    else:
        missing_reviewers = sorted(set("ABCDE") - set(str(item) for item in reviewers))
        extra_reviewers = sorted(set(str(item) for item in reviewers) - set("ABCDE"))
        if missing_reviewers:
            errors.append(f"missing reviewers={missing_reviewers}")
        if extra_reviewers:
            errors.append(f"extra reviewers={extra_reviewers}")
        for reviewer, row in reviewers.items():
            if not isinstance(row, dict):
                errors.append(f"reviewer {reviewer} row is not object")
                continue
            remaining = int(row.get("remaining", 0) or 0)
            next_rows = row.get("next_todo_rows")
            if remaining > 0 and not next_rows:
                errors.append(f"reviewer {reviewer} has remaining={remaining} but no next_todo_rows")
            if next_rows and not isinstance(next_rows, list):
                errors.append(f"reviewer {reviewer} next_todo_rows is not list")
                continue
            for todo in (next_rows or [])[:5]:
                if not isinstance(todo, dict):
                    errors.append(f"reviewer {reviewer} next_todo row is not object")
                    continue
                for key in ("batch", "path", "line", "source_dataset", "sample_id", "qa_idx"):
                    if todo.get(key) in (None, ""):
                        errors.append(f"reviewer {reviewer} next_todo missing {key}")
    return errors


def reviewer_todos_check_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("purpose") != "validate_read_only_reviewer_todo_indexes_are_fresh":
        errors.append(f"purpose={report.get('purpose')!r}")
    if report.get("errors"):
        errors.append(f"report errors={report.get('errors')!r}")

    checker_path = Path(__file__).with_name("check_locomo_human_audit_reviewer_todos.py")
    if checker_path.is_file() and report.get("checker_sha256") != sha256_file(checker_path):
        errors.append("checker_sha256 mismatch")

    expected_paths = {
        "manifest": root / "human_audit_reviewer_todos" / "manifest.json",
        "assignment_summary": root / "human_audit_assignment_risk_summary.json",
        "audit_packet": root / "human_audit_packet.jsonl",
    }
    for key, path in expected_paths.items():
        row = report.get(key)
        if not isinstance(row, dict):
            errors.append(f"{key} missing")
            continue
        if row.get("path") != str(path):
            errors.append(f"{key}.path={row.get('path')!r} expected={str(path)!r}")
        if not path.is_file():
            errors.append(f"{key} file missing: {path}")
        elif row.get("sha256") != sha256_file(path):
            errors.append(f"{key}.sha256 mismatch")

    manifest_path = root / "human_audit_reviewer_todos" / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    manifest_reviewers = manifest.get("reviewers") if isinstance(manifest, dict) else {}
    report_reviewers = report.get("reviewers") if isinstance(report, dict) else {}
    if not isinstance(manifest_reviewers, dict):
        errors.append("manifest reviewers missing")
        manifest_reviewers = {}
    if not isinstance(report_reviewers, dict):
        errors.append("report reviewers missing")
        report_reviewers = {}
    expected_reviewers = set(str(item) for item in manifest_reviewers)
    observed_reviewers = set(str(item) for item in report_reviewers)
    missing_reviewers = sorted(expected_reviewers - observed_reviewers)
    extra_reviewers = sorted(observed_reviewers - expected_reviewers)
    if missing_reviewers:
        errors.append(f"report missing reviewers={missing_reviewers}")
    if extra_reviewers:
        errors.append(f"report extra reviewers={extra_reviewers}")
    total_from_report = 0
    for reviewer, manifest_row in manifest_reviewers.items():
        report_row = report_reviewers.get(str(reviewer), {})
        if not isinstance(report_row, dict):
            errors.append(f"reviewer {reviewer} report row is not object")
            continue
        expected_count = int((manifest_row or {}).get("todo_rows_exported", 0) or 0)
        actual_count = int(report_row.get("expected_todo_rows", -1) or -1)
        if actual_count != expected_count:
            errors.append(f"reviewer {reviewer} expected_todo_rows={actual_count} manifest={expected_count}")
        total_from_report += max(actual_count, 0)
    if int(report.get("total_todo_rows", -1) or -1) != total_from_report:
        errors.append(f"total_todo_rows={report.get('total_todo_rows')!r} reviewer_total={total_from_report}")
    return errors


def missing_required_report_items(report: dict[str, Any], key: str, expected: list[str]) -> list[str]:
    observed = report.get(key)
    if not isinstance(observed, list):
        return [f"{key} missing or not list"]
    observed_set = {str(item) for item in observed}
    return [f"{key} missing {item!r}" for item in expected if item not in observed_set]


def human_audit_handoff_docs_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    errors.extend(
        missing_required_report_items(
            report,
            "required_next_steps_references",
            REQUIRED_HANDOFF_NEXT_STEPS_REFERENCES,
        )
    )
    errors.extend(
        missing_required_report_items(
            report,
            "required_runbook_skipped_references",
            REQUIRED_HANDOFF_RUNBOOK_SKIPPED_REFERENCES,
        )
    )
    errors.extend(
        missing_required_report_items(
            report,
            "full_csv_finalizer_docs",
            sorted(HANDOFF_FULL_CSV_FINALIZER_DOCS),
        )
    )
    errors.extend(
        missing_required_report_items(
            report,
            "required_full_csv_finalizer_references",
            REQUIRED_HANDOFF_FULL_CSV_FINALIZER_REFERENCES,
        )
    )
    errors.extend(
        missing_required_report_items(
            report,
            "required_primary_readme_references",
            REQUIRED_HANDOFF_PRIMARY_README_REFERENCES,
        )
    )
    errors.extend(
        missing_required_report_items(
            report,
            "required_baseline_results_references",
            REQUIRED_HANDOFF_BASELINE_RESULTS_REFERENCES,
        )
    )
    handoff_doc_paths = [
        root / "NEXT_STEPS.md",
        root / "RUNBOOK.md",
        root / "HUMAN_AUDIT_HANDOFF.md",
        root / "HUMAN_AUDIT_QUICKSTART_ZH.md",
        root / "HUMAN_AUDIT_WORKPLAN.md",
        root / "HUMAN_AUDIT_ASSIGNMENTS.md",
        root / "human_audit_review_guide.md",
        root / "human_audit_batches" / "README.md",
    ]
    boundary_doc_paths = {
        "baseline_results_doc": root / "baseline_results" / "README.md",
        "primary_readme_doc": root / "primary" / "README.md",
        "recent_session_readme_doc": root / "recent_session_ablation" / "README.md",
    }
    observed_doc_hashes = {
        str(item.get("path")): item.get("sha256")
        for item in report.get("required_docs", [])
        if isinstance(item, dict)
    }
    for doc_path in handoff_doc_paths:
        if not doc_path.exists():
            errors.append(f"missing doc={doc_path}")
        elif observed_doc_hashes.get(str(doc_path)) != sha256_file(doc_path):
            errors.append(f"{doc_path} sha256 mismatch")
    for report_key, doc_path in boundary_doc_paths.items():
        record = report.get(report_key)
        if not isinstance(record, dict):
            errors.append(f"{report_key} missing")
        elif not doc_path.exists():
            errors.append(f"missing doc={doc_path}")
        elif record.get("path") != str(doc_path):
            errors.append(f"{report_key}.path={record.get('path')!r} expected={str(doc_path)!r}")
        elif record.get("sha256") != sha256_file(doc_path):
            errors.append(f"{doc_path} sha256 mismatch")
    if report.get("batch_csv_count") != 10:
        errors.append(f"batch_csv_count={report.get('batch_csv_count')!r}")
    if report.get("batch_review_md_count") != 10:
        errors.append(f"batch_review_md_count={report.get('batch_review_md_count')!r}")
    if report.get("batch_review_md_with_fact_ledger_support") != 10:
        errors.append(
            "batch_review_md_with_fact_ledger_support="
            f"{report.get('batch_review_md_with_fact_ledger_support')!r}"
        )
    observed_review_hashes = {
        str(item.get("path")): item
        for item in report.get("batch_review_docs", [])
        if isinstance(item, dict)
    }
    for review_path in sorted((root / "human_audit_batch_reviews").glob("batch_*.md")):
        record = observed_review_hashes.get(str(review_path))
        if not record:
            errors.append(f"{review_path} missing from batch_review_docs")
            continue
        if record.get("sha256") != sha256_file(review_path):
            errors.append(f"{review_path} sha256 mismatch")
        if record.get("has_fact_ledger_support") is not True:
            errors.append(f"{review_path} missing fact-ledger support in report")
    return errors


def gate(name: str, passed: bool, evidence: str, *, blocking: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "blocking": blocking,
        "evidence": evidence,
    }


def baseline_methods(report: dict[str, Any]) -> set[str]:
    methods: set[str] = set()
    raw_methods = report.get("methods")
    if isinstance(raw_methods, list):
        methods.update(str(item) for item in raw_methods)
    for key in ("results", "rows", "per_method"):
        value = report.get(key)
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    method = row.get("method") or row.get("model") or row.get("name")
                    if method:
                        methods.add(str(method))
        elif isinstance(value, dict):
            methods.update(str(item) for item in value)
    return methods


def baseline_settings_errors(report: dict[str, Any], summary_json: Path, expected_dataset: Path | None) -> list[str]:
    errors: list[str] = []
    settings_file = report.get("settings_file")
    if not settings_file:
        return ["missing settings_file"]
    settings_path = Path(str(settings_file))
    if not settings_path.is_file():
        return [f"settings_file missing: {settings_path}"]
    if report.get("settings_sha256") != sha256_file(settings_path):
        errors.append("settings_sha256 does not match settings_file")
    settings = load_json(settings_path)
    if settings.get("status") != "predeclared":
        errors.append(f"settings.status={settings.get('status')!r} expected='predeclared'")
    errors.extend(anti_tuning_contract_errors(settings))
    if expected_dataset is not None:
        errors.extend(fixed_baseline_execution_contract_errors(settings, str(expected_dataset)))
    dataset_settings = settings.get("dataset", {})
    if expected_dataset is not None and dataset_settings.get("audited_primary") != str(expected_dataset):
        errors.append(
            f"settings.dataset.audited_primary={dataset_settings.get('audited_primary')!r} "
            f"expected={str(expected_dataset)!r}"
        )
    if dataset_settings.get("input_policy") != "conversation_only":
        errors.append("settings.dataset.input_policy must be conversation_only")
    if dataset_settings.get("summary_visible") is not False:
        errors.append("settings.dataset.summary_visible must be false")
    fixed = settings.get("fixed_baselines", {})
    if fixed.get("summary_json") != str(summary_json):
        errors.append(f"settings.fixed_baselines.summary_json={fixed.get('summary_json')!r} expected={str(summary_json)!r}")
    if report.get("metric_metadata_file") != fixed.get("metric_metadata_jsonl"):
        errors.append(
            f"metric_metadata_file={report.get('metric_metadata_file')!r} "
            f"expected={fixed.get('metric_metadata_jsonl')!r}"
        )
    if report.get("settings_source") != fixed.get("settings_source"):
        errors.append(
            f"settings_source={report.get('settings_source')!r} expected={fixed.get('settings_source')!r}"
        )
    required_methods = {str(item) for item in fixed.get("required_methods", [])}
    if required_methods != REQUIRED_BASELINE_METHODS:
        errors.append(f"settings.required_methods={sorted(required_methods)} expected={sorted(REQUIRED_BASELINE_METHODS)}")
    common = fixed.get("common", {})
    if common.get("input_policy") != "conversation_only":
        errors.append("settings.fixed_baselines.common.input_policy must be conversation_only")
    if common.get("summary_visible") is not False:
        errors.append("settings.fixed_baselines.common.summary_visible must be false")
    cat5_metrics = {str(item) for item in common.get("cat5_metrics", [])}
    if cat5_metrics != {"refusal_accuracy", "unsupported_claim_rate"}:
        errors.append("settings.fixed_baselines.common.cat5_metrics must contain refusal and unsupported-claim metrics")
    report_group_by = {str(item) for item in settings.get("reporting", {}).get("group_by", [])}
    if report_group_by != REQUIRED_REPORT_GROUP_BY:
        errors.append(f"settings.reporting.group_by={sorted(report_group_by)} expected={sorted(REQUIRED_REPORT_GROUP_BY)}")
    return errors


def anti_tuning_contract_errors(settings: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = settings.get("anti_tuning_contract", {})
    if contract.get("benchmark_tuning_allowed") is not False:
        errors.append("anti_tuning_contract.benchmark_tuning_allowed must be false")
    if contract.get("frozen_before_final_runs") is not True:
        errors.append("anti_tuning_contract.frozen_before_final_runs must be true")
    frozen_items = {str(item) for item in contract.get("do_not_tune_on_this_benchmark", [])}
    missing = sorted(REQUIRED_ANTI_TUNING_ITEMS - frozen_items)
    if missing:
        errors.append(f"anti_tuning_contract.do_not_tune_on_this_benchmark missing {missing}")
    service_check = str(contract.get("allowed_service_check", "")).lower()
    if "tiny chat" not in service_check or "dataset" not in service_check:
        errors.append("anti_tuning_contract.allowed_service_check must limit service checks to tiny chat outside dataset/metrics")
    return errors


def fixed_baseline_execution_contract_errors(settings: dict[str, Any], expected_dataset: str) -> list[str]:
    errors: list[str] = []
    model = settings.get("model", {})
    if model.get("served_model") != REQUIRED_FIXED_MODEL:
        errors.append(f"model.served_model={model.get('served_model')!r} expected={REQUIRED_FIXED_MODEL!r}")
    if model.get("temperature") != 0:
        errors.append("model.temperature must be 0")

    fixed = settings.get("fixed_baselines", {})
    contract = fixed.get("execution_contract")
    if not isinstance(contract, dict):
        errors.append("fixed_baselines.execution_contract is required")
        contract = {}

    if contract.get("final_dataset") != expected_dataset:
        errors.append(
            f"fixed_baselines.execution_contract.final_dataset={contract.get('final_dataset')!r} "
            f"expected={expected_dataset!r}"
        )
    if contract.get("model") != REQUIRED_FIXED_MODEL:
        errors.append("fixed_baselines.execution_contract.model must match Qwen/Qwen3-8B")
    if contract.get("input_policy") != "conversation_only":
        errors.append("fixed_baselines.execution_contract.input_policy must be conversation_only")
    if contract.get("summary_visible") is not False:
        errors.append("fixed_baselines.execution_contract.summary_visible must be false")
    if contract.get("summary_builder") != "scripts/build_locomo_baseline_summary.py":
        errors.append("fixed_baselines.execution_contract.summary_builder must be build_locomo_baseline_summary.py")
    if contract.get("legacy_locomo10_defaults_forbidden") is not True:
        errors.append("fixed_baselines.execution_contract.legacy_locomo10_defaults_forbidden must be true")

    prediction_contract = contract.get("prediction_jsonl_contract", {})
    identity_fields = {str(item) for item in prediction_contract.get("identity_fields", [])}
    if not REQUIRED_PREDICTION_IDENTITY_FIELDS <= identity_fields:
        errors.append(
            "fixed_baselines.execution_contract.prediction_jsonl_contract.identity_fields "
            "must include sample_id and qa_idx"
        )
    row_fields = {str(item) for item in prediction_contract.get("required_row_fields", [])}
    if not REQUIRED_PREDICTION_ROW_FIELDS <= row_fields:
        errors.append(
            "fixed_baselines.execution_contract.prediction_jsonl_contract.required_row_fields "
            "must include sample_id, qa_idx, model, and dataset_sha256"
        )
    if prediction_contract.get("required_model") != REQUIRED_FIXED_MODEL:
        errors.append(
            "fixed_baselines.execution_contract.prediction_jsonl_contract.required_model "
            "must match Qwen/Qwen3-8B"
        )
    prediction_options = {str(item) for item in prediction_contract.get("prediction_field_options", [])}
    if not REQUIRED_PREDICTION_FIELD_OPTIONS <= prediction_options:
        errors.append(
            "fixed_baselines.execution_contract.prediction_jsonl_contract.prediction_field_options "
            "must include prediction, answer, and response"
        )

    methods = fixed.get("methods", {})
    if not isinstance(methods, dict):
        errors.append("fixed_baselines.methods must be an object")
        return errors
    for method in sorted(REQUIRED_BASELINE_METHODS):
        config = methods.get(method)
        if not isinstance(config, dict):
            errors.append(f"fixed_baselines.methods.{method} is required")
            continue
        if "runner" in config:
            errors.append(
                f"fixed_baselines.methods.{method}.runner must not point to a legacy script; "
                "final runs must provide per-method prediction JSONL files under the execution contract"
            )
        if "script" in config:
            errors.append(
                f"fixed_baselines.methods.{method}.script must not define a final runner; "
                "use the execution contract and summary builder"
            )

    optional_methods = fixed.get("optional_methods", {})
    memgas_policy = optional_methods.get("MemGAS") if isinstance(optional_methods, dict) else None
    if not isinstance(memgas_policy, dict):
        errors.append("fixed_baselines.optional_methods.MemGAS is required")
    else:
        for key, expected in OPTIONAL_MEMGAS_POLICY.items():
            if memgas_policy.get(key) != expected:
                errors.append(
                    f"fixed_baselines.optional_methods.MemGAS.{key}="
                    f"{memgas_policy.get(key)!r} expected={expected!r}"
                )
    return errors


def fixed_eval_settings_errors(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"settings file missing: {path}"]
    settings = load_json(path)
    if settings.get("status") != "predeclared":
        errors.append(f"status={settings.get('status')!r} expected='predeclared'")
    dataset = settings.get("dataset", {})
    expected_audited = str(root / "primary" / "multilingual_locomo_style_eval_audited.json")
    if dataset.get("audited_primary") != expected_audited:
        errors.append(f"dataset.audited_primary={dataset.get('audited_primary')!r} expected={expected_audited!r}")
    if dataset.get("input_policy") != "conversation_only":
        errors.append("dataset.input_policy must be conversation_only")
    if dataset.get("summary_visible") is not False:
        errors.append("dataset.summary_visible must be false")
    errors.extend(anti_tuning_contract_errors(settings))
    errors.extend(fixed_baseline_execution_contract_errors(settings, expected_audited))
    recent = settings.get("recent_session_diagnostic", {})
    if set(map(str, recent.get("contexts", []))) != {"full_conversation", "last_session_only", "last_3_sessions_only"}:
        errors.append("recent_session_diagnostic.contexts must contain full/last/last_3")
    if set(map(str, recent.get("categories", []))) != {"1", "2", "3", "4"}:
        errors.append("recent_session_diagnostic.categories must be 1,2,3,4")
    signal = recent.get("long_memory_signal", {})
    if signal.get("metric") != "mean_token_f1":
        errors.append("recent_session_diagnostic.long_memory_signal.metric must be mean_token_f1")
    for field in (
        "min_full_mean_token_f1_for_ratio_check",
        "max_last_session_ratio_of_full",
        "max_last_3_sessions_ratio_of_full",
    ):
        try:
            float(signal[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"recent_session_diagnostic.long_memory_signal.{field} must be numeric")
    fixed = settings.get("fixed_baselines", {})
    if fixed.get("summary_json") != str(root / "baseline_results" / "summary.json"):
        errors.append(f"fixed_baselines.summary_json={fixed.get('summary_json')!r}")
    if fixed.get("metric_metadata_jsonl") != str(root / "baseline_results" / "metric_metadata.jsonl"):
        errors.append(f"fixed_baselines.metric_metadata_jsonl={fixed.get('metric_metadata_jsonl')!r}")
    if {str(item) for item in fixed.get("required_methods", [])} != REQUIRED_BASELINE_METHODS:
        errors.append("fixed_baselines.required_methods does not match required baseline methods")
    common = fixed.get("common", {})
    if common.get("input_policy") != "conversation_only":
        errors.append("fixed_baselines.common.input_policy must be conversation_only")
    if common.get("summary_visible") is not False:
        errors.append("fixed_baselines.common.summary_visible must be false")
    cat5_metrics = {str(item) for item in common.get("cat5_metrics", [])}
    if cat5_metrics != {"refusal_accuracy", "unsupported_claim_rate"}:
        errors.append("fixed_baselines.common.cat5_metrics must contain refusal and unsupported-claim metrics")
    report_group_by = {str(item) for item in settings.get("reporting", {}).get("group_by", [])}
    if report_group_by != REQUIRED_REPORT_GROUP_BY:
        errors.append("reporting.group_by does not match required report dimensions")
    return errors


def construction_report_errors(path: Path, artifact: str) -> list[str]:
    if not path.is_file():
        return [f"construction report missing: {path}"]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    required_fragments = [
        f"# Construction Report: {artifact}",
        "Model calls: 0",
        "## Summary",
        "## Notes",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            errors.append(f"missing fragment {fragment!r}")
    for fragment in REQUIRED_CONSTRUCTION_REPORT_FRAGMENTS.get(artifact, []):
        if fragment not in text:
            errors.append(f"missing source-specific fragment {fragment!r}")
    if len(text.strip()) < 200:
        errors.append("construction report is unexpectedly short")
    return errors


def primary_output_uniqueness_errors(root: Path) -> list[str]:
    primary_root = root / "primary"
    if not primary_root.is_dir():
        return [f"primary directory missing: {primary_root}"]
    expected_direct = {
        *(f"{artifact}.json" for artifact in SOURCE_ARTIFACTS),
        "multilingual_locomo_style_eval.json",
        "multilingual_locomo_style_eval_audited.json",
    }
    direct_json = {path.name for path in primary_root.glob("*.json")}
    unexpected_direct = sorted(direct_json - expected_direct)
    errors: list[str] = []
    if unexpected_direct:
        errors.append(f"primary directory has unexpected direct JSON files {unexpected_direct}")

    audited_source_root = primary_root / "audited_sources"
    if audited_source_root.exists():
        expected_audited_sources = {f"{artifact}.json" for artifact in SOURCE_ARTIFACTS}
        audited_json = {path.name for path in audited_source_root.glob("*.json")}
        unexpected_audited = sorted(audited_json - expected_audited_sources)
        if unexpected_audited:
            errors.append(f"audited_sources has unexpected JSON files {unexpected_audited}")
    return errors


def no_stale_final_outputs_before_audit_errors(root: Path) -> list[str]:
    """Reject final-output leftovers while the audit apply report is not applied."""

    audit_apply_report_path = root / "human_audit_apply_report.json"
    audit_apply_report = load_json(audit_apply_report_path) if audit_apply_report_path.exists() else {}
    if audit_apply_report.get("status") == "applied":
        return []

    final_paths = [
        root / "primary" / "multilingual_locomo_style_eval_audited.json",
        root / "primary" / "audited_sources",
        root / "baseline_results" / "metric_metadata.jsonl",
        root / "baseline_results" / "summary.json",
        root / "baseline_results" / "predictions",
        root / "baseline_results" / "normalized",
        root / "baseline_results" / "normalization_summaries",
        root / "recent_session_ablation" / "model_results_summary.json",
        root / "recent_session_ablation" / "model_prediction_records.jsonl",
    ]
    stale = [str(path) for path in final_paths if path.exists()]
    if not stale:
        return []
    return [
        "stale final outputs exist before human audit is applied: "
        f"{stale}; human_audit_apply_report.status={audit_apply_report.get('status')!r}"
    ]


def skipped_audit_stop_point_report_errors(report: dict[str, Any], root: Path) -> list[str]:
    """Validate the skipped-audit stop-point report while audit is not applied.

    The report records release_gate_report.json as one of its inputs, but this
    release gate rewrites that report. To avoid a self-referential hash cycle,
    this gate verifies only the non-circular input hashes here; the current
    release-gate blocker set is validated by this script's own checks.
    """

    audit_apply_report_path = root / "human_audit_apply_report.json"
    audit_apply_report = load_json(audit_apply_report_path) if audit_apply_report_path.exists() else {}
    if audit_apply_report.get("status") == "applied":
        return []

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("purpose") != "safe_stop_point_when_manual_human_audit_is_skipped":
        errors.append(f"purpose={report.get('purpose')!r}")

    checker_path = Path(__file__).with_name("check_locomo_skipped_audit_stop_point.py")
    if not checker_path.is_file():
        errors.append(f"checker missing: {checker_path}")
    elif report.get("checker_sha256") != sha256_file(checker_path):
        errors.append("checker_sha256 mismatch")

    expected_blockers = {str(item) for item in report.get("expected_blockers", [])}
    if expected_blockers != REQUIRED_SKIPPED_AUDIT_BLOCKERS:
        errors.append(
            f"expected_blockers={sorted(expected_blockers)} expected={sorted(REQUIRED_SKIPPED_AUDIT_BLOCKERS)}"
        )

    input_files = report.get("input_files")
    if not isinstance(input_files, dict):
        errors.append("input_files missing")
    else:
        observed_inputs = {str(item) for item in input_files}
        missing_inputs = sorted(SKIPPED_AUDIT_INPUT_FILES - observed_inputs)
        extra_inputs = sorted(observed_inputs - SKIPPED_AUDIT_INPUT_FILES)
        if missing_inputs:
            errors.append(f"input_files missing {missing_inputs}")
        if extra_inputs:
            errors.append(f"input_files unexpected {extra_inputs}")
        for rel in sorted(SKIPPED_AUDIT_INPUT_FILES & observed_inputs):
            state = input_files.get(rel)
            path = root / rel
            if not isinstance(state, dict):
                errors.append(f"input_files[{rel}] is not object")
                continue
            if state.get("path") != str(path):
                errors.append(f"input_files[{rel}].path={state.get('path')!r} expected={str(path)!r}")
            if state.get("exists") is not True:
                errors.append(f"input_files[{rel}].exists={state.get('exists')!r} expected=True")
            if not path.is_file():
                errors.append(f"input file missing now: {path}")
                continue
            if rel in SKIPPED_AUDIT_NON_CIRCULAR_INPUT_FILES and state.get("sha256") != sha256_file(path):
                errors.append(f"input_files[{rel}].sha256 mismatch")

    final_outputs_state = report.get("final_outputs_state")
    if not isinstance(final_outputs_state, dict):
        errors.append("final_outputs_state missing")
    else:
        observed_outputs = {str(item) for item in final_outputs_state}
        missing_outputs = sorted(SKIPPED_AUDIT_FINAL_OUTPUTS - observed_outputs)
        extra_outputs = sorted(observed_outputs - SKIPPED_AUDIT_FINAL_OUTPUTS)
        if missing_outputs:
            errors.append(f"final_outputs_state missing {missing_outputs}")
        if extra_outputs:
            errors.append(f"final_outputs_state unexpected {extra_outputs}")
        for rel in sorted(SKIPPED_AUDIT_FINAL_OUTPUTS & observed_outputs):
            state = final_outputs_state.get(rel)
            path = root / rel
            if not isinstance(state, dict):
                errors.append(f"final_outputs_state[{rel}] is not object")
                continue
            if state.get("path") != str(path):
                errors.append(f"final_outputs_state[{rel}].path={state.get('path')!r} expected={str(path)!r}")
            if state.get("exists") is not False:
                errors.append(f"final_outputs_state[{rel}].exists={state.get('exists')!r} expected=False")
            if path.exists():
                errors.append(f"final output exists now: {path}")
    return errors


def index_samples(rows: Any, label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not isinstance(rows, list):
        return {}, [f"{label} must be a JSON list"]
    indexed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{label}[{row_idx}] must be object")
            continue
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            errors.append(f"{label}[{row_idx}] missing sample_id")
            continue
        if sample_id in indexed:
            errors.append(f"{label}: duplicate sample_id={sample_id!r}")
        indexed[sample_id] = row
    return indexed, errors


def combined_primary_partition_errors(root: Path) -> list[str]:
    primary_root = root / "primary"
    combined_path = primary_root / "multilingual_locomo_style_eval.json"
    if not combined_path.is_file():
        return [f"combined primary missing: {combined_path}"]
    errors: list[str] = []
    combined_index, index_errors = index_samples(load_json(combined_path), "combined primary")
    errors.extend(index_errors)
    source_index: dict[str, dict[str, Any]] = {}
    for artifact in SOURCE_ARTIFACTS:
        source_path = primary_root / f"{artifact}.json"
        if not source_path.is_file():
            errors.append(f"source primary missing: {source_path}")
            continue
        artifact_index, index_errors = index_samples(load_json(source_path), str(source_path))
        errors.extend(index_errors)
        overlap = sorted(set(source_index) & set(artifact_index))
        if overlap:
            errors.append(f"duplicate sample_ids across source primaries: {overlap[:10]}")
        source_index.update(artifact_index)

    missing = sorted(set(source_index) - set(combined_index))
    extra = sorted(set(combined_index) - set(source_index))
    if missing:
        errors.append(f"combined primary missing source sample_ids: {missing[:20]}")
    if extra:
        errors.append(f"combined primary has extra sample_ids: {extra[:20]}")
    mismatched = [
        sample_id
        for sample_id in sorted(set(source_index) & set(combined_index))
        if source_index[sample_id] != combined_index[sample_id]
    ]
    if mismatched:
        errors.append(f"combined primary sample content mismatches: {mismatched[:20]}")
    return errors


def single_qa_set_eval_split_errors(root: Path) -> tuple[list[str], dict[str, Any]]:
    primary_path = root / "primary" / "multilingual_locomo_style_eval.json"
    errors: list[str] = []
    summary: dict[str, Any] = {
        "primary": str(primary_path),
        "samples": 0,
        "primary_qa": 0,
        "split_counts": {},
        "qa_audit_rows": 0,
        "qa_set_counts": {},
    }
    if not primary_path.is_file():
        return [f"combined primary missing: {primary_path}"], summary

    rows = load_json(primary_path)
    if not isinstance(rows, list):
        return [f"{primary_path} must be a JSON list"], summary

    split_counts: Counter[str] = Counter()
    primary_qa = 0
    for row_idx, sample in enumerate(rows):
        if not isinstance(sample, dict):
            errors.append(f"{primary_path}[{row_idx}] must be object")
            continue
        split = str(sample.get("split"))
        split_counts[split] += 1
        if split != REQUIRED_SPLIT:
            errors.append(f"{primary_path}[{row_idx}] split={split!r} expected {REQUIRED_SPLIT!r}")
        qa_rows = sample.get("qa", [])
        if not isinstance(qa_rows, list):
            errors.append(f"{primary_path}[{row_idx}].qa must be list")
            continue
        primary_qa += len(qa_rows)

    qa_set_counts: Counter[str] = Counter()
    qa_audit_rows = 0
    for artifact in SOURCE_ARTIFACTS:
        audit_path = root / "sidecars" / artifact / f"{artifact}_qa_audit.jsonl"
        if not audit_path.is_file():
            errors.append(f"qa audit sidecar missing: {audit_path}")
            continue
        for row_idx, row in enumerate(iter_jsonl(audit_path), start=1):
            qa_audit_rows += 1
            qa_set = str(row.get("qa_set"))
            qa_set_counts[qa_set] += 1
            if qa_set != REQUIRED_QA_SET:
                errors.append(
                    f"{audit_path}:{row_idx}: qa_set={qa_set!r} expected {REQUIRED_QA_SET!r}"
                )

    if qa_audit_rows != primary_qa:
        errors.append(f"qa_audit_rows={qa_audit_rows} does not match primary_qa={primary_qa}")

    summary.update(
        {
            "samples": len(rows),
            "primary_qa": primary_qa,
            "split_counts": dict(sorted(split_counts.items())),
            "qa_audit_rows": qa_audit_rows,
            "qa_set_counts": dict(sorted(qa_set_counts.items())),
        }
    )
    return errors, summary


def perltqa_specific_ratio_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("source_dataset") != "PerLTQA":
        errors.append(f"source_dataset={report.get('source_dataset')!r} expected='PerLTQA'")
    if report.get("artifact") != "PerLTQA-LoCoMo-style-eval":
        errors.append(f"artifact={report.get('artifact')!r} expected='PerLTQA-LoCoMo-style-eval'")
    expected_files = {
        "primary_json": root / "primary" / "multilingual_locomo_style_eval.json",
        "provenance": root
        / "sidecars"
        / "PerLTQA-LoCoMo-style-eval"
        / "PerLTQA-LoCoMo-style-eval_provenance.jsonl",
        "fact_ledger": root
        / "sidecars"
        / "PerLTQA-LoCoMo-style-eval"
        / "PerLTQA-LoCoMo-style-eval_fact_ledger.jsonl",
        "qa_audit": root
        / "sidecars"
        / "PerLTQA-LoCoMo-style-eval"
        / "PerLTQA-LoCoMo-style-eval_qa_audit.jsonl",
    }
    errors.extend(validation_report_errors(report, expected_files))
    ratios = report.get("ratios")
    if not isinstance(ratios, dict):
        errors.append("ratios missing")
    else:
        required_ratios = {
            "original_turn_evidence_ratio",
            "memory_anchor_evidence_ratio",
            "synthetic_bridge_turn_ratio",
            "answer_fact_original_backed_ratio",
        }
        missing = sorted(required_ratios - set(ratios))
        if missing:
            errors.append(f"ratios missing {missing}")
        for field in required_ratios & set(ratios):
            try:
                value = float(ratios[field])
            except (TypeError, ValueError):
                errors.append(f"ratios.{field} must be numeric")
                continue
            if value < 0.0 or value > 1.0:
                errors.append(f"ratios.{field}={value} outside [0,1]")
        if float(ratios.get("answer_fact_original_backed_ratio", -1.0)) != 1.0:
            errors.append(
                "ratios.answer_fact_original_backed_ratio must be 1.0 for PerLTQA PlanMode D"
            )
    counts = report.get("counts")
    if not isinstance(counts, dict):
        errors.append("counts missing")
    else:
        for field in ("samples", "qa_total", "answerable_qa", "turns_total"):
            if not isinstance(counts.get(field), int) or int(counts.get(field, 0)) <= 0:
                errors.append(f"counts.{field} must be a positive integer")
    if report.get("errors"):
        errors.append(f"report_errors={report.get('errors')!r}")
    return errors


def baseline_validation_errors(report: dict[str, Any], expected_dataset: Path | None, summary_json: Path) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "completed":
        errors.append(f"status={report.get('status')!r}")
    if not report.get("settings_source"):
        errors.append("missing settings_source")
    if report.get("input_policy") != "conversation_only":
        errors.append(f"input_policy={report.get('input_policy')!r} expected='conversation_only'")
    if report.get("summary_visible") is not False:
        errors.append(f"summary_visible={report.get('summary_visible')!r} expected=False")
    if report.get("model") != REQUIRED_FIXED_MODEL:
        errors.append(f"model={report.get('model')!r} expected={REQUIRED_FIXED_MODEL!r}")
    if expected_dataset is not None and report.get("dataset") != str(expected_dataset):
        errors.append(f"dataset={report.get('dataset')!r} expected={str(expected_dataset)!r}")
    errors.extend(baseline_settings_errors(report, summary_json, expected_dataset))
    metadata_file = report.get("metric_metadata_file")
    if not metadata_file:
        errors.append("missing metric_metadata_file")
    else:
        metadata_path = Path(str(metadata_file))
        if not metadata_path.is_file():
            errors.append(f"metric_metadata_file missing: {metadata_path}")
        else:
            if report.get("metric_metadata_sha256") != sha256_file(metadata_path):
                errors.append("metric_metadata_sha256 does not match metric_metadata_file")
    dataset_value = report.get("dataset")
    expected_counts: dict[str, int] | None = None
    if not dataset_value:
        errors.append("missing dataset")
    dataset_path = Path(str(dataset_value or "__missing_dataset__"))
    if dataset_path.is_file():
        expected_counts = dataset_counts(dataset_path)
        if metadata_file:
            metadata_path = Path(str(metadata_file))
            if metadata_path.is_file() and count_jsonl_rows(metadata_path) != expected_counts["qa_count"]:
                errors.append(
                    f"metric metadata rows={count_jsonl_rows(metadata_path)} expected={expected_counts['qa_count']}"
                )
            if metadata_path.is_file():
                errors.extend(metadata_dataset_alignment_errors(metadata_path, dataset_path))
        errors.extend(prediction_files_errors(report, dataset_path))
    else:
        errors.append(f"dataset missing: {dataset_path}")
    expected_groups: dict[str, set[str]] = {
        "by_source_dataset": set(),
        "by_language": set(),
        "by_category": set(),
        "by_cross_session": set(),
        "by_evidence_provenance": set(),
    }
    has_cat5 = False
    metadata_path_for_groups = Path(str(metadata_file or ""))
    if metadata_path_for_groups.is_file():
        expected_groups, has_cat5 = metadata_group_keys(metadata_path_for_groups)

    results = report.get("results")
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty list")
        return errors

    methods_with_rows: set[str] = set()
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"results[{index}] is not an object")
            continue
        missing_fields = sorted(REQUIRED_BASELINE_RESULT_FIELDS - set(row))
        if missing_fields:
            errors.append(f"results[{index}] missing {missing_fields}")
            continue
        method = str(row["method"])
        methods_with_rows.add(method)
        if expected_counts is not None:
            if row["qa_count"] != expected_counts["qa_count"]:
                errors.append(f"{method}: qa_count={row['qa_count']} expected={expected_counts['qa_count']}")
            if row["answerable_qa_count"] != expected_counts["answerable_qa_count"]:
                errors.append(
                    f"{method}: answerable_qa_count={row['answerable_qa_count']} "
                    f"expected={expected_counts['answerable_qa_count']}"
                )
            if row["cat5_qa_count"] != expected_counts["cat5_qa_count"]:
                errors.append(
                    f"{method}: cat5_qa_count={row['cat5_qa_count']} expected={expected_counts['cat5_qa_count']}"
                )
        errors.extend(metric_object_errors(row["overall_answerable"], f"{method}: overall_answerable"))
        for field in (
            "by_source_dataset",
            "by_language",
            "by_category",
            "by_cross_session",
            "by_evidence_provenance",
        ):
            errors.extend(group_result_errors(row[field], f"{method}: {field}", expected_groups[field]))
        if has_cat5:
            for field in ("cat5_refusal", "cat5_unsupported_claim"):
                errors.extend(metric_object_errors(row[field], f"{method}: {field}"))

    missing_rows = sorted(REQUIRED_BASELINE_METHODS - methods_with_rows)
    if missing_rows:
        errors.append(f"missing result rows for {missing_rows}")
    return errors


def recent_session_settings_errors(report: dict[str, Any], expected_settings_path: Path) -> list[str]:
    errors: list[str] = []
    if report.get("settings_file") != str(expected_settings_path):
        errors.append(f"settings_file={report.get('settings_file')!r} expected={str(expected_settings_path)!r}")
        return errors
    if not expected_settings_path.is_file():
        errors.append(f"settings file missing: {expected_settings_path}")
        return errors
    if report.get("settings_sha256") != sha256_file(expected_settings_path):
        errors.append("settings_sha256 does not match fixed_eval_settings.json")
    settings = load_json(expected_settings_path)
    if report.get("settings_source") != settings.get("fixed_baselines", {}).get("settings_source"):
        errors.append(f"settings_source={report.get('settings_source')!r}")
    recent = settings.get("recent_session_diagnostic", {})
    if report.get("model") != settings.get("model", {}).get("served_model"):
        errors.append(f"model={report.get('model')!r} expected={settings.get('model', {}).get('served_model')!r}")
    if {str(item) for item in report.get("categories", [])} != {str(item) for item in recent.get("categories", [])}:
        errors.append(f"categories={report.get('categories')!r} expected={recent.get('categories')!r}")
    if report.get("max_context_chars") != recent.get("max_context_chars"):
        errors.append(f"max_context_chars={report.get('max_context_chars')!r} expected={recent.get('max_context_chars')!r}")
    if report.get("max_answer_tokens") != recent.get("max_answer_tokens"):
        errors.append(f"max_answer_tokens={report.get('max_answer_tokens')!r} expected={recent.get('max_answer_tokens')!r}")
    if float(report.get("request_timeout", -1)) != float(recent.get("request_timeout_seconds", -2)):
        errors.append(f"request_timeout={report.get('request_timeout')!r} expected={recent.get('request_timeout_seconds')!r}")
    if report.get("workers") != recent.get("workers"):
        errors.append(f"workers={report.get('workers')!r} expected={recent.get('workers')!r}")
    if report.get("settings_errors"):
        errors.append(f"settings_errors={report.get('settings_errors')!r}")
    return errors


def recent_session_result_errors(report: dict[str, Any], expected_dataset: Path | None, expected_settings_path: Path) -> list[str]:
    errors: list[str] = []
    runner_script = Path(__file__).with_name("run_locomo_recent_session_model_diagnostic.py")
    if report.get("runner_script_sha256") != sha256_file(runner_script):
        errors.append("runner_script_sha256 does not match run_locomo_recent_session_model_diagnostic.py")
    if report.get("input_policy") != "conversation_only":
        errors.append(f"input_policy={report.get('input_policy')!r} expected='conversation_only'")
    if report.get("summary_visible") is not False:
        errors.append(f"summary_visible={report.get('summary_visible')!r} expected=False")
    if report.get("input_fields_rendered") != ["conversation"]:
        errors.append(f"input_fields_rendered={report.get('input_fields_rendered')!r} expected=['conversation']")
    excluded_fields = report.get("input_fields_excluded")
    required_excluded_fields = {"observation", "session_summary", "event_summary", "sidecars"}
    if not isinstance(excluded_fields, list):
        errors.append("input_fields_excluded missing")
    else:
        missing_excluded = sorted(required_excluded_fields - {str(item) for item in excluded_fields})
        if missing_excluded:
            errors.append(f"input_fields_excluded missing {missing_excluded}")
    if report.get("prompt_policy") != "conversation_history_only_direct_answer":
        errors.append(f"prompt_policy={report.get('prompt_policy')!r} expected='conversation_history_only_direct_answer'")
    if report.get("context_renderer") != "conversation.session_i_date_time_and_turn_dia_id_speaker_text_only":
        errors.append(
            "context_renderer="
            f"{report.get('context_renderer')!r} expected='conversation.session_i_date_time_and_turn_dia_id_speaker_text_only'"
        )
    if report.get("status") != "completed":
        errors.append(f"status={report.get('status')!r}")
    if report.get("limit_samples") not in (0, None):
        errors.append(f"limit_samples={report.get('limit_samples')!r}")
    if report.get("limit_qa_per_sample") not in (0, None):
        errors.append(f"limit_qa_per_sample={report.get('limit_qa_per_sample')!r}")
    if report.get("summary", {}).get("errors") not in (0, None):
        errors.append(f"errors={report.get('summary', {}).get('errors')!r}")
    errors.extend(recent_session_settings_errors(report, expected_settings_path))
    settings = load_json(expected_settings_path) if expected_settings_path.is_file() else {}
    signal = settings.get("recent_session_diagnostic", {}).get("long_memory_signal", {})
    contexts = report.get("summary", {}).get("contexts", {})
    full_mean = contexts.get("full_conversation", {}).get("mean_token_f1")
    if full_mean is None:
        errors.append("summary.contexts.full_conversation.mean_token_f1 missing")
    else:
        full_value = float(full_mean)
        min_full = float(signal.get("min_full_mean_token_f1_for_ratio_check", 0.05))
        if full_value < min_full:
            errors.append(f"full_conversation mean_token_f1={full_value:.6f} below interpretable minimum={min_full}")
        else:
            for context_name, threshold_field in (
                ("last_session_only", "max_last_session_ratio_of_full"),
                ("last_3_sessions_only", "max_last_3_sessions_ratio_of_full"),
            ):
                context_mean = contexts.get(context_name, {}).get("mean_token_f1")
                if context_mean is None:
                    errors.append(f"summary.contexts.{context_name}.mean_token_f1 missing")
                    continue
                ratio = float(context_mean) / full_value if full_value else 1.0
                threshold = float(signal.get(threshold_field, 1.0))
                if ratio > threshold:
                    errors.append(
                        f"{context_name} mean_token_f1/full={ratio:.6f} exceeds {threshold_field}={threshold}"
                    )

    if expected_dataset is not None:
        if report.get("ablation_input") != str(expected_dataset):
            errors.append(f"ablation_input={report.get('ablation_input')!r} expected={str(expected_dataset)!r}")
        if expected_dataset.is_file() and report.get("ablation_input_sha256") != sha256_file(expected_dataset):
            errors.append("ablation_input_sha256 does not match expected dataset")

        categories = {str(item) for item in report.get("categories", [])}
        if categories:
            expected_per_context = dataset_answerable_count_for_categories(expected_dataset, categories)
            expected_records = expected_per_context * 3
            if report.get("expected_records") != expected_records:
                errors.append(
                    f"expected_records={report.get('expected_records')!r} expected={expected_records}"
                )
            context_counts = report.get("context_counts", {})
            for context_name in ("full_conversation", "last_session_only", "last_3_sessions_only"):
                if context_counts.get(context_name) != expected_per_context:
                    errors.append(
                        f"{context_name} count={context_counts.get(context_name)!r} "
                        f"expected={expected_per_context}"
                    )
            errors.extend(recent_session_prediction_record_errors(report, expected_dataset, categories))
    return errors


def audited_primary_validation_errors(path: Path | None) -> list[str]:
    if path is None:
        return ["audited output path missing"]
    if not path.exists():
        return [f"audited output missing: {path}"]
    from validate_locomo_style_eval import validate_primary

    report = validate_primary(path)
    return list(report.get("errors", []))


def audited_source_validation_errors(audit_apply: dict[str, Any], root: Path) -> list[str]:
    if audit_apply.get("status") != "applied":
        return ["audit not applied"]
    combined_path = Path(str(audit_apply.get("output_json", ""))) if audit_apply.get("output_json") else None
    if combined_path is None or not combined_path.is_file():
        return [f"audited combined output missing: {audit_apply.get('output_json')!r}"]
    output_files = audit_apply.get("output_source_files")
    if not isinstance(output_files, dict):
        return ["audit apply report missing output_source_files"]

    errors: list[str] = []
    from validate_locomo_style_eval import validate_primary

    expected_by_source = {
        "PerLTQA": "PerLTQA-LoCoMo-style-eval",
        "OPELA": "OPELA-LoCoMo-style-eval",
        "JLongChat": "JLongChat-LoCoMo-style-eval",
        "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
    }
    combined_rows = load_json(combined_path)
    combined_ids = {str(sample.get("sample_id")) for sample in combined_rows}
    combined_qa = sum(len(sample.get("qa", [])) for sample in combined_rows)
    source_ids: set[str] = set()
    source_qa = 0
    for source, artifact in expected_by_source.items():
        raw_path = output_files.get(source)
        if not raw_path:
            errors.append(f"missing audited source output for {source}")
            continue
        path = Path(str(raw_path))
        expected_path = root / "primary" / "audited_sources" / f"{artifact}.json"
        if path != expected_path:
            errors.append(f"{source}: output path={path} expected={expected_path}")
        if not path.is_file():
            errors.append(f"{source}: audited source file missing: {path}")
            continue
        report = validate_primary(path)
        if report.get("errors"):
            errors.extend(f"{source}: {error}" for error in report["errors"][:20])
        try:
            rows = load_json(path)
        except Exception as exc:  # noqa: BLE001 - report parse failure in release gate.
            errors.append(f"{source}: failed to read audited source file: {type(exc).__name__}: {exc}")
            continue
        bad_sources = sorted({str(sample.get("source_dataset")) for sample in rows if sample.get("source_dataset") != source})
        if bad_sources:
            errors.append(f"{source}: audited file contains other source_dataset values: {bad_sources}")
        ids = [str(sample.get("sample_id")) for sample in rows]
        duplicate_ids = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicate_ids:
            errors.append(f"{source}: duplicate sample_id values: {duplicate_ids[:10]}")
        overlap = sorted(source_ids & set(ids))
        if overlap:
            errors.append(f"{source}: sample_id overlaps with another audited source file: {overlap[:10]}")
        source_ids.update(ids)
        source_qa += sum(len(sample.get("qa", [])) for sample in rows)
    missing_from_sources = sorted(combined_ids - source_ids)
    extra_in_sources = sorted(source_ids - combined_ids)
    if missing_from_sources:
        errors.append(f"audited source files missing combined sample_ids: {missing_from_sources[:20]}")
    if extra_in_sources:
        errors.append(f"audited source files contain sample_ids absent from combined output: {extra_in_sources[:20]}")
    if source_qa != combined_qa:
        errors.append(f"audited source QA total={source_qa} does not match combined QA total={combined_qa}")
    return errors


def audited_apply_integrity_errors(report_path: Path, audit_apply: dict[str, Any], root: Path) -> list[str]:
    if audit_apply.get("status") != "applied":
        return ["audit not applied"]
    if not report_path.is_file():
        return [f"audited apply integrity report missing: {report_path}"]

    report = load_json(report_path)
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")

    expected_original = str(root / "primary" / "multilingual_locomo_style_eval.json")
    expected_audit = str(root / "human_audit_packet.jsonl")
    expected_audited = str(audit_apply.get("output_json"))
    if report.get("original_primary") != expected_original:
        errors.append(f"original_primary={report.get('original_primary')!r} expected={expected_original!r}")
    if report.get("audit_jsonl") != expected_audit:
        errors.append(f"audit_jsonl={report.get('audit_jsonl')!r} expected={expected_audit!r}")
    if report.get("audited_primary") != expected_audited:
        errors.append(f"audited_primary={report.get('audited_primary')!r} expected={expected_audited!r}")
    expected_sidecar_root = str(root / "sidecars")
    if report.get("sidecar_root") != expected_sidecar_root:
        errors.append(f"sidecar_root={report.get('sidecar_root')!r} expected={expected_sidecar_root!r}")
    sidecar_root = root / "sidecars"
    if sidecar_root.is_dir() and report.get("sidecar_trace_files_sha256") != sidecar_audit_trace_files_sha256(sidecar_root):
        errors.append("sidecar_trace_files_sha256 mismatch")
    if not isinstance(report.get("trace_checked_qa"), int) or int(report.get("trace_checked_qa", 0)) <= 0:
        errors.append(f"trace_checked_qa={report.get('trace_checked_qa')!r} expected positive integer")
    if report.get("errors"):
        errors.append(f"report_errors={report.get('errors')!r}")
    return errors


def metric_metadata_errors(summary_path: Path, audit_apply: dict[str, Any]) -> list[str]:
    if audit_apply.get("status") != "applied":
        return ["audit not applied"]
    if not summary_path.is_file():
        return [f"metric metadata summary missing: {summary_path}"]

    report = load_json(summary_path)
    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"status={report.get('status')!r}")

    audited_path = Path(str(audit_apply.get("output_json", "")))
    if report.get("primary_json") != str(audited_path):
        errors.append(f"primary_json={report.get('primary_json')!r} expected={str(audited_path)!r}")
    if audited_path.is_file() and report.get("primary_sha256") != sha256_file(audited_path):
        errors.append("primary_sha256 does not match audited output")

    output_jsonl = Path(str(report.get("output_jsonl", "")))
    if not output_jsonl.is_file():
        errors.append(f"metric metadata jsonl missing: {output_jsonl}")
    elif report.get("output_jsonl_sha256") != sha256_file(output_jsonl):
        errors.append("output_jsonl_sha256 does not match metric metadata file")

    if audited_path.is_file():
        counts = dataset_counts(audited_path)
        if report.get("rows") != counts["qa_count"]:
            errors.append(f"rows={report.get('rows')!r} expected={counts['qa_count']}")
    for field in ("by_source_dataset", "by_language", "by_category", "by_cross_session", "by_evidence_provenance"):
        if not isinstance(report.get(field), dict) or not report.get(field):
            errors.append(f"{field} must be a non-empty object")
    if report.get("errors"):
        errors.append(f"report_errors={report.get('errors')!r}")
    return errors


def human_audit_results_errors(report: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    audit_packet_path = root / "human_audit_packet.jsonl"
    audit_validator_path = Path(__file__).with_name("validate_locomo_human_audit_results.py")
    sidecar_root = root / "sidecars"
    if report.get("status") != "completed":
        errors.append(f"status={report.get('status')!r} incomplete={report.get('incomplete_count')!r}")
    if report.get("input_jsonl") != str(audit_packet_path):
        errors.append(f"input_jsonl={report.get('input_jsonl')!r} expected={str(audit_packet_path)!r}")
    if audit_packet_path.is_file() and report.get("input_jsonl_sha256") != sha256_file(audit_packet_path):
        errors.append("input_jsonl_sha256 mismatch")
    if report.get("validator") != str(audit_validator_path):
        errors.append(f"validator={report.get('validator')!r} expected={str(audit_validator_path)!r}")
    if audit_validator_path.is_file() and report.get("validator_sha256") != sha256_file(audit_validator_path):
        errors.append("validator_sha256 mismatch")
    if report.get("sidecar_root") != str(sidecar_root):
        errors.append(f"sidecar_root={report.get('sidecar_root')!r} expected={str(sidecar_root)!r}")
    if sidecar_root.is_dir() and report.get("sidecar_trace_files_sha256") != sidecar_trace_files_sha256(sidecar_root):
        errors.append("sidecar_trace_files_sha256 mismatch")
    if report.get("errors"):
        errors.append(f"report_errors={report.get('errors')!r}")
    return errors


def audit_packet_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("source_dataset")), str(row.get("sample_id")), int(row.get("qa_idx", -1)))


def keyed_jsonl_errors(path: Path) -> tuple[list[tuple[str, str, int]], list[str]]:
    if not path.is_file():
        return [], [f"missing: {path}"]
    keys: list[tuple[str, str, int]] = []
    errors: list[str] = []
    for row_idx, row in enumerate(iter_jsonl(path), start=1):
        try:
            keys.append(audit_packet_key(row))
        except (TypeError, ValueError):
            errors.append(f"{path}: row {row_idx} has invalid audit key")
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        errors.append(f"{path}: duplicate audit keys={duplicate_count}")
    return keys, errors


def human_audit_packet_queue_errors(root: Path) -> list[str]:
    queue_path = root / "human_audit_queue.jsonl"
    packet_path = root / "human_audit_packet.jsonl"
    queue_keys, errors = keyed_jsonl_errors(queue_path)
    packet_keys, packet_errors = keyed_jsonl_errors(packet_path)
    errors.extend(packet_errors)
    queue_set = set(queue_keys)
    packet_set = set(packet_keys)
    missing = sorted(queue_set - packet_set)
    extra = sorted(packet_set - queue_set)
    if missing:
        errors.append(f"audit packet missing queue QA keys; first={missing[:10]}")
    if extra:
        errors.append(f"audit packet has QA keys not in queue; first={extra[:10]}")
    if len(packet_keys) != len(queue_keys):
        errors.append(f"audit packet rows={len(packet_keys)} expected queue rows={len(queue_keys)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/locomo_style_eval"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    checks: list[dict[str, Any]] = []

    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    checks.append(gate("manifest_exists", manifest_path.exists(), str(manifest_path)))
    checks.append(
        gate(
            "construction_has_zero_model_calls",
            manifest.get("model_calls") == 0 and manifest.get("local_model_required_for_construction") is False,
            f"model_calls={manifest.get('model_calls')} local_model_required={manifest.get('local_model_required_for_construction')}",
        )
    )
    benchmark_claim = str(manifest.get("benchmark_claim", ""))
    checks.append(
        gate(
            "manifest_has_non_equivalence_claim",
            "not claimed to be LoCoMo-equivalent" in benchmark_claim,
            f"benchmark_claim={manifest.get('benchmark_claim')!r}",
        )
    )
    no_model_guard_path = root / "no_model_construction_guard.json"
    no_model_guard = load_json(no_model_guard_path) if no_model_guard_path.exists() else {}
    no_model_errors = no_model_guard_errors(no_model_guard)
    checks.append(
        gate(
            "no_model_construction_guard_passed",
            not no_model_errors,
            f"{no_model_guard_path}: errors={no_model_errors[:10]}",
        )
    )
    no_model_guard_selftest_path = root / "release_gate_no_model_guard_selftest.json"
    no_model_guard_selftest = load_json(no_model_guard_selftest_path) if no_model_guard_selftest_path.exists() else {}
    no_model_guard_selftest_errors: list[str] = []
    if no_model_guard_selftest.get("status") != "passed":
        no_model_guard_selftest_errors.append(f"status={no_model_guard_selftest.get('status')!r}")
    if no_model_guard_selftest.get("gate_script_sha256") != sha256_file(Path(__file__)):
        no_model_guard_selftest_errors.append("gate_script_sha256 mismatch")
    checks.append(
        gate(
            "release_gate_no_model_guard_selftest_passed",
            not no_model_guard_selftest_errors,
            f"{no_model_guard_selftest_path}: errors={no_model_guard_selftest_errors}",
        )
    )
    style_validator_selftest_path = root / "locomo_style_validator_selftest.json"
    style_validator_selftest = (
        load_json(style_validator_selftest_path) if style_validator_selftest_path.exists() else {}
    )
    style_validator_selftest_errors: list[str] = []
    if style_validator_selftest.get("status") != "passed":
        style_validator_selftest_errors.append(f"status={style_validator_selftest.get('status')!r}")
    validator_path = Path(__file__).with_name("validate_locomo_style_eval.py")
    if (
        validator_path.is_file()
        and style_validator_selftest.get("validator_sha256") != sha256_file(validator_path)
    ):
        style_validator_selftest_errors.append("validator_sha256 mismatch")
    selftest_path = Path(__file__).with_name("selftest_locomo_style_validator.py")
    if (
        selftest_path.is_file()
        and style_validator_selftest.get("selftest_sha256") != sha256_file(selftest_path)
    ):
        style_validator_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "locomo_style_validator_selftest_passed",
            not style_validator_selftest_errors,
            f"{style_validator_selftest_path}: errors={style_validator_selftest_errors}",
        )
    )
    audited_source_gate_selftest_path = root / "audited_source_validation_gate_selftest.json"
    audited_source_gate_selftest = (
        load_json(audited_source_gate_selftest_path) if audited_source_gate_selftest_path.exists() else {}
    )
    audited_source_gate_selftest_errors: list[str] = []
    if audited_source_gate_selftest.get("status") != "passed":
        audited_source_gate_selftest_errors.append(f"status={audited_source_gate_selftest.get('status')!r}")
    if audited_source_gate_selftest.get("gate_script_sha256") != sha256_file(Path(__file__)):
        audited_source_gate_selftest_errors.append("gate_script_sha256 mismatch")
    checks.append(
        gate(
            "audited_source_validation_gate_selftest_passed",
            not audited_source_gate_selftest_errors,
            f"{audited_source_gate_selftest_path}: errors={audited_source_gate_selftest_errors}",
        )
    )
    checks.append(
        gate(
            "manifest_not_claiming_final_release",
            manifest.get("status") == "bootstrap_harness_artifact_not_final_audited_release",
            f"status={manifest.get('status')}",
            blocking=False,
        )
    )
    stale_final_errors = no_stale_final_outputs_before_audit_errors(root)
    checks.append(
        gate(
            "no_stale_final_outputs_before_audit_apply",
            not stale_final_errors,
            f"errors={stale_final_errors[:10]}",
        )
    )
    skipped_audit_stop_point_report_path = root / "skipped_audit_stop_point_report.json"
    skipped_audit_stop_point_report = (
        load_json(skipped_audit_stop_point_report_path)
        if skipped_audit_stop_point_report_path.exists()
        else {}
    )
    skipped_audit_stop_point_errors = skipped_audit_stop_point_report_errors(
        skipped_audit_stop_point_report,
        root,
    )
    checks.append(
        gate(
            "skipped_audit_stop_point_report_fresh",
            not skipped_audit_stop_point_errors,
            f"{skipped_audit_stop_point_report_path}: errors={skipped_audit_stop_point_errors[:10]}",
        )
    )
    skipped_audit_stop_point_selftest_path = root / "skipped_audit_stop_point_selftest.json"
    skipped_audit_stop_point_selftest = (
        load_json(skipped_audit_stop_point_selftest_path)
        if skipped_audit_stop_point_selftest_path.exists()
        else {}
    )
    skipped_audit_stop_point_selftest_errors: list[str] = []
    skipped_audit_checker_path = Path(__file__).with_name("check_locomo_skipped_audit_stop_point.py")
    skipped_audit_selftest_script_path = Path(__file__).with_name("selftest_locomo_skipped_audit_stop_point.py")
    if skipped_audit_stop_point_selftest.get("status") != "passed":
        skipped_audit_stop_point_selftest_errors.append(
            f"status={skipped_audit_stop_point_selftest.get('status')!r}"
        )
    if (
        skipped_audit_checker_path.is_file()
        and skipped_audit_stop_point_selftest.get("checker_sha256") != sha256_file(skipped_audit_checker_path)
    ):
        skipped_audit_stop_point_selftest_errors.append("checker_sha256 mismatch")
    if (
        skipped_audit_selftest_script_path.is_file()
        and skipped_audit_stop_point_selftest.get("selftest_sha256") != sha256_file(skipped_audit_selftest_script_path)
    ):
        skipped_audit_stop_point_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "skipped_audit_stop_point_selftest_passed",
            not skipped_audit_stop_point_selftest_errors,
            f"{skipped_audit_stop_point_selftest_path}: errors={skipped_audit_stop_point_selftest_errors}",
        )
    )
    fixed_settings_path = root / "fixed_eval_settings.json"
    fixed_settings_errors = fixed_eval_settings_errors(fixed_settings_path, root)
    checks.append(
        gate(
            "fixed_eval_settings_predeclared",
            not fixed_settings_errors,
            f"{fixed_settings_path}: errors={fixed_settings_errors[:10]}",
        )
    )
    primary_uniqueness_errors = primary_output_uniqueness_errors(root)
    checks.append(
        gate(
            "primary_output_uniqueness",
            not primary_uniqueness_errors,
            f"{root / 'primary'}: errors={primary_uniqueness_errors[:10]}",
        )
    )

    primary_root = root / "primary"
    for artifact in SOURCE_ARTIFACTS:
        primary_path = primary_root / f"{artifact}.json"
        checks.append(gate(f"primary_exists_{artifact}", primary_path.exists(), str(primary_path)))
        sidecar_root = root / "sidecars" / artifact
        for suffix in ("fact_ledger", "provenance", "qa_audit", "hash_check"):
            path = sidecar_root / f"{artifact}_{suffix}.jsonl"
            checks.append(gate(f"sidecar_exists_{artifact}_{suffix}", path.exists(), str(path)))
        construction_report_path = sidecar_root / f"{artifact}_construction_report.md"
        construction_report_check_errors = construction_report_errors(construction_report_path, artifact)
        checks.append(
            gate(
                f"sidecar_exists_{artifact}_construction_report",
                not construction_report_check_errors,
                f"{construction_report_path}: errors={construction_report_check_errors[:10]}",
            )
        )
        validation_path = sidecar_root / f"{artifact}_validation.json"
        passed = False
        evidence = str(validation_path)
        if validation_path.exists():
            report = load_json(validation_path)
            validation_errors = validation_report_errors(
                report,
                {
                    "primary": primary_path,
                    "provenance": sidecar_root / f"{artifact}_provenance.jsonl",
                    "fact_ledger": sidecar_root / f"{artifact}_fact_ledger.jsonl",
                    "qa_audit": sidecar_root / f"{artifact}_qa_audit.jsonl",
                },
            )
            passed = not validation_errors
            evidence = f"{validation_path}: errors={validation_errors[:10]}"
        checks.append(gate(f"strict_validation_passed_{artifact}", passed, evidence))

    combined_validation_path = root / "sidecars" / "multilingual_locomo_style_eval_validation.json"
    if not combined_validation_path.exists():
        combined_validation_path = root / "sidecars" / "multilingual_locomo_style_eval" / "multilingual_locomo_style_eval_validation.json"
    combined_report = load_json(combined_validation_path) if combined_validation_path.exists() else {}
    combined_validation_errors = validation_report_errors(
        combined_report,
        {"primary": primary_root / "multilingual_locomo_style_eval.json"},
    )
    checks.append(
        gate(
            "combined_primary_validation_passed",
            not combined_validation_errors,
            f"{combined_validation_path}: errors={combined_validation_errors[:10]}",
        )
    )
    combined_partition_errors = combined_primary_partition_errors(root)
    checks.append(
        gate(
            "combined_primary_partitions_source_files",
            not combined_partition_errors,
            f"{primary_root / 'multilingual_locomo_style_eval.json'}: errors={combined_partition_errors[:10]}",
        )
    )
    qa_set_split_errors, qa_set_split_summary = single_qa_set_eval_split_errors(root)
    checks.append(
        gate(
            "single_qa_set_eval_split",
            not qa_set_split_errors,
            (
                f"samples={qa_set_split_summary['samples']} "
                f"primary_qa={qa_set_split_summary['primary_qa']} "
                f"qa_audit_rows={qa_set_split_summary['qa_audit_rows']} "
                f"splits={qa_set_split_summary['split_counts']} "
                f"qa_sets={qa_set_split_summary['qa_set_counts']} "
                f"errors={qa_set_split_errors[:10]}"
            ),
        )
    )

    smoke_path = root / "no_model_loader_smoke.json"
    smoke = load_json(smoke_path) if smoke_path.exists() else {}
    smoke_input_errors = validation_report_errors(
        smoke,
        {"primary_json": root / "primary" / "multilingual_locomo_style_eval.json"},
    )
    if smoke.get("input_fields_rendered") != ["conversation"]:
        smoke_input_errors.append(
            f"input_fields_rendered={smoke.get('input_fields_rendered')!r} expected=['conversation']"
        )
    excluded_fields = smoke.get("input_fields_excluded")
    required_excluded_fields = {"observation", "session_summary", "event_summary", "sidecars"}
    if not isinstance(excluded_fields, list):
        smoke_input_errors.append("input_fields_excluded missing")
    else:
        missing_excluded = sorted(required_excluded_fields - set(str(item) for item in excluded_fields))
        if missing_excluded:
            smoke_input_errors.append(f"input_fields_excluded missing {missing_excluded}")
    checks.append(
        gate(
            "no_model_loader_smoke_passed",
            not smoke_input_errors and smoke.get("model_calls") == 0,
            (
                f"{smoke_path}: input_errors={smoke_input_errors[:5]} "
                f"status={smoke.get('status')} model_calls={smoke.get('model_calls')}"
            ),
        )
    )
    baseline_loader_smoke_path = root / "baseline_loader_compat_smoke.json"
    baseline_loader_smoke = load_json(baseline_loader_smoke_path) if baseline_loader_smoke_path.exists() else {}
    baseline_loader_errors = validation_report_errors(
        baseline_loader_smoke,
        {"primary_json": root / "primary" / "multilingual_locomo_style_eval.json"},
        allowed_extra_keys={"loader_files"},
    )
    baseline_loader_errors.extend(
        file_list_report_errors(
            baseline_loader_smoke,
            {
                "loader_files": [
                    Path("scripts/locomo_2026_sota.py"),
                    Path("baseline/A-MEM/load_dataset.py"),
                    Path("baseline/SimpleMem/OmniSimpleMem/benchmarks/locomo/run_locomo.py"),
                ]
            },
        )
    )
    checks.append(
        gate(
            "baseline_loader_compat_smoke_passed",
            not baseline_loader_errors and baseline_loader_smoke.get("model_calls") == 0,
            (
                f"{baseline_loader_smoke_path}: errors={baseline_loader_errors[:10]} "
                f"status={baseline_loader_smoke.get('status')} "
                f"model_calls={baseline_loader_smoke.get('model_calls')}"
            ),
        )
    )

    diagnostic_path = root / "long_memory_evidence_locality_diagnostic.json"
    diagnostic = load_json(diagnostic_path) if diagnostic_path.exists() else {}
    overall = diagnostic.get("overall", {})
    diagnostic_input_errors = validation_report_errors(
        diagnostic,
        {"primary_json": root / "primary" / "multilingual_locomo_style_eval.json"},
    )
    checks.append(
        gate(
            "construction_long_memory_diagnostic_passed",
            not diagnostic_input_errors
            and bool(overall)
            and overall.get("last_session_sufficient_ratio", 1.0) <= 0.2
            and overall.get("last_three_sessions_sufficient_ratio", 1.0) <= 0.5
            and not diagnostic.get("weak_samples_last_three_ratio_ge_0_8"),
            (
                f"{diagnostic_path}: input_errors={diagnostic_input_errors[:5]} "
                f"last_session={overall.get('last_session_sufficient_ratio')} "
                f"last_three={overall.get('last_three_sessions_sufficient_ratio')} "
                f"weak_samples={len(diagnostic.get('weak_samples_last_three_ratio_ge_0_8', []))}"
            ),
        )
    )
    expected_primary_files = [root / "primary" / f"{artifact}.json" for artifact in SOURCE_ARTIFACTS]
    expected_provenance_files = [
        root / "sidecars" / artifact / f"{artifact}_provenance.jsonl"
        for artifact in SOURCE_ARTIFACTS
    ]
    expected_qa_audit_files = [
        root / "sidecars" / artifact / f"{artifact}_qa_audit.jsonl"
        for artifact in SOURCE_ARTIFACTS
    ]
    planmode_path = root / "planmode_provenance_summary.json"
    planmode_report = load_json(planmode_path) if planmode_path.exists() else {}
    planmode_errors = file_list_report_errors(
        planmode_report,
        {
            "provenance_files": expected_provenance_files,
            "qa_audit_files": expected_qa_audit_files,
        },
    )
    checks.append(
        gate(
            "planmode_provenance_passed",
            not planmode_errors,
            f"{planmode_path}: errors={planmode_errors[:10]}",
        )
    )
    planmode_selftest_path = root / "planmode_provenance_selftest.json"
    planmode_selftest = load_json(planmode_selftest_path) if planmode_selftest_path.exists() else {}
    planmode_selftest_errors: list[str] = []
    planmode_checker_path = Path(__file__).with_name("check_locomo_planmode_provenance.py")
    if planmode_selftest.get("status") != "passed":
        planmode_selftest_errors.append(f"status={planmode_selftest.get('status')!r}")
    if (
        planmode_checker_path.is_file()
        and planmode_selftest.get("checker_sha256") != sha256_file(planmode_checker_path)
    ):
        planmode_selftest_errors.append("checker_sha256 mismatch")
    planmode_selftest_script_path = Path(__file__).with_name("selftest_locomo_planmode_provenance.py")
    if (
        planmode_selftest_script_path.is_file()
        and planmode_selftest.get("selftest_sha256") != sha256_file(planmode_selftest_script_path)
    ):
        planmode_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "planmode_provenance_selftest_passed",
            not planmode_selftest_errors,
            f"{planmode_selftest_path}: errors={planmode_selftest_errors}",
        )
    )
    perltqa_original_qa_exclusion_path = root / "perltqa_original_qa_exclusion_report.json"
    perltqa_original_qa_exclusion_report = (
        load_json(perltqa_original_qa_exclusion_path)
        if perltqa_original_qa_exclusion_path.exists()
        else {}
    )
    perltqa_original_qa_exclusion_errors = validation_report_errors(
        perltqa_original_qa_exclusion_report,
        {
            "primary_json": root / "primary" / "PerLTQA-LoCoMo-style-eval.json",
            "source_qa": Path("datasets/PerLTQA/Dataset/zh/perltqa.json"),
            "construction_report": root
            / "sidecars"
            / "PerLTQA-LoCoMo-style-eval"
            / "PerLTQA-LoCoMo-style-eval_construction_report.md",
        },
    )
    checks.append(
        gate(
            "perltqa_original_qa_exclusion_passed",
            not perltqa_original_qa_exclusion_errors,
            f"{perltqa_original_qa_exclusion_path}: errors={perltqa_original_qa_exclusion_errors[:10]}",
        )
    )
    perltqa_original_qa_exclusion_selftest_path = root / "perltqa_original_qa_exclusion_selftest.json"
    perltqa_original_qa_exclusion_selftest = (
        load_json(perltqa_original_qa_exclusion_selftest_path)
        if perltqa_original_qa_exclusion_selftest_path.exists()
        else {}
    )
    perltqa_original_qa_exclusion_selftest_errors: list[str] = []
    perltqa_original_qa_exclusion_checker_path = Path(__file__).with_name(
        "check_locomo_perltqa_original_qa_exclusion.py"
    )
    if perltqa_original_qa_exclusion_selftest.get("status") != "passed":
        perltqa_original_qa_exclusion_selftest_errors.append(
            f"status={perltqa_original_qa_exclusion_selftest.get('status')!r}"
        )
    if (
        perltqa_original_qa_exclusion_checker_path.is_file()
        and perltqa_original_qa_exclusion_selftest.get("checker_sha256")
        != sha256_file(perltqa_original_qa_exclusion_checker_path)
    ):
        perltqa_original_qa_exclusion_selftest_errors.append("checker_sha256 mismatch")
    perltqa_original_qa_exclusion_selftest_script_path = Path(__file__).with_name(
        "selftest_locomo_perltqa_original_qa_exclusion.py"
    )
    if (
        perltqa_original_qa_exclusion_selftest_script_path.is_file()
        and perltqa_original_qa_exclusion_selftest.get("selftest_sha256")
        != sha256_file(perltqa_original_qa_exclusion_selftest_script_path)
    ):
        perltqa_original_qa_exclusion_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "perltqa_original_qa_exclusion_selftest_passed",
            not perltqa_original_qa_exclusion_selftest_errors,
            (
                f"{perltqa_original_qa_exclusion_selftest_path}: "
                f"errors={perltqa_original_qa_exclusion_selftest_errors}"
            ),
        )
    )
    perltqa_fact_ledger_coverage_path = root / "perltqa_fact_ledger_coverage_report.json"
    perltqa_fact_ledger_coverage_report = (
        load_json(perltqa_fact_ledger_coverage_path)
        if perltqa_fact_ledger_coverage_path.exists()
        else {}
    )
    perltqa_fact_ledger_coverage_errors = validation_report_errors(
        perltqa_fact_ledger_coverage_report,
        {
            "primary_json": root / "primary" / "PerLTQA-LoCoMo-style-eval.json",
            "source_memory": Path("datasets/PerLTQA/Dataset/zh/perltmem.json"),
            "source_qa": Path("datasets/PerLTQA/Dataset/zh/perltqa.json"),
            "fact_ledger": root
            / "sidecars"
            / "PerLTQA-LoCoMo-style-eval"
            / "PerLTQA-LoCoMo-style-eval_fact_ledger.jsonl",
            "provenance": root
            / "sidecars"
            / "PerLTQA-LoCoMo-style-eval"
            / "PerLTQA-LoCoMo-style-eval_provenance.jsonl",
        },
    )
    checks.append(
        gate(
            "perltqa_fact_ledger_coverage_passed",
            not perltqa_fact_ledger_coverage_errors,
            f"{perltqa_fact_ledger_coverage_path}: errors={perltqa_fact_ledger_coverage_errors[:10]}",
        )
    )
    perltqa_fact_ledger_coverage_selftest_path = root / "perltqa_fact_ledger_coverage_selftest.json"
    perltqa_fact_ledger_coverage_selftest = (
        load_json(perltqa_fact_ledger_coverage_selftest_path)
        if perltqa_fact_ledger_coverage_selftest_path.exists()
        else {}
    )
    perltqa_fact_ledger_coverage_selftest_errors: list[str] = []
    perltqa_fact_ledger_coverage_checker_path = Path(__file__).with_name(
        "check_locomo_perltqa_fact_ledger_coverage.py"
    )
    if perltqa_fact_ledger_coverage_selftest.get("status") != "passed":
        perltqa_fact_ledger_coverage_selftest_errors.append(
            f"status={perltqa_fact_ledger_coverage_selftest.get('status')!r}"
        )
    if (
        perltqa_fact_ledger_coverage_checker_path.is_file()
        and perltqa_fact_ledger_coverage_selftest.get("checker_sha256")
        != sha256_file(perltqa_fact_ledger_coverage_checker_path)
    ):
        perltqa_fact_ledger_coverage_selftest_errors.append("checker_sha256 mismatch")
    perltqa_fact_ledger_coverage_selftest_script_path = Path(__file__).with_name(
        "selftest_locomo_perltqa_fact_ledger_coverage.py"
    )
    if (
        perltqa_fact_ledger_coverage_selftest_script_path.is_file()
        and perltqa_fact_ledger_coverage_selftest.get("selftest_sha256")
        != sha256_file(perltqa_fact_ledger_coverage_selftest_script_path)
    ):
        perltqa_fact_ledger_coverage_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "perltqa_fact_ledger_coverage_selftest_passed",
            not perltqa_fact_ledger_coverage_selftest_errors,
            (
                f"{perltqa_fact_ledger_coverage_selftest_path}: "
                f"errors={perltqa_fact_ledger_coverage_selftest_errors}"
            ),
        )
    )
    summary_placeholder_path = root / "summary_placeholder_report.json"
    summary_placeholder_report = (
        load_json(summary_placeholder_path) if summary_placeholder_path.exists() else {}
    )
    summary_placeholder_errors = file_list_report_errors(
        summary_placeholder_report,
        {"primary_files": [*expected_primary_files, root / "primary" / "multilingual_locomo_style_eval.json"]},
    )
    checks.append(
        gate(
            "summary_placeholders_empty",
            not summary_placeholder_errors,
            f"{summary_placeholder_path}: errors={summary_placeholder_errors[:10]}",
        )
    )
    summary_placeholder_selftest_path = root / "summary_placeholder_selftest.json"
    summary_placeholder_selftest = (
        load_json(summary_placeholder_selftest_path)
        if summary_placeholder_selftest_path.exists()
        else {}
    )
    summary_placeholder_selftest_errors: list[str] = []
    summary_placeholder_checker_path = Path(__file__).with_name("check_locomo_summary_placeholders.py")
    if summary_placeholder_selftest.get("status") != "passed":
        summary_placeholder_selftest_errors.append(f"status={summary_placeholder_selftest.get('status')!r}")
    if (
        summary_placeholder_checker_path.is_file()
        and summary_placeholder_selftest.get("checker_sha256")
        != sha256_file(summary_placeholder_checker_path)
    ):
        summary_placeholder_selftest_errors.append("checker_sha256 mismatch")
    summary_placeholder_selftest_script_path = Path(__file__).with_name("selftest_locomo_summary_placeholders.py")
    if (
        summary_placeholder_selftest_script_path.is_file()
        and summary_placeholder_selftest.get("selftest_sha256")
        != sha256_file(summary_placeholder_selftest_script_path)
    ):
        summary_placeholder_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "summary_placeholder_selftest_passed",
            not summary_placeholder_selftest_errors,
            f"{summary_placeholder_selftest_path}: errors={summary_placeholder_selftest_errors}",
        )
    )
    opela_temporal_policy_path = root / "opela_temporal_policy_report.json"
    opela_temporal_policy_report = (
        load_json(opela_temporal_policy_path) if opela_temporal_policy_path.exists() else {}
    )
    opela_temporal_policy_errors = validation_report_errors(
        opela_temporal_policy_report,
        {
            "primary_json": root / "primary" / "OPELA-LoCoMo-style-eval.json",
            "provenance": root
            / "sidecars"
            / "OPELA-LoCoMo-style-eval"
            / "OPELA-LoCoMo-style-eval_provenance.jsonl",
            "source_csv": Path("datasets/OPELA/data/oplea_open_data.csv"),
        },
    )
    checks.append(
        gate(
            "opela_temporal_policy_passed",
            not opela_temporal_policy_errors,
            f"{opela_temporal_policy_path}: errors={opela_temporal_policy_errors[:10]}",
        )
    )
    opela_temporal_policy_selftest_path = root / "opela_temporal_policy_selftest.json"
    opela_temporal_policy_selftest = (
        load_json(opela_temporal_policy_selftest_path)
        if opela_temporal_policy_selftest_path.exists()
        else {}
    )
    opela_temporal_policy_selftest_errors: list[str] = []
    opela_temporal_checker_path = Path(__file__).with_name("check_locomo_opela_temporal_policy.py")
    if opela_temporal_policy_selftest.get("status") != "passed":
        opela_temporal_policy_selftest_errors.append(f"status={opela_temporal_policy_selftest.get('status')!r}")
    if (
        opela_temporal_checker_path.is_file()
        and opela_temporal_policy_selftest.get("checker_sha256")
        != sha256_file(opela_temporal_checker_path)
    ):
        opela_temporal_policy_selftest_errors.append("checker_sha256 mismatch")
    opela_temporal_selftest_path = Path(__file__).with_name("selftest_locomo_opela_temporal_policy.py")
    if (
        opela_temporal_selftest_path.is_file()
        and opela_temporal_policy_selftest.get("selftest_sha256")
        != sha256_file(opela_temporal_selftest_path)
    ):
        opela_temporal_policy_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "opela_temporal_policy_selftest_passed",
            not opela_temporal_policy_selftest_errors,
            f"{opela_temporal_policy_selftest_path}: errors={opela_temporal_policy_selftest_errors}",
        )
    )
    opela_evidence_policy_path = root / "opela_evidence_policy_report.json"
    opela_evidence_policy_report = (
        load_json(opela_evidence_policy_path) if opela_evidence_policy_path.exists() else {}
    )
    opela_evidence_policy_errors = validation_report_errors(
        opela_evidence_policy_report,
        {
            "primary_json": root / "primary" / "OPELA-LoCoMo-style-eval.json",
            "fact_ledger": root
            / "sidecars"
            / "OPELA-LoCoMo-style-eval"
            / "OPELA-LoCoMo-style-eval_fact_ledger.jsonl",
            "qa_audit": root
            / "sidecars"
            / "OPELA-LoCoMo-style-eval"
            / "OPELA-LoCoMo-style-eval_qa_audit.jsonl",
        },
    )
    checks.append(
        gate(
            "opela_evidence_policy_passed",
            not opela_evidence_policy_errors,
            f"{opela_evidence_policy_path}: errors={opela_evidence_policy_errors[:10]}",
        )
    )
    opela_evidence_policy_selftest_path = root / "opela_evidence_policy_selftest.json"
    opela_evidence_policy_selftest = (
        load_json(opela_evidence_policy_selftest_path)
        if opela_evidence_policy_selftest_path.exists()
        else {}
    )
    opela_evidence_policy_selftest_errors: list[str] = []
    opela_evidence_checker_path = Path(__file__).with_name("check_locomo_opela_evidence_policy.py")
    if opela_evidence_policy_selftest.get("status") != "passed":
        opela_evidence_policy_selftest_errors.append(
            f"status={opela_evidence_policy_selftest.get('status')!r}"
        )
    if (
        opela_evidence_checker_path.is_file()
        and opela_evidence_policy_selftest.get("checker_sha256")
        != sha256_file(opela_evidence_checker_path)
    ):
        opela_evidence_policy_selftest_errors.append("checker_sha256 mismatch")
    opela_evidence_selftest_path = Path(__file__).with_name("selftest_locomo_opela_evidence_policy.py")
    if (
        opela_evidence_selftest_path.is_file()
        and opela_evidence_policy_selftest.get("selftest_sha256")
        != sha256_file(opela_evidence_selftest_path)
    ):
        opela_evidence_policy_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "opela_evidence_policy_selftest_passed",
            not opela_evidence_policy_selftest_errors,
            f"{opela_evidence_policy_selftest_path}: errors={opela_evidence_policy_selftest_errors}",
        )
    )
    del1l2im_source_policy_path = root / "del1l2im_source_policy_report.json"
    del1l2im_source_policy_report = (
        load_json(del1l2im_source_policy_path) if del1l2im_source_policy_path.exists() else {}
    )
    del1l2im_source_policy_errors = validation_report_errors(
        del1l2im_source_policy_report,
        {
            "primary_json": root / "primary" / "deL1L2IM-LoCoMo-style-eval.json",
            "provenance": root
            / "sidecars"
            / "deL1L2IM-LoCoMo-style-eval"
            / "deL1L2IM-LoCoMo-style-eval_provenance.jsonl",
            "fact_ledger": root
            / "sidecars"
            / "deL1L2IM-LoCoMo-style-eval"
            / "deL1L2IM-LoCoMo-style-eval_fact_ledger.jsonl",
            "qa_audit": root
            / "sidecars"
            / "deL1L2IM-LoCoMo-style-eval"
            / "deL1L2IM-LoCoMo-style-eval_qa_audit.jsonl",
        },
    )
    checks.append(
        gate(
            "del1l2im_source_policy_passed",
            not del1l2im_source_policy_errors,
            f"{del1l2im_source_policy_path}: errors={del1l2im_source_policy_errors[:10]}",
        )
    )
    del1l2im_source_policy_selftest_path = root / "del1l2im_source_policy_selftest.json"
    del1l2im_source_policy_selftest = (
        load_json(del1l2im_source_policy_selftest_path)
        if del1l2im_source_policy_selftest_path.exists()
        else {}
    )
    del1l2im_source_policy_selftest_errors: list[str] = []
    del1l2im_source_checker_path = Path(__file__).with_name("check_locomo_del1l2im_source_policy.py")
    if del1l2im_source_policy_selftest.get("status") != "passed":
        del1l2im_source_policy_selftest_errors.append(
            f"status={del1l2im_source_policy_selftest.get('status')!r}"
        )
    if (
        del1l2im_source_checker_path.is_file()
        and del1l2im_source_policy_selftest.get("checker_sha256")
        != sha256_file(del1l2im_source_checker_path)
    ):
        del1l2im_source_policy_selftest_errors.append("checker_sha256 mismatch")
    del1l2im_source_selftest_path = Path(__file__).with_name(
        "selftest_locomo_del1l2im_source_policy.py"
    )
    if (
        del1l2im_source_selftest_path.is_file()
        and del1l2im_source_policy_selftest.get("selftest_sha256")
        != sha256_file(del1l2im_source_selftest_path)
    ):
        del1l2im_source_policy_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "del1l2im_source_policy_selftest_passed",
            not del1l2im_source_policy_selftest_errors,
            f"{del1l2im_source_policy_selftest_path}: errors={del1l2im_source_policy_selftest_errors}",
        )
    )
    perltqa_ratio_path = root / "perltqa_specific_ratios.json"
    perltqa_ratio_report = load_json(perltqa_ratio_path) if perltqa_ratio_path.exists() else {}
    perltqa_ratio_errors = perltqa_specific_ratio_errors(perltqa_ratio_report, root)
    checks.append(
        gate(
            "perltqa_specific_ratios_reported",
            not perltqa_ratio_errors,
            f"{perltqa_ratio_path}: errors={perltqa_ratio_errors[:10]}",
        )
    )
    expected_hash_check_files = [
        root / "sidecars" / artifact / f"{artifact}_hash_check.jsonl"
        for artifact in SOURCE_ARTIFACTS
    ]
    hash_coverage_path = root / "hash_coverage_report.json"
    hash_coverage = load_json(hash_coverage_path) if hash_coverage_path.exists() else {}
    hash_coverage_errors = file_list_report_errors(
        hash_coverage,
        {
            "provenance_files": expected_provenance_files,
            "hash_check_files": expected_hash_check_files,
        },
    )
    checks.append(
        gate(
            "hash_coverage_passed",
            not hash_coverage_errors,
            f"{hash_coverage_path}: errors={hash_coverage_errors[:10]}",
        )
    )
    hash_coverage_selftest_path = root / "hash_coverage_selftest.json"
    hash_coverage_selftest = (
        load_json(hash_coverage_selftest_path) if hash_coverage_selftest_path.exists() else {}
    )
    hash_coverage_selftest_errors: list[str] = []
    hash_coverage_checker_path = Path(__file__).with_name("check_locomo_hash_coverage.py")
    if hash_coverage_selftest.get("status") != "passed":
        hash_coverage_selftest_errors.append(f"status={hash_coverage_selftest.get('status')!r}")
    if (
        hash_coverage_checker_path.is_file()
        and hash_coverage_selftest.get("checker_sha256") != sha256_file(hash_coverage_checker_path)
    ):
        hash_coverage_selftest_errors.append("checker_sha256 mismatch")
    hash_coverage_selftest_script_path = Path(__file__).with_name("selftest_locomo_hash_coverage.py")
    if (
        hash_coverage_selftest_script_path.is_file()
        and hash_coverage_selftest.get("selftest_sha256") != sha256_file(hash_coverage_selftest_script_path)
    ):
        hash_coverage_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "hash_coverage_selftest_passed",
            not hash_coverage_selftest_errors,
            f"{hash_coverage_selftest_path}: errors={hash_coverage_selftest_errors}",
        )
    )
    alignment_path = root / "primary_sidecar_alignment_report.json"
    alignment_report = load_json(alignment_path) if alignment_path.exists() else {}
    alignment_errors = file_list_report_errors(
        alignment_report,
        {
            "primary_files": expected_primary_files,
            "provenance_files": expected_provenance_files,
        },
    )
    checks.append(
        gate(
            "primary_sidecar_alignment_passed",
            not alignment_errors,
            f"{alignment_path}: errors={alignment_errors[:10]}",
        )
    )
    alignment_selftest_path = root / "primary_sidecar_alignment_selftest.json"
    alignment_selftest = load_json(alignment_selftest_path) if alignment_selftest_path.exists() else {}
    alignment_selftest_errors: list[str] = []
    alignment_checker_path = Path(__file__).with_name("check_locomo_primary_sidecar_alignment.py")
    if alignment_selftest.get("status") != "passed":
        alignment_selftest_errors.append(f"status={alignment_selftest.get('status')!r}")
    if (
        alignment_checker_path.is_file()
        and alignment_selftest.get("checker_sha256") != sha256_file(alignment_checker_path)
    ):
        alignment_selftest_errors.append("checker_sha256 mismatch")
    alignment_selftest_script_path = Path(__file__).with_name("selftest_locomo_primary_sidecar_alignment.py")
    if (
        alignment_selftest_script_path.is_file()
        and alignment_selftest.get("selftest_sha256") != sha256_file(alignment_selftest_script_path)
    ):
        alignment_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "primary_sidecar_alignment_selftest_passed",
            not alignment_selftest_errors,
            f"{alignment_selftest_path}: errors={alignment_selftest_errors}",
        )
    )
    source_replay_path = root / "source_replay_report.json"
    source_replay_report = load_json(source_replay_path) if source_replay_path.exists() else {}
    source_replay_errors = file_list_report_errors(
        source_replay_report,
        {
            "provenance_files": expected_provenance_files,
            "raw_source_files": collect_source_replay_raw_files(root),
        },
    )
    checks.append(
        gate(
            "source_replay_passed",
            not source_replay_errors,
            f"{source_replay_path}: errors={source_replay_errors[:10]}",
        )
    )
    source_replay_selftest_path = root / "source_replay_selftest.json"
    source_replay_selftest = (
        load_json(source_replay_selftest_path) if source_replay_selftest_path.exists() else {}
    )
    source_replay_selftest_errors: list[str] = []
    source_replay_checker_path = Path(__file__).with_name("check_locomo_source_replay.py")
    if source_replay_selftest.get("status") != "passed":
        source_replay_selftest_errors.append(f"status={source_replay_selftest.get('status')!r}")
    if (
        source_replay_checker_path.is_file()
        and source_replay_selftest.get("checker_sha256") != sha256_file(source_replay_checker_path)
    ):
        source_replay_selftest_errors.append("checker_sha256 mismatch")
    source_replay_selftest_script_path = Path(__file__).with_name("selftest_locomo_source_replay.py")
    if (
        source_replay_selftest_script_path.is_file()
        and source_replay_selftest.get("selftest_sha256") != sha256_file(source_replay_selftest_script_path)
    ):
        source_replay_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "source_replay_selftest_passed",
            not source_replay_selftest_errors,
            f"{source_replay_selftest_path}: errors={source_replay_selftest_errors}",
        )
    )
    session_order_path = root / "session_order_report.json"
    session_order_report = load_json(session_order_path) if session_order_path.exists() else {}
    session_order_errors = file_list_report_errors(
        session_order_report,
        {
            "primary_files": expected_primary_files,
            "provenance_files": expected_provenance_files,
        },
    )
    checks.append(
        gate(
            "session_order_passed",
            not session_order_errors,
            f"{session_order_path}: errors={session_order_errors[:10]}",
        )
    )
    session_order_selftest_path = root / "session_order_selftest.json"
    session_order_selftest = (
        load_json(session_order_selftest_path) if session_order_selftest_path.exists() else {}
    )
    session_order_selftest_errors: list[str] = []
    session_order_checker_path = Path(__file__).with_name("check_locomo_session_order.py")
    if session_order_selftest.get("status") != "passed":
        session_order_selftest_errors.append(f"status={session_order_selftest.get('status')!r}")
    if (
        session_order_checker_path.is_file()
        and session_order_selftest.get("checker_sha256") != sha256_file(session_order_checker_path)
    ):
        session_order_selftest_errors.append("checker_sha256 mismatch")
    session_order_selftest_script_path = Path(__file__).with_name("selftest_locomo_session_order.py")
    if (
        session_order_selftest_script_path.is_file()
        and session_order_selftest.get("selftest_sha256") != sha256_file(session_order_selftest_script_path)
    ):
        session_order_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "session_order_selftest_passed",
            not session_order_selftest_errors,
            f"{session_order_selftest_path}: errors={session_order_selftest_errors}",
        )
    )
    qa_trace_path = root / "qa_trace_integrity_report.json"
    qa_trace_report = load_json(qa_trace_path) if qa_trace_path.exists() else {}
    qa_trace_errors = file_list_report_errors(
        qa_trace_report,
        {
            "primary_files": expected_primary_files,
            "provenance_files": expected_provenance_files,
            "fact_ledger_files": [
                root / "sidecars" / artifact / f"{artifact}_fact_ledger.jsonl"
                for artifact in SOURCE_ARTIFACTS
            ],
            "qa_audit_files": [
                root / "sidecars" / artifact / f"{artifact}_qa_audit.jsonl"
                for artifact in SOURCE_ARTIFACTS
            ],
        },
    )
    checks.append(
        gate(
            "qa_trace_integrity_passed",
            not qa_trace_errors,
            f"{qa_trace_path}: errors={qa_trace_errors[:10]}",
        )
    )
    qa_trace_selftest_path = root / "qa_trace_integrity_selftest.json"
    qa_trace_selftest = load_json(qa_trace_selftest_path) if qa_trace_selftest_path.exists() else {}
    qa_trace_selftest_errors: list[str] = []
    qa_trace_checker_path = Path(__file__).with_name("check_locomo_qa_trace_integrity.py")
    if qa_trace_selftest.get("status") != "passed":
        qa_trace_selftest_errors.append(f"status={qa_trace_selftest.get('status')!r}")
    if (
        qa_trace_checker_path.is_file()
        and qa_trace_selftest.get("checker_sha256") != sha256_file(qa_trace_checker_path)
    ):
        qa_trace_selftest_errors.append("checker_sha256 mismatch")
    checks.append(
        gate(
            "qa_trace_integrity_selftest_passed",
            not qa_trace_selftest_errors,
            f"{qa_trace_selftest_path}: errors={qa_trace_selftest_errors}",
        )
    )
    qa_quality_path = root / "qa_quality_report.json"
    qa_quality = load_json(qa_quality_path) if qa_quality_path.exists() else {}
    qa_quality_input_errors = validation_report_errors(
        qa_quality,
        {"primary_json": root / "primary" / "multilingual_locomo_style_eval.json"},
    )
    checks.append(
        gate(
            "qa_quality_passed",
            not qa_quality_input_errors and qa_quality.get("status") == "passed",
            (
                f"{qa_quality_path}: input_errors={qa_quality_input_errors[:5]} "
                f"status={qa_quality.get('status')} "
                f"unique_ratio={qa_quality.get('unique_question_ratio')} "
                f"max_duplicate={qa_quality.get('max_duplicate_question_count')} "
                f"qa_per_sample_min={qa_quality.get('qa_per_sample', {}).get('min')} "
                f"qa_per_sample_max={qa_quality.get('qa_per_sample', {}).get('max')} "
                f"qa_per_sample_out_of_range={qa_quality.get('qa_per_sample', {}).get('out_of_range_count')}"
            ),
        )
    )
    qa_quality_selftest_path = root / "qa_quality_selftest.json"
    qa_quality_selftest = load_json(qa_quality_selftest_path) if qa_quality_selftest_path.exists() else {}
    qa_quality_selftest_errors: list[str] = []
    qa_quality_checker_path = Path(__file__).with_name("check_locomo_qa_quality.py")
    qa_quality_selftest_script_path = Path(__file__).with_name("selftest_locomo_qa_quality.py")
    if qa_quality_selftest.get("status") != "passed":
        qa_quality_selftest_errors.append(f"status={qa_quality_selftest.get('status')!r}")
    if (
        qa_quality_checker_path.is_file()
        and qa_quality_selftest.get("checker_sha256") != sha256_file(qa_quality_checker_path)
    ):
        qa_quality_selftest_errors.append("checker_sha256 mismatch")
    if (
        qa_quality_selftest_script_path.is_file()
        and qa_quality_selftest.get("selftest_sha256") != sha256_file(qa_quality_selftest_script_path)
    ):
        qa_quality_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "qa_quality_selftest_passed",
            not qa_quality_selftest_errors,
            f"{qa_quality_selftest_path}: errors={qa_quality_selftest_errors}",
        )
    )

    audit_summary_path = root / "human_audit_queue_summary.json"
    audit_summary = load_json(audit_summary_path) if audit_summary_path.exists() else {}
    checks.append(
        gate(
            "human_audit_queue_exists",
            audit_summary.get("selected_qa", 0) > 0,
            f"{audit_summary_path}: selected_qa={audit_summary.get('selected_qa')}",
        )
    )
    audit_queue_coverage_path = root / "human_audit_queue_coverage.json"
    audit_queue_coverage = load_json(audit_queue_coverage_path) if audit_queue_coverage_path.exists() else {}
    audit_queue_errors = audit_queue_coverage_errors(audit_queue_coverage, root)
    checks.append(
        gate(
            "human_audit_queue_coverage_passed",
            not audit_queue_errors,
            f"{audit_queue_coverage_path}: errors={audit_queue_errors[:10]}",
        )
    )
    audit_queue_coverage_selftest_path = root / "human_audit_queue_coverage_selftest.json"
    audit_queue_coverage_selftest = (
        load_json(audit_queue_coverage_selftest_path) if audit_queue_coverage_selftest_path.exists() else {}
    )
    audit_queue_coverage_selftest_errors: list[str] = []
    audit_queue_validator_path = Path(__file__).with_name("validate_locomo_human_audit_queue.py")
    if audit_queue_coverage_selftest.get("status") != "passed":
        audit_queue_coverage_selftest_errors.append(f"status={audit_queue_coverage_selftest.get('status')!r}")
    if (
        audit_queue_validator_path.is_file()
        and audit_queue_coverage_selftest.get("validator_sha256") != sha256_file(audit_queue_validator_path)
    ):
        audit_queue_coverage_selftest_errors.append("validator_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_queue_coverage_selftest_passed",
            not audit_queue_coverage_selftest_errors,
            f"{audit_queue_coverage_selftest_path}: errors={audit_queue_coverage_selftest_errors}",
        )
    )
    audit_packet_queue_errors = human_audit_packet_queue_errors(root)
    checks.append(
        gate(
            "human_audit_packet_matches_queue",
            not audit_packet_queue_errors,
            f"human_audit_packet.jsonl vs human_audit_queue.jsonl errors={audit_packet_queue_errors[:10]}",
        )
    )
    audit_validator_selftest_path = root / "human_audit_validator_selftest.json"
    audit_validator_selftest = (
        load_json(audit_validator_selftest_path) if audit_validator_selftest_path.exists() else {}
    )
    audit_validator_selftest_errors: list[str] = []
    audit_packet_path = root / "human_audit_packet.jsonl"
    audit_validator_path = Path(__file__).with_name("validate_locomo_human_audit_results.py")
    sidecar_root = root / "sidecars"
    if audit_validator_selftest.get("status") != "passed":
        audit_validator_selftest_errors.append(f"status={audit_validator_selftest.get('status')!r}")
    if audit_packet_path.is_file() and audit_validator_selftest.get("audit_packet_sha256") != sha256_file(audit_packet_path):
        audit_validator_selftest_errors.append("audit_packet_sha256 mismatch")
    if audit_validator_path.is_file() and audit_validator_selftest.get("validator_sha256") != sha256_file(audit_validator_path):
        audit_validator_selftest_errors.append("validator_sha256 mismatch")
    if sidecar_root.is_dir() and audit_validator_selftest.get("sidecar_trace_files_sha256") != sidecar_trace_files_sha256(sidecar_root):
        audit_validator_selftest_errors.append("sidecar_trace_files_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_validator_selftest_passed",
            not audit_validator_selftest_errors,
            f"{audit_validator_selftest_path}: errors={audit_validator_selftest_errors}",
        )
    )
    audit_results_gate_selftest_path = root / "human_audit_results_gate_selftest.json"
    audit_results_gate_selftest = (
        load_json(audit_results_gate_selftest_path) if audit_results_gate_selftest_path.exists() else {}
    )
    audit_results_gate_selftest_errors: list[str] = []
    if audit_results_gate_selftest.get("status") != "passed":
        audit_results_gate_selftest_errors.append(f"status={audit_results_gate_selftest.get('status')!r}")
    if audit_results_gate_selftest.get("gate_script_sha256") != sha256_file(Path(__file__)):
        audit_results_gate_selftest_errors.append("gate_script_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_results_gate_selftest_passed",
            not audit_results_gate_selftest_errors,
            f"{audit_results_gate_selftest_path}: errors={audit_results_gate_selftest_errors}",
        )
    )
    audit_csv_workflow_selftest_path = root / "human_audit_csv_workflow_selftest.json"
    audit_csv_workflow_selftest = (
        load_json(audit_csv_workflow_selftest_path) if audit_csv_workflow_selftest_path.exists() else {}
    )
    audit_csv_errors = audit_csv_workflow_selftest_errors(audit_csv_workflow_selftest)
    checks.append(
        gate(
            "human_audit_csv_workflow_selftest_passed",
            not audit_csv_errors,
            f"{audit_csv_workflow_selftest_path}: errors={audit_csv_errors}",
        )
    )
    audit_flags_selftest_path = root / "human_audit_flags_selftest.json"
    audit_flags_selftest = load_json(audit_flags_selftest_path) if audit_flags_selftest_path.exists() else {}
    audit_flags_selftest_errors: list[str] = []
    audit_flags_script_path = Path(__file__).with_name("summarize_locomo_human_audit_flags.py")
    audit_flags_selftest_script_path = Path(__file__).with_name("selftest_locomo_human_audit_flags.py")
    if audit_flags_selftest.get("status") != "passed":
        audit_flags_selftest_errors.append(f"status={audit_flags_selftest.get('status')!r}")
    if (
        audit_flags_script_path.is_file()
        and audit_flags_selftest.get("summary_script_sha256") != sha256_file(audit_flags_script_path)
    ):
        audit_flags_selftest_errors.append("summary_script_sha256 mismatch")
    if (
        audit_flags_selftest_script_path.is_file()
        and audit_flags_selftest.get("selftest_sha256") != sha256_file(audit_flags_selftest_script_path)
    ):
        audit_flags_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_flags_selftest_passed",
            not audit_flags_selftest_errors,
            f"{audit_flags_selftest_path}: errors={audit_flags_selftest_errors}",
        )
    )
    assignment_risk_path = root / "human_audit_assignment_risk_summary.json"
    assignment_risk = load_json(assignment_risk_path) if assignment_risk_path.exists() else {}
    assignment_risk_errors = assignment_risk_summary_errors(assignment_risk, root)
    checks.append(
        gate(
            "human_audit_assignment_risk_summary_passed",
            not assignment_risk_errors,
            f"{assignment_risk_path}: errors={assignment_risk_errors[:10]}",
        )
    )
    reviewer_todos_check_path = root / "human_audit_reviewer_todos_check.json"
    reviewer_todos_check = load_json(reviewer_todos_check_path) if reviewer_todos_check_path.exists() else {}
    reviewer_todos_errors = reviewer_todos_check_errors(reviewer_todos_check, root)
    checks.append(
        gate(
            "human_audit_reviewer_todos_fresh",
            not reviewer_todos_errors,
            f"{reviewer_todos_check_path}: errors={reviewer_todos_errors[:10]}",
        )
    )
    reviewer_todos_selftest_path = root / "human_audit_reviewer_todos_check_selftest.json"
    reviewer_todos_selftest = load_json(reviewer_todos_selftest_path) if reviewer_todos_selftest_path.exists() else {}
    reviewer_todos_selftest_errors: list[str] = []
    reviewer_todos_checker_path = Path(__file__).with_name("check_locomo_human_audit_reviewer_todos.py")
    reviewer_todos_selftest_script_path = Path(__file__).with_name(
        "selftest_locomo_human_audit_reviewer_todos_check.py"
    )
    if reviewer_todos_selftest.get("status") != "passed":
        reviewer_todos_selftest_errors.append(f"status={reviewer_todos_selftest.get('status')!r}")
    if (
        reviewer_todos_checker_path.is_file()
        and reviewer_todos_selftest.get("checker_sha256") != sha256_file(reviewer_todos_checker_path)
    ):
        reviewer_todos_selftest_errors.append("checker_sha256 mismatch")
    if (
        reviewer_todos_selftest_script_path.is_file()
        and reviewer_todos_selftest.get("selftest_sha256") != sha256_file(reviewer_todos_selftest_script_path)
    ):
        reviewer_todos_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_reviewer_todos_selftest_passed",
            not reviewer_todos_selftest_errors,
            f"{reviewer_todos_selftest_path}: errors={reviewer_todos_selftest_errors}",
        )
    )
    handoff_docs_report_path = root / "human_audit_handoff_docs_report.json"
    handoff_docs_report = load_json(handoff_docs_report_path) if handoff_docs_report_path.exists() else {}
    handoff_docs_errors = human_audit_handoff_docs_errors(handoff_docs_report, root)
    checks.append(
        gate(
            "human_audit_handoff_docs_passed",
            not handoff_docs_errors,
            f"{handoff_docs_report_path}: errors={handoff_docs_errors[:10]}",
        )
    )
    handoff_docs_selftest_path = root / "human_audit_handoff_docs_selftest.json"
    handoff_docs_selftest = load_json(handoff_docs_selftest_path) if handoff_docs_selftest_path.exists() else {}
    handoff_docs_selftest_errors: list[str] = []
    handoff_docs_checker_path = Path(__file__).with_name("check_locomo_human_audit_handoff_docs.py")
    handoff_docs_selftest_script_path = Path(__file__).with_name("selftest_locomo_human_audit_handoff_docs.py")
    if handoff_docs_selftest.get("status") != "passed":
        handoff_docs_selftest_errors.append(f"status={handoff_docs_selftest.get('status')!r}")
    if (
        handoff_docs_checker_path.is_file()
        and handoff_docs_selftest.get("checker_sha256") != sha256_file(handoff_docs_checker_path)
    ):
        handoff_docs_selftest_errors.append("checker_sha256 mismatch")
    if (
        handoff_docs_selftest_script_path.is_file()
        and handoff_docs_selftest.get("selftest_sha256") != sha256_file(handoff_docs_selftest_script_path)
    ):
        handoff_docs_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "human_audit_handoff_docs_selftest_passed",
            not handoff_docs_selftest_errors,
            f"{handoff_docs_selftest_path}: errors={handoff_docs_selftest_errors}",
        )
    )
    release_blocker_selftest_path = root / "release_blocker_summary_selftest.json"
    release_blocker_selftest = load_json(release_blocker_selftest_path) if release_blocker_selftest_path.exists() else {}
    release_blocker_selftest_errors: list[str] = []
    release_blocker_script_path = Path(__file__).with_name("summarize_locomo_release_blockers.py")
    release_blocker_selftest_script_path = Path(__file__).with_name("selftest_locomo_release_blocker_summary.py")
    if release_blocker_selftest.get("status") != "passed":
        release_blocker_selftest_errors.append(f"status={release_blocker_selftest.get('status')!r}")
    if (
        release_blocker_script_path.is_file()
        and release_blocker_selftest.get("summary_script_sha256") != sha256_file(release_blocker_script_path)
    ):
        release_blocker_selftest_errors.append("summary_script_sha256 mismatch")
    if (
        release_blocker_selftest_script_path.is_file()
        and release_blocker_selftest.get("selftest_sha256") != sha256_file(release_blocker_selftest_script_path)
    ):
        release_blocker_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "release_blocker_summary_selftest_passed",
            not release_blocker_selftest_errors,
            f"{release_blocker_selftest_path}: errors={release_blocker_selftest_errors}",
        )
    )
    dataset_card_selftest_path = root / "dataset_card_summary_selftest.json"
    dataset_card_selftest = load_json(dataset_card_selftest_path) if dataset_card_selftest_path.exists() else {}
    dataset_card_selftest_errors: list[str] = []
    dataset_card_script_path = Path(__file__).with_name("summarize_locomo_dataset_card.py")
    dataset_card_selftest_script_path = Path(__file__).with_name("selftest_locomo_dataset_card_summary.py")
    if dataset_card_selftest.get("status") != "passed":
        dataset_card_selftest_errors.append(f"status={dataset_card_selftest.get('status')!r}")
    if (
        dataset_card_script_path.is_file()
        and dataset_card_selftest.get("summary_script_sha256") != sha256_file(dataset_card_script_path)
    ):
        dataset_card_selftest_errors.append("summary_script_sha256 mismatch")
    if (
        dataset_card_selftest_script_path.is_file()
        and dataset_card_selftest.get("selftest_sha256") != sha256_file(dataset_card_selftest_script_path)
    ):
        dataset_card_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "dataset_card_summary_selftest_passed",
            not dataset_card_selftest_errors,
            f"{dataset_card_selftest_path}: errors={dataset_card_selftest_errors}",
        )
    )
    goal_traceability_selftest_path = root / "goal_traceability_selftest.json"
    goal_traceability_selftest = (
        load_json(goal_traceability_selftest_path) if goal_traceability_selftest_path.exists() else {}
    )
    goal_traceability_selftest_errors: list[str] = []
    goal_traceability_script_path = Path(__file__).with_name("summarize_locomo_goal_traceability.py")
    goal_traceability_selftest_script_path = Path(__file__).with_name("selftest_locomo_goal_traceability.py")
    if goal_traceability_selftest.get("status") != "passed":
        goal_traceability_selftest_errors.append(f"status={goal_traceability_selftest.get('status')!r}")
    if (
        goal_traceability_script_path.is_file()
        and goal_traceability_selftest.get("summary_script_sha256") != sha256_file(goal_traceability_script_path)
    ):
        goal_traceability_selftest_errors.append("summary_script_sha256 mismatch")
    if (
        goal_traceability_selftest_script_path.is_file()
        and goal_traceability_selftest.get("selftest_sha256") != sha256_file(goal_traceability_selftest_script_path)
    ):
        goal_traceability_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "goal_traceability_selftest_passed",
            not goal_traceability_selftest_errors,
            f"{goal_traceability_selftest_path}: errors={goal_traceability_selftest_errors}",
        )
    )
    goal_traceability_matrix_path = root / "goal_traceability_matrix.json"
    goal_traceability_matrix = (
        load_json(goal_traceability_matrix_path) if goal_traceability_matrix_path.exists() else {}
    )
    goal_doc_path = Path("docs/locomo_style_eval_goal.md")
    expected_release_gate_report_path = root / "release_gate_report.json"
    matrix_errors = goal_traceability_matrix_errors(
        goal_traceability_matrix,
        matrix_path=goal_traceability_matrix_path,
        goal_doc_path=goal_doc_path,
        release_gate_report_path=expected_release_gate_report_path,
    )
    checks.append(
        gate(
            "goal_traceability_matrix_fresh",
            not matrix_errors,
            f"{goal_traceability_matrix_path}: errors={matrix_errors}",
        )
    )
    baseline_validator_selftest_path = root / "baseline_results_validator_selftest.json"
    baseline_validator_selftest = (
        load_json(baseline_validator_selftest_path) if baseline_validator_selftest_path.exists() else {}
    )
    baseline_validator_path = Path(__file__).with_name("validate_locomo_baseline_results.py")
    baseline_validator_selftest_errors: list[str] = []
    if baseline_validator_selftest.get("status") != "passed":
        baseline_validator_selftest_errors.append(f"status={baseline_validator_selftest.get('status')!r}")
    if (
        baseline_validator_path.is_file()
        and baseline_validator_selftest.get("validator_sha256") != sha256_file(baseline_validator_path)
    ):
        baseline_validator_selftest_errors.append("validator_sha256 mismatch")
    checks.append(
        gate(
            "baseline_results_validator_selftest_passed",
            not baseline_validator_selftest_errors,
            f"{baseline_validator_selftest_path}: errors={baseline_validator_selftest_errors}",
        )
    )
    recent_session_gate_selftest_path = root / "recent_session_result_gate_selftest.json"
    recent_session_gate_selftest = (
        load_json(recent_session_gate_selftest_path) if recent_session_gate_selftest_path.exists() else {}
    )
    recent_session_gate_selftest_errors: list[str] = []
    if recent_session_gate_selftest.get("status") != "passed":
        recent_session_gate_selftest_errors.append(f"status={recent_session_gate_selftest.get('status')!r}")
    if recent_session_gate_selftest.get("gate_script_sha256") != sha256_file(Path(__file__)):
        recent_session_gate_selftest_errors.append("gate_script_sha256 mismatch")
    checks.append(
        gate(
            "recent_session_result_gate_selftest_passed",
            not recent_session_gate_selftest_errors,
            f"{recent_session_gate_selftest_path}: errors={recent_session_gate_selftest_errors}",
        )
    )
    recent_session_runner_selftest_path = root / "recent_session_runner_settings_selftest.json"
    recent_session_runner_selftest = (
        load_json(recent_session_runner_selftest_path) if recent_session_runner_selftest_path.exists() else {}
    )
    recent_session_runner_selftest_errors: list[str] = []
    recent_session_runner_path = Path(__file__).with_name("run_locomo_recent_session_model_diagnostic.py")
    recent_session_runner_selftest_script = Path(__file__).with_name("selftest_locomo_recent_session_runner_settings.py")
    if recent_session_runner_selftest.get("status") != "passed":
        recent_session_runner_selftest_errors.append(f"status={recent_session_runner_selftest.get('status')!r}")
    if (
        recent_session_runner_path.is_file()
        and recent_session_runner_selftest.get("runner_script_sha256") != sha256_file(recent_session_runner_path)
    ):
        recent_session_runner_selftest_errors.append("runner_script_sha256 mismatch")
    if (
        recent_session_runner_selftest_script.is_file()
        and recent_session_runner_selftest.get("selftest_sha256") != sha256_file(recent_session_runner_selftest_script)
    ):
        recent_session_runner_selftest_errors.append("selftest_sha256 mismatch")
    checks.append(
        gate(
            "recent_session_runner_settings_selftest_passed",
            not recent_session_runner_selftest_errors,
            f"{recent_session_runner_selftest_path}: errors={recent_session_runner_selftest_errors}",
        )
    )
    experiment_preflight_selftest_path = root / "experiment_preflight_selftest.json"
    experiment_preflight_selftest = (
        load_json(experiment_preflight_selftest_path) if experiment_preflight_selftest_path.exists() else {}
    )
    experiment_preflight_selftest_errors: list[str] = []
    experiment_preflight_path = Path(__file__).with_name("preflight_locomo_style_experiment.py")
    if experiment_preflight_selftest.get("status") != "passed":
        experiment_preflight_selftest_errors.append(f"status={experiment_preflight_selftest.get('status')!r}")
    if (
        experiment_preflight_path.is_file()
        and experiment_preflight_selftest.get("preflight_script_sha256") != sha256_file(experiment_preflight_path)
    ):
        experiment_preflight_selftest_errors.append("preflight_script_sha256 mismatch")
    checks.append(
        gate(
            "experiment_preflight_selftest_passed",
            not experiment_preflight_selftest_errors,
            f"{experiment_preflight_selftest_path}: errors={experiment_preflight_selftest_errors}",
        )
    )
    audit_apply_integrity_selftest_path = root / "audit_apply_integrity_selftest.json"
    audit_apply_integrity_selftest = (
        load_json(audit_apply_integrity_selftest_path) if audit_apply_integrity_selftest_path.exists() else {}
    )
    audit_apply_integrity_selftest_errors: list[str] = []
    apply_script_path = Path(__file__).with_name("apply_locomo_human_audit_results.py")
    integrity_script_path = Path(__file__).with_name("check_locomo_audited_apply_integrity.py")
    if audit_apply_integrity_selftest.get("status") != "passed":
        audit_apply_integrity_selftest_errors.append(f"status={audit_apply_integrity_selftest.get('status')!r}")
    if (
        apply_script_path.is_file()
        and audit_apply_integrity_selftest.get("apply_script_sha256") != sha256_file(apply_script_path)
    ):
        audit_apply_integrity_selftest_errors.append("apply_script_sha256 mismatch")
    if (
        integrity_script_path.is_file()
        and audit_apply_integrity_selftest.get("integrity_script_sha256") != sha256_file(integrity_script_path)
    ):
        audit_apply_integrity_selftest_errors.append("integrity_script_sha256 mismatch")
    checks.append(
        gate(
            "audit_apply_integrity_selftest_passed",
            not audit_apply_integrity_selftest_errors,
            f"{audit_apply_integrity_selftest_path}: errors={audit_apply_integrity_selftest_errors}",
        )
    )
    post_audit_pipeline_selftest_path = root / "post_audit_pipeline_selftest.json"
    post_audit_pipeline_selftest = (
        load_json(post_audit_pipeline_selftest_path) if post_audit_pipeline_selftest_path.exists() else {}
    )
    post_audit_pipeline_selftest_errors: list[str] = []
    post_audit_pipeline_path = Path(__file__).with_name("run_locomo_post_audit_pipeline.py")
    post_audit_pipeline_selftest_script = Path(__file__).with_name("selftest_locomo_post_audit_pipeline.py")
    if post_audit_pipeline_selftest.get("status") != "passed":
        post_audit_pipeline_selftest_errors.append(f"status={post_audit_pipeline_selftest.get('status')!r}")
    if (
        post_audit_pipeline_path.is_file()
        and post_audit_pipeline_selftest.get("pipeline_script_sha256") != sha256_file(post_audit_pipeline_path)
    ):
        post_audit_pipeline_selftest_errors.append("pipeline_script_sha256 mismatch")
    if (
        post_audit_pipeline_selftest_script.is_file()
        and post_audit_pipeline_selftest.get("selftest_script_sha256") != sha256_file(post_audit_pipeline_selftest_script)
    ):
        post_audit_pipeline_selftest_errors.append("selftest_script_sha256 mismatch")
    checks.append(
        gate(
            "post_audit_pipeline_selftest_passed",
            not post_audit_pipeline_selftest_errors,
            f"{post_audit_pipeline_selftest_path}: errors={post_audit_pipeline_selftest_errors}",
        )
    )
    metric_metadata_builder_selftest_path = root / "metric_metadata_builder_selftest.json"
    metric_metadata_builder_selftest = (
        load_json(metric_metadata_builder_selftest_path) if metric_metadata_builder_selftest_path.exists() else {}
    )
    metric_metadata_builder_selftest_errors: list[str] = []
    metric_metadata_builder_path = Path(__file__).with_name("build_locomo_metric_metadata.py")
    if metric_metadata_builder_selftest.get("status") != "passed":
        metric_metadata_builder_selftest_errors.append(f"status={metric_metadata_builder_selftest.get('status')!r}")
    if (
        metric_metadata_builder_path.is_file()
        and metric_metadata_builder_selftest.get("builder_sha256") != sha256_file(metric_metadata_builder_path)
    ):
        metric_metadata_builder_selftest_errors.append("builder_sha256 mismatch")
    checks.append(
        gate(
            "metric_metadata_builder_selftest_passed",
            not metric_metadata_builder_selftest_errors,
            f"{metric_metadata_builder_selftest_path}: errors={metric_metadata_builder_selftest_errors}",
        )
    )
    baseline_summary_builder_selftest_path = root / "baseline_summary_builder_selftest.json"
    baseline_summary_builder_selftest = (
        load_json(baseline_summary_builder_selftest_path)
        if baseline_summary_builder_selftest_path.exists()
        else {}
    )
    baseline_summary_builder_selftest_errors: list[str] = []
    baseline_summary_builder_path = Path(__file__).with_name("build_locomo_baseline_summary.py")
    baseline_validator_path_for_builder = Path(__file__).with_name("validate_locomo_baseline_results.py")
    if baseline_summary_builder_selftest.get("status") != "passed":
        baseline_summary_builder_selftest_errors.append(
            f"status={baseline_summary_builder_selftest.get('status')!r}"
        )
    if (
        baseline_summary_builder_path.is_file()
        and baseline_summary_builder_selftest.get("builder_sha256") != sha256_file(baseline_summary_builder_path)
    ):
        baseline_summary_builder_selftest_errors.append("builder_sha256 mismatch")
    if (
        baseline_validator_path_for_builder.is_file()
        and baseline_summary_builder_selftest.get("validator_sha256")
        != sha256_file(baseline_validator_path_for_builder)
    ):
        baseline_summary_builder_selftest_errors.append("validator_sha256 mismatch")
    checks.append(
        gate(
            "baseline_summary_builder_selftest_passed",
            not baseline_summary_builder_selftest_errors,
            f"{baseline_summary_builder_selftest_path}: errors={baseline_summary_builder_selftest_errors}",
        )
    )
    prediction_normalizer_selftest_path = root / "prediction_normalizer_selftest.json"
    prediction_normalizer_selftest = (
        load_json(prediction_normalizer_selftest_path) if prediction_normalizer_selftest_path.exists() else {}
    )
    prediction_normalizer_selftest_errors: list[str] = []
    prediction_normalizer_path = Path(__file__).with_name("normalize_locomo_baseline_predictions.py")
    if prediction_normalizer_selftest.get("status") != "passed":
        prediction_normalizer_selftest_errors.append(f"status={prediction_normalizer_selftest.get('status')!r}")
    if (
        prediction_normalizer_path.is_file()
        and prediction_normalizer_selftest.get("normalizer_sha256") != sha256_file(prediction_normalizer_path)
    ):
        prediction_normalizer_selftest_errors.append("normalizer_sha256 mismatch")
    checks.append(
        gate(
            "prediction_normalizer_selftest_passed",
            not prediction_normalizer_selftest_errors,
            f"{prediction_normalizer_selftest_path}: errors={prediction_normalizer_selftest_errors}",
        )
    )
    audit_results_path = root / "human_audit_results_summary.json"
    audit_results = load_json(audit_results_path) if audit_results_path.exists() else {}
    audit_results_errors_list = human_audit_results_errors(audit_results, root)
    checks.append(
        gate(
            "human_audit_completed",
            not audit_results_errors_list,
            f"{audit_results_path}: errors={audit_results_errors_list[:10]}",
        )
    )
    audit_apply_path = root / "human_audit_apply_report.json"
    audit_apply = load_json(audit_apply_path) if audit_apply_path.exists() else {}
    audited_output_path = Path(str(audit_apply.get("output_json", ""))) if audit_apply.get("output_json") else None
    checks.append(
        gate(
            "human_audit_applied",
            audit_apply.get("status") == "applied" and audited_output_path is not None and audited_output_path.exists(),
            f"{audit_apply_path}: status={audit_apply.get('status')} output={audit_apply.get('output_json')}",
        )
    )
    audited_primary_errors = (
        audited_primary_validation_errors(audited_output_path)
        if audit_apply.get("status") == "applied"
        else ["audit not applied"]
    )
    checks.append(
        gate(
            "audited_primary_validation_passed",
            audit_apply.get("status") == "applied" and not audited_primary_errors,
            f"output={audit_apply.get('output_json')} errors={audited_primary_errors[:10]}",
        )
    )
    audited_source_errors = audited_source_validation_errors(audit_apply, root)
    checks.append(
        gate(
            "audited_source_files_validation_passed",
            audit_apply.get("status") == "applied" and not audited_source_errors,
            f"output_source_files={audit_apply.get('output_source_files')} errors={audited_source_errors[:10]}",
        )
    )
    audited_apply_integrity_path = root / "audited_apply_integrity_report.json"
    audited_apply_integrity_errors_list = audited_apply_integrity_errors(audited_apply_integrity_path, audit_apply, root)
    checks.append(
        gate(
            "audited_apply_integrity_passed",
            audit_apply.get("status") == "applied" and not audited_apply_integrity_errors_list,
            f"{audited_apply_integrity_path}: errors={audited_apply_integrity_errors_list[:10]}",
        )
    )
    metric_metadata_summary_path = root / "baseline_results" / "metric_metadata_summary.json"
    metric_metadata_errors_list = metric_metadata_errors(metric_metadata_summary_path, audit_apply)
    checks.append(
        gate(
            "metric_metadata_created",
            audit_apply.get("status") == "applied" and not metric_metadata_errors_list,
            f"{metric_metadata_summary_path}: errors={metric_metadata_errors_list[:10]}",
        )
    )

    ablation_manifest_path = root / "recent_session_ablation" / "recent_session_ablation_manifest.json"
    ablation_manifest = load_json(ablation_manifest_path) if ablation_manifest_path.exists() else {}
    ablation_files = ablation_manifest.get("files", {})
    expected_ablation_dataset = None
    if audit_apply.get("status") == "applied" and audit_apply.get("output_json"):
        expected_ablation_dataset = Path(str(audit_apply["output_json"]))
    ablation_input_errors: list[str] = []
    if expected_ablation_dataset is not None:
        if ablation_manifest.get("input") != str(expected_ablation_dataset):
            ablation_input_errors.append(
                f"input={ablation_manifest.get('input')!r} expected={str(expected_ablation_dataset)!r}"
            )
        if expected_ablation_dataset.is_file() and ablation_manifest.get("input_sha256") != sha256_file(expected_ablation_dataset):
            ablation_input_errors.append("input_sha256 does not match audited dataset")
    checks.append(
        gate(
            "recent_session_ablation_files_created",
            all(Path(path).exists() for path in ablation_files.values())
            and len(ablation_files) == 3
            and not ablation_input_errors,
            f"{ablation_manifest_path}: files={ablation_files} input_errors={ablation_input_errors}",
        )
    )
    recent_session_smoke = root / "recent_session_ablation" / "model_smoke_summary.json"
    recent_session_smoke_report = load_json(recent_session_smoke) if recent_session_smoke.exists() else {}
    smoke_errors = recent_session_smoke_report.get("summary", {}).get("errors")
    checks.append(
        gate(
            "recent_session_model_smoke_ran",
            recent_session_smoke.exists() and smoke_errors == 0,
            f"{recent_session_smoke}: status={recent_session_smoke_report.get('status')} errors={smoke_errors}",
            blocking=False,
        )
    )
    recent_session_results = root / "recent_session_ablation" / "model_results_summary.json"
    recent_session_report = load_json(recent_session_results) if recent_session_results.exists() else {}
    recent_session_errors = (
        recent_session_result_errors(recent_session_report, expected_ablation_dataset, root / "fixed_eval_settings.json")
        if recent_session_results.exists()
        else ["summary missing"]
    )
    checks.append(
        gate(
            "recent_session_model_results_exist",
            recent_session_report.get("status") == "completed" and not recent_session_errors,
            f"{recent_session_results}: status={recent_session_report.get('status')} errors={recent_session_errors}",
        )
    )

    baseline_results = root / "baseline_results" / "summary.json"
    baseline_report = load_json(baseline_results) if baseline_results.exists() else {}
    observed_methods = baseline_methods(baseline_report)
    missing_methods = sorted(REQUIRED_BASELINE_METHODS - observed_methods)
    expected_baseline_dataset = None
    if audit_apply.get("status") == "applied" and audit_apply.get("output_json"):
        expected_baseline_dataset = Path(str(audit_apply["output_json"]))
    baseline_errors = (
        baseline_validation_errors(baseline_report, expected_baseline_dataset, baseline_results)
        if baseline_results.exists()
        else ["summary missing"]
    )
    checks.append(
        gate(
            "fixed_baseline_results_exist",
            baseline_report.get("status") == "completed" and not missing_methods and not baseline_errors,
            (
                f"{baseline_results}: status={baseline_report.get('status')} "
                f"missing_methods={missing_methods} schema_errors={baseline_errors}"
            ),
        )
    )

    blocking_failed = [item for item in checks if item["blocking"] and item["status"] != "passed"]
    report = {
        "root": str(root),
        "status": "release_ready" if not blocking_failed else "blocked",
        "blocking_failed": [item["name"] for item in blocking_failed],
        "checks": checks,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if blocking_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
