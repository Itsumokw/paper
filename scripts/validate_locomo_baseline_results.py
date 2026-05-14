#!/usr/bin/env python3
"""Validate fixed-setting baseline results for the LoCoMo-style eval release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_METHODS = {"Full Context", "A-MEM", "Mem0", "SimpleMem", "HiGMem"}
REQUIRED_RESULT_FIELDS = {
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
REQUIRED_AUDITED_DATASET_FILENAME = "multilingual_locomo_style_eval_audited.json"
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_counts(dataset_path: Path) -> dict[str, int]:
    samples = load_json(dataset_path)
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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fail(message: str) -> int:
    print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False, indent=2))
    return 1


def anti_tuning_contract_errors(settings: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = settings.get("anti_tuning_contract", {})
    if contract.get("benchmark_tuning_allowed") is not False:
        errors.append("settings.anti_tuning_contract.benchmark_tuning_allowed must be false")
    if contract.get("frozen_before_final_runs") is not True:
        errors.append("settings.anti_tuning_contract.frozen_before_final_runs must be true")
    frozen_items = {str(item) for item in contract.get("do_not_tune_on_this_benchmark", [])}
    missing = sorted(REQUIRED_ANTI_TUNING_ITEMS - frozen_items)
    if missing:
        errors.append(f"settings.anti_tuning_contract.do_not_tune_on_this_benchmark missing {missing}")
    service_check = str(contract.get("allowed_service_check", "")).lower()
    if "tiny chat" not in service_check or "dataset" not in service_check:
        errors.append("settings.anti_tuning_contract.allowed_service_check must limit service checks to tiny chat outside dataset/metrics")
    return errors


def fixed_baseline_execution_contract_errors(settings: dict[str, Any], expected_dataset: str) -> list[str]:
    errors: list[str] = []
    model = settings.get("model", {})
    if model.get("served_model") != REQUIRED_FIXED_MODEL:
        errors.append(f"settings.model.served_model={model.get('served_model')!r} expected={REQUIRED_FIXED_MODEL!r}")
    if model.get("temperature") != 0:
        errors.append("settings.model.temperature must be 0")

    fixed = settings.get("fixed_baselines", {})
    contract = fixed.get("execution_contract")
    if not isinstance(contract, dict):
        errors.append("settings.fixed_baselines.execution_contract is required")
        contract = {}

    if contract.get("final_dataset") != expected_dataset:
        errors.append(
            f"settings.fixed_baselines.execution_contract.final_dataset={contract.get('final_dataset')!r} "
            f"expected={expected_dataset!r}"
        )
    if contract.get("model") != REQUIRED_FIXED_MODEL:
        errors.append("settings.fixed_baselines.execution_contract.model must match Qwen/Qwen3-8B")
    if contract.get("input_policy") != "conversation_only":
        errors.append("settings.fixed_baselines.execution_contract.input_policy must be conversation_only")
    if contract.get("summary_visible") is not False:
        errors.append("settings.fixed_baselines.execution_contract.summary_visible must be false")
    if contract.get("summary_builder") != "scripts/build_locomo_baseline_summary.py":
        errors.append("settings.fixed_baselines.execution_contract.summary_builder must be build_locomo_baseline_summary.py")
    if contract.get("legacy_locomo10_defaults_forbidden") is not True:
        errors.append("settings.fixed_baselines.execution_contract.legacy_locomo10_defaults_forbidden must be true")

    prediction_contract = contract.get("prediction_jsonl_contract", {})
    identity_fields = {str(item) for item in prediction_contract.get("identity_fields", [])}
    if not REQUIRED_PREDICTION_IDENTITY_FIELDS <= identity_fields:
        errors.append(
            "settings.fixed_baselines.execution_contract.prediction_jsonl_contract.identity_fields "
            "must include sample_id and qa_idx"
        )
    row_fields = {str(item) for item in prediction_contract.get("required_row_fields", [])}
    if not REQUIRED_PREDICTION_ROW_FIELDS <= row_fields:
        errors.append(
            "settings.fixed_baselines.execution_contract.prediction_jsonl_contract.required_row_fields "
            "must include sample_id, qa_idx, model, and dataset_sha256"
        )
    if prediction_contract.get("required_model") != REQUIRED_FIXED_MODEL:
        errors.append(
            "settings.fixed_baselines.execution_contract.prediction_jsonl_contract.required_model "
            "must match Qwen/Qwen3-8B"
        )
    prediction_options = {str(item) for item in prediction_contract.get("prediction_field_options", [])}
    if not REQUIRED_PREDICTION_FIELD_OPTIONS <= prediction_options:
        errors.append(
            "settings.fixed_baselines.execution_contract.prediction_jsonl_contract.prediction_field_options "
            "must include prediction, answer, and response"
        )

    methods = fixed.get("methods", {})
    if not isinstance(methods, dict):
        errors.append("settings.fixed_baselines.methods must be an object")
        return errors
    for method in sorted(REQUIRED_METHODS):
        config = methods.get(method)
        if not isinstance(config, dict):
            errors.append(f"settings.fixed_baselines.methods.{method} is required")
            continue
        if "runner" in config:
            errors.append(
                f"settings.fixed_baselines.methods.{method}.runner must not point to a legacy script; "
                "final runs must provide per-method prediction JSONL files under the execution contract"
            )
        if "script" in config:
            errors.append(
                f"settings.fixed_baselines.methods.{method}.script must not define a final runner; "
                "use the execution contract and summary builder"
            )

    optional_methods = fixed.get("optional_methods", {})
    memgas_policy = optional_methods.get("MemGAS") if isinstance(optional_methods, dict) else None
    if not isinstance(memgas_policy, dict):
        errors.append("settings.fixed_baselines.optional_methods.MemGAS is required")
    else:
        for key, expected in OPTIONAL_MEMGAS_POLICY.items():
            if memgas_policy.get(key) != expected:
                errors.append(
                    f"settings.fixed_baselines.optional_methods.MemGAS.{key}="
                    f"{memgas_policy.get(key)!r} expected={expected!r}"
                )
    return errors


def settings_errors(report: dict[str, Any], summary_json: Path, expected_dataset: str) -> list[str]:
    errors: list[str] = []
    settings_file = report.get("settings_file")
    if not settings_file:
        return ["settings_file is required"]
    settings_path = Path(str(settings_file))
    if not settings_path.is_file():
        return [f"settings_file not found: {settings_path}"]

    expected_hash = sha256_file(settings_path)
    if report.get("settings_sha256") != expected_hash:
        errors.append("settings_sha256 does not match settings_file")

    settings = load_json(settings_path)
    if settings.get("status") != "predeclared":
        errors.append(f"settings.status={settings.get('status')!r} expected='predeclared'")
    errors.extend(anti_tuning_contract_errors(settings))
    errors.extend(fixed_baseline_execution_contract_errors(settings, expected_dataset))

    dataset_settings = settings.get("dataset", {})
    if dataset_settings.get("audited_primary") != expected_dataset:
        errors.append(
            f"settings.dataset.audited_primary={dataset_settings.get('audited_primary')!r} "
            f"expected={expected_dataset!r}"
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
    if required_methods != REQUIRED_METHODS:
        errors.append(f"settings required_methods={sorted(required_methods)} expected={sorted(REQUIRED_METHODS)}")
    common = fixed.get("common", {})
    if common.get("input_policy") != "conversation_only":
        errors.append("settings.fixed_baselines.common.input_policy must be conversation_only")
    if common.get("summary_visible") is not False:
        errors.append("settings.fixed_baselines.common.summary_visible must be false")
    cat5_metrics = {str(item) for item in common.get("cat5_metrics", [])}
    if cat5_metrics != {"refusal_accuracy", "unsupported_claim_rate"}:
        errors.append(
            "settings.fixed_baselines.common.cat5_metrics must be "
            "['refusal_accuracy', 'unsupported_claim_rate']"
        )
    report_group_by = {str(item) for item in settings.get("reporting", {}).get("group_by", [])}
    if report_group_by != REQUIRED_REPORT_GROUP_BY:
        errors.append(f"settings.reporting.group_by={sorted(report_group_by)} expected={sorted(REQUIRED_REPORT_GROUP_BY)}")
    return errors


def expected_metadata_index(dataset_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    dataset = load_json(dataset_path)
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in dataset:
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


def metric_metadata_errors(report: dict[str, Any], expected_counts: dict[str, int], dataset_path: Path) -> list[str]:
    errors: list[str] = []
    metadata_file = report.get("metric_metadata_file")
    if not metadata_file:
        return ["metric_metadata_file is required"]
    metadata_path = Path(str(metadata_file))
    if not metadata_path.is_file():
        return [f"metric_metadata_file not found: {metadata_path}"]
    if report.get("metric_metadata_sha256") != sha256_file(metadata_path):
        errors.append("metric_metadata_sha256 does not match metric_metadata_file")

    required_row_fields = {
        "source_dataset",
        "language",
        "sample_id",
        "qa_idx",
        "category",
        "answerable",
        "whether_cross_session",
        "evidence_provenance",
    }
    expected_rows = expected_metadata_index(dataset_path)
    seen_keys: set[tuple[str, int]] = set()
    row_count = 0
    missing_field_examples: list[str] = []
    alignment_examples: list[str] = []
    for row_count, row in enumerate(iter_jsonl(metadata_path), start=1):
        missing = sorted(required_row_fields - set(row))
        if missing and len(missing_field_examples) < 10:
            missing_field_examples.append(f"row {row_count} missing {missing}")
            continue
        try:
            key = (str(row["sample_id"]), int(row["qa_idx"]))
        except (TypeError, ValueError):
            if len(alignment_examples) < 10:
                alignment_examples.append(f"row {row_count} has invalid sample_id/qa_idx")
            continue
        if key in seen_keys:
            if len(alignment_examples) < 10:
                alignment_examples.append(f"row {row_count} duplicate key={key}")
            continue
        seen_keys.add(key)
        expected = expected_rows.get(key)
        if expected is None:
            if len(alignment_examples) < 10:
                alignment_examples.append(f"row {row_count} key={key} not found in dataset")
            continue
        for field in ("source_dataset", "language", "category", "answerable"):
            if row.get(field) != expected[field]:
                if len(alignment_examples) < 10:
                    alignment_examples.append(
                        f"row {row_count} key={key} {field}={row.get(field)!r} expected={expected[field]!r}"
                    )
                break
    if row_count != expected_counts["qa_count"]:
        errors.append(f"metric metadata rows={row_count} expected={expected_counts['qa_count']}")
    missing_keys = set(expected_rows) - seen_keys
    if missing_keys:
        errors.append(f"metric metadata missing dataset QA keys; first={sorted(missing_keys)[:10]}")
    errors.extend(missing_field_examples)
    errors.extend(alignment_examples)
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
    missing_methods = sorted(REQUIRED_METHODS - observed_methods)
    extra_methods = sorted(observed_methods - REQUIRED_METHODS)
    if missing_methods:
        errors.append(f"prediction_files missing required methods: {missing_methods}")
    if extra_methods:
        errors.append(f"prediction_files has unexpected methods: {extra_methods}")

    expected_keys = set(expected_metadata_index(dataset_path))
    expected_dataset_sha256 = sha256_file(dataset_path)
    for method in sorted(REQUIRED_METHODS & observed_methods):
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


def audited_dataset_errors(dataset_path: Path) -> list[str]:
    if dataset_path.name != REQUIRED_AUDITED_DATASET_FILENAME:
        return [
            f"dataset filename must be {REQUIRED_AUDITED_DATASET_FILENAME!r}; "
            f"got {dataset_path.name!r}. Final baseline results must use the human-audited eval file."
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--dataset", default=None, help="Expected dataset path recorded in the baseline summary.")
    args = parser.parse_args()

    if not args.summary_json.exists():
        return fail(f"summary not found: {args.summary_json}")

    report = load_json(args.summary_json)
    if report.get("status") != "completed":
        return fail(f"status must be completed, got {report.get('status')!r}")
    expected_dataset = args.dataset or report.get("dataset")
    if not expected_dataset:
        return fail("dataset is required")
    if report.get("dataset") != expected_dataset:
        return fail(f"dataset must be {expected_dataset!r}, got {report.get('dataset')!r}")
    dataset_path = Path(expected_dataset)
    if not dataset_path.is_file():
        return fail(f"dataset not found: {dataset_path}")
    audit_errors = audited_dataset_errors(dataset_path)
    if audit_errors:
        return fail("; ".join(audit_errors))
    expected_counts = dataset_counts(dataset_path)
    if not report.get("settings_source"):
        return fail("settings_source is required and must identify fixed/predeclared settings")
    if report.get("input_policy") != "conversation_only":
        return fail("input_policy must be 'conversation_only'")
    if report.get("summary_visible") is not False:
        return fail("summary_visible must be false for the main fixed-baseline table")
    if report.get("model") != REQUIRED_FIXED_MODEL:
        return fail(f"model must be {REQUIRED_FIXED_MODEL!r}, got {report.get('model')!r}")
    setting_errors = settings_errors(report, args.summary_json, expected_dataset)
    if setting_errors:
        return fail("; ".join(setting_errors))
    metadata_errors = metric_metadata_errors(report, expected_counts, dataset_path)
    if metadata_errors:
        return fail("; ".join(metadata_errors))
    prediction_errors = prediction_files_errors(report, dataset_path)
    if prediction_errors:
        return fail("; ".join(prediction_errors))
    expected_groups, has_cat5 = metadata_group_keys(Path(str(report["metric_metadata_file"])))

    methods = {str(item) for item in report.get("methods", [])}
    missing_methods = sorted(REQUIRED_METHODS - methods)
    if missing_methods:
        return fail(f"missing required methods: {missing_methods}")

    results = report.get("results")
    if not isinstance(results, list) or not results:
        return fail("results must be a non-empty list")

    methods_with_rows: set[str] = set()
    row_errors: list[str] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            row_errors.append(f"results[{index}] is not an object")
            continue
        missing_fields = sorted(REQUIRED_RESULT_FIELDS - set(row))
        if missing_fields:
            row_errors.append(f"results[{index}] missing fields: {missing_fields}")
            continue
        method = str(row["method"])
        methods_with_rows.add(method)
        if row["qa_count"] != expected_counts["qa_count"]:
            row_errors.append(f"{method}: qa_count={row['qa_count']} expected={expected_counts['qa_count']}")
        if row["answerable_qa_count"] != expected_counts["answerable_qa_count"]:
            row_errors.append(
                f"{method}: answerable_qa_count={row['answerable_qa_count']} "
                f"expected={expected_counts['answerable_qa_count']}"
            )
        if row["cat5_qa_count"] != expected_counts["cat5_qa_count"]:
            row_errors.append(
                f"{method}: cat5_qa_count={row['cat5_qa_count']} expected={expected_counts['cat5_qa_count']}"
            )
        for error in metric_object_errors(row["overall_answerable"], f"{method}: overall_answerable"):
            row_errors.append(error)
        for field in (
            "by_source_dataset",
            "by_language",
            "by_category",
            "by_cross_session",
            "by_evidence_provenance",
        ):
            for error in group_result_errors(row[field], f"{method}: {field}", expected_groups[field]):
                row_errors.append(error)
        if has_cat5:
            for field in ("cat5_refusal", "cat5_unsupported_claim"):
                for error in metric_object_errors(row[field], f"{method}: {field}"):
                    row_errors.append(error)

    missing_result_rows = sorted(REQUIRED_METHODS - methods_with_rows)
    if missing_result_rows:
        row_errors.append(f"missing result rows for methods: {missing_result_rows}")
    if row_errors:
        return fail("; ".join(row_errors))

    print(
        json.dumps(
            {
                "status": "passed",
                "summary_json": str(args.summary_json),
                "dataset": expected_dataset,
                "settings_file": report.get("settings_file"),
                "settings_sha256": report.get("settings_sha256"),
                "methods": sorted(methods_with_rows),
                **expected_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
