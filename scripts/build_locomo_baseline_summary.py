#!/usr/bin/env python3
"""Build the fixed-baseline summary.json from per-method prediction JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REQUIRED_METHODS = ["Full Context", "A-MEM", "Mem0", "SimpleMem", "HiGMem"]
REQUIRED_FIXED_MODEL = "Qwen/Qwen3-8B"
REFUSAL_PATTERNS = (
    "not enough information",
    "cannot answer",
    "can't answer",
    "unsupported",
    "unknown",
    "insufficient information",
    "no information",
    "not mentioned",
    "无法回答",
    "无法确定",
    "信息不足",
    "没有足够",
    "未提及",
    "不清楚",
    "不知道",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def metric_tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    cjk_chars = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    if cjk_chars:
        ascii_words = re.findall(r"[a-z0-9]+", text)
        return cjk_chars + ascii_words
    return text.split()


def token_f1(prediction: Any, reference: Any) -> float:
    pred = set(metric_tokens(prediction))
    ref = set(metric_tokens(reference))
    if not pred or not ref:
        return 0.0
    common = pred & ref
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_refusal(prediction: Any) -> bool:
    text = normalize_text(prediction)
    if not text:
        return False
    return any(pattern in text for pattern in REFUSAL_PATTERNS)


def dataset_index(dataset: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in dataset:
        sample_id = str(sample.get("sample_id"))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            category = int(qa.get("category"))
            indexed[(sample_id, qa_idx)] = {
                "source_dataset": str(sample.get("source_dataset")),
                "language": str(sample.get("language")),
                "category": category,
                "answerable": category != 5,
                "reference": str(
                    qa.get("adversarial_answer" if category == 5 else "answer", "")
                ),
            }
    return indexed


def metadata_index(metadata_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in iter_jsonl(metadata_path):
        key = (str(row.get("sample_id")), int(row.get("qa_idx")))
        indexed[key] = row
    return indexed


def parse_method_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("method cannot be empty")
    return method, Path(path)


def prediction_key(row: dict[str, Any], row_idx: int, errors: list[str], method: str) -> tuple[str, int] | None:
    try:
        return (str(row["sample_id"]), int(row["qa_idx"]))
    except (KeyError, TypeError, ValueError):
        errors.append(f"{method}: prediction row {row_idx} missing valid sample_id/qa_idx")
        return None


def load_predictions(
    method: str,
    path: Path,
    expected_keys: set[tuple[str, int]],
    expected_dataset_sha256: str,
    errors: list[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{method}: prediction file missing: {path}")
        return {}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row_idx, row in enumerate(iter_jsonl(path), start=1):
        key = prediction_key(row, row_idx, errors, method)
        if key is None:
            continue
        if key in rows:
            errors.append(f"{method}: duplicate prediction key={key}")
            continue
        if key not in expected_keys:
            errors.append(f"{method}: prediction key={key} not found in dataset")
            continue
        if "prediction" not in row and "answer" not in row and "response" not in row:
            errors.append(f"{method}: prediction key={key} missing prediction/answer/response field")
            continue
        if row.get("model") != REQUIRED_FIXED_MODEL:
            errors.append(
                f"{method}: prediction key={key} model={row.get('model')!r} "
                f"expected={REQUIRED_FIXED_MODEL!r}"
            )
            continue
        if row.get("dataset_sha256") != expected_dataset_sha256:
            errors.append(
                f"{method}: prediction key={key} dataset_sha256={row.get('dataset_sha256')!r} "
                f"expected={expected_dataset_sha256!r}"
            )
            continue
        rows[key] = row
    missing = sorted(expected_keys - set(rows))
    if missing:
        errors.append(f"{method}: missing predictions for {len(missing)} QA; first={missing[:10]}")
    return rows


def prediction_text(row: dict[str, Any]) -> str:
    for field in ("prediction", "answer", "response"):
        if field in row:
            return str(row.get(field, ""))
    return ""


def mean_metric(scores: list[float]) -> dict[str, Any]:
    return {
        "count": len(scores),
        "mean_token_f1": round(mean(scores), 6) if scores else 0.0,
    }


def grouped_metrics(records: list[dict[str, Any]], group_field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_field])].append(float(record["token_f1"]))
    return {key: mean_metric(values) for key, values in sorted(grouped.items())}


def method_summary(
    method: str,
    predictions: dict[tuple[str, int], dict[str, Any]],
    dataset_rows: dict[tuple[str, int], dict[str, Any]],
    metadata_rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    answerable_records: list[dict[str, Any]] = []
    cat5_records: list[dict[str, Any]] = []
    for key, dataset_row in sorted(dataset_rows.items()):
        metadata = metadata_rows[key]
        pred = prediction_text(predictions[key])
        if dataset_row["answerable"]:
            answerable_records.append(
                {
                    "token_f1": token_f1(pred, dataset_row["reference"]),
                    "source_dataset": metadata["source_dataset"],
                    "language": metadata["language"],
                    "category": str(metadata["category"]),
                    "whether_cross_session": str(metadata["whether_cross_session"]).lower(),
                    "evidence_provenance": str(metadata["evidence_provenance"]),
                }
            )
        else:
            refused = is_refusal(pred)
            cat5_records.append({"refused": refused, "unsupported_claim": not refused})

    refusal_count = sum(1 for row in cat5_records if row["refused"])
    unsupported_count = sum(1 for row in cat5_records if row["unsupported_claim"])
    cat5_total = len(cat5_records)
    return {
        "method": method,
        "qa_count": len(dataset_rows),
        "answerable_qa_count": len(answerable_records),
        "cat5_qa_count": cat5_total,
        "overall_answerable": mean_metric([float(row["token_f1"]) for row in answerable_records]),
        "by_source_dataset": grouped_metrics(answerable_records, "source_dataset"),
        "by_language": grouped_metrics(answerable_records, "language"),
        "by_category": grouped_metrics(answerable_records, "category"),
        "by_cross_session": grouped_metrics(answerable_records, "whether_cross_session"),
        "by_evidence_provenance": grouped_metrics(answerable_records, "evidence_provenance"),
        "cat5_refusal": {
            "count": cat5_total,
            "refused": refusal_count,
            "accuracy": round(refusal_count / cat5_total, 6) if cat5_total else 0.0,
        },
        "cat5_unsupported_claim": {
            "count": cat5_total,
            "unsupported_claims": unsupported_count,
            "rate": round(unsupported_count / cat5_total, 6) if cat5_total else 0.0,
        },
    }


def metadata_alignment_errors(
    dataset_rows: dict[tuple[str, int], dict[str, Any]],
    metadata_rows: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    missing = sorted(set(dataset_rows) - set(metadata_rows))
    extra = sorted(set(metadata_rows) - set(dataset_rows))
    if missing:
        errors.append(f"metric metadata missing dataset QA keys; first={missing[:10]}")
    if extra:
        errors.append(f"metric metadata has extra QA keys; first={extra[:10]}")
    for key in sorted(set(dataset_rows) & set(metadata_rows)):
        expected = dataset_rows[key]
        observed = metadata_rows[key]
        for field in ("source_dataset", "language", "category", "answerable"):
            if observed.get(field) != expected[field]:
                errors.append(f"metadata key={key} {field}={observed.get(field)!r} expected={expected[field]!r}")
                break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metric-metadata", type=Path, required=True)
    parser.add_argument("--settings-file", type=Path, default=Path("datasets/locomo_style_eval/fixed_eval_settings.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/baseline_results/summary.json"))
    parser.add_argument(
        "--prediction-jsonl",
        action="append",
        type=parse_method_prediction,
        default=[],
        metavar="METHOD=PATH",
        help="Per-method prediction JSONL. Each row needs sample_id, qa_idx, and prediction/answer/response.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.dataset.is_file():
        errors.append(f"dataset missing: {args.dataset}")
        dataset = []
    else:
        dataset = load_json(args.dataset)
    if not args.metric_metadata.is_file():
        errors.append(f"metric metadata missing: {args.metric_metadata}")
        metadata_rows = {}
    else:
        metadata_rows = metadata_index(args.metric_metadata)
    if not args.settings_file.is_file():
        errors.append(f"settings file missing: {args.settings_file}")
        settings = {}
    else:
        settings = load_json(args.settings_file)
    model = (
        settings.get("fixed_baselines", {})
        .get("execution_contract", {})
        .get("model")
        or settings.get("model", {}).get("served_model")
    )
    if model != REQUIRED_FIXED_MODEL:
        errors.append(f"settings model={model!r} expected={REQUIRED_FIXED_MODEL!r}")

    dataset_rows = dataset_index(dataset) if isinstance(dataset, list) else {}
    dataset_sha256 = sha256_file(args.dataset) if args.dataset.is_file() else ""
    errors.extend(metadata_alignment_errors(dataset_rows, metadata_rows))

    method_paths = dict(args.prediction_jsonl)
    missing_methods = sorted(set(REQUIRED_METHODS) - set(method_paths))
    extra_methods = sorted(set(method_paths) - set(REQUIRED_METHODS))
    if missing_methods:
        errors.append(f"missing prediction files for required methods: {missing_methods}")
    if extra_methods:
        errors.append(f"unexpected prediction methods: {extra_methods}")

    results: list[dict[str, Any]] = []
    prediction_files: dict[str, dict[str, str]] = {}
    for method in REQUIRED_METHODS:
        path = method_paths.get(method)
        if path is None:
            continue
        if path.is_file():
            prediction_files[method] = {"path": str(path), "sha256": sha256_file(path)}
        predictions = load_predictions(method, path, set(dataset_rows), dataset_sha256, errors)
        if set(predictions) == set(dataset_rows) and set(dataset_rows) <= set(metadata_rows):
            results.append(method_summary(method, predictions, dataset_rows, metadata_rows))

    report = {
        "status": "completed" if not errors else "failed",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256 if args.dataset.is_file() else None,
        "model": model,
        "settings_source": "predeclared_fixed_eval_settings",
        "settings_file": str(args.settings_file),
        "settings_sha256": sha256_file(args.settings_file) if args.settings_file.is_file() else None,
        "metric_metadata_file": str(args.metric_metadata),
        "metric_metadata_sha256": sha256_file(args.metric_metadata) if args.metric_metadata.is_file() else None,
        "prediction_files": prediction_files,
        "input_policy": "conversation_only",
        "summary_visible": False,
        "methods": REQUIRED_METHODS,
        "results": results,
        "errors": errors,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
