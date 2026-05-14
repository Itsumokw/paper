#!/usr/bin/env python3
"""Validate LoCoMo core acceptance artifacts against the local reproduction checklist."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASELINES = ("full_context", "amem", "mem0", "simplemem", "higmem", "memgas")
REQUIRED_TOP_LEVEL = (
    "overall_summary.json",
    "overall_summary.md",
    "leaderboard_f1.md",
    "leaderboard_efficiency.md",
    "failure_report.md",
)
REQUIRED_BASELINE_FILES = ("predictions.json", "metrics.json", "run.log", "status.json", "command.env", "diff_audit.md")
REQUIRED_METRIC_FIELDS = ("overall", "categories", "runtime", "tokens", "memory_retrieval", "reliability")
TEXT_METRICS = ("f1", "bleu1", "rouge_l", "bertscore_f1")
EXPECTED_QA = 1540


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("records", "per_item", "individual_results", "detailed_results", "results", "qa"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def command_value(text: str, key: str) -> str | None:
    matches = re.findall(rf"^{re.escape(key)}=(.*)$", text, flags=re.MULTILINE)
    return matches[-1].strip() if matches else None


def metric_mean(metrics: dict[str, Any], name: str) -> Any:
    value = (metrics.get("overall") or {}).get(name)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def prediction_text(record: dict[str, Any]) -> str:
    for key in ("prediction", "model_answer", "answer"):
        if key in record:
            return "" if record.get(key) is None else str(record.get(key))
    return ""


def validate(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    baseline_reports: dict[str, Any] = {}
    bertscore_models: set[str] = set()
    bertscore_devices: set[str] = set()
    bertscore_batches: set[str] = set()

    for name in REQUIRED_TOP_LEVEL:
        if not (root / name).exists():
            errors.append(f"missing top-level artifact: {name}")

    for baseline in BASELINES:
        bdir = root / baseline
        report: dict[str, Any] = {"path": str(bdir), "errors": [], "warnings": []}
        baseline_reports[baseline] = report
        if not bdir.is_dir():
            report["errors"].append("baseline artifact directory missing")
            errors.append(f"{baseline}: artifact directory missing")
            continue

        for name in REQUIRED_BASELINE_FILES:
            if not (bdir / name).exists():
                report["errors"].append(f"missing {name}")

        metrics_path = bdir / "metrics.json"
        predictions_path = bdir / "predictions.json"
        status_path = bdir / "status.json"
        command_path = bdir / "command.env"
        audit_path = bdir / "diff_audit.md"

        metrics: dict[str, Any] = {}
        status: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = read_json(metrics_path)
        if status_path.exists():
            status = read_json(status_path)

        if metrics:
            for field in REQUIRED_METRIC_FIELDS:
                if field not in metrics:
                    report["errors"].append(f"metrics.json missing field {field}")
            if metrics.get("status") != "completed":
                report["errors"].append(f"status is {metrics.get('status')!r}, not completed")
            if metrics.get("count") != EXPECTED_QA:
                report["errors"].append(f"metrics count {metrics.get('count')} != {EXPECTED_QA}")
            for metric in TEXT_METRICS:
                if metric_mean(metrics, metric) is None:
                    report["errors"].append(f"missing overall {metric}")
            categories = metrics.get("categories") or {}
            for category in ("1", "2", "3", "4"):
                if category not in categories:
                    report["errors"].append(f"missing category {category} metrics")
                    continue
                for metric in TEXT_METRICS:
                    value = categories.get(category, {}).get(metric)
                    mean = value.get("mean") if isinstance(value, dict) else value
                    if mean is None:
                        report["errors"].append(f"missing category {category} {metric}")
            reliability = metrics.get("reliability") or {}
            if reliability.get("fatal_error_count"):
                report["errors"].append(f"fatal_error_count={reliability.get('fatal_error_count')}")
            bertscore = metrics.get("bertscore") or {}
            model = bertscore.get("model")
            if model:
                bertscore_models.add(str(model))
            elif baseline != "full_context":
                report["warnings"].append("BERTScore model path not recorded in metrics.json")

        if predictions_path.exists():
            records = flatten_records(read_json(predictions_path))
            if len(records) != EXPECTED_QA:
                report["errors"].append(f"prediction rows {len(records)} != {EXPECTED_QA}")
            empty = [idx for idx, row in enumerate(records) if not prediction_text(row).strip()]
            if empty:
                report["errors"].append(f"empty predictions found: first rows {empty[:10]}")

        if status and status.get("status") != "completed":
            report["errors"].append(f"status.json status is {status.get('status')!r}")

        if command_path.exists():
            text = command_path.read_text(encoding="utf-8", errors="ignore")
            if "max_tokens=512" in text or "MAX_TOKENS=512" in text:
                report["errors"].append("command.env contains max_tokens=512-style setting")
            for key in ("BERTSCORE_MODEL", "BERTSCORE_DEVICE", "BERTSCORE_BATCH_SIZE"):
                value = command_value(text, key)
                if value is None:
                    report["warnings"].append(f"command.env missing {key}")
                elif key == "BERTSCORE_DEVICE":
                    bertscore_devices.add(value)
                elif key == "BERTSCORE_BATCH_SIZE":
                    bertscore_batches.add(value)
            if "VLLM_MAX_MODEL_LEN=" not in text and "LOCOMO_HIGH_UTIL_VLLM_MAX_MODEL_LEN=" not in text:
                report["warnings"].append("command.env missing vLLM max_model_len")

        if audit_path.exists():
            audit = audit_path.read_text(encoding="utf-8", errors="ignore")
            for phrase in ("Diff Audit", "Classification", "Semantic Impact", "Paper Disclosure"):
                if phrase not in audit:
                    report["errors"].append(f"diff_audit.md missing section/phrase: {phrase}")

        if report["errors"]:
            errors.extend(f"{baseline}: {msg}" for msg in report["errors"])
        if report["warnings"]:
            warnings.extend(f"{baseline}: {msg}" for msg in report["warnings"])

    if len(bertscore_models) > 1:
        errors.append(f"inconsistent BERTScore models: {sorted(bertscore_models)}")
    if len(bertscore_devices) > 1:
        errors.append(f"inconsistent BERTScore devices: {sorted(bertscore_devices)}")
    if len(bertscore_batches) > 1:
        errors.append(f"inconsistent BERTScore batch sizes: {sorted(bertscore_batches)}")

    return {
        "root": str(root),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "baselines": baseline_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = validate(args.artifact_root)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
