#!/usr/bin/env python3
"""Audit and summarize multilingual LoCoMo-style 3B/8B baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, Any]] = {
    "perltqa": {
        "label": "PerLTQA",
        "language": "zh",
        "samples": 10,
        "cat14_qa": 320,
        "source": "datasets/locomo_style_eval_repaired_20260513/primary/PerLTQA-LoCoMo-style-eval.json",
    },
    "opela": {
        "label": "OPELA",
        "language": "ko",
        "samples": 10,
        "cat14_qa": 200,
        "source": "datasets/locomo_style_eval_repaired_20260513/primary/OPELA-LoCoMo-style-eval.json",
    },
    "jlongchat": {
        "label": "JLongChat",
        "language": "ja",
        "samples": 10,
        "cat14_qa": 200,
        "source": "datasets/locomo_style_eval_repaired_20260513/primary/JLongChat-LoCoMo-style-eval.json",
    },
    "del1l2im": {
        "label": "deL1L2IM",
        "language": "de",
        "samples": 9,
        "cat14_qa": 180,
        "source": "datasets/locomo_style_eval_repaired_20260513/primary/deL1L2IM-LoCoMo-style-eval.json",
    },
}

MODELS: dict[str, dict[str, str]] = {
    "qwen25_3b": {"label": "Qwen2.5-3B", "openai_model": "Qwen/Qwen2.5-3B-Instruct"},
    "qwen3_8b": {"label": "Qwen3-8B", "openai_model": "Qwen/Qwen3-8B"},
}

METHODS: dict[str, dict[str, str]] = {
    "full_context": {
        "label": "Full Context",
        "pred": "full_context_fixed/full_context/full_context_predictions_flat.json",
        "metrics": "full_context_fixed/full_context/full_context_metrics_cat1_4.json",
        "judge": "full_context_fixed/full_context/full_context_judge_metrics_cat1_4.json",
    },
    "amem": {
        "label": "A-MEM",
        "pred": "amem_official/official_predictions_cat1_4.json",
        "metrics": "amem_official/official_metrics_cat1_4.json",
        "judge": "amem_official/official_judge_metrics_cat1_4.json",
    },
    "mem0": {
        "label": "Mem0",
        "pred": "mem0_fixed/mem0/mem0_predictions_flat.json",
        "metrics": "mem0_fixed/mem0/mem0_metrics_cat1_4.json",
        "judge": "mem0_fixed/mem0/mem0_judge_metrics_cat1_4.json",
    },
    "simplemem": {
        "label": "SimpleMem",
        "pred": "simplemem/normalized_predictions_cat1_4.json",
        "metrics": "simplemem/simplemem_metrics_cat1_4.json",
        "judge": "simplemem/simplemem_judge_metrics_cat1_4.json",
    },
    "higmem": {
        "label": "HiGMem",
        "pred": "higmem/normalized_predictions_cat1_4.json",
        "metrics": "higmem/higmem_metrics_cat1_4.json",
        "judge": "higmem/higmem_judge_metrics_cat1_4.json",
    },
    "memgas": {
        "label": "MemGAS",
        "pred": "memgas/normalized_predictions_cat1_4.json",
        "metrics": "memgas/normalized_metrics_cat1_4.json",
        "judge": "memgas/normalized_judge_metrics_cat1_4.json",
    },
}

METRIC_KEYS = ("f1", "bleu1", "rouge_l", "bertscore_f1")
JUDGE_KEYS = ("judge_score", "judge_correct", "judge_acceptable")
PREDICTION_KEYS = ("prediction", "model_answer", "answer")


@dataclass
class PredictionAudit:
    count: int | None
    empty_predictions: int | None
    error: str | None = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_records(data: Any) -> list[Any] | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(data, dict) and isinstance(data.get("individual_results"), list):
        return data["individual_results"]
    if isinstance(data, dict) and isinstance(data.get("detailed_results"), list):
        return data["detailed_results"]
    return None


def audit_predictions(path: Path) -> PredictionAudit:
    if not path.exists():
        return PredictionAudit(None, None, "missing predictions")
    try:
        records = get_records(load_json(path))
    except Exception as exc:  # noqa: BLE001
        return PredictionAudit(None, None, f"invalid prediction JSON: {exc}")
    if records is None:
        return PredictionAudit(None, None, "prediction JSON has no records/list")
    empty = 0
    for row in records:
        if not isinstance(row, dict):
            empty += 1
            continue
        value = ""
        for key in PREDICTION_KEYS:
            if key in row:
                value = str(row.get(key) or "")
                break
        if not value.strip():
            empty += 1
    return PredictionAudit(len(records), empty)


def metric_mean_and_count(metrics: dict[str, Any], key: str) -> tuple[float | None, int | None]:
    overall = metrics.get("overall") if isinstance(metrics, dict) else None
    if not isinstance(overall, dict):
        return None, None
    value = overall.get(key)
    if not isinstance(value, dict):
        return None, None
    mean = value.get("mean")
    count = value.get("count")
    try:
        mean_out = float(mean)
    except (TypeError, ValueError):
        mean_out = None
    try:
        count_out = int(count)
    except (TypeError, ValueError):
        count_out = None
    return mean_out, count_out


def audit_metrics(path: Path) -> tuple[dict[str, float | None], dict[str, int | None], str | None]:
    if not path.exists():
        return {}, {}, "missing metrics"
    try:
        metrics = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {}, {}, f"invalid metrics JSON: {exc}"
    values: dict[str, float | None] = {}
    counts: dict[str, int | None] = {}
    for key in METRIC_KEYS:
        values[key], counts[key] = metric_mean_and_count(metrics, key)
    missing = [key for key in METRIC_KEYS if values.get(key) is None or counts.get(key) is None]
    if missing:
        return values, counts, f"missing metric fields: {','.join(missing)}"
    return values, counts, None


def audit_judge_metrics(
    path: Path,
    expected_protocol: str | None,
) -> tuple[dict[str, float | None], dict[str, int | None], int | None, str | None]:
    if not path.exists():
        return {}, {}, None, "missing judge metrics"
    try:
        metrics = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {}, {}, None, f"invalid judge JSON: {exc}"
    overall = metrics.get("overall") if isinstance(metrics, dict) else None
    if not isinstance(overall, dict):
        return {}, {}, None, "judge JSON missing overall"
    judge_info = metrics.get("judge") if isinstance(metrics, dict) else None
    protocol = judge_info.get("protocol") if isinstance(judge_info, dict) else None
    if expected_protocol and protocol != expected_protocol:
        return {}, {}, None, f"judge protocol {protocol!r} != {expected_protocol!r}"
    values: dict[str, float | None] = {}
    counts: dict[str, int | None] = {}
    for key in JUDGE_KEYS:
        values[key], counts[key] = metric_mean_and_count(metrics, key)
    missing = [key for key in JUDGE_KEYS if values.get(key) is None or counts.get(key) is None]
    judge_errors_raw = overall.get("judge_errors")
    try:
        judge_errors = int(judge_errors_raw)
    except (TypeError, ValueError):
        judge_errors = None
        missing.append("judge_errors")
    if missing:
        return values, counts, judge_errors, f"missing judge fields: {','.join(missing)}"
    if judge_errors:
        return values, counts, judge_errors, f"judge errors {judge_errors}"
    return values, counts, judge_errors, None


def iter_keys(selected: list[str] | None, all_keys: list[str]) -> list[str]:
    if not selected:
        return all_keys
    unknown = sorted(set(selected) - set(all_keys))
    if unknown:
        raise SystemExit(f"Unknown selection: {', '.join(unknown)}")
    return selected


def audit_run(
    run_root: Path,
    model_keys: list[str],
    dataset_keys: list[str],
    method_keys: list[str],
    skip_judge: bool,
    judge_protocol: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        for dataset_key in dataset_keys:
            expected = int(DATASETS[dataset_key]["cat14_qa"])
            for method_key in method_keys:
                method = METHODS[method_key]
                task_root = run_root / model_key / dataset_key
                pred_path = task_root / method["pred"]
                metrics_path = task_root / method["metrics"]
                judge_path = task_root / method["judge"]
                pred = audit_predictions(pred_path)
                metric_values, metric_counts, metric_error = audit_metrics(metrics_path)
                judge_values: dict[str, float | None] = {}
                judge_counts: dict[str, int | None] = {}
                judge_errors: int | None = None
                judge_error: str | None = None
                if not skip_judge:
                    judge_values, judge_counts, judge_errors, judge_error = audit_judge_metrics(
                        judge_path,
                        judge_protocol,
                    )

                reasons = []
                if pred.error:
                    reasons.append(pred.error)
                if pred.count != expected:
                    reasons.append(f"prediction count {pred.count} != {expected}")
                if pred.empty_predictions not in (0, None):
                    reasons.append(f"empty predictions {pred.empty_predictions}")
                if metric_error:
                    reasons.append(metric_error)
                bad_metric_counts = [
                    f"{key}:{metric_counts.get(key)}"
                    for key in METRIC_KEYS
                    if metric_counts.get(key) != expected
                ]
                if bad_metric_counts:
                    reasons.append(f"metric counts != {expected}: {','.join(bad_metric_counts)}")
                if judge_error:
                    reasons.append(judge_error)
                if not skip_judge:
                    bad_judge_counts = [
                        f"{key}:{judge_counts.get(key)}"
                        for key in JUDGE_KEYS
                        if judge_counts.get(key) != expected
                    ]
                    if bad_judge_counts:
                        reasons.append(f"judge counts != {expected}: {','.join(bad_judge_counts)}")

                complete = not reasons
                row: dict[str, Any] = {
                    "status": "complete" if complete else "incomplete",
                    "reason": "; ".join(reasons),
                    "model": model_key,
                    "model_label": MODELS[model_key]["label"],
                    "dataset": dataset_key,
                    "dataset_label": DATASETS[dataset_key]["label"],
                    "language": DATASETS[dataset_key]["language"],
                    "method": method_key,
                    "method_label": method["label"],
                    "expected_cat14_qa": expected,
                    "prediction_count": pred.count,
                    "empty_predictions": pred.empty_predictions,
                    "prediction_path": str(pred_path),
                    "metrics_path": str(metrics_path),
                    "judge_path": str(judge_path),
                    "judge_errors": judge_errors,
                }
                for key in METRIC_KEYS:
                    row[key] = metric_values.get(key)
                    row[f"{key}_count"] = metric_counts.get(key)
                for key in JUDGE_KEYS:
                    row[key] = judge_values.get(key)
                    row[f"{key}_count"] = judge_counts.get(key)
                rows.append(row)
    return rows


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["status"] == "complete"]
    payload = {
        "run_root": str(run_root),
        "complete": len(completed),
        "total": len(rows),
        "ok": len(completed) == len(rows),
        "rows": rows,
    }
    (summary_dir / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "status",
        "model",
        "dataset",
        "language",
        "method",
        "expected_cat14_qa",
        "prediction_count",
        "f1",
        "bleu1",
        "rouge_l",
        "bertscore_f1",
        "judge_score",
        "judge_correct",
        "judge_acceptable",
        "reason",
        "metrics_path",
        "judge_path",
    ]
    with (summary_dir / "results_f1_desc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["status"] != "complete", -(r.get("f1") or -1.0))):
            writer.writerow({col: row.get(col) for col in columns})

    md_lines = [
        "| Model | Dataset | Lang | Method | F1 | BLEU-1 | ROUGE-L | Judge | Judge OK | Count | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["status"] != "complete", -(r.get("f1") or -1.0))):
        fmt = lambda value: "" if value is None else f"{value:.4f}"  # noqa: E731
        md_lines.append(
            "| {model} | {dataset} | {lang} | {method} | {f1} | {bleu1} | {rouge_l} | {judge} | {judge_ok} | {count} | {status} |".format(
                model=row["model_label"],
                dataset=row["dataset_label"],
                lang=row["language"],
                method=row["method_label"],
                f1=fmt(row.get("f1")),
                bleu1=fmt(row.get("bleu1")),
                rouge_l=fmt(row.get("rouge_l")),
                judge=fmt(row.get("judge_score")),
                judge_ok=fmt(row.get("judge_acceptable")),
                count=row.get("prediction_count") if row.get("prediction_count") is not None else "",
                status=row["status"],
            )
        )
    (summary_dir / "results_f1_desc.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model", action="append", choices=sorted(MODELS))
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--method", action="append", choices=sorted(METHODS))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--skip-judge", action="store_true", help="Do not require LLM-as-a-judge metrics")
    parser.add_argument(
        "--judge-protocol",
        default="locomo_binary",
        help="Required judge protocol when judge metrics are enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_keys = iter_keys(args.model, list(MODELS))
    dataset_keys = iter_keys(args.dataset, list(DATASETS))
    method_keys = iter_keys(args.method, list(METHODS))
    rows = audit_run(
        args.run_root,
        model_keys,
        dataset_keys,
        method_keys,
        args.skip_judge,
        None if args.skip_judge else args.judge_protocol,
    )
    complete = sum(1 for row in rows if row["status"] == "complete")
    if not args.no_write:
        write_outputs(args.run_root, rows)
    if not args.quiet:
        print(f"complete={complete}/{len(rows)} run_root={args.run_root}")
        for row in rows:
            if row["status"] != "complete":
                print(
                    "INCOMPLETE {model}/{dataset}/{method}: {reason}".format(
                        model=row["model"],
                        dataset=row["dataset"],
                        method=row["method"],
                        reason=row["reason"],
                    )
                )
    if args.fail_on_incomplete and complete != len(rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
