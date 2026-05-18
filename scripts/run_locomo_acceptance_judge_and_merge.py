#!/usr/bin/env python3
"""Backfill LoCoMo core baseline judge metrics and merge them with the 48 grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/stu0032/paper")
EXPECTED_LOCOMO_QA = 1540
JUDGE_PROTOCOL = "locomo_binary"
JUDGE_PROTOCOL_VERSION = "lightmem-locomo-judge-v1"

TEXT_METRICS = ("f1", "bleu1", "rouge_l", "bertscore_f1")
JUDGE_METRICS = ("judge_score", "judge_correct", "judge_acceptable")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    openai_model: str
    root: Path
    model_path: Path
    alt_served_model: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    prediction_key: str
    reference_key: str


MODELS = (
    ModelSpec(
        key="qwen25_3b",
        label="Qwen2.5-3B",
        openai_model="Qwen/Qwen2.5-3B-Instruct",
        root=REPO_ROOT / "runs/locomo_core_acceptance_qwen25_3b_32000_qa8192/20260512_122426/acceptance_artifacts",
        model_path=REPO_ROOT / "models/Qwen2.5-3B-Instruct-clean",
        alt_served_model="Qwen2.5-3B-Instruct",
    ),
    ModelSpec(
        key="qwen3_8b",
        label="Qwen3-8B",
        openai_model="Qwen/Qwen3-8B",
        root=REPO_ROOT / "runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/acceptance_artifacts",
        model_path=REPO_ROOT / "models/Qwen3-8B",
        alt_served_model="Qwen3-8B",
    ),
)

METHODS = (
    MethodSpec("full_context", "Full Context", "model_answer", "golden_answer"),
    MethodSpec("amem", "A-MEM", "prediction", "reference"),
    MethodSpec("mem0", "Mem0", "model_answer", "golden_answer"),
    MethodSpec("simplemem", "SimpleMem", "prediction", "reference"),
    MethodSpec("higmem", "HiGMem", "prediction", "reference"),
    MethodSpec("memgas", "MemGAS", "prediction", "reference"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("records", "per_item", "individual_results", "detailed_results", "results", "qa"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def metric_mean_and_count(data: dict[str, Any], key: str) -> tuple[float | None, int | None]:
    overall = data.get("overall") if isinstance(data, dict) else None
    value = overall.get(key) if isinstance(overall, dict) else None
    if not isinstance(value, dict):
        return None, None
    try:
        mean = float(value.get("mean"))
    except (TypeError, ValueError):
        mean = None
    try:
        count = int(value.get("count"))
    except (TypeError, ValueError):
        count = None
    return mean, count


def judge_valid(path: Path, expected_count: int = EXPECTED_LOCOMO_QA) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing judge metrics"
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid judge JSON: {exc}"
    judge = data.get("judge") if isinstance(data, dict) else None
    if not isinstance(judge, dict):
        return False, "judge JSON missing judge metadata"
    if judge.get("protocol") != JUDGE_PROTOCOL:
        return False, f"judge protocol {judge.get('protocol')!r} != {JUDGE_PROTOCOL!r}"
    if judge.get("protocol_version") != JUDGE_PROTOCOL_VERSION:
        return False, "judge protocol version mismatch"
    overall = data.get("overall") if isinstance(data, dict) else None
    if not isinstance(overall, dict):
        return False, "judge JSON missing overall"
    if int(overall.get("judge_errors") or 0) != 0:
        return False, f"judge errors {overall.get('judge_errors')}"
    if int(data.get("count") or 0) != expected_count:
        return False, f"judge top-level count {data.get('count')} != {expected_count}"
    for key in JUDGE_METRICS:
        _, count = metric_mean_and_count(data, key)
        if count != expected_count:
            return False, f"{key} count {count} != {expected_count}"
    return True, ""


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def served_model_ready(openai_model: str, base_url: str) -> bool:
    code = (
        "from openai import OpenAI\n"
        "import os, sys\n"
        "client=OpenAI(api_key='EMPTY', base_url=os.environ['BASE_URL'], timeout=5, max_retries=0)\n"
        "models=[m.id for m in client.models.list().data]\n"
        "sys.exit(0 if os.environ['OPENAI_MODEL'] in models else 2)\n"
    )
    env = os.environ.copy()
    env["BASE_URL"] = base_url
    env["OPENAI_MODEL"] = openai_model
    result = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/python"), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_vllm(model: ModelSpec, args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    if args.no_start_vllm and served_model_ready(model.openai_model, base_url):
        return
    if args.no_start_vllm:
        raise SystemExit(f"{model.openai_model} is not served at {base_url}")
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_BASE_URL": base_url,
            "VLLM_MODEL_PATH": str(model.model_path),
            "VLLM_SERVED_MODEL": model.openai_model,
            "VLLM_ALT_SERVED_MODEL": model.alt_served_model,
            "VLLM_LOG_DIR": str(args.output_dir / "vllm_logs"),
            "VLLM_FORCE_RESTART": "0" if served_model_ready(model.openai_model, base_url) else "1",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.90" if model.key == "qwen3_8b" else "0.88",
            "VLLM_MAX_MODEL_LEN": str(args.vllm_max_model_len),
            "VLLM_MAX_NUM_SEQS": str(args.vllm_max_num_seqs),
            "VLLM_MAX_NUM_BATCHED_TOKENS": str(args.vllm_max_num_batched_tokens),
            "VLLM_GENERATION_CONFIG": "vllm",
            "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS": '{"enable_thinking": false}',
        }
    )
    if model.key == "qwen25_3b":
        env["VLLM_EXTRA_ARGS"] = "--enforce-eager"
    run(["bash", "scripts/start_vllm_qwen25_3b.sh"], env=env)


def run_judge_for(model: ModelSpec, method: MethodSpec, args: argparse.Namespace) -> None:
    method_dir = model.root / method.key
    prediction_path = method_dir / "predictions.json"
    output_path = method_dir / "judge_metrics.json"
    ok, reason = judge_valid(output_path)
    if ok:
        print(f"SKIP judge complete: {model.key}/{method.key}", flush=True)
        return
    print(f"RUN judge: {model.key}/{method.key} ({reason})", flush=True)
    if not prediction_path.exists():
        raise SystemExit(f"Missing predictions: {prediction_path}")
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": args.api_key,
            "OPENAI_BASE_URL": args.base_url.rstrip("/"),
            "LLM_JUDGE_WORKERS": str(args.workers),
        }
    )
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        "scripts/compute_locomo_llm_judge_metrics.py",
        "--input",
        str(prediction_path),
        "--output",
        str(output_path),
        "--prediction-key",
        method.prediction_key,
        "--reference-key",
        method.reference_key,
        "--question-key",
        "question",
        "--category-key",
        "category",
        "--model",
        model.openai_model,
        "--base-url",
        args.base_url.rstrip("/"),
        "--api-key",
        args.api_key,
        "--protocol",
        JUDGE_PROTOCOL,
        "--max-workers",
        str(args.workers),
        "--max-retries",
        str(args.max_retries),
        "--resume",
        "--fail-on-error",
    ]
    log_path = method_dir / "judge_metrics.log"
    print("+", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    ok, reason = judge_valid(output_path)
    if not ok:
        raise SystemExit(f"Judge output invalid for {model.key}/{method.key}: {reason}")


def locomo_row(model: ModelSpec, method: MethodSpec) -> dict[str, Any]:
    method_dir = model.root / method.key
    pred_path = method_dir / "predictions.json"
    metrics_path = method_dir / "metrics.json"
    judge_path = method_dir / "judge_metrics.json"
    reasons: list[str] = []
    pred_count: int | None = None
    empty_predictions: int | None = None
    if pred_path.exists():
        records = flatten_records(load_json(pred_path))
        pred_count = len(records)
        empty_predictions = sum(1 for row in records if not str(row.get(method.prediction_key) or "").strip())
    else:
        reasons.append("missing predictions")
    metrics: dict[str, Any] = load_json(metrics_path) if metrics_path.exists() else {}
    if not metrics:
        reasons.append("missing text metrics")
    judge: dict[str, Any] = load_json(judge_path) if judge_path.exists() else {}
    judge_ok, judge_reason = judge_valid(judge_path)
    if not judge_ok:
        reasons.append(judge_reason)
    row: dict[str, Any] = {
        "suite": "locomo_core",
        "model": model.key,
        "model_label": model.label,
        "dataset": "locomo",
        "dataset_label": "LoCoMo",
        "language": "en",
        "method": method.key,
        "method_label": method.label,
        "expected_count": EXPECTED_LOCOMO_QA,
        "prediction_count": pred_count,
        "empty_predictions": empty_predictions,
        "status": "complete" if not reasons and pred_count == EXPECTED_LOCOMO_QA and empty_predictions == 0 else "incomplete",
        "reason": "; ".join(reasons),
        "prediction_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "judge_path": str(judge_path),
    }
    for key in TEXT_METRICS:
        row[key], row[f"{key}_count"] = metric_mean_and_count(metrics, key)
    for key in JUDGE_METRICS:
        row[key], row[f"{key}_count"] = metric_mean_and_count(judge, key)
    overall = judge.get("overall") if isinstance(judge, dict) else {}
    row["judge_errors"] = overall.get("judge_errors") if isinstance(overall, dict) else None
    if pred_count != EXPECTED_LOCOMO_QA:
        row["status"] = "incomplete"
        row["reason"] = (row["reason"] + "; " if row["reason"] else "") + f"prediction count {pred_count} != {EXPECTED_LOCOMO_QA}"
    return row


def load_48_rows(run_root: Path) -> list[dict[str, Any]]:
    audit_path = run_root / "summary/audit.json"
    audit = load_json(audit_path)
    rows = []
    for row in audit.get("rows", []):
        out = dict(row)
        out["suite"] = "multilingual_48"
        out["expected_count"] = out.get("expected_cat14_qa")
        out.setdefault("judge_errors", None)
        rows.append(out)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "suite",
        "status",
        "model",
        "model_label",
        "dataset",
        "dataset_label",
        "language",
        "method",
        "method_label",
        "expected_count",
        "prediction_count",
        "empty_predictions",
        "f1",
        "bleu1",
        "rouge_l",
        "bertscore_f1",
        "judge_score",
        "judge_correct",
        "judge_acceptable",
        "judge_errors",
        "reason",
        "metrics_path",
        "judge_path",
        "prediction_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Suite | Model | Dataset | Method | F1 | BLEU-1 | ROUGE-L | BERTScore-F1 | Judge score | Judge correct | Count | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {model} | {dataset} | {method} | {f1} | {bleu1} | {rouge_l} | {bertscore} | {judge} | {judge_correct} | {count} | {status} |".format(
                suite=row.get("suite", ""),
                model=row.get("model_label") or row.get("model", ""),
                dataset=row.get("dataset_label") or row.get("dataset", ""),
                method=row.get("method_label") or row.get("method", ""),
                f1=fmt(row.get("f1")),
                bleu1=fmt(row.get("bleu1")),
                rouge_l=fmt(row.get("rouge_l")),
                bertscore=fmt(row.get("bertscore_f1")),
                judge=fmt(row.get("judge_score")),
                judge_correct=fmt(row.get("judge_correct")),
                count=fmt(row.get("prediction_count")),
                status=row.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    locomo_rows = [locomo_row(model, method) for model in MODELS for method in METHODS]
    rows_48 = load_48_rows(args.multilingual_48_root)
    combined_rows = locomo_rows + rows_48

    locomo_rows_sorted = sorted(locomo_rows, key=lambda r: (r["model"], r["method"]))
    rows_48_sorted = sorted(rows_48, key=lambda r: (r.get("model", ""), r.get("dataset", ""), r.get("method", "")))
    combined_sorted = sorted(combined_rows, key=lambda r: (r.get("suite", ""), r.get("model", ""), r.get("dataset", ""), r.get("method", "")))

    write_csv(output_dir / "locomo_core_3b_8b_with_llm_judge.csv", locomo_rows_sorted)
    write_md(output_dir / "locomo_core_3b_8b_with_llm_judge.md", "LoCoMo Core 3B/8B With LLM-as-a-Judge", locomo_rows_sorted)
    write_json(output_dir / "locomo_core_3b_8b_with_llm_judge.json", {"rows": locomo_rows_sorted})

    write_csv(output_dir / "multilingual_48_with_llm_judge.csv", rows_48_sorted)
    write_md(output_dir / "multilingual_48_with_llm_judge.md", "Multilingual LoCoMo-Style 48 With LLM-as-a-Judge", rows_48_sorted)
    write_json(output_dir / "multilingual_48_with_llm_judge.json", {"rows": rows_48_sorted})

    write_csv(output_dir / "combined_locomo_core_and_48_with_llm_judge.csv", combined_sorted)
    write_md(output_dir / "combined_locomo_core_and_48_with_llm_judge.md", "Combined LoCoMo Core And 48 With LLM-as-a-Judge", combined_sorted)
    write_json(output_dir / "combined_locomo_core_and_48_with_llm_judge.json", {"rows": combined_sorted})

    summary = {
        "output_dir": str(output_dir),
        "locomo_core": {
            "complete": sum(1 for row in locomo_rows if row["status"] == "complete"),
            "total": len(locomo_rows),
            "rows_file": str(output_dir / "locomo_core_3b_8b_with_llm_judge.csv"),
        },
        "multilingual_48": {
            "run_root": str(args.multilingual_48_root),
            "complete": sum(1 for row in rows_48 if row["status"] == "complete"),
            "total": len(rows_48),
            "rows_file": str(output_dir / "multilingual_48_with_llm_judge.csv"),
        },
        "combined": {
            "complete": sum(1 for row in combined_rows if row["status"] == "complete"),
            "total": len(combined_rows),
            "rows_file": str(output_dir / "combined_locomo_core_and_48_with_llm_judge.csv"),
        },
        "judge_protocol": JUDGE_PROTOCOL,
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "source_roots": {
            "qwen25_3b_locomo": str(MODELS[0].root),
            "qwen3_8b_locomo": str(MODELS[1].root),
            "multilingual_48": str(args.multilingual_48_root),
        },
    }
    write_json(output_dir / "manifest.json", summary)

    for src_name in ("audit.json", "results_f1_desc.csv", "results_f1_desc.md"):
        src = args.multilingual_48_root / "summary" / src_name
        if src.exists():
            shutil.copy2(src, output_dir / f"source_48_{src_name}")

    readme_lines = [
        "# LoCoMo LLM Judge + 48 Results",
        "",
        "This directory merges the original LoCoMo core baseline results with the completed multilingual LoCoMo-style 48-result grid.",
        "",
        f"- LoCoMo core judge rows: {summary['locomo_core']['complete']}/{summary['locomo_core']['total']}",
        f"- Multilingual 48 rows: {summary['multilingual_48']['complete']}/{summary['multilingual_48']['total']}",
        f"- Judge protocol: `{JUDGE_PROTOCOL}` / `{JUDGE_PROTOCOL_VERSION}`",
        "",
        "Primary files:",
        "",
        "- `combined_locomo_core_and_48_with_llm_judge.md`",
        "- `combined_locomo_core_and_48_with_llm_judge.csv`",
        "- `locomo_core_3b_8b_with_llm_judge.md`",
        "- `multilingual_48_with_llm_judge.md`",
        "- `manifest.json`",
        "- `completion_audit.md`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-judge", action="store_true", help="Run missing LoCoMo judge metrics before merging")
    parser.add_argument("--no-start-vllm", action="store_true", help="Use an already-running OpenAI-compatible server")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LOCOMO_ACCEPTANCE_JUDGE_WORKERS", "8")))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--multilingual-48-root",
        type=Path,
        default=REPO_ROOT / "runs/multilingual_locomo_style_repaired_48/20260513_123300",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/locomo_llm_judge_plus_48",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.multilingual_48_root = args.multilingual_48_root.resolve()
    if args.run_judge:
        for model in MODELS:
            ensure_vllm(model, args)
            for method in METHODS:
                run_judge_for(model, method, args)
    summary = build_outputs(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_incomplete and summary["combined"]["complete"] != summary["combined"]["total"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
