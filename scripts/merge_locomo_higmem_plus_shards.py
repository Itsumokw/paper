#!/usr/bin/env python3
"""Merge LoCoMo HiGMemPlus shard outputs without rerunning model calls."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACE_FILES = [
    "retrieved_evidence.jsonl",
    "component_traces.jsonl",
    "graph_traces.jsonl",
    "repair_traces.jsonl",
    "episode_traces.jsonl",
    "route_traces.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "f1": statistics.mean(float(row.get("f1", 0.0)) for row in rows),
        "bleu1": statistics.mean(float(row.get("bleu1", 0.0)) for row in rows),
        "judge_accuracy_proxy": sum(1 for row in rows if row.get("answer_correct_proxy")) / len(rows),
        "evidence_support_rate": sum(1 for row in rows if row.get("evidence_support")) / len(rows),
        "drill_down_rate": sum(1 for row in rows if row.get("drill_down")) / len(rows),
        "avg_context_tokens": statistics.mean(int(row.get("retrieved_context_tokens_approx", 0)) for row in rows),
        "avg_latency_seconds": statistics.mean(float(row.get("latency_seconds", 0.0)) for row in rows),
        "avg_retrieval_seconds": statistics.mean(float(row.get("retrieval_seconds", 0.0)) for row in rows),
        "avg_answer_seconds": statistics.mean(float(row.get("answer_seconds", 0.0)) for row in rows),
        "avg_total_tokens": statistics.mean(int((row.get("answer_token_usage") or {}).get("total_tokens", 0)) for row in rows),
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"overall": summarize_rows(rows), "by_category": {}, "by_error_type": {}}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category_name") or row.get("category"))].append(row)
        by_error[str(row.get("error_type") or "unknown")].append(row)
    metrics["by_category"] = {key: summarize_rows(value) for key, value in sorted(by_category.items())}
    metrics["by_error_type"] = {key: summarize_rows(value) for key, value in sorted(by_error.items())}
    metrics["bad_cases"] = [
        {
            "row_id": row.get("row_id"),
            "question": row.get("question"),
            "prediction": row.get("prediction"),
            "reference": row.get("reference"),
            "category_name": row.get("category_name"),
            "error_type": row.get("error_type"),
            "evidence_support": row.get("evidence_support"),
            "sufficiency_status": row.get("sufficiency_status"),
            "f1": row.get("f1"),
        }
        for row in rows
        if not row.get("answer_correct_proxy")
    ][:120]
    return metrics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    command_envs: list[str] = []
    dataset_manifest: dict[str, Any] | None = None
    token_usage: Counter[str] = Counter()
    shard_stats: dict[str, Any] = {}

    for shard in args.shards:
        method_dir = shard / args.method
        if not method_dir.exists():
            raise FileNotFoundError(method_dir)
        prediction_rows.extend(read_jsonl(method_dir / "raw_predictions.jsonl"))
        env_path = method_dir / "command.env"
        if env_path.exists():
            command_envs.append(f"### {method_dir}\n{env_path.read_text(encoding='utf-8').rstrip()}")
        manifest_path = method_dir / "dataset_manifest.json"
        if manifest_path.exists() and dataset_manifest is None:
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats_path = method_dir / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            shard_stats[str(method_dir)] = stats
            token_usage.update({key: int(value) for key, value in (stats.get("token_usage") or {}).items()})

    ids = [str(row.get("row_id")) for row in prediction_rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate row_id values: {duplicates[:10]}")
    prediction_rows.sort(key=lambda row: (int(str(row.get("sample_id", 0))), int(row.get("qa_index", 0))))
    write_jsonl(args.output_dir / "raw_predictions.jsonl", prediction_rows)

    for name in TRACE_FILES:
        rows: list[dict[str, Any]] = []
        for shard in args.shards:
            rows.extend(read_jsonl(shard / args.method / name))
        rows.sort(key=lambda row: str(row.get("row_id", "")))
        write_jsonl(args.output_dir / name, rows)

    if dataset_manifest:
        dataset_manifest["merged_from_shards"] = [str(shard / args.method) for shard in args.shards]
        write_json(args.output_dir / "dataset_manifest.json", dataset_manifest)
    (args.output_dir / "command.env").write_text("\n\n".join(command_envs) + "\n", encoding="utf-8")
    metrics = build_metrics(prediction_rows)
    metrics["dataset"] = dataset_manifest or {}
    metrics["artifact_paths"] = {
        "raw_predictions": str(args.output_dir / "raw_predictions.jsonl"),
        "retrieved_evidence": str(args.output_dir / "retrieved_evidence.jsonl"),
        "component_traces": str(args.output_dir / "component_traces.jsonl"),
        "graph_traces": str(args.output_dir / "graph_traces.jsonl"),
        "repair_traces": str(args.output_dir / "repair_traces.jsonl"),
        "episode_traces": str(args.output_dir / "episode_traces.jsonl"),
        "route_traces": str(args.output_dir / "route_traces.jsonl"),
        "metrics": str(args.output_dir / "metrics.json"),
        "stats": str(args.output_dir / "stats.json"),
        "summary": str(args.output_dir / "summary.md"),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    stats = {
        "method": args.method,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "merged_from_shards": [str(shard / args.method) for shard in args.shards],
        "total_rows": len(prediction_rows),
        "token_usage": dict(token_usage),
        "shards": shard_stats,
    }
    write_json(args.output_dir / "stats.json", stats)
    lines = [
        f"# LoCoMo HiGMemPlus {args.method} Merged Shards",
        "",
        f"- Count: {metrics['overall'].get('count')}",
        f"- F1: {metrics['overall'].get('f1')}",
        f"- Judge accuracy proxy: {metrics['overall'].get('judge_accuracy_proxy')}",
        f"- Evidence support rate: {metrics['overall'].get('evidence_support_rate')}",
        f"- Drill-down rate: {metrics['overall'].get('drill_down_rate')}",
        f"- Avg latency seconds: {metrics['overall'].get('avg_latency_seconds')}",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "overall": metrics["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
