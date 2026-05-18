#!/usr/bin/env python3
"""Merge LongDialQA HiGMemPlus show-shard outputs without rerunning model calls."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_longdialqa_baseline import build_metrics


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


def sort_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("show") or ""),
        int(row.get("session_ordinal") or 0),
        int(row.get("ask_turn_index") or 0),
        str(row.get("qa_id") or ""),
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    command_envs: list[str] = []
    dataset_manifest: dict[str, Any] | None = None
    token_usage: Counter[str] = Counter()
    shard_stats: dict[str, Any] = {}

    for shard in args.shards:
        method_dir = shard / args.method if (shard / args.method).exists() else shard
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

    ids = [str(row.get("qa_id")) for row in prediction_rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate qa_id values: {duplicates[:10]}")
    prediction_rows.sort(key=sort_key)
    write_jsonl(args.output_dir / "raw_predictions.jsonl", prediction_rows)

    for name in TRACE_FILES:
        rows: list[dict[str, Any]] = []
        for shard in args.shards:
            method_dir = shard / args.method if (shard / args.method).exists() else shard
            rows.extend(read_jsonl(method_dir / name))
        rows.sort(key=lambda row: str(row.get("qa_id") or ""))
        write_jsonl(args.output_dir / name, rows)

    if dataset_manifest:
        dataset_manifest["merged_from_shards"] = [
            str(shard / args.method if (shard / args.method).exists() else shard) for shard in args.shards
        ]
        dataset_manifest["selected_rows"] = len(prediction_rows)
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
        "merged_from_shards": [
            str(shard / args.method if (shard / args.method).exists() else shard) for shard in args.shards
        ],
        "total_rows": len(prediction_rows),
        "token_usage": dict(token_usage),
        "shards": shard_stats,
    }
    write_json(args.output_dir / "stats.json", stats)
    lines = [
        f"# LongDialQA HiGMemPlus {args.method} Merged Shards",
        "",
        f"- Count: {metrics['overall'].get('count')}",
        f"- Accuracy: {metrics['overall'].get('accuracy')}",
        f"- Mean token F1: {metrics['overall'].get('mean_token_f1')}",
        f"- Evidence recall any: {metrics['overall'].get('evidence_recall_any')}",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "overall": metrics["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
