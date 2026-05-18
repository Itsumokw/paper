#!/usr/bin/env python3
"""Merge show-sharded LongDialQA/DialSim baseline artifacts.

The baseline runner writes complete artifacts per shard.  This utility
combines those artifacts into one baseline directory without re-evaluating
model outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from run_longdialqa_baseline import build_metrics, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", required=True, help="Shard directory names under --run-root.")
    parser.add_argument("--output-name", required=True, help="Merged directory name under --run-root.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=None,
        help="Canonical subset/full manifest for merged dataset metadata.",
    )
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_canonical_dataset(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    manifest = read_json(manifest_path)
    subset_path = Path(manifest["subset_path"])
    if not subset_path.is_absolute():
        subset_path = Path.cwd() / subset_path
    fraction = float(manifest.get("fraction", 0.0) or 0.0)
    if manifest.get("split_label") == "full" or fraction >= 0.999:
        subset_label = "full benchmark reproduction"
    elif fraction < 0.5:
        subset_label = "5% smoke subset result"
    else:
        subset_label = "50% subset result"
    normalized_manifest = manifest.get("normalized_manifest") or "datasets/DialSim/longdialqa_normalized_v1.1_seed0/manifest.json"
    normalized_manifest_path = Path(normalized_manifest)
    return {
        "normalized_manifest": normalized_manifest,
        "normalized_manifest_sha256": sha256_file(normalized_manifest_path) if normalized_manifest_path.exists() else None,
        "subset_sha256": sha256_file(subset_path),
        "subset_manifest": str(manifest_path),
        "subset_manifest_sha256": sha256_file(manifest_path),
        "subset_rows": len(read_json(subset_path) if subset_path.suffix.lower() == ".json" else read_jsonl(subset_path)),
        "subset_label": subset_label,
    }


def sum_token_usage(stats_rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for stats in stats_rows:
        total.update({key: int(value) for key, value in (stats.get("token_usage") or {}).items()})
    return dict(total)


def sum_prediction_token_usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        usage = row.get("answer_token_usage") or {}
        total.update({key: int(value) for key, value in usage.items()})
    return dict(total)


def prediction_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("qa_id") or row.get("question_id") or "")


def prediction_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    ordinal = row.get("global_question_ordinal")
    if ordinal is None:
        ordinal = row.get("session_ordinal", 0)
    return (str(row.get("show")), int(ordinal or 0), prediction_id(row))


def main() -> int:
    args = parse_args()
    out_dir = args.run_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    command_envs: list[str] = []

    for shard_name in args.shards:
        shard_dir = args.run_root / shard_name
        if not shard_dir.exists():
            raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")
        pred_path = shard_dir / "raw_predictions.jsonl"
        metrics_path = shard_dir / "metrics.json"
        stats_path = shard_dir / "stats.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"Shard is incomplete: {shard_dir}")
        pred_rows.extend(read_jsonl(pred_path))
        retrieval_rows.extend(read_jsonl(shard_dir / "retrieved_context.jsonl"))
        failure_rows.extend(read_jsonl(shard_dir / "failure_retry_logs.jsonl"))
        if metrics_path.exists():
            metrics_rows.append(read_json(metrics_path))
        if stats_path.exists():
            stats_rows.append(read_json(stats_path))
        env_path = shard_dir / "command.env"
        if env_path.exists():
            command_envs.append(f"### {shard_name}\n{env_path.read_text(encoding='utf-8').rstrip()}")

    ids = [prediction_id(row) for row in pred_rows]
    duplicate_ids = sorted([qid for qid, count in Counter(ids).items() if count > 1])
    if duplicate_ids and not args.allow_duplicates:
        raise ValueError(f"Duplicate QA ids in shards: {duplicate_ids[:10]}")

    pred_rows.sort(key=prediction_sort_key)
    retrieval_rows.sort(key=prediction_sort_key)

    pred_path = out_dir / "raw_predictions.jsonl"
    retrieval_path = out_dir / "retrieved_context.jsonl"
    failures_path = out_dir / "failure_retry_logs.jsonl"
    metrics_path = out_dir / "metrics.json"
    stats_path = out_dir / "stats.json"
    summary_path = out_dir / "summary.md"
    run_log_path = out_dir / "run.log"
    command_env_path = out_dir / "command.env"

    write_jsonl(pred_path, pred_rows)
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(failures_path, failure_rows)
    command_env_path.write_text("\n\n".join(command_envs) + "\n", encoding="utf-8")

    metrics = build_metrics(pred_rows)
    first_dataset = load_canonical_dataset(args.subset_manifest) or (metrics_rows[0].get("dataset", {}) if metrics_rows else {})
    if args.subset_manifest is None:
        for shard_metrics in metrics_rows[1:]:
            dataset = shard_metrics.get("dataset", {})
            for key in ["normalized_manifest_sha256", "subset_sha256", "subset_manifest_sha256", "subset_label"]:
                if first_dataset.get(key) != dataset.get(key):
                    raise ValueError(f"Dataset metadata mismatch for {key}")
    metrics["dataset"] = dict(first_dataset)
    metrics["dataset"]["merged_from_shards"] = args.shards
    metrics["dataset"]["merged_count"] = len(pred_rows)
    metrics["artifact_paths"] = {
        "raw_predictions": str(pred_path),
        "retrieved_context": str(retrieval_path),
        "metrics": str(metrics_path),
        "stats": str(stats_path),
        "failures": str(failures_path),
        "command_env": str(command_env_path),
        "run_log": str(run_log_path),
        "summary_md": str(summary_path),
        "subset": first_dataset.get("subset_path") or first_dataset.get("subset"),
    }

    runtime_values = [float(stats.get("runtime_seconds", 0.0) or 0.0) for stats in stats_rows]
    runtime_values = [float(stats.get("runtime_seconds", 0.0) or 0.0) for stats in stats_rows]
    prediction_runtime = sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in pred_rows)
    stats = {
        "baseline": args.baseline,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "merged_from_shards": args.shards,
        "parallel_wall_runtime_seconds_estimate": max(runtime_values) if runtime_values else None,
        "sum_shard_runtime_seconds": sum(runtime_values) if runtime_values else None,
        "sum_prediction_latency_seconds": prediction_runtime,
        "token_usage": sum_token_usage(stats_rows) if stats_rows else sum_prediction_token_usage(pred_rows),
        "shards": {name: stats for name, stats in zip(args.shards, stats_rows, strict=False)},
    }

    write_json(metrics_path, metrics)
    write_json(stats_path, stats)
    run_log_path.write_text(
        "\n".join(
            [
                f"baseline={args.baseline}",
                f"merged_from_shards={','.join(args.shards)}",
                f"raw_predictions={len(pred_rows)}",
                f"failures={len(failure_rows)}",
                f"overall={json.dumps(metrics['overall'], ensure_ascii=False, sort_keys=True)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        "\n".join(
            [
                f"# LongDialQA/DialSim {args.baseline} Merged Baseline",
                "",
                f"Result label: {metrics.get('dataset', {}).get('subset_label')}",
                f"Merged shards: `{', '.join(args.shards)}`",
                f"Count: {metrics['overall'].get('count')}",
                f"Accuracy: {metrics['overall'].get('accuracy')}",
                f"Mean retrieved K: {metrics['overall'].get('mean_retrieved_k')}",
                f"Mean retrieved tokens approx: {metrics['overall'].get('mean_context_tokens_approx')}",
                f"Parallel wall runtime seconds estimate: {stats['parallel_wall_runtime_seconds_estimate']}",
                "",
                "## Artifacts",
                "",
            ]
            + [f"- {key}: `{value}`" for key, value in metrics["artifact_paths"].items()]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_dir), "overall": metrics["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
