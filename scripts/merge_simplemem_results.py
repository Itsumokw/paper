from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _aggregate_metrics(all_metrics: list[dict[str, Any]], all_categories: list[int]) -> dict[str, Any]:
    """Mirror test_locomo10.aggregate_metrics without importing heavy runtime deps."""
    if not all_metrics:
        return {}

    aggregates: dict[str, list[float]] = defaultdict(list)
    category_aggregates: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                fv = float(value)
                aggregates[metric_name].append(fv)
                category_aggregates[category][metric_name].append(fv)

    results: dict[str, Any] = {"overall": {}}
    for metric_name, values in aggregates.items():
        if values:
            results["overall"][metric_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    for category in sorted(category_aggregates):
        bucket = {}
        for metric_name, values in category_aggregates[category].items():
            if values:
                bucket[metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        results[f"category_{category}"] = bucket

    return results


def _load_result(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "detailed_results" not in obj or "summary" not in obj:
        raise ValueError(f"{path} is not a valid SimpleMem result file")
    return obj


def merge_results(result_files: list[Path], output_file: Path) -> None:
    merged_detailed: list[dict[str, Any]] = []
    total_samples = 0
    by_source: list[dict[str, Any]] = []

    for path in result_files:
        obj = _load_result(path)
        detailed = obj.get("detailed_results", [])
        summary = obj.get("summary", {})
        total_samples += int(summary.get("num_samples", 0) or 0)
        merged_detailed.extend(detailed)
        by_source.append(
            {
                "file": str(path),
                "num_samples": int(summary.get("num_samples", 0) or 0),
                "num_questions": int(summary.get("num_questions", len(detailed)) or len(detailed)),
            }
        )

    retrieval_times = [float(x.get("retrieval_time", 0.0) or 0.0) for x in merged_detailed]
    answer_times = [float(x.get("answer_time", 0.0) or 0.0) for x in merged_detailed]
    total_times = [float(x.get("total_time", 0.0) or 0.0) for x in merged_detailed]

    metrics_list: list[dict[str, Any]] = []
    categories: list[int] = []
    for row in merged_detailed:
        metrics = row.get("metrics", {}) or {}
        if metrics:
            metrics_list.append(metrics)
            categories.append(int(row.get("category", 0) or 0))

    merged_obj = {
        "summary": {
            "num_samples": total_samples,
            "num_questions": len(merged_detailed),
            "avg_retrieval_time": statistics.mean(retrieval_times) if retrieval_times else 0.0,
            "avg_answer_time": statistics.mean(answer_times) if answer_times else 0.0,
            "avg_total_time": statistics.mean(total_times) if total_times else 0.0,
        },
        "aggregated_metrics": _aggregate_metrics(metrics_list, categories),
        "detailed_results": merged_detailed,
        "merge_info": {"source_files": by_source},
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(merged_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple SimpleMem result.json files into one.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input result.json files to merge (e.g. first5/result.json remaining5/result.json)",
    )
    parser.add_argument("--output", required=True, help="Merged output result.json path")
    args = parser.parse_args()

    input_paths = [Path(x).resolve() for x in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input result file not found: {p}")

    output_path = Path(args.output).resolve()
    merge_results(input_paths, output_path)
    print(f"Merged {len(input_paths)} files -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

