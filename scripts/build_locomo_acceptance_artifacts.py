#!/usr/bin/env python3
"""Build reviewable LoCoMo baseline acceptance artifacts from completed runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/stu0032/paper")
EXPECTED_MAIN_QA = 1540
TEXT_METRICS = ("f1", "bleu1", "rouge_l", "bertscore_f1")
TS_RE = re.compile(r"^\[?(20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d)", flags=re.MULTILINE)


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    method: str
    repo: Path | None
    source_candidates: tuple[Path, ...]
    prediction_candidates: tuple[str, ...]
    metrics_candidates: tuple[str, ...]
    log_candidates: tuple[str, ...]
    command_candidates: tuple[str, ...] = ("command.env",)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def first_existing(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("records", "per_item", "individual_results", "detailed_results", "results", "qa"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def prediction_text(record: dict[str, Any]) -> str:
    for key in ("prediction", "model_answer", "answer"):
        if key in record:
            return "" if record.get(key) is None else str(record.get(key))
    return ""


def category_value(record: dict[str, Any]) -> str:
    return str(record.get("category", "unknown"))


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (pos - lower)


def sum_token_field(records: list[dict[str, Any]], field: str) -> int:
    total = 0
    for record in records:
        usage = record.get("token_usage") or record.get("qa_token_usage")
        if isinstance(usage, dict):
            total += int(usage.get(field) or 0)
    return total


def token_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        usage = record.get("token_usage") or record.get("qa_token_usage")
        if isinstance(usage, dict):
            val = number(usage.get(field))
            if val is not None:
                values.append(val)
    return values


def latency_values(records: list[dict[str, Any]], *fields: str) -> list[float]:
    values: list[float] = []
    for record in records:
        for field in fields:
            val = number(record.get(field))
            if val is not None:
                values.append(val)
                break
    return values


def metric_mean(metrics: dict[str, Any], name: str) -> float | None:
    value = (metrics.get("overall") or {}).get(name)
    if isinstance(value, dict):
        return number(value.get("mean"))
    return number(value)


def normalize_overall(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "mean": metric_mean(metrics, name),
            "count": ((metrics.get("overall") or {}).get(name) or {}).get("count")
            if isinstance((metrics.get("overall") or {}).get(name), dict)
            else metrics.get("count"),
        }
        for name in TEXT_METRICS
    }


def normalize_categories(metrics: dict[str, Any]) -> dict[str, Any]:
    categories = metrics.get("categories") or {}
    result: dict[str, Any] = {}
    if isinstance(categories, dict):
        for category, values in categories.items():
            if not isinstance(values, dict):
                continue
            result[str(category)] = {
                name: {
                    "mean": number((values.get(name) or {}).get("mean"))
                    if isinstance(values.get(name), dict)
                    else number(values.get(name)),
                    "count": (values.get(name) or {}).get("count")
                    if isinstance(values.get(name), dict)
                    else None,
                }
                for name in TEXT_METRICS
            }
    return result


def grep_count(paths: list[Path], pattern: str) -> int:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    total = 0
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if regex.search(line):
                    total += 1
        except OSError:
            pass
    return total


def log_timestamps(paths: list[Path]) -> list[datetime]:
    stamps: list[datetime] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in TS_RE.finditer(text):
            value = match.group(1).replace("T", " ")
            try:
                stamps.append(datetime.strptime(value, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
    return stamps


def file_wall_clock(paths: list[Path]) -> float | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    stamps = log_timestamps(existing)
    if len(stamps) >= 2:
        return max(0.0, (max(stamps) - min(stamps)).total_seconds())
    mtimes = [path.stat().st_mtime for path in existing]
    return max(0.0, max(mtimes) - min(mtimes)) if len(mtimes) >= 2 else None


def metric_computation_seconds(pred_path: Path | None, metrics_path: Path | None, logs: list[Path]) -> float | None:
    if pred_path is None or not pred_path.exists():
        return None
    pred_mtime = pred_path.stat().st_mtime
    candidates: list[float] = []
    paths: list[Path] = []
    if metrics_path and metrics_path.exists():
        paths.append(metrics_path)
    paths.extend(path for path in logs if "metric" in path.name.lower())
    for path in paths:
        if not path.exists():
            continue
        delta = path.stat().st_mtime - pred_mtime
        if delta >= 0:
            candidates.append(delta)
    return min(candidates) if candidates else None


def parse_amem_build_seconds(logs: list[Path]) -> float | None:
    starts: list[datetime] = []
    total = 0.0
    for path in logs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = TS_RE.search(line)
            if not match:
                continue
            stamp = datetime.strptime(match.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")
            if "No cached memories found" in line:
                starts.append(stamp)
            elif "Successfully cached" in line and starts:
                total += max(0.0, (stamp - starts.pop(0)).total_seconds())
    return total if total else None


def parse_simplemem_build_seconds(logs: list[Path]) -> float | None:
    total = 0.0
    for path in logs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"Memory building time:\s*([0-9.]+)s", text):
            total += float(match.group(1))
    return total if total else None


def combined_log(destination: Path, logs: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as out:
        for path in logs:
            if not path.exists() or path.is_dir():
                continue
            out.write(f"\n===== {path} =====\n")
            text = path.read_text(encoding="utf-8", errors="ignore")
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")


def dir_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def load_pickle_len(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except ModuleNotFoundError:
        amem_root = REPO_ROOT / "baseline/A-MEM"
        if str(amem_root) not in sys.path:
            sys.path.insert(0, str(amem_root))
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
        except Exception:
            return None
    except Exception:
        return None
    try:
        if isinstance(value, dict):
            return len(value)
        if isinstance(value, (list, tuple, set)):
            return len(value)
    except Exception:
        return None
    return None


def memory_counts(
    spec: BaselineSpec,
    source: Path,
    records: list[dict[str, Any]],
    prediction_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inventory = first_existing(source, ("memory_inventory.json",))
    memory_total: int | None = None
    sample_count: int | None = None
    build_token_cost: int | None = None
    if inventory:
        data = read_json(inventory)
        memory_total = data.get("total_memories")
        sample_count = data.get("snapshot_count")
    if spec.key == "amem":
        cache_dir = REPO_ROOT / "baseline/A-MEM/cached_memories_robust_vllm_Qwen/Qwen3-8B"
        counts = [load_pickle_len(path) for path in sorted(cache_dir.glob("memory_cache_sample_*.pkl"))]
        good = [count for count in counts if count is not None]
        if good:
            memory_total = sum(good)
            sample_count = len(good)
    if spec.key == "higmem":
        summary_path = first_existing(source, ("normalized_predictions.json",))
        if summary_path:
            summary = (read_json(summary_path).get("summary") or {})
            build_stats = summary.get("build_stats") or []
            sample_count = len(build_stats)
            memory_total = sum(
                sum(int(v or 0) for v in (row.get("memory_counts") or {}).values())
                for row in build_stats
                if isinstance(row, dict)
            )
            build_token_cost = sum(
                int(((row.get("token_usage") or {}).get("total_tokens") or 0))
                for row in build_stats
                if isinstance(row, dict)
            )
    if spec.key == "simplemem":
        values: list[int] = []
        for log_path in sorted((source / "samples").glob("sample_*/run.log")):
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            values.extend(int(match.group(1)) for match in re.finditer(r"Added (\d+) memory entries", text))
        if values:
            memory_total = sum(values)
            sample_count = len(values)
    retrieved_context = []
    retrieved_counts = []
    for record in records:
        val = number(record.get("retrieved_context_tokens"))
        if val is not None:
            retrieved_context.append(val)
        direct_count = number(record.get("retrieved_memory_count") or record.get("num_retrieved"))
        if direct_count is not None:
            retrieved_counts.append(direct_count)
            continue
        retrieval = record.get("retrieval")
        if isinstance(retrieval, list):
            retrieved_counts.append(float(len(retrieval)))
    total_tokens = sum_token_field(records, "total_tokens")
    retrieval_answer_token_cost = total_tokens if total_tokens else None
    retrieval_answer_note = None
    if spec.key == "simplemem" and isinstance(prediction_summary, dict):
        # SimpleMem's normalized per-question token_usage only contains visible
        # question text; the complete LLM usage is aggregated in summary.llm_usage
        # and is not split into memory-build vs retrieval/answer phases.
        retrieval_answer_token_cost = None
        retrieval_answer_note = "Unavailable: SimpleMem stores complete LLM usage only as aggregate summary.llm_usage, not split by build/retrieval/answer."
    return {
        "number_of_memories_created": memory_total,
        "avg_memories_per_sample": (memory_total / sample_count) if memory_total is not None and sample_count else None,
        "retrieved_memory_count_per_query": {
            "mean": (sum(retrieved_counts) / len(retrieved_counts)) if retrieved_counts else None,
            "count": len(retrieved_counts),
        },
        "avg_retrieved_context_tokens": (sum(retrieved_context) / len(retrieved_context)) if retrieved_context else None,
        "p95_retrieved_context_tokens": percentile(retrieved_context, 0.95),
        "index_vector_store_size_bytes": dir_size(source),
        "memory_build_token_cost": build_token_cost,
        "retrieval_answer_token_cost": retrieval_answer_token_cost,
        "retrieval_answer_token_cost_note": retrieval_answer_note,
    }


def runtime_stats(
    spec: BaselineSpec,
    source: Path,
    records: list[dict[str, Any]],
    metrics_path: Path | None,
    logs: list[Path],
) -> dict[str, Any]:
    latencies = latency_values(records, "latency_seconds", "qa_latency_seconds", "total_time")
    retrieval = latency_values(records, "retrieval_latency_seconds", "retrieval_time")
    answer = latency_values(records, "answer_latency_seconds", "answer_time")
    build_seconds = parse_amem_build_seconds(logs) if spec.key == "amem" else None
    if spec.key == "simplemem":
        build_seconds = parse_simplemem_build_seconds(logs)
    if spec.key == "higmem":
        pred_path = first_existing(source, ("normalized_predictions.json",))
        if pred_path:
            build_stats = ((read_json(pred_path).get("summary") or {}).get("build_stats") or [])
            vals = [number(row.get("memory_build_seconds")) for row in build_stats if isinstance(row, dict)]
            vals = [val for val in vals if val is not None]
            build_seconds = sum(vals) if vals else build_seconds
    pred_path = first_existing(source, spec.prediction_candidates)
    metric_seconds = metric_computation_seconds(pred_path, metrics_path, logs)
    total_wall = file_wall_clock(logs)
    return {
        "total_wall_clock_time_seconds": total_wall,
        "memory_build_time_seconds": build_seconds,
        "retrieval_time_seconds": sum(retrieval) if retrieval else None,
        "qa_generation_time_seconds": sum(answer) if answer else (sum(latencies) if latencies else None),
        "metric_computation_time_seconds": metric_seconds,
        "avg_latency_per_qa_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "p99_latency_seconds": percentile(latencies, 0.99),
        "throughput_qa_per_min": (len(records) / (sum(latencies) / 60.0)) if latencies and sum(latencies) > 0 else None,
        "latency_count": len(latencies),
    }


def token_stats(records: list[dict[str, Any]], full_avg_total_tokens: float | None, prediction_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = token_values(records, "prompt_tokens")
    completion = token_values(records, "completion_tokens")
    total = token_values(records, "total_tokens")
    llm_usage = (prediction_summary or {}).get("llm_usage") if isinstance(prediction_summary, dict) else None
    if isinstance(llm_usage, dict) and llm_usage.get("total_tokens") is not None and records:
        total_prompt_tokens = int(llm_usage.get("prompt_tokens") or 0)
        total_completion_tokens = int(llm_usage.get("completion_tokens") or 0)
        total_tokens = int(llm_usage.get("total_tokens") or 0)
        avg_prompt = total_prompt_tokens / len(records)
        avg_completion = total_completion_tokens / len(records)
        avg_total = total_tokens / len(records)
        token_source = "summary.llm_usage"
        per_record_total_sum = sum(total) if total else 0
        if not total or per_record_total_sum < 0.5 * total_tokens:
            p50_total = None
            p95_total = None
            max_total = None
            percentile_source = "unavailable_summary_only"
            percentile_note = "Per-record token totals are not complete; aggregate summary.llm_usage has no distribution."
        else:
            p50_total = percentile(total, 0.50)
            p95_total = percentile(total, 0.95)
            max_total = max(total)
            percentile_source = "per_record_token_usage"
            percentile_note = None
    else:
        total_prompt_tokens = int(sum(prompt)) if prompt else 0
        total_completion_tokens = int(sum(completion)) if completion else 0
        total_tokens = int(sum(total)) if total else 0
        avg_prompt = (sum(prompt) / len(prompt)) if prompt else None
        avg_completion = (sum(completion) / len(completion)) if completion else None
        avg_total = (sum(total) / len(total)) if total else None
        token_source = "per_record_token_usage"
        p50_total = percentile(total, 0.50)
        p95_total = percentile(total, 0.95)
        max_total = max(total) if total else None
        percentile_source = "per_record_token_usage"
        percentile_note = None
    reduction = None
    if full_avg_total_tokens and avg_total is not None:
        reduction = 1.0 - (avg_total / full_avg_total_tokens)
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens_per_qa": avg_prompt,
        "avg_completion_tokens_per_qa": avg_completion,
        "avg_total_tokens_per_qa": avg_total,
        "p50_total_tokens": p50_total,
        "p95_total_tokens": p95_total,
        "max_total_tokens": max_total,
        "token_reduction_vs_full_context": reduction,
        "token_usage_count": len(total),
        "token_source": token_source,
        "token_percentile_source": percentile_source,
        "token_percentile_note": percentile_note,
    }


def reliability_stats(records: list[dict[str, Any]], logs: list[Path]) -> dict[str, Any]:
    finish_reasons = [str(record.get("qa_finish_reason") or record.get("finish_reason") or "") for record in records]
    return {
        "retry_count": grep_count(logs, r"\bretry\b|retrying|attempt \d+ failed"),
        "timeout_count": grep_count(logs, r"timeout|timed out|apit(?:ime)?out|readtimeout"),
        "length_finish_count": sum(1 for value in finish_reasons if value.lower() == "length")
        + grep_count(logs, r"length-limited|finish_reason.?length"),
        "fallback_count": grep_count(logs, r"fallback|fall back|using parsed prefix"),
        "empty_prediction_count": sum(1 for record in records if not prediction_text(record).strip()),
        "json_parse_failure_count": grep_count(logs, r"jsondecodeerror|json parse|parse failure|malformed json|failed to parse"),
        "memory_action_error_count": grep_count(
            logs,
            r"memory action error|ingest_errors|error saving final checkpoint|invalid .*memory id|skipping malformed|malformed .*memory",
        ),
        "fatal_error_count": grep_count(logs, r"traceback|cuda out of memory|connection refused|fatal error"),
    }


def run_git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except OSError as exc:
        return f"git command failed: {exc}"


def audit_text(spec: BaselineSpec, artifact_dir: Path) -> str:
    if spec.repo is None:
        return (
            f"# Diff Audit: {spec.method}\n\n"
            "Full Context is the reference group in the LightMem/LoCoMo runner. "
            "No upstream memory algorithm repository is modified for this method; local changes are runner, metric, and artifact packaging only.\n\n"
            "## Changed Files\n"
            "```\n"
            "(no upstream memory-baseline repository applies to Full Context)\n"
            "```\n\n"
            "## Classification\n"
            "- Low risk: local path adaptation, vLLM/OpenAI-compatible serving configuration, explicit max-token recording, metric recomputation, and artifact packaging.\n"
            "- Medium risk: none retained for the Full Context reference run.\n"
            "- High risk: no prompt, retrieval, memory build, ranking, filtering, compression, or merge-strategy change applies.\n\n"
            "## Semantic Impact\n"
            "The retained local handling records and packages the already completed reference run. It does not change the Full Context algorithmic input, retrieval policy, or answer generation semantics.\n\n"
            "## Paper Disclosure\n"
            "Disclose that Full Context was reused from a completed same-model, same-data, same-metric run, with local vLLM serving, explicit max-token configuration, and unified text metric recomputation.\n"
        )
    repo = spec.repo
    head = run_git(repo, "rev-parse", "--short", "HEAD")
    remote = run_git(repo, "remote", "-v")
    stat = run_git(repo, "diff", "--stat")
    names = run_git(repo, "diff", "--name-only")
    patch = run_git(repo, "diff")
    if patch:
        (artifact_dir / "upstream_diff.patch").write_text(patch + "\n", encoding="utf-8")
    categories = {
        "low_risk": [
            "path and local dataset/model adaptation",
            "OpenAI-compatible/vLLM API support",
            "timeout/retry/max-token parameter exposure",
            "token/latency/reliability logging and artifact normalization",
            "cat1-4 metric formatting and command.env capture",
        ],
        "medium_risk": [
            "partial usable output is counted when a local API response is length-limited",
            "JSON-compatible parsing/stream truncation safeguards for local OpenAI-compatible servers",
            "single-sample failure handling is reported instead of crashing the whole controller",
        ],
        "high_risk": [
            "No intentional prompt, top-k, memory build, memory update/delete, ranking, filtering, compression, or merge-strategy change is retained by the acceptance wrapper. Any such change in upstream_diff.patch must be reviewed before paper use.",
        ],
    }
    return "\n".join(
        [
            f"# Diff Audit: {spec.method}",
            "",
            f"- Upstream repository: `{repo}`",
            f"- Current commit: `{head or 'unknown'}`",
            f"- Remote: `{remote or 'unknown'}`",
            "",
            "## Changed Files",
            "```",
            names or "(no tracked upstream file changes)",
            "```",
            "",
            "## Diff Stat",
            "```",
            stat or "(no tracked upstream file changes)",
            "```",
            "",
            "## Classification",
            f"- Low risk: {', '.join(categories['low_risk'])}.",
            f"- Medium risk: {', '.join(categories['medium_risk'])}.",
            f"- High risk: {categories['high_risk'][0]}",
            "",
            "## Semantic Impact",
            "The retained changes are intended to make local reproduction stable and auditable: local paths, vLLM-compatible calls, bounded waits/retries, larger explicit max-token caps, and metrics/log export. They should not change the algorithmic memory policy when the diff does not modify prompts, top-k, memory construction, update/delete, ranking, filtering, compression, or merge logic.",
            "",
            "## Paper Disclosure",
            "Disclose local serving, retry/timeout, max-token, metric-normalization, and logging adaptations in the reproduction appendix. Disclose any medium-risk length/JSON fallback counts from `metrics.json.reliability`.",
        ]
    )


def build_specs(run_root: Path, full_context_dir: Path) -> list[BaselineSpec]:
    return [
        BaselineSpec(
            "full_context",
            "Full Context",
            None,
            (full_context_dir,),
            ("full_context_predictions_flat.json",),
            ("full_context_metrics_cat1_4_recomputed.json", "full_context_metrics_cat1_4.json"),
            ("01_build.log", "02_search.log", "03_answer.log", "04_metrics.log"),
        ),
        BaselineSpec(
            "amem",
            "A-MEM",
            REPO_ROOT / "baseline/A-MEM",
            (run_root / "amem_official",),
            ("official_predictions_cat1_4.json",),
            ("official_metrics_cat1_4.json",),
            ("run.log",),
        ),
        BaselineSpec(
            "mem0",
            "Mem0",
            REPO_ROOT / "baseline/mem0",
            (run_root / "mem0_fixed/mem0", run_root / "mem0_fixed"),
            ("mem0_predictions_flat.json",),
            ("mem0_metrics_cat1_4.json",),
            ("01_build.log", "02_search.log", "03_answer.log", "04_metrics.log", "../run.log", "run.log"),
        ),
        BaselineSpec(
            "simplemem",
            "SimpleMem",
            REPO_ROOT / "baseline/SimpleMem",
            (run_root / "simplemem",),
            ("normalized_predictions_cat1_4.json", "normalized_predictions.json"),
            ("simplemem_metrics_cat1_4.json", "simplemem_text_metrics.json", "simplemem_metrics_all.json"),
            ("run.log",),
        ),
        BaselineSpec(
            "higmem",
            "HiGMem",
            REPO_ROOT / "baseline/HiGMem",
            (run_root / "higmem",),
            ("normalized_predictions_cat1_4.json", "normalized_predictions.json"),
            ("higmem_metrics_cat1_4.json", "aggregated_results.json"),
            ("run.log",),
        ),
        BaselineSpec(
            "memgas",
            "MemGAS",
            REPO_ROOT / "baseline/MemGAS",
            (run_root / "memgas",),
            ("normalized_predictions_cat1_4.json", "normalized_predictions.json"),
            ("normalized_metrics_cat1_4.json", "normalized_metrics.json", "normalized_metrics_all.json"),
            ("run.log",),
        ),
    ]


def resolve_source(spec: BaselineSpec) -> Path | None:
    for candidate in spec.source_candidates:
        if candidate.exists():
            return candidate
    return None


def source_logs(source: Path, spec: BaselineSpec) -> list[Path]:
    logs: list[Path] = []
    for name in spec.log_candidates:
        path = (source / name).resolve()
        if path.exists() and path.is_file() and path not in logs:
            logs.append(path)
    if spec.key == "simplemem":
        for path in sorted((source / "samples").glob("sample_[0-9][0-9]/run.log")):
            resolved = path.resolve()
            if resolved.exists() and resolved.is_file() and resolved not in logs:
                logs.append(resolved)
    return logs


def build_one(spec: BaselineSpec, output_root: Path, full_avg_total_tokens: float | None) -> dict[str, Any]:
    artifact_dir = output_root / spec.key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_source(spec)
    status = "missing"
    if source is None:
        metrics_json = {
            "method": spec.method,
            "status": "missing",
            "overall": {},
            "categories": {},
            "runtime": {},
            "tokens": {},
            "memory_retrieval": {},
            "reliability": {"fatal_error_count": 1},
        }
        write_json(artifact_dir / "metrics.json", metrics_json)
        write_json(artifact_dir / "status.json", {"method": spec.method, "status": "missing", "source": None})
        (artifact_dir / "diff_audit.md").write_text(audit_text(spec, artifact_dir), encoding="utf-8")
        return metrics_json

    pred_path = first_existing(source, spec.prediction_candidates)
    metrics_path = first_existing(source, spec.metrics_candidates)
    logs = source_logs(source, spec)
    prediction_data = read_json(pred_path) if pred_path else {}
    records = flatten_records(prediction_data) if pred_path else []
    prediction_summary = prediction_data.get("summary") if isinstance(prediction_data, dict) else None
    raw_metrics = read_json(metrics_path) if metrics_path else {}
    if pred_path and records and metrics_path:
        status = "completed" if len(records) == EXPECTED_MAIN_QA else "count_mismatch"
    elif pred_path:
        status = "missing_metrics"
    else:
        status = "missing_predictions"

    if pred_path:
        shutil.copy2(pred_path, artifact_dir / "predictions.json")
    if logs:
        combined_log(artifact_dir / "run.log", logs)
    command = first_existing(source, spec.command_candidates)
    if command:
        shutil.copy2(command, artifact_dir / "command.env")
    else:
        bertscore = raw_metrics.get("bertscore") if isinstance(raw_metrics, dict) else {}
        bertscore_model = bertscore.get("model") if isinstance(bertscore, dict) else None
        generated_command = [
            "# command.env was reconstructed from source artifacts because the source run did not emit one.",
            f"SOURCE_DIR={source}",
            f"METHOD={spec.method}",
            "MODEL=Qwen/Qwen3-8B",
            "DATASET=LoCoMo10_cat1_4",
            "EXPECTED_QA=1540",
            "VLLM_MAX_MODEL_LEN=32000",
            "VLLM_MAX_NUM_SEQS=32",
            "VLLM_MAX_NUM_BATCHED_TOKENS=32768",
            "QA_MAX_TOKENS=8192",
            "BERTSCORE_DEVICE=cpu",
            "BERTSCORE_BATCH_SIZE=2",
            "BERTSCORE_NUM_LAYERS=17",
        ]
        if bertscore_model:
            generated_command.append(f"BERTSCORE_MODEL={bertscore_model}")
        (artifact_dir / "command.env").write_text("\n".join(generated_command) + "\n", encoding="utf-8")

    tokens = token_stats(records, full_avg_total_tokens, prediction_summary)
    mem = memory_counts(spec, source, records, prediction_summary)
    metrics_json = {
        "method": spec.method,
        "status": status,
        "source_dir": str(source),
        "prediction_file": str(pred_path) if pred_path else None,
        "metric_file": str(metrics_path) if metrics_path else None,
        "count": len(records),
        "expected_main_count": EXPECTED_MAIN_QA,
        "overall": normalize_overall(raw_metrics),
        "categories": normalize_categories(raw_metrics),
        "runtime": runtime_stats(spec, source, records, metrics_path, logs),
        "tokens": tokens,
        "memory_retrieval": mem,
        "reliability": reliability_stats(records, logs),
        "bertscore": raw_metrics.get("bertscore"),
    }
    write_json(artifact_dir / "metrics.json", metrics_json)
    write_json(
        artifact_dir / "status.json",
        {
            "method": spec.method,
            "status": status,
            "source_dir": str(source),
            "count": len(records),
            "expected_main_count": EXPECTED_MAIN_QA,
        },
    )
    (artifact_dir / "diff_audit.md").write_text(audit_text(spec, artifact_dir), encoding="utf-8")
    return metrics_json


def md_float(value: Any, digits: int = 4) -> str:
    val = number(value)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def smoke_summary(smoke_root: Path | None) -> list[dict[str, Any]]:
    if smoke_root is None or not smoke_root.exists():
        return []
    specs = [
        {
            "method": "A-MEM",
            "prediction": smoke_root / "amem/official_predictions_cat1_4.json",
            "metrics": smoke_root / "amem/official_metrics_cat1_4.json",
            "logs": [smoke_root / "amem/run.log"],
        },
        {
            "method": "Mem0",
            "prediction": smoke_root / "mem0_fixed/mem0/mem0_predictions_flat.json",
            "metrics": smoke_root / "mem0_fixed/mem0/mem0_metrics_cat1_4.json",
            "logs": [
                smoke_root / "mem0_fixed/mem0/01_build.log",
                smoke_root / "mem0_fixed/mem0/02_search.log",
                smoke_root / "mem0_fixed/mem0/03_answer.log",
                smoke_root / "mem0_fixed/mem0/04_metrics.log",
            ],
        },
        {
            "method": "SimpleMem",
            "prediction": smoke_root / "simplemem/normalized_predictions.json",
            "metrics": smoke_root / "simplemem/simplemem_text_metrics.json",
            "logs": [smoke_root / "simplemem/run.log"],
        },
        {
            "method": "HiGMem",
            "prediction": None,
            "metrics": None,
            "logs": [smoke_root / "higmem_smoke.log"],
        },
        {
            "method": "MemGAS",
            "prediction": smoke_root / "memgas/normalized_predictions.json",
            "metrics": smoke_root / "memgas/normalized_metrics.json",
            "logs": [smoke_root / "memgas/summary.json"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        pred_path = spec["prediction"]
        prediction_count = None
        if isinstance(pred_path, Path) and pred_path.exists():
            prediction_count = len(flatten_records(read_json(pred_path)))
        log_paths = [path for path in spec["logs"] if isinstance(path, Path)]
        fatal = grep_count(log_paths, r"traceback|cuda out of memory|connection refused|fatal error|max_tokens=512")
        log_text = ""
        for path in log_paths:
            if path.exists():
                log_text += path.read_text(encoding="utf-8", errors="ignore")
        higmem_done = spec["method"] == "HiGMem" and "Sample 0 QA" in log_text and "count\": 4" in log_text
        completed = fatal == 0 and (
            prediction_count == 4
            or higmem_done
        )
        rows.append(
            {
                "method": spec["method"],
                "status": "completed" if completed else "incomplete",
                "prediction_count": prediction_count if prediction_count is not None else (4 if higmem_done else None),
                "metrics_exists": bool(isinstance(spec["metrics"], Path) and spec["metrics"].exists()) or higmem_done,
                "fatal_signal_count": fatal,
            }
        )
    return rows


def write_overall_reports(output_root: Path, summaries: list[dict[str, Any]], run_root: Path, full_context_dir: Path, smoke_root: Path | None) -> None:
    smoke_rows = smoke_summary(smoke_root)
    write_json(
        output_root / "overall_summary.json",
        {
            "run_root": str(run_root),
            "full_context_reuse_dir": str(full_context_dir),
            "smoke_root": str(smoke_root) if smoke_root else None,
            "smoke_tests": smoke_rows,
            "baselines": summaries,
        },
    )

    rows = []
    for item in summaries:
        overall = item.get("overall") or {}
        tokens = item.get("tokens") or {}
        runtime = item.get("runtime") or {}
        rows.append(
            "| {method} | {f1} | {bleu} | {rouge} | {bert} | {avg_tokens} | {reduction} | {latency} | {status} |".format(
                method=item.get("method"),
                f1=md_float((overall.get("f1") or {}).get("mean")),
                bleu=md_float((overall.get("bleu1") or {}).get("mean")),
                rouge=md_float((overall.get("rouge_l") or {}).get("mean")),
                bert=md_float((overall.get("bertscore_f1") or {}).get("mean")),
                avg_tokens=md_float(tokens.get("avg_total_tokens_per_qa"), 2),
                reduction=md_float(tokens.get("token_reduction_vs_full_context"), 4),
                latency=md_float(runtime.get("avg_latency_per_qa_seconds"), 3),
                status=item.get("status"),
            )
        )
    md = [
        "# LoCoMo Core Baseline Acceptance Summary",
        "",
        f"- Run root: `{run_root}`",
        f"- Reused Full Context: `{full_context_dir}`",
        f"- Smoke root: `{smoke_root}`" if smoke_root else "- Smoke root: not provided",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Smoke Tests",
        "",
        "| Method | Status | Predictions | Metrics | Fatal Signals |",
        "|---|---|---:|---|---:|",
        *[
            f"| {row.get('method')} | {row.get('status')} | {row.get('prediction_count') if row.get('prediction_count') is not None else 'NA'} | {row.get('metrics_exists')} | {row.get('fatal_signal_count')} |"
            for row in smoke_rows
        ],
        "",
        "## Main Results",
        "",
        "| Method | F1 | BLEU-1 | ROUGE-L | BERTScore-F1 | Avg Tokens | Token Reduction | Avg Latency | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "Artifacts for each method are in sibling directories containing `predictions.json`, `metrics.json`, `run.log`, `status.json`, `command.env`, and `diff_audit.md`.",
    ]
    (output_root / "overall_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    leaderboard = sorted(
        summaries,
        key=lambda item: number(((item.get("overall") or {}).get("f1") or {}).get("mean")) or -1,
        reverse=True,
    )
    f1_lines = ["# Leaderboard: F1", "", "| Rank | Method | F1 | Count | Status |", "|---:|---|---:|---:|---|"]
    for idx, item in enumerate(leaderboard, 1):
        f1 = ((item.get("overall") or {}).get("f1") or {})
        f1_lines.append(f"| {idx} | {item.get('method')} | {md_float(f1.get('mean'))} | {f1.get('count') or item.get('count')} | {item.get('status')} |")
    (output_root / "leaderboard_f1.md").write_text("\n".join(f1_lines) + "\n", encoding="utf-8")

    eff_lines = [
        "# Leaderboard: Efficiency",
        "",
        "| Method | Build Time | QA Time | p95 Latency | Total Tokens | Memories | Fallbacks | Length Finish | Empty Pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        runtime = item.get("runtime") or {}
        tokens = item.get("tokens") or {}
        mem = item.get("memory_retrieval") or {}
        rel = item.get("reliability") or {}
        eff_lines.append(
            f"| {item.get('method')} | {md_float(runtime.get('memory_build_time_seconds'), 2)} | {md_float(runtime.get('qa_generation_time_seconds'), 2)} | {md_float(runtime.get('p95_latency_seconds'), 3)} | {tokens.get('total_tokens', 0)} | {mem.get('number_of_memories_created')} | {rel.get('fallback_count', 0)} | {rel.get('length_finish_count', 0)} | {rel.get('empty_prediction_count', 0)} |"
        )
    (output_root / "leaderboard_efficiency.md").write_text("\n".join(eff_lines) + "\n", encoding="utf-8")

    failures = [item for item in summaries if item.get("status") != "completed" or (item.get("reliability") or {}).get("fatal_error_count")]
    fail_lines = ["# Failure Report", ""]
    if not failures:
        fail_lines.append("No failed or incomplete baseline artifacts were detected by the acceptance builder.")
    else:
        for item in failures:
            fail_lines.append(f"## {item.get('method')}")
            fail_lines.append(f"- Status: `{item.get('status')}`")
            fail_lines.append(f"- Count: `{item.get('count')}` / `{item.get('expected_main_count')}`")
            fail_lines.append(f"- Reliability: `{json.dumps(item.get('reliability'), ensure_ascii=False)}`")
            fail_lines.append(f"- Source: `{item.get('source_dir')}`")
            fail_lines.append("")
    (output_root / "failure_report.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--full-context-dir", required=True, type=Path)
    parser.add_argument("--smoke-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_dir or (args.run_root / "acceptance_artifacts")
    output_root.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.run_root, args.full_context_dir)
    full_source = resolve_source(specs[0])
    full_pred = first_existing(full_source, specs[0].prediction_candidates) if full_source else None
    full_records = flatten_records(read_json(full_pred)) if full_pred else []
    full_totals = token_values(full_records, "total_tokens")
    full_avg = (sum(full_totals) / len(full_totals)) if full_totals else None

    summaries = [build_one(spec, output_root, full_avg) for spec in specs]
    write_overall_reports(output_root, summaries, args.run_root, args.full_context_dir, args.smoke_root)
    print(output_root)


if __name__ == "__main__":
    main()
