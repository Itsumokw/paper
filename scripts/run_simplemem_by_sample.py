#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/stu0032/paper")
DEFAULT_DATASET = WORKSPACE / "datasets" / "locomo" / "data" / "locomo10.json"
DEFAULT_WRAPPER = WORKSPACE / "scripts" / "run_simplemem_qwen25_3b_full.sh"
MERGE_SCRIPT = WORKSPACE / "scripts" / "merge_simplemem_results.py"
METRICS_SCRIPT = WORKSPACE / "scripts" / "compute_locomo_text_metrics.py"


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"LoCoMo dataset must be a JSON list: {path}")
    return data


def _category_set(value: str) -> set[str] | None:
    if value == "all":
        return None
    categories = {item.strip() for item in value.split(",") if item.strip()}
    if not categories:
        raise ValueError("--categories must be 'all' or a non-empty comma-separated list")
    return categories


def _write_sample_dataset(sample: dict[str, Any], dst: Path, categories: set[str] | None) -> int:
    item = json.loads(json.dumps(sample, ensure_ascii=False))
    if categories is not None:
        item["qa"] = [qa for qa in item.get("qa", []) if str(qa.get("category")) in categories]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps([item], ensure_ascii=False, indent=2), encoding="utf-8")
    return len(item.get("qa", []))


def _valid_result(path: Path, expected_qa: int) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = obj.get("summary") or {}
    details = obj.get("detailed_results") or []
    return summary.get("num_samples") == 1 and summary.get("num_questions") == expected_qa and len(details) == expected_qa


def _estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    value = str(text)
    return max(1, len(value) // 4) if value else 0


def _write_normalized(result_path: Path, output_path: Path) -> None:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    records = []
    for idx, row in enumerate(data.get("detailed_results") or []):
        prediction = row.get("answer", "")
        records.append(
            {
                "source": "simplemem",
                "qa_idx": idx,
                "category": row.get("category"),
                "question": row.get("question", ""),
                "prediction": prediction,
                "reference": row.get("reference", ""),
                "latency_seconds": row.get("total_time"),
                "retrieval_latency_seconds": row.get("retrieval_time"),
                "answer_latency_seconds": row.get("answer_time"),
                "retrieved_memory_count": row.get("num_retrieved"),
                "token_usage": {
                    "prompt_tokens": _estimate_tokens(row.get("question", "")),
                    "completion_tokens": _estimate_tokens(prediction),
                    "total_tokens": _estimate_tokens(row.get("question", "")) + _estimate_tokens(prediction),
                    "note": (
                        "SimpleMem per-question prompt context is not saved by the upstream evaluator; "
                        "prompt_tokens here count the visible question only. Full LLM-call aggregate is in "
                        "result.summary.llm_usage."
                    ),
                },
            }
        )
    output_path.write_text(
        json.dumps(
            {
                "records": records,
                "summary": {
                    "method": "SimpleMem",
                    "source_result": str(result_path),
                    "num_records": len(records),
                    "latency": data.get("summary", {}),
                    "llm_usage": (data.get("summary") or {}).get("llm_usage"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_cat14(src: Path, dst: Path, expected_qa: int) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    records = [row for row in data.get("records", []) if str(row.get("category")) in {"1", "2", "3", "4"}]
    if len(records) != expected_qa:
        raise RuntimeError(f"Expected {expected_qa} SimpleMem cat1-4 rows, got {len(records)}")
    dst.write_text(json.dumps({"records": records, "summary": data.get("summary", {})}, ensure_ascii=False, indent=2))


def _run_metrics(python: Path, normalized: Path, output: Path, skip_bertscore: bool) -> None:
    cmd = [
        str(python),
        str(METRICS_SCRIPT),
        "--input",
        str(normalized),
        "--output",
        str(output),
        "--prediction-key",
        "prediction",
        "--reference-key",
        "reference",
    ]
    if skip_bertscore:
        cmd.append("--skip-bertscore")
    else:
        cmd.extend(
            [
                "--bertscore-model",
                os.environ.get("BERTSCORE_MODEL", "roberta-large"),
                "--bertscore-batch-size",
                os.environ.get("BERTSCORE_BATCH_SIZE", "4"),
                "--bertscore-num-layers",
                os.environ.get("BERTSCORE_NUM_LAYERS", "17"),
                "--bertscore-device",
                os.environ.get("BERTSCORE_DEVICE", "cpu"),
                "--fail-on-bertscore-error",
            ]
        )
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SimpleMem on LoCoMo one sample at a time and merge results.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--python", default=str(WORKSPACE / ".venv" / "bin" / "python"))
    parser.add_argument("--wrapper", default=str(DEFAULT_WRAPPER))
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--end-sample", type=int, default=-1, help="Inclusive sample index, -1 means last.")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    run_root = Path(args.run_root).resolve()
    python = Path(args.python).resolve()
    wrapper = Path(args.wrapper).resolve()
    categories = _category_set(args.categories)
    samples = _load_dataset(dataset)
    end_sample = args.end_sample if args.end_sample >= 0 else len(samples) - 1
    if args.start_sample < 0 or end_sample >= len(samples) or args.start_sample > end_sample:
        parser.error(f"invalid sample range {args.start_sample}..{end_sample} for {len(samples)} samples")

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "command.env").write_text(
        "\n".join(
            [
                f"SIMPLEMEM_BY_SAMPLE=1",
                f"SIMPLEMEM_SOURCE_DATASET={dataset}",
                f"SIMPLEMEM_CATEGORIES={args.categories}",
                f"SIMPLEMEM_RUN_ROOT={run_root}",
                f"SIMPLEMEM_START_SAMPLE={args.start_sample}",
                f"SIMPLEMEM_END_SAMPLE={end_sample}",
                f"SIMPLEMEM_MAX_OUTPUT_TOKENS={os.environ.get('SIMPLEMEM_MAX_OUTPUT_TOKENS', '')}",
                f"SIMPLEMEM_TEST_WORKERS={os.environ.get('SIMPLEMEM_TEST_WORKERS', '')}",
                f"SIMPLEMEM_BUILD_WORKERS={os.environ.get('SIMPLEMEM_BUILD_WORKERS', '')}",
                f"SIMPLEMEM_RETRIEVAL_WORKERS={os.environ.get('SIMPLEMEM_RETRIEVAL_WORKERS', '')}",
                f"OPENAI_MODEL={os.environ.get('OPENAI_MODEL', '')}",
                f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', '')}",
                f"LOCOMO_HIGH_UTILIZATION_AFTER_AMEM={os.environ.get('LOCOMO_HIGH_UTILIZATION_AFTER_AMEM', '')}",
                f"LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART={os.environ.get('LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_GPU_MEMORY_UTILIZATION={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_GPU_MEMORY_UTILIZATION', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_MAX_MODEL_LEN={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_MAX_MODEL_LEN', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_SEQS={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_SEQS', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_BATCHED_TOKENS={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_BATCHED_TOKENS', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_GENERATION_CONFIG={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_GENERATION_CONFIG', '')}",
                f"LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS={os.environ.get('LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS', '')}",
                f"VLLM_MAX_NUM_SEQS={os.environ.get('VLLM_MAX_NUM_SEQS', '')}",
                f"VLLM_MAX_NUM_BATCHED_TOKENS={os.environ.get('VLLM_MAX_NUM_BATCHED_TOKENS', '')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sample_infos: list[tuple[int, Path, Path, int]] = []
    total_expected_qa = 0
    for sample_idx in range(args.start_sample, end_sample + 1):
        sample_dataset = run_root / "sample_datasets" / f"sample_{sample_idx:02d}.json"
        expected_qa = _write_sample_dataset(samples[sample_idx], sample_dataset, categories)
        total_expected_qa += expected_qa
        sample_out = run_root / "samples" / f"sample_{sample_idx:02d}"
        sample_infos.append((sample_idx, sample_dataset, sample_out, expected_qa))

    manifest = {
        "dataset": str(dataset),
        "run_root": str(run_root),
        "categories": args.categories,
        "start_sample": args.start_sample,
        "end_sample": end_sample,
        "expected_samples": len(sample_infos),
        "expected_qa": total_expected_qa,
        "samples": [
            {
                "sample_idx": sample_idx,
                "dataset": str(sample_dataset),
                "output_dir": str(sample_out),
                "expected_qa": expected_qa,
            }
            for sample_idx, sample_dataset, sample_out, expected_qa in sample_infos
        ],
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for sample_idx, sample_dataset, sample_out, expected_qa in sample_infos:
        result_path = sample_out / "result.json"
        if _valid_result(result_path, expected_qa) and not args.force_rerun:
            print(f"[simplemem-by-sample] skip sample {sample_idx}: {result_path}")
            continue
        env = os.environ.copy()
        env.update(
            {
                "SIMPLEMEM_RUN_ROOT": str(sample_out),
                "SIMPLEMEM_SOURCE_DATASET": str(sample_dataset),
                "SIMPLEMEM_CATEGORIES": "all",
                "SIMPLEMEM_EXPECTED_SAMPLES": "1",
                "SIMPLEMEM_EXPECTED_QA": str(expected_qa),
                "SIMPLEMEM_EXPECTED_CAT14_QA": str(expected_qa),
                "SIMPLEMEM_SKIP_BERTSCORE": "1",
            }
        )
        sample_out.mkdir(parents=True, exist_ok=True)
        print(f"[simplemem-by-sample] running sample {sample_idx} expected_qa={expected_qa}")
        subprocess.run(["bash", str(wrapper)], cwd=str(WORKSPACE), env=env, check=True)
        if not _valid_result(result_path, expected_qa):
            raise RuntimeError(f"sample {sample_idx} did not produce a valid result: {result_path}")

    result_files = [sample_out / "result.json" for _, _, sample_out, _ in sample_infos]
    subprocess.run(
        [str(python), str(MERGE_SCRIPT), "--inputs", *[str(path) for path in result_files], "--output", str(run_root / "result.json")],
        cwd=str(WORKSPACE),
        check=True,
    )

    merged = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
    if merged.get("summary", {}).get("num_questions") != total_expected_qa:
        raise RuntimeError(f"merged SimpleMem QA count mismatch: {merged.get('summary')}")
    if len(merged.get("detailed_results", [])) != total_expected_qa:
        raise RuntimeError("merged SimpleMem detailed result count mismatch")
    if any(not str(row.get("answer") or "").strip() for row in merged.get("detailed_results", [])):
        raise RuntimeError("merged SimpleMem contains empty answers")

    _write_normalized(run_root / "result.json", run_root / "normalized_predictions.json")
    _write_cat14(run_root / "normalized_predictions.json", run_root / "normalized_predictions_cat1_4.json", total_expected_qa)

    skip_final_bertscore = os.environ.get("SIMPLEMEM_SKIP_BERTSCORE", "0") == "1"
    _run_metrics(python, run_root / "normalized_predictions.json", run_root / "simplemem_metrics_all.json", skip_final_bertscore)
    _run_metrics(python, run_root / "normalized_predictions_cat1_4.json", run_root / "simplemem_metrics_cat1_4.json", skip_final_bertscore)
    print(f"[simplemem-by-sample] completed: {run_root / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
