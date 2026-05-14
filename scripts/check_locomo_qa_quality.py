#!/usr/bin/env python3
"""Heuristic QA-quality checks for LoCoMo-style eval primary JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/qa_quality_report.json"))
    parser.add_argument("--min-unique-question-ratio", type=float, default=0.80)
    parser.add_argument("--max-duplicate-question-count", type=int, default=10)
    parser.add_argument("--max-short-answerable-answers", type=int, default=5)
    parser.add_argument("--min-qa-per-sample", type=int, default=20)
    parser.add_argument("--max-qa-per-sample", type=int, default=40)
    args = parser.parse_args()

    data = load_json(args.primary_json)
    question_counts: Counter[str] = Counter()
    by_source_questions: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_sample_qa_counts: dict[str, list[int]] = defaultdict(list)
    sample_qa_counts: list[dict[str, Any]] = []
    answer_lengths: list[int] = []
    short_answers: list[dict[str, Any]] = []
    qa_count_out_of_range: list[dict[str, Any]] = []

    total_qa = 0
    answerable_qa = 0
    for sample in data:
        source = str(sample.get("source_dataset"))
        sample_id = str(sample.get("sample_id"))
        qas = sample.get("qa", [])
        qa_count = len(qas) if isinstance(qas, list) else 0
        by_source_sample_qa_counts[source].append(qa_count)
        sample_qa_record = {
            "source_dataset": source,
            "sample_id": sample_id,
            "qa": qa_count,
        }
        sample_qa_counts.append(sample_qa_record)
        if qa_count < args.min_qa_per_sample or qa_count > args.max_qa_per_sample:
            qa_count_out_of_range.append(sample_qa_record)
        for qa_idx, qa in enumerate(qas):
            total_qa += 1
            question = str(qa.get("question", "")).strip()
            question_counts[question] += 1
            by_source_questions[source][question] += 1
            if qa.get("category") != 5:
                answerable_qa += 1
                answer = str(qa.get("answer", ""))
                answer_lengths.append(len(answer))
                if len(answer.strip()) < 3:
                    short_answers.append(
                        {
                            "source_dataset": source,
                            "sample_id": sample_id,
                            "qa_idx": qa_idx,
                            "category": qa.get("category"),
                            "question": question,
                            "answer": answer,
                        }
                    )

    unique_questions = len(question_counts)
    duplicate_questions = {q: c for q, c in question_counts.items() if c > 1}
    max_duplicate = max(duplicate_questions.values()) if duplicate_questions else 1
    unique_ratio = unique_questions / total_qa if total_qa else 0.0

    errors: list[str] = []
    warnings: list[str] = []
    if unique_ratio < args.min_unique_question_ratio:
        errors.append(
            f"unique_question_ratio={unique_ratio:.4f} below threshold={args.min_unique_question_ratio:.4f}"
        )
    if max_duplicate > args.max_duplicate_question_count:
        errors.append(
            f"max_duplicate_question_count={max_duplicate} exceeds threshold={args.max_duplicate_question_count}"
        )
    if len(short_answers) > args.max_short_answerable_answers:
        errors.append(
            f"short answerable answers={len(short_answers)} exceeds threshold={args.max_short_answerable_answers}"
        )
    if qa_count_out_of_range:
        examples = ", ".join(
            f"{row['source_dataset']}/{row['sample_id']}={row['qa']}"
            for row in qa_count_out_of_range[:10]
        )
        errors.append(
            "sample QA counts outside "
            f"[{args.min_qa_per_sample}, {args.max_qa_per_sample}]: {examples}"
        )
    if duplicate_questions:
        warnings.append("duplicate questions remain; inspect top_duplicate_questions")

    qa_counts = [row["qa"] for row in sample_qa_counts]
    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(args.primary_json),
        },
        "primary_json": str(args.primary_json),
        "thresholds": {
            "min_unique_question_ratio": args.min_unique_question_ratio,
            "max_duplicate_question_count": args.max_duplicate_question_count,
            "max_short_answerable_answers": args.max_short_answerable_answers,
            "min_qa_per_sample": args.min_qa_per_sample,
            "max_qa_per_sample": args.max_qa_per_sample,
        },
        "total_qa": total_qa,
        "answerable_qa": answerable_qa,
        "qa_per_sample": {
            "samples": len(sample_qa_counts),
            "min": min(qa_counts) if qa_counts else 0,
            "mean": round(mean(qa_counts), 2) if qa_counts else 0.0,
            "median": median(qa_counts) if qa_counts else 0.0,
            "max": max(qa_counts) if qa_counts else 0,
            "out_of_range_count": len(qa_count_out_of_range),
            "out_of_range_examples": qa_count_out_of_range[:20],
            "by_source": {
                source: {
                    "samples": len(counts),
                    "min": min(counts) if counts else 0,
                    "mean": round(mean(counts), 2) if counts else 0.0,
                    "median": median(counts) if counts else 0.0,
                    "max": max(counts) if counts else 0,
                }
                for source, counts in sorted(by_source_sample_qa_counts.items())
            },
        },
        "unique_questions": unique_questions,
        "duplicate_question_excess_count": sum(count - 1 for count in duplicate_questions.values()),
        "unique_question_ratio": round(unique_ratio, 6),
        "max_duplicate_question_count": max_duplicate,
        "top_duplicate_questions": [
            {"count": count, "question": question}
            for question, count in question_counts.most_common(20)
            if count > 1
        ],
        "by_source": {
            source: {
                "qa": sum(counter.values()),
                "unique_questions": len(counter),
                "duplicate_question_excess_count": sum(count - 1 for count in counter.values() if count > 1),
                "max_duplicate_question_count": max(counter.values()) if counter else 0,
            }
            for source, counter in sorted(by_source_questions.items())
        },
        "answer_length": {
            "min": min(answer_lengths) if answer_lengths else 0,
            "mean": round(mean(answer_lengths), 2) if answer_lengths else 0.0,
            "median": median(answer_lengths) if answer_lengths else 0.0,
            "max": max(answer_lengths) if answer_lengths else 0,
            "short_answerable_answers": len(short_answers),
            "short_answer_examples": short_answers[:20],
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
