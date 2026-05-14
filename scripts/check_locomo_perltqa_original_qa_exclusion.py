#!/usr/bin/env python3
"""Check that PerLTQA original QA questions are not copied into final eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def iter_source_qa(node: Any) -> Iterator[dict[str, str]]:
    if isinstance(node, dict):
        if "Question" in node and "Answer" in node:
            yield {
                "question": normalize_text(node.get("Question")),
                "answer": normalize_text(node.get("Answer")),
            }
        for value in node.values():
            yield from iter_source_qa(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_source_qa(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary/PerLTQA-LoCoMo-style-eval.json"),
    )
    parser.add_argument(
        "--source-qa",
        type=Path,
        default=Path("datasets/PerLTQA/Dataset/zh/perltqa.json"),
    )
    parser.add_argument(
        "--construction-report",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/PerLTQA-LoCoMo-style-eval/"
            "PerLTQA-LoCoMo-style-eval_construction_report.md"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/perltqa_original_qa_exclusion_report.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    source_qas = list(iter_source_qa(json.loads(args.source_qa.read_text(encoding="utf-8"))))
    source_questions = {row["question"] for row in source_qas if row["question"]}
    source_pairs = {
        (row["question"], row["answer"])
        for row in source_qas
        if row["question"] and row["answer"]
    }

    primary = json.loads(args.primary_json.read_text(encoding="utf-8"))
    perltqa_qa_count = 0
    direct_question_matches: list[dict[str, Any]] = []
    direct_question_answer_matches: list[dict[str, Any]] = []
    for sample in primary:
        if sample.get("source_dataset") != "PerLTQA":
            continue
        sample_id = str(sample.get("sample_id"))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            perltqa_qa_count += 1
            question = normalize_text(qa.get("question"))
            answer = normalize_text(qa.get("answer"))
            if question in source_questions:
                direct_question_matches.append(
                    {"sample_id": sample_id, "qa_idx": qa_idx, "question": question}
                )
            if (question, answer) in source_pairs:
                direct_question_answer_matches.append(
                    {"sample_id": sample_id, "qa_idx": qa_idx, "question": question}
                )

    if direct_question_matches:
        errors.append(
            "PerLTQA final eval contains questions copied exactly from original PerLTQA QA; "
            f"count={len(direct_question_matches)} first={direct_question_matches[:5]}"
        )
    if direct_question_answer_matches:
        errors.append(
            "PerLTQA final eval contains exact original PerLTQA question+answer pairs; "
            f"count={len(direct_question_answer_matches)} first={direct_question_answer_matches[:5]}"
        )

    report_text = args.construction_report.read_text(encoding="utf-8")
    required_phrase = "original PerLTQA QA is not copied into final eval"
    forbidden_phrase = "original QA is only included when mapped"
    if required_phrase not in report_text:
        errors.append(f"construction report missing phrase: {required_phrase!r}")
    if forbidden_phrase in report_text:
        errors.append(f"construction report still contains forbidden phrase: {forbidden_phrase!r}")

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(args.primary_json),
            "source_qa": file_record(args.source_qa),
            "construction_report": file_record(args.construction_report),
        },
        "source_qa_question_count": len(source_questions),
        "source_qa_pair_count": len(source_pairs),
        "perltqa_final_qa_count": perltqa_qa_count,
        "direct_question_match_count": len(direct_question_matches),
        "direct_question_answer_match_count": len(direct_question_answer_matches),
        "direct_question_match_examples": direct_question_matches[:10],
        "direct_question_answer_match_examples": direct_question_answer_matches[:10],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
