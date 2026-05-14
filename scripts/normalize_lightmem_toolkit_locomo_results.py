#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def first_answer(answer_list: Any) -> str:
    if isinstance(answer_list, (list, tuple)) and answer_list:
        return str(answer_list[0])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flatten LightMem memory_toolkits LoCoMo evaluation output for text metrics."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("LightMem toolkit evaluation output must be a list")

    records = []
    for idx, item in enumerate(data):
        qa_pair = item.get("qa_pair", {})
        metadata = qa_pair.get("metadata", {}) if isinstance(qa_pair, dict) else {}
        category_id = metadata.get("category_id")
        records.append(
            {
                "index": idx,
                "method": args.method,
                "model": args.model,
                "user_id": item.get("user_id"),
                "question": qa_pair.get("question", ""),
                "golden_answer": first_answer(qa_pair.get("answer_list", [])),
                "model_answer": item.get("prediction", ""),
                "qa_error": item.get("qa_error"),
                "qa_finish_reason": item.get("qa_finish_reason"),
                "latency_seconds": item.get("qa_latency_seconds"),
                "token_usage": item.get("qa_token_usage"),
                "category": category_id if category_id is not None else metadata.get("category", "unknown"),
                "question_type": metadata.get("question_type") or metadata.get("category"),
                "is_correct": item.get("is_correct"),
                "judge_response": item.get("judge_response"),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": args.method,
                "model": args.model,
                "source": str(Path(args.input).resolve()),
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(records)} flattened records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
