#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from string import Template
from typing import Any

from openai import OpenAI


_THREAD_LOCAL = threading.local()

LOCOMO_JUDGE_PROMPT = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: "
    "(1) a question (posed by one user to another user), "
    "(2) a 'gold' (ground truth) answer, "
    "(3) a generated answer "
    "which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
    "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
    "Now it's time for the real question:\n"
    "Question: $question\n"
    "Gold answer: $golden_answers\n"
    "Generated answer: $prediction\n\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
    "Just return the label CORRECT or WRONG in a json format with the key as 'label'."
)

PROTOCOL_VERSIONS = {
    "locomo_binary": "lightmem-locomo-judge-v1",
    "semantic_3way": "custom-semantic-3way-v1",
}


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise TypeError("Input JSON must be a dict or list")

    records: list[dict[str, Any]] = []
    for key in (
        "records",
        "per_item",
        "question_answering_records",
        "individual_results",
        "results",
        "qa",
    ):
        value = data.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    if records:
        return records

    for value in data.values():
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def trim_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def record_key(record: dict[str, Any], index: int, args: argparse.Namespace) -> str:
    payload = {
        "protocol": args.protocol,
        "protocol_version": PROTOCOL_VERSIONS[args.protocol],
        "index": record.get("index", index),
        "user_id": record.get("user_id"),
        "question": record.get(args.question_key, ""),
        "reference": record.get(args.reference_key, ""),
        "prediction": record.get(args.prediction_key, ""),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def client_for_thread(args: argparse.Namespace) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url.rstrip("/"),
            timeout=args.request_timeout,
            max_retries=0,
        )
        _THREAD_LOCAL.client = client
    return client


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    score_match = re.search(r'"?score"?\s*:\s*(-?\d+(?:\.\d+)?)', text, flags=re.I)
    verdict_match = re.search(r'"?verdict"?\s*:\s*"?([a-zA-Z]+)', text, flags=re.I)
    if score_match:
        verdict = verdict_match.group(1).lower() if verdict_match else ""
        return {
            "score": float(score_match.group(1)),
            "verdict": verdict or ("correct" if float(score_match.group(1)) >= 1 else "partial" if float(score_match.group(1)) >= 0.5 else "incorrect"),
            "reason": text.strip()[:240],
        }
    if "correct" in lowered and "incorrect" not in lowered:
        return {"score": 1.0, "verdict": "correct", "reason": text.strip()[:240]}
    if "partial" in lowered:
        return {"score": 0.5, "verdict": "partial", "reason": text.strip()[:240]}
    return {"score": 0.0, "verdict": "incorrect", "reason": text.strip()[:240]}


def normalized_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        score = float(match.group(0)) if match else 0.0
    return max(0.0, min(1.0, score))


def parse_locomo_binary_response(content: str) -> tuple[float, str, str]:
    label = ""
    reason = ""
    try:
        parsed = extract_json_object(content)
        label = str(parsed.get("label") or parsed.get("verdict") or "").strip().lower()
        reason = str(parsed.get("reason") or parsed.get("explanation") or "").strip()
    except Exception:  # noqa: BLE001
        label = ""

    text = label or content.lower()
    if re.search(r"\b(wrong|incorrect|no)\b", text) or re.search(r"\bnot\s+correct\b", text):
        return 0.0, "wrong", reason or content.strip()[:500]
    if re.search(r"\b(correct|yes)\b", text):
        return 1.0, "correct", reason or content.strip()[:500]
    return 0.0, "wrong", reason or content.strip()[:500]


def build_messages(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    question = trim_text(record.get(args.question_key, ""), args.max_field_chars)
    reference = trim_text(record.get(args.reference_key, ""), args.max_field_chars)
    prediction = trim_text(record.get(args.prediction_key, ""), args.max_field_chars)
    category = str(record.get(args.category_key, "unknown"))

    if args.protocol == "locomo_binary":
        return [
            {
                "role": "user",
                "content": Template(LOCOMO_JUDGE_PROMPT).substitute(
                    question=question,
                    golden_answers=reference,
                    prediction=prediction,
                ),
            }
        ]

    payload = {
        "category": category,
        "question": question,
        "reference_answer": reference,
        "candidate_answer": prediction,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict but semantics-aware judge for LoCoMo-style "
                "memory question answering. Decide whether the candidate answer "
                "answers the question with the same information as the reference. "
                "Accept paraphrases, equivalent wording, harmless extra context, "
                "and punctuation/casing differences. For Chinese, Japanese, and "
                "Korean, judge semantic equivalence instead of whitespace overlap. "
                "Use score 1 for fully correct, 0.5 for partially correct but "
                "missing a key detail, and 0 for wrong, contradicted, empty, or "
                "unrelated answers. Return only JSON with keys score, verdict, "
                "reason."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def judge_record(index: int, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prediction = str(record.get(args.prediction_key, "") or "").strip()
    reference = str(record.get(args.reference_key, "") or "").strip()
    item = {
        "record_index": record.get("index", index),
        "record_key": record_key(record, index, args),
        "category": str(record.get(args.category_key, "unknown")),
        "question": record.get(args.question_key, ""),
        "reference": reference,
        "prediction": prediction,
        "judge_score": None,
        "judge_correct": None,
        "judge_acceptable": None,
        "judge_verdict": None,
        "judge_reason": None,
        "judge_raw_response": None,
        "judge_error": None,
        "judge_protocol": args.protocol,
    }

    if not prediction:
        item.update(
            {
                "judge_score": 0.0,
                "judge_correct": False,
                "judge_acceptable": False,
                "judge_verdict": "incorrect",
                "judge_reason": "Empty candidate answer.",
            }
        )
        return item
    if not reference:
        item.update(
            {
                "judge_score": 0.0,
                "judge_correct": False,
                "judge_acceptable": False,
                "judge_verdict": "incorrect",
                "judge_reason": "Missing reference answer.",
            }
        )
        return item

    last_error: str | None = None
    for attempt in range(1, args.max_retries + 2):
        try:
            response = client_for_thread(args).chat.completions.create(
                model=args.model,
                messages=build_messages(record, args),
                temperature=0,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content or ""
            if args.protocol == "locomo_binary":
                score, verdict, reason = parse_locomo_binary_response(content)
            else:
                parsed = extract_json_object(content)
                score = normalized_score(parsed.get("score", 0.0))
                verdict = str(parsed.get("verdict") or "").strip().lower()
                if verdict not in {"correct", "partial", "incorrect"}:
                    verdict = "correct" if score >= 1.0 else "partial" if score >= 0.5 else "incorrect"
                reason = str(parsed.get("reason") or "").strip()[:500]
            item.update(
                {
                    "judge_score": score,
                    "judge_correct": score >= args.correct_threshold,
                    "judge_acceptable": score >= args.acceptable_threshold,
                    "judge_verdict": verdict,
                    "judge_reason": reason,
                    "judge_raw_response": content,
                }
            )
            return item
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= args.max_retries:
                time.sleep(args.retry_sleep * attempt)

    item["judge_error"] = last_error
    return item


def load_cached_items(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for item in data.get("per_item", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        key = item.get("record_key")
        if key and item.get("judge_error") is None and item.get("judge_score") is not None:
            cached[str(key)] = item
    return cached


def summarize_scores(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    scores = [float(item["judge_score"]) for item in items if item.get("judge_score") is not None]
    correct = [1.0 if item.get("judge_correct") else 0.0 for item in items if item.get("judge_score") is not None]
    acceptable = [
        1.0 if item.get("judge_acceptable") else 0.0
        for item in items
        if item.get("judge_score") is not None
    ]
    return {
        "judge_score": {"mean": mean(scores) if scores else 0.0, "count": len(scores)},
        "judge_correct": {
            "mean": mean(correct) if correct else 0.0,
            "count": len(correct),
            "threshold": args.correct_threshold,
        },
        "judge_acceptable": {
            "mean": mean(acceptable) if acceptable else 0.0,
            "count": len(acceptable),
            "threshold": args.acceptable_threshold,
        },
        "judge_errors": sum(1 for item in items if item.get("judge_error")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute LLM-as-a-judge metrics for LoCoMo-style QA predictions."
    )
    parser.add_argument("--input", required=True, help="Input prediction JSON")
    parser.add_argument("--output", required=True, help="Output judge metrics JSON")
    parser.add_argument("--prediction-key", default="model_answer")
    parser.add_argument("--reference-key", default="golden_answer")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--category-key", default="category")
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOL_VERSIONS),
        default=os.environ.get("LOCOMO_JUDGE_PROTOCOL")
        or os.environ.get("LLM_JUDGE_PROTOCOL")
        or "locomo_binary",
        help="Judge protocol. locomo_binary matches LightMem's LoCoMo locomo-judge prompt.",
    )
    parser.add_argument("--model", default=os.environ.get("LLM_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("LLM_JUDGE_WORKERS", "8")))
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-field-chars", type=int, default=3000)
    parser.add_argument("--correct-threshold", type=float, default=1.0)
    parser.add_argument("--acceptable-threshold", type=float, default=0.5)
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for smoke runs")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-item judge rows from output")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = flatten_records(json.loads(input_path.read_text(encoding="utf-8")))
    if args.max_records > 0:
        records = records[: args.max_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_items(output_path) if args.resume else {}
    per_item: list[dict[str, Any] | None] = [None] * len(records)
    futures = {}
    started = time.time()
    reused_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for index, record in enumerate(records):
            key = record_key(record, index, args)
            if key in cached:
                per_item[index] = cached[key]
                reused_count += 1
                continue
            futures[executor.submit(judge_record, index, record, args)] = index

        completed = len(records) - len(futures)
        for future in as_completed(futures):
            index = futures[future]
            per_item[index] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(records):
                print(f"judged {completed}/{len(records)}", flush=True)

    items = [item for item in per_item if item is not None]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_category[str(item.get("category", "unknown"))].append(item)

    result = {
        "input": str(input_path.resolve()),
        "count": len(items),
        "judge": {
            "model": args.model,
            "base_url": args.base_url,
            "protocol": args.protocol,
            "protocol_version": PROTOCOL_VERSIONS[args.protocol],
            "prompt_name": "locomo-judge" if args.protocol == "locomo_binary" else "custom-semantic-3way",
            "max_workers": args.max_workers,
            "correct_threshold": args.correct_threshold,
            "acceptable_threshold": args.acceptable_threshold,
        },
        "overall": summarize_scores(items, args),
        "categories": {
            category: summarize_scores(category_items, args)
            for category, category_items in sorted(by_category.items())
        },
        "runtime": {
            "seconds": time.time() - started,
            "reused_cached": reused_count,
        },
        "per_item": items,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, ensure_ascii=False, indent=2))

    if args.fail_on_error and result["overall"]["judge_errors"]:
        raise SystemExit(f"Judge errors: {result['overall']['judge_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
