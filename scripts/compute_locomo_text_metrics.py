#!/usr/bin/env python3
import argparse
import json
import re
import string
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def simple_tokenize(text: Any) -> list[str]:
    """Tokenize English/space-separated text and CJK text without whitespace.

    The original LoCoMo metric used whitespace tokens for English answers.
    Multilingual LoCoMo-style answers often contain Chinese, Japanese, or Korean
    with no spaces, so splitting only on whitespace turns whole sentences into
    single tokens and drives F1/ROUGE to zero even for partial matches.
    """
    value = normalize_answer(text)
    tokens: list[str] = []
    for chunk in value.split():
        if CJK_RE.search(chunk):
            tokens.extend(match.group(0) for match in TOKEN_RE.finditer(chunk))
        else:
            tokens.append(chunk)
    return tokens


def normalize_answer(text: Any) -> str:
    """LoCoMo-style answer normalization used for all text metrics."""
    value = "" if text is None else str(text)
    value = value.lower()
    value = value.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    value = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = " ".join(value.split())
    return value


def f1_score(prediction: Any, reference: Any) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def bleu1_score(prediction: Any, reference: Any) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = [simple_tokenize(reference)]
    try:
        return sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return 0.0


def rouge_l_score(prediction: Any, reference: Any) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    prev = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        curr = [0]
        for idx, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise TypeError("Input JSON must be a dict or list")
    records: list[dict[str, Any]] = []
    for key in ("records", "per_item", "question_answering_records", "individual_results", "results", "qa"):
        value = data.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    if records:
        return records
    for value in data.values():
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(values) if values else 0.0,
        "count": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LoCoMo token F1 and BLEU-1 from generated QA outputs."
    )
    parser.add_argument("--input", required=True, help="Input QA output JSON")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    parser.add_argument(
        "--prediction-key",
        default="model_answer",
        help="Prediction field name in each QA record",
    )
    parser.add_argument(
        "--reference-key",
        default="golden_answer",
        help="Reference answer field name in each QA record",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="Question field name in each QA record",
    )
    parser.add_argument(
        "--category-key",
        default="category",
        help="Category field name in each QA record",
    )
    parser.add_argument(
        "--bertscore-model",
        default="roberta-large",
        help="BERTScore model_type to use",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="BERTScore batch size",
    )
    parser.add_argument(
        "--bertscore-num-layers",
        type=int,
        default=None,
        help="Explicit BERTScore layer count; required when model_type is a local path",
    )
    parser.add_argument(
        "--bertscore-device",
        default="cpu",
        help="Device for BERTScore calculation; default cpu avoids vLLM GPU OOM",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore calculation",
    )
    parser.add_argument(
        "--fail-on-bertscore-error",
        action="store_true",
        help="Exit nonzero if BERTScore calculation is enabled but fails",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = flatten_records(json.load(f))

    per_item = []
    by_category: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "bleu1": [], "rouge_l": [], "bertscore_f1": []}
    )
    overall = {"f1": [], "bleu1": [], "rouge_l": [], "bertscore_f1": []}
    predictions: list[str] = []
    references: list[str] = []
    latencies: list[float] = []
    token_usages: list[dict[str, Any]] = []

    for record in records:
        category = str(record.get(args.category_key, "unknown"))
        prediction = record.get(args.prediction_key, "")
        reference = record.get(args.reference_key, "")
        item_f1 = f1_score(prediction, reference)
        item_bleu1 = bleu1_score(prediction, reference)
        item_rouge_l = rouge_l_score(prediction, reference)

        per_item.append(
            {
                "category": category,
                "question": record.get(args.question_key, ""),
                "f1": item_f1,
                "bleu1": item_bleu1,
                "rouge_l": item_rouge_l,
            }
        )
        by_category[category]["f1"].append(item_f1)
        by_category[category]["bleu1"].append(item_bleu1)
        by_category[category]["rouge_l"].append(item_rouge_l)
        overall["f1"].append(item_f1)
        overall["bleu1"].append(item_bleu1)
        overall["rouge_l"].append(item_rouge_l)
        predictions.append(str(prediction))
        references.append(str(reference))
        latency = record.get("latency_seconds")
        if latency is None:
            latency = record.get("qa_latency_seconds")
        if latency is not None:
            try:
                latencies.append(float(latency))
            except (TypeError, ValueError):
                pass
        token_usage = record.get("token_usage")
        if token_usage is None:
            token_usage = record.get("qa_token_usage")
        if isinstance(token_usage, dict):
            token_usages.append(token_usage)

    bertscore_error = None
    if not args.skip_bertscore and per_item:
        try:
            from bert_score import BERTScorer

            scorer = BERTScorer(
                model_type=args.bertscore_model,
                num_layers=args.bertscore_num_layers,
                lang="en",
                rescale_with_baseline=False,
                device=args.bertscore_device,
            )
            _, _, f1_values = scorer.score(
                predictions,
                references,
                batch_size=args.bertscore_batch_size,
            )
            for item, score in zip(per_item, f1_values.tolist()):
                item["bertscore_f1"] = float(score)
                category = str(item["category"])
                by_category[category]["bertscore_f1"].append(float(score))
                overall["bertscore_f1"].append(float(score))
        except Exception as exc:  # noqa: BLE001
            bertscore_error = str(exc)
            if args.fail_on_bertscore_error:
                raise
            for item in per_item:
                item["bertscore_f1"] = None
    else:
        for item in per_item:
            item["bertscore_f1"] = None

    result = {
        "input": str(Path(args.input).resolve()),
        "count": len(per_item),
        "bertscore": {
            "enabled": not args.skip_bertscore,
            "model": args.bertscore_model if not args.skip_bertscore else None,
            "error": bertscore_error,
        },
        "overall": {
            "f1": summarize(overall["f1"]),
            "bleu1": summarize(overall["bleu1"]),
            "rouge_l": summarize(overall["rouge_l"]),
            "bertscore_f1": summarize(overall["bertscore_f1"]),
        },
        "runtime": {
            "latency": {
                "mean_seconds": mean(latencies) if latencies else 0.0,
                "total_seconds": sum(latencies),
                "max_seconds": max(latencies) if latencies else 0.0,
                "count": len(latencies),
            },
            "tokens": {
                "avg_prompt_tokens": (
                    mean([float(item.get("prompt_tokens") or 0) for item in token_usages])
                    if token_usages else 0.0
                ),
                "avg_completion_tokens": (
                    mean([float(item.get("completion_tokens") or 0) for item in token_usages])
                    if token_usages else 0.0
                ),
                "avg_total_tokens": (
                    mean([float(item.get("total_tokens") or 0) for item in token_usages])
                    if token_usages else 0.0
                ),
                "total_prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in token_usages),
                "total_completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in token_usages),
                "total_tokens": sum(int(item.get("total_tokens") or 0) for item in token_usages),
                "count": len(token_usages),
            },
        },
        "categories": {
            category: {
                "f1": summarize(values["f1"]),
                "bleu1": summarize(values["bleu1"]),
                "rouge_l": summarize(values["rouge_l"]),
                "bertscore_f1": summarize(values["bertscore_f1"]),
            }
            for category, values in sorted(by_category.items())
        },
        "per_item": per_item,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, indent=2))
    if not args.skip_bertscore and bertscore_error:
        raise SystemExit(f"BERTScore failed: {bertscore_error}")


if __name__ == "__main__":
    main()
