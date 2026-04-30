#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def simple_tokenize(text: Any) -> list[str]:
    return (
        str(text)
        .lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


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
    try:
        pred_tokens = nltk.word_tokenize(str(prediction).lower())
        ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    except Exception:
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
    for key in ("records", "per_item", "question_answering_records", "results", "qa"):
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
        "--bertscore-device",
        default="cpu",
        help="Device for BERTScore calculation; default cpu avoids vLLM GPU OOM",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore calculation",
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

    bertscore_error = None
    if not args.skip_bertscore and per_item:
        try:
            from bert_score import BERTScorer

            scorer = BERTScorer(
                model_type=args.bertscore_model,
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


if __name__ == "__main__":
    main()
