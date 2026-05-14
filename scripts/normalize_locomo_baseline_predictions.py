#!/usr/bin/env python3
"""Normalize baseline prediction outputs into fixed-eval prediction JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PREDICTION_FIELDS = ("prediction", "answer", "response", "model_answer", "model_output", "output")
DEFAULT_FIXED_MODEL = "Qwen/Qwen3-8B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = list(iter_jsonl(path))
    else:
        payload = load_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("records", "individual_results", "results", "predictions", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
            else:
                raise ValueError("JSON object must contain one of records/individual_results/results/predictions/rows")
        else:
            raise ValueError("input must be a JSON list, JSON object with records, or JSONL rows")
    output = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {idx} is not an object")
        output.append(row)
    return output


def normalize_question(value: Any) -> str:
    return " ".join(str(value or "").split())


def dataset_indexes(dataset_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, str], list[int]]]:
    data = load_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON list")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    by_question: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample in data:
        sample_id = str(sample.get("sample_id"))
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            question = normalize_question(qa.get("question"))
            by_key[(sample_id, qa_idx)] = {"question": question}
            by_question[(sample_id, question)].append(qa_idx)
    return by_key, by_question


def prediction_text(row: dict[str, Any]) -> tuple[str, str | None]:
    for field in PREDICTION_FIELDS:
        if field in row:
            return str(row.get(field, "")), field
    return "", None


def resolve_key(
    row: dict[str, Any],
    row_idx: int,
    by_key: dict[tuple[str, int], dict[str, Any]],
    by_question: dict[tuple[str, str], list[int]],
    errors: list[str],
) -> tuple[str, int] | None:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id and row.get("qa_idx") not in (None, ""):
        try:
            qa_idx = int(row.get("qa_idx"))
        except (TypeError, ValueError):
            errors.append(f"row {row_idx}: qa_idx={row.get('qa_idx')!r} is not an integer")
            return None
        key = (sample_id, qa_idx)
        if key not in by_key:
            errors.append(f"row {row_idx}: key={key} not found in dataset")
            return None
        return key

    question = normalize_question(row.get("question"))
    if sample_id and question:
        matches = by_question.get((sample_id, question), [])
        if len(matches) == 1:
            return (sample_id, matches[0])
        if not matches:
            errors.append(f"row {row_idx}: sample_id/question pair not found in dataset")
        else:
            errors.append(f"row {row_idx}: sample_id/question pair is ambiguous for {len(matches)} QA")
        return None

    errors.append(f"row {row_idx}: cannot resolve QA key; need sample_id+qa_idx or unique sample_id+question")
    return None


def normalize_predictions(dataset_path: Path, input_path: Path, model: str) -> tuple[list[dict[str, Any]], list[str]]:
    by_key, by_question = dataset_indexes(dataset_path)
    dataset_sha256 = sha256_file(dataset_path)
    rows = load_records(input_path)
    errors: list[str] = []
    output: list[dict[str, Any]] = []
    seen: Counter[tuple[str, int]] = Counter()
    for row_idx, row in enumerate(rows, start=1):
        key = resolve_key(row, row_idx, by_key, by_question, errors)
        text, source_field = prediction_text(row)
        if source_field is None:
            errors.append(f"row {row_idx}: missing prediction field {PREDICTION_FIELDS}")
        if key is None or source_field is None:
            continue
        seen[key] += 1
        output.append(
            {
                "sample_id": key[0],
                "qa_idx": key[1],
                "model": model,
                "dataset_sha256": dataset_sha256,
                "prediction": text,
            }
        )

    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate prediction keys: {duplicates[:10]}")
    missing = sorted(set(by_key) - set(seen))
    if missing:
        errors.append(f"missing predictions for {len(missing)} QA; first={missing[:10]}")
    extra = sorted(set(seen) - set(by_key))
    if extra:
        errors.append(f"unexpected prediction keys: {extra[:10]}")
    return output, errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=DEFAULT_FIXED_MODEL,
        help="Model identity to stamp into every normalized prediction row.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    output_rows: list[dict[str, Any]] = []
    try:
        output_rows, errors = normalize_predictions(args.dataset, args.input, args.model)
    except Exception as exc:  # noqa: BLE001 - normalization should emit a structured failure.
        errors = [f"{type(exc).__name__}: {exc}"]

    status = "normalized" if not errors else "failed"
    if status == "normalized":
        write_jsonl(args.output_jsonl, output_rows)

    report = {
        "status": status,
        "dataset": str(args.dataset),
        "input": str(args.input),
        "output_jsonl": str(args.output_jsonl) if status == "normalized" else None,
        "model": args.model,
        "dataset_sha256": sha256_file(args.dataset) if args.dataset.is_file() else None,
        "rows": len(output_rows),
        "errors": errors,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "normalized" else 1


if __name__ == "__main__":
    raise SystemExit(main())
