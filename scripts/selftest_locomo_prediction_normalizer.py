#!/usr/bin/env python3
"""Self-test baseline prediction normalization for fixed LoCoMo-style eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_dataset(path: Path, *, duplicate_question: bool = False) -> None:
    second_question = "What does Alice like?" if duplicate_question else "What does Bob remember?"
    write_json(
        path,
        [
            {
                "sample_id": "fixture_0",
                "source_dataset": "Fixture",
                "language": "en",
                "split": "eval",
                "conversation": {},
                "observation": {},
                "session_summary": {},
                "event_summary": {},
                "qa": [
                    {"question": "What does Alice like?", "answer": "tea", "category": 1, "evidence": ["D1:1"]},
                    {"question": second_question, "answer": "tea", "category": 2, "evidence": ["D1:1", "D2:1"]},
                ],
            }
        ],
    )


def run_json(command: list[str]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    if completed.stderr.strip() and completed.stdout.strip():
        parsed["stderr"] = completed.stderr
    return completed.returncode, parsed


def case_result(
    name: str,
    returncode: int,
    result: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = "; ".join(str(item) for item in result.get("errors", []))
    observed_success = returncode == 0 and result.get("status") == "normalized"
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None or expected_error_fragment in errors
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "observed_status": result.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": result.get("errors", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalizer", type=Path, default=Path("scripts/normalize_locomo_baseline_predictions.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/prediction_normalizer_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_prediction_normalizer_selftest_") as tmp:
        tempdir = Path(tmp)
        dataset = tempdir / "dataset.json"
        write_dataset(dataset)

        keyed_input = tempdir / "keyed.jsonl"
        write_jsonl(
            keyed_input,
            [
                {"sample_id": "fixture_0", "qa_idx": 0, "prediction": "tea"},
                {"sample_id": "fixture_0", "qa_idx": 1, "response": "Bob remembers tea"},
            ],
        )
        keyed_output = tempdir / "keyed_out.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(keyed_input),
                "--output-jsonl",
                str(keyed_output),
            ]
        )
        keyed_rows = read_jsonl(keyed_output) if keyed_output.exists() else []
        keyed_case = case_result(
            "sample_id_qa_idx_rows_are_normalized",
            rc,
            result,
            expect_success=True,
        )
        expected_dataset_sha256 = sha256_file(dataset)
        if not all(row.get("model") == "Qwen/Qwen3-8B" for row in keyed_rows):
            keyed_case["status"] = "failed"
            keyed_case["errors"] = ["normalized rows must include model=Qwen/Qwen3-8B"]
        if not all(row.get("dataset_sha256") == expected_dataset_sha256 for row in keyed_rows):
            keyed_case["status"] = "failed"
            keyed_case["errors"] = ["normalized rows must include dataset_sha256 for the target dataset"]
        cases.append(
            keyed_case | {"output_rows": keyed_rows}
        )

        object_input = tempdir / "records_object.json"
        write_json(
            object_input,
            {
                "records": [
                    {"sample_id": "fixture_0", "question": "What does Alice like?", "model_answer": "tea"},
                    {"sample_id": "fixture_0", "question": "What does Bob remember?", "answer": "tea"},
                ]
            },
        )
        object_output = tempdir / "records_object_out.jsonl"
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(object_input),
                "--output-jsonl",
                str(object_output),
            ]
        )
        cases.append(
            case_result(
                "unique_sample_question_rows_are_normalized",
                rc,
                result,
                expect_success=True,
            )
        )

        missing_input = tempdir / "missing.jsonl"
        write_jsonl(missing_input, [{"sample_id": "fixture_0", "qa_idx": 0, "prediction": "tea"}])
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(missing_input),
                "--output-jsonl",
                str(tempdir / "missing_out.jsonl"),
            ]
        )
        cases.append(
            case_result(
                "missing_predictions_are_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="missing predictions",
            )
        )

        duplicate_input = tempdir / "duplicate_keyed.jsonl"
        write_jsonl(
            duplicate_input,
            [
                {"sample_id": "fixture_0", "qa_idx": 0, "prediction": "tea"},
                {"sample_id": "fixture_0", "qa_idx": 0, "prediction": "tea again"},
                {"sample_id": "fixture_0", "qa_idx": 1, "prediction": "Bob remembers tea"},
            ],
        )
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(duplicate_input),
                "--output-jsonl",
                str(tempdir / "duplicate_keyed_out.jsonl"),
            ]
        )
        cases.append(
            case_result(
                "duplicate_prediction_keys_are_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="duplicate prediction keys",
            )
        )

        unknown_key_input = tempdir / "unknown_key.jsonl"
        write_jsonl(
            unknown_key_input,
            [
                {"sample_id": "fixture_0", "qa_idx": 0, "prediction": "tea"},
                {"sample_id": "fixture_0", "qa_idx": 1, "prediction": "Bob remembers tea"},
                {"sample_id": "fixture_0", "qa_idx": 99, "prediction": "extra"},
            ],
        )
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(unknown_key_input),
                "--output-jsonl",
                str(tempdir / "unknown_key_out.jsonl"),
            ]
        )
        cases.append(
            case_result(
                "unknown_prediction_keys_are_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="not found in dataset",
            )
        )

        missing_field_input = tempdir / "missing_prediction_field.jsonl"
        write_jsonl(
            missing_field_input,
            [
                {"sample_id": "fixture_0", "qa_idx": 0},
                {"sample_id": "fixture_0", "qa_idx": 1, "prediction": "Bob remembers tea"},
            ],
        )
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(dataset),
                "--input",
                str(missing_field_input),
                "--output-jsonl",
                str(tempdir / "missing_prediction_field_out.jsonl"),
            ]
        )
        cases.append(
            case_result(
                "missing_prediction_fields_are_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="missing prediction field",
            )
        )

        duplicate_dataset = tempdir / "duplicate_dataset.json"
        write_dataset(duplicate_dataset, duplicate_question=True)
        ambiguous_input = tempdir / "ambiguous.json"
        write_json(
            ambiguous_input,
            {
                "records": [
                    {"sample_id": "fixture_0", "question": "What does Alice like?", "prediction": "tea"},
                    {"sample_id": "fixture_0", "question": "What does Alice like?", "prediction": "tea"},
                ]
            },
        )
        rc, result = run_json(
            [
                sys.executable,
                str(args.normalizer),
                "--dataset",
                str(duplicate_dataset),
                "--input",
                str(ambiguous_input),
                "--output-jsonl",
                str(tempdir / "ambiguous_out.jsonl"),
            ]
        )
        cases.append(
            case_result(
                "ambiguous_question_mapping_is_rejected",
                rc,
                result,
                expect_success=False,
                expected_error_fragment="ambiguous",
            )
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "normalizer": str(args.normalizer),
        "normalizer_sha256": sha256_file(args.normalizer),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
