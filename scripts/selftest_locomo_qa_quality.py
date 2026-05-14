#!/usr/bin/env python3
"""Self-test heuristic QA-quality checks for LoCoMo-style eval primary JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qa_row(
    idx: int,
    *,
    sample_id: str = "fixture",
    question: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    return {
        "question": question if question is not None else f"What fact is checked in {sample_id} QA {idx}?",
        "answer": answer if answer is not None else f"Answer {idx}",
        "category": 1,
        "evidence": ["D1:1"],
    }


def sample(sample_id: str, qa_count: int) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_dataset": "Fixture",
        "language": "xx",
        "split": "eval",
        "conversation": {},
        "observation": {},
        "session_summary": {},
        "event_summary": {},
        "qa": [qa_row(idx, sample_id=sample_id) for idx in range(qa_count)],
    }


def fixture_data() -> list[dict[str, Any]]:
    return [
        sample("fixture_min_20", 20),
        sample("fixture_max_40", 40),
    ]


def write_primary(path: Path, mutate: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
    data = fixture_data()
    if mutate is not None:
        mutate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_checker(checker: Path, primary: Path, output: Path) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            str(primary),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if output.exists():
        summary = json.loads(output.read_text(encoding="utf-8"))
    else:
        summary = {
            "status": "checker_did_not_write_report",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return completed.returncode, summary


def case_result(
    name: str,
    returncode: int,
    summary: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = [str(item) for item in summary.get("errors", [])]
    observed_success = returncode == 0 and summary.get("status") == "passed"
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None
            or any(expected_error_fragment in error for error in errors)
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "checker_status": summary.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": errors[:10],
        "qa_per_sample": summary.get("qa_per_sample"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default=Path("scripts/check_locomo_qa_quality.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/qa_quality_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_qa_quality_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate: Callable[[list[dict[str, Any]]], None] | None,
            expect_success: bool,
            expected_error_fragment: str | None = None,
        ) -> None:
            primary = tempdir / name / "primary.json"
            output = tempdir / name / "report.json"
            write_primary(primary, mutate)
            rc, summary = run_checker(args.checker, primary, output)
            cases.append(
                case_result(
                    name,
                    rc,
                    summary,
                    expect_success=expect_success,
                    expected_error_fragment=expected_error_fragment,
                )
            )

        run_case("valid_20_to_40_qa_per_sample_is_accepted", None, True)

        def below_minimum(data: list[dict[str, Any]]) -> None:
            data[0]["qa"] = data[0]["qa"][:19]

        run_case(
            "sample_below_20_qa_is_rejected",
            below_minimum,
            False,
            "sample QA counts outside",
        )

        def above_maximum(data: list[dict[str, Any]]) -> None:
            data[1]["qa"].append(qa_row(40, sample_id=str(data[1]["sample_id"])))

        run_case(
            "sample_above_40_qa_is_rejected",
            above_maximum,
            False,
            "sample QA counts outside",
        )

        def duplicate_questions(data: list[dict[str, Any]]) -> None:
            for idx in range(11):
                data[0]["qa"][idx]["question"] = "Repeated question?"

        run_case(
            "duplicate_question_threshold_is_rejected",
            duplicate_questions,
            False,
            "max_duplicate_question_count",
        )

        def too_many_short_answers(data: list[dict[str, Any]]) -> None:
            for idx in range(6):
                data[0]["qa"][idx]["answer"] = "x"

        run_case(
            "short_answer_threshold_is_rejected",
            too_many_short_answers,
            False,
            "short answerable answers",
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "checker": str(args.checker),
        "checker_sha256": sha256_file(args.checker),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
