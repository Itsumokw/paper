#!/usr/bin/env python3
"""Self-test PerLTQA original-QA exclusion checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


GOOD_PHRASE = "original PerLTQA QA is not copied into final eval"
OLD_PHRASE = "original QA is only included when mapped"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_qa_fixture() -> list[dict[str, Any]]:
    return [
        {
            "张三": {
                "profile": [
                    {
                        "Question": "张三的职业是什么？",
                        "Answer": "张三是工程师。",
                        "Reference Memory": "Occupation",
                    }
                ],
                "events": {
                    "1": [
                        {
                            "Question": "张三和谁一起看电影？",
                            "Answer": "张三和李四一起看电影。",
                            "Reference Memory": "['1']",
                        }
                    ]
                },
            }
        }
    ]


def primary_fixture() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "perltqa_fixture_0",
            "source_dataset": "PerLTQA",
            "language": "zh",
            "split": "eval",
            "conversation": {
                "speaker_a": "张三",
                "speaker_b": "其他参与者",
                "session_1_date_time": "2024-01-01",
                "session_1": [{"speaker": "张三", "dia_id": "D1:1", "text": "记忆锚点：张三是工程师。"}],
            },
            "observation": {"session_1_observation": []},
            "session_summary": {"session_1_summary": ""},
            "event_summary": {"events_session_1": []},
            "qa": [
                {
                    "question": "根据第 1 个 session，张三的长期记忆记录了哪类事实？",
                    "answer": "张三是工程师。",
                    "category": 1,
                    "evidence": ["D1:1"],
                }
            ],
        }
    ]


def write_case(
    tempdir: Path,
    name: str,
    mutate_primary: Callable[[list[dict[str, Any]]], None] | None = None,
    report_text: str | None = None,
) -> dict[str, Path]:
    case_dir = tempdir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / "perltqa.json"
    primary_path = case_dir / "primary.json"
    construction_report_path = case_dir / "construction_report.md"
    primary = deepcopy(primary_fixture())
    if mutate_primary is not None:
        mutate_primary(primary)
    source_path.write_text(json.dumps(source_qa_fixture(), ensure_ascii=False, indent=2), encoding="utf-8")
    primary_path.write_text(json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8")
    construction_report_path.write_text(report_text or f"- {GOOD_PHRASE}\n", encoding="utf-8")
    return {
        "source_qa": source_path,
        "primary_json": primary_path,
        "construction_report": construction_report_path,
        "output": case_dir / "report.json",
    }


def run_checker(checker: Path, paths: dict[str, Path]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--primary-json",
            str(paths["primary_json"]),
            "--source-qa",
            str(paths["source_qa"]),
            "--construction-report",
            str(paths["construction_report"]),
            "--output",
            str(paths["output"]),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if paths["output"].exists():
        summary = json.loads(paths["output"].read_text(encoding="utf-8"))
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
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = [str(item) for item in summary.get("errors", [])]
    observed_success = returncode == 0 and summary.get("status") == "passed"
    fragment_found = (
        expected_error_fragment is None
        or any(expected_error_fragment in error for error in errors)
    )
    passed = observed_success if expect_success else (not observed_success and fragment_found)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "checker_status": summary.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checker",
        type=Path,
        default=Path("scripts/check_locomo_perltqa_original_qa_exclusion.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/perltqa_original_qa_exclusion_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_perltqa_original_qa_selftest_") as tmp:
        tempdir = Path(tmp)

        paths = write_case(tempdir, "valid_rewritten_perltqa_question_is_accepted")
        rc, summary = run_checker(args.checker, paths)
        cases.append(case_result("valid_rewritten_perltqa_question_is_accepted", rc, summary, True))

        def copy_question(primary: list[dict[str, Any]]) -> None:
            primary[0]["qa"][0]["question"] = "张三的职业是什么？"

        paths = write_case(tempdir, "copied_original_question_is_rejected", copy_question)
        rc, summary = run_checker(args.checker, paths)
        cases.append(
            case_result(
                "copied_original_question_is_rejected",
                rc,
                summary,
                False,
                "questions copied exactly from original PerLTQA QA",
            )
        )

        def copy_question_answer(primary: list[dict[str, Any]]) -> None:
            primary[0]["qa"][0]["question"] = "张三的职业是什么？"
            primary[0]["qa"][0]["answer"] = "张三是工程师。"

        paths = write_case(tempdir, "copied_original_question_answer_pair_is_rejected", copy_question_answer)
        rc, summary = run_checker(args.checker, paths)
        cases.append(
            case_result(
                "copied_original_question_answer_pair_is_rejected",
                rc,
                summary,
                False,
                "exact original PerLTQA question+answer pairs",
            )
        )

        paths = write_case(
            tempdir,
            "missing_construction_report_phrase_is_rejected",
            report_text="- PerLTQA PlanMode D without the required wording.\n",
        )
        rc, summary = run_checker(args.checker, paths)
        cases.append(
            case_result(
                "missing_construction_report_phrase_is_rejected",
                rc,
                summary,
                False,
                "construction report missing phrase",
            )
        )

        paths = write_case(
            tempdir,
            "old_construction_report_phrase_is_rejected",
            report_text=f"- {GOOD_PHRASE}; {OLD_PHRASE}.\n",
        )
        rc, summary = run_checker(args.checker, paths)
        cases.append(
            case_result(
                "old_construction_report_phrase_is_rejected",
                rc,
                summary,
                False,
                "construction report still contains forbidden phrase",
            )
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
