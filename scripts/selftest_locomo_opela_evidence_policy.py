#!/usr/bin/env python3
"""Self-test OPELA evidence-policy checker."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_fixture() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "opela_fixture",
            "source_dataset": "OPELA",
            "language": "ko",
            "split": "eval",
            "conversation": {
                "speaker_a": "persona",
                "speaker_b": "user",
                "session_1_date_time": "OPELA virtual session 1; pause_hours=[1]",
                "session_1": [{"speaker": "user", "dia_id": "D1:1", "text": "hello"}],
            },
            "observation": {"session_1_observation": []},
            "session_summary": {"session_1_summary": ""},
            "event_summary": {"events_session_1": []},
            "qa": [
                {"question": "What was said?", "answer": "hello", "category": 1, "evidence": ["D1:1"]},
                {"question": "Unsupported swap?", "category": 5, "evidence": [], "adversarial_answer": "unknown"},
            ],
        }
    ]


def fact_ledger_fixture() -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture",
            "fact_id": "f_turn",
            "source_type": "original_turn",
            "source_text": "hello",
        },
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture",
            "fact_id": "f_memory",
            "source_type": "original_memory",
            "source_text": "summary-only",
        },
    ]


def qa_audit_fixture() -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture",
            "qa_idx": 0,
            "qa_set": "locomo_style_main",
            "question": "What was said?",
            "answer": "hello",
            "category": 1,
            "evidence": ["D1:1"],
            "answer_facts": [{"fact": "hello", "supported_by": ["D1:1"], "source_fact_id": "f_turn"}],
            "evidence_detail": [
                {"dia_id": "D1:1", "source_origin": "original_turn", "supports_answer_fact": ["f_turn"]}
            ],
            "negative_evidence": [],
            "adversarial_reason": None,
        },
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture",
            "qa_idx": 1,
            "qa_set": "locomo_style_main",
            "question": "Unsupported swap?",
            "category": 5,
            "evidence": [],
            "answer_facts": [],
            "evidence_detail": [],
            "negative_evidence": ["D1:1"],
            "adversarial_reason": "unsupported_fact",
        },
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_case(
    tempdir: Path,
    name: str,
    mutate_primary: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_fact_ledger: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_qa_audit: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Path]:
    case_dir = tempdir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    primary = deepcopy(primary_fixture())
    facts = deepcopy(fact_ledger_fixture())
    qa_audit = deepcopy(qa_audit_fixture())
    if mutate_primary is not None:
        mutate_primary(primary)
    if mutate_fact_ledger is not None:
        mutate_fact_ledger(facts)
    if mutate_qa_audit is not None:
        mutate_qa_audit(qa_audit)
    paths = {
        "primary_json": case_dir / "primary.json",
        "fact_ledger": case_dir / "fact_ledger.jsonl",
        "qa_audit": case_dir / "qa_audit.jsonl",
        "output": case_dir / "report.json",
    }
    write_json(paths["primary_json"], primary)
    write_jsonl(paths["fact_ledger"], facts)
    write_jsonl(paths["qa_audit"], qa_audit)
    return paths


def run_checker(checker: Path, paths: dict[str, Path]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--primary-json",
            str(paths["primary_json"]),
            "--fact-ledger",
            str(paths["fact_ledger"]),
            "--qa-audit",
            str(paths["qa_audit"]),
            "--output",
            str(paths["output"]),
        ],
        check=False,
        capture_output=True,
        text=True,
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
    fragment_found = expected_error_fragment is None or any(
        expected_error_fragment in error for error in errors
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
    parser.add_argument("--checker", type=Path, default=Path("scripts/check_locomo_opela_evidence_policy.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/opela_evidence_policy_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_opela_evidence_policy_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            expect_success: bool,
            expected_error_fragment: str | None = None,
            mutate_primary: Callable[[list[dict[str, Any]]], None] | None = None,
            mutate_fact_ledger: Callable[[list[dict[str, Any]]], None] | None = None,
            mutate_qa_audit: Callable[[list[dict[str, Any]]], None] | None = None,
        ) -> None:
            paths = write_case(
                tempdir,
                name,
                mutate_primary=mutate_primary,
                mutate_fact_ledger=mutate_fact_ledger,
                mutate_qa_audit=mutate_qa_audit,
            )
            rc, summary = run_checker(args.checker, paths)
            cases.append(case_result(name, rc, summary, expect_success, expected_error_fragment))

        run_case("valid_original_turn_answerable_and_cat5_are_accepted", True)

        def memory_backed_answer(qa_audit: list[dict[str, Any]]) -> None:
            qa_audit[0]["answer_facts"][0]["source_fact_id"] = "f_memory"

        run_case(
            "summary_or_memory_only_answer_fact_is_rejected",
            False,
            "expected 'original_turn'",
            mutate_qa_audit=memory_backed_answer,
        )

        def no_original_turn_evidence(qa_audit: list[dict[str, Any]]) -> None:
            qa_audit[0]["evidence_detail"][0]["source_origin"] = "original_memory"

        run_case(
            "answerable_without_original_turn_evidence_is_rejected",
            False,
            "no original_turn evidence_detail",
            mutate_qa_audit=no_original_turn_evidence,
        )

        def cat5_with_ordinary_evidence(qa_audit: list[dict[str, Any]]) -> None:
            qa_audit[1]["evidence"] = ["D1:1"]

        run_case(
            "cat5_with_ordinary_evidence_is_rejected",
            False,
            "category 5 has ordinary evidence",
            mutate_qa_audit=cat5_with_ordinary_evidence,
        )

        def drop_audit_row(qa_audit: list[dict[str, Any]]) -> None:
            qa_audit.pop()

        run_case(
            "qa_audit_missing_primary_key_is_rejected",
            False,
            "qa_audit missing primary OPELA QA keys",
            mutate_qa_audit=drop_audit_row,
        )

    failed = [case for case in cases if case["status"] != "passed"]
    report = {
        "status": "passed" if not failed else "failed",
        "checker": str(args.checker),
        "checker_sha256": sha256_file(args.checker),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
        "errors": [case for case in failed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
