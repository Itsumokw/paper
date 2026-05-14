#!/usr/bin/env python3
"""Self-test PerLTQA fact-ledger coverage checker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import check_locomo_perltqa_fact_ledger_coverage as checker


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def fixture_source_memory() -> list[dict[str, Any]]:
    return [
        {
            "profile": {"Protagonist": "Alice", "Age": "20"},
            "social_relationship": {"r1": {"name": "Bob", "relation": "friend"}},
            "events": {"e1": {"content": "Alice won a debate with Bob's help."}},
            "dialogues": {
                "e1#0": {
                    "events": "e1",
                    "contents": {"2024-01-01": ["Alice:Thanks for helping.", "Bob:You did the work."]},
                }
            },
        },
        {
            "profile": {"Protagonist": "Carol", "Age": "31"},
            "social_relationship": {"r2": {"name": "Dana", "relation": "mentor"}},
            "events": {"e2": {"content": "Carol practiced German with Dana."}},
            "dialogues": {
                "e2#0": {
                    "events": "e2",
                    "contents": {"2024-01-02": ["Carol:I practiced the grammar rule.", "Dana:Good progress."]},
                }
            },
        },
    ]


def fixture_source_qa() -> list[dict[str, Any]]:
    return [
        {
            "Alice": {
                "memory": [
                    {
                        "Question": "Who helped Alice with the debate?",
                        "Answer": "Bob helped Alice.",
                        "Reference Memory": "Alice won a debate with Bob's help.",
                    }
                ]
            }
        }
    ]


def fixture_primary() -> list[dict[str, Any]]:
    return [
        {"sample_id": "perltqa_0000", "source_dataset": "PerLTQA", "qa": []},
        {"sample_id": "perltqa_0001", "source_dataset": "PerLTQA", "qa": []},
    ]


def expected_rows(memory: list[dict[str, Any]], source_qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qas_by_person: dict[str, Any] = {}
    for row in source_qa:
        for person, value in row.items():
            qas_by_person[person] = value
    rows: list[dict[str, Any]] = []
    for raw_idx, item in enumerate(memory):
        sample_id = f"perltqa_{raw_idx:04d}"
        protagonist = item["profile"]["Protagonist"]
        for fact in checker.expected_perltqa_facts(sample_id, item, qas_by_person.get(protagonist, {})):
            fact["source_dataset"] = "PerLTQA"
            fact["sample_id"] = sample_id
            rows.append(fact)
    return rows


def provenance_rows_from_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in facts:
        if row.get("source_type") != "original_turn":
            continue
        sample_id = str(row["fact_id"]).split("_turn_", 1)[0]
        rows.append(
            {
                "source_dataset": "PerLTQA",
                "sample_id": sample_id,
                "source_origin": "original_turn",
                "source_fact_id": row["fact_id"],
                "dia_id": row["dia_id"],
            }
        )
    return rows


def run_checker(
    tempdir: Path,
    *,
    facts: list[dict[str, Any]],
    provenance: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any], str]:
    primary = tempdir / "primary.json"
    memory = tempdir / "perltmem.json"
    source_qa = tempdir / "perltqa.json"
    ledger = tempdir / "fact_ledger.jsonl"
    provenance_path = tempdir / "provenance.jsonl"
    output = tempdir / "report.json"
    write_json(primary, fixture_primary())
    write_json(memory, fixture_source_memory())
    write_json(source_qa, fixture_source_qa())
    write_jsonl(ledger, facts)
    write_jsonl(provenance_path, provenance if provenance is not None else provenance_rows_from_facts(facts))
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(checker.__file__)),
            "--primary-json",
            str(primary),
            "--source-memory",
            str(memory),
            "--source-qa",
            str(source_qa),
            "--fact-ledger",
            str(ledger),
            "--provenance",
            str(provenance_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    return completed.returncode, report, completed.stderr


def case_result(
    name: str,
    returncode: int,
    report: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = report.get("errors", [])
    observed_success = returncode == 0 and report.get("status") == "passed"
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None
            or any(expected_error_fragment in str(error) for error in errors)
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "expected_error_fragment": expected_error_fragment,
        "returncode": returncode,
        "checker_status": report.get("status"),
        "errors": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/perltqa_fact_ledger_coverage_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    base_facts = expected_rows(fixture_source_memory(), fixture_source_qa())
    with tempfile.TemporaryDirectory(prefix="locomo_perltqa_fact_ledger_selftest_") as tmp:
        tempdir = Path(tmp)

        returncode, report, _ = run_checker(tempdir / "valid", facts=base_facts)
        cases.append(case_result("valid_fixture_is_accepted", returncode, report, expect_success=True))

        bad = [row for row in base_facts if row.get("source_type") != "original_relationship"]
        returncode, report, _ = run_checker(tempdir / "missing_relationship", facts=bad)
        cases.append(
            case_result(
                "missing_relationship_fact_is_rejected",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="original_relationship",
            )
        )

        bad = [row for row in base_facts if row.get("source_type") != "original_qa"]
        returncode, report, _ = run_checker(tempdir / "missing_original_qa", facts=bad)
        cases.append(
            case_result(
                "missing_original_qa_fact_is_rejected_when_source_has_qa",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="original_qa",
            )
        )

        bad = [row for row in base_facts if row.get("source_type") != "original_turn"]
        returncode, report, _ = run_checker(tempdir / "missing_turns", facts=bad, provenance=[])
        cases.append(
            case_result(
                "missing_original_turn_fact_is_rejected",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="original_turn",
            )
        )

        bad = [row for row in base_facts if row.get("source_type") != "original_event"]
        returncode, report, _ = run_checker(tempdir / "missing_events", facts=bad)
        cases.append(
            case_result(
                "missing_dialogue_event_fact_is_rejected",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="original_event",
            )
        )

        bad = deepcopy(base_facts)
        for row in bad:
            if row.get("source_type") == "original_turn":
                row["source_text"] = "__corrupted_turn__"
                break
        returncode, report, _ = run_checker(tempdir / "turn_mismatch", facts=bad)
        cases.append(
            case_result(
                "mismatched_turn_text_is_rejected",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="mismatched content",
            )
        )

        bad_provenance = []
        returncode, report, _ = run_checker(tempdir / "missing_turn_provenance", facts=base_facts, provenance=bad_provenance)
        cases.append(
            case_result(
                "missing_original_turn_provenance_is_rejected",
                returncode,
                report,
                expect_success=False,
                expected_error_fragment="missing provenance",
            )
        )

    failed_cases = [case for case in cases if case["status"] != "passed"]
    result = {
        "status": "passed" if not failed_cases else "failed",
        "checker": str(Path(checker.__file__)),
        "checker_sha256": checker.sha256_file(Path(checker.__file__)),
        "selftest": str(Path(__file__)),
        "selftest_sha256": checker.sha256_file(Path(__file__)),
        "cases": cases,
        "errors": [f"{case['name']}: {case['errors']}" for case in failed_cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
