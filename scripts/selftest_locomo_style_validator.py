#!/usr/bin/env python3
"""Self-test primary and sidecar validation for LoCoMo-style eval files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from validate_locomo_style_eval import sha256_file, sha256_text


VALIDATOR = Path(__file__).with_name("validate_locomo_style_eval.py")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def fixture(tempdir: Path) -> dict[str, Path]:
    primary = tempdir / "primary.json"
    provenance = tempdir / "provenance.jsonl"
    fact_ledger = tempdir / "fact_ledger.jsonl"
    qa_audit = tempdir / "qa_audit.jsonl"

    sample = {
        "sample_id": "fixture_sample",
        "source_dataset": "PerLTQA",
        "language": "zh",
        "split": "eval",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1_date_time": "2024-01-01",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "A likes tea."},
                {"speaker": "B", "dia_id": "D1:2", "text": "B asked about coffee."},
            ],
        },
        "observation": {},
        "session_summary": {},
        "event_summary": {},
        "qa": [
            {
                "question": "What does A like?",
                "answer": "A likes tea.",
                "category": 1,
                "evidence": ["D1:1"],
            },
            {
                "question": "Does the conversation say A won an Olympic medal?",
                "category": 5,
                "evidence": [],
                "adversarial_answer": "No supporting evidence in the conversation.",
            },
        ],
    }
    write_json(primary, [sample])

    write_jsonl(
        provenance,
        [
            {
                "source_dataset": "PerLTQA",
                "sample_id": "fixture_sample",
                "dia_id": "D1:1",
                "session_id": "session_1",
                "turn_index": 1,
                "text": "A likes tea.",
                "raw_text_hash": sha256_text("A likes tea."),
                "source_origin": "original_turn",
                "source_file": "fixtures/source.json",
                "source_record_id": "fixture_record",
                "source_turn_id": "turn_1",
                "source_fact_id": "f_tea",
            },
            {
                "source_dataset": "PerLTQA",
                "sample_id": "fixture_sample",
                "dia_id": "D1:2",
                "session_id": "session_1",
                "turn_index": 2,
                "text": "B asked about coffee.",
                "raw_text_hash": sha256_text("B asked about coffee."),
                "source_origin": "original_turn",
                "source_file": "fixtures/source.json",
                "source_record_id": "fixture_record",
                "source_turn_id": "turn_2",
                "source_fact_id": "f_coffee",
            },
        ],
    )
    write_jsonl(
        fact_ledger,
        [
            {
                "source_dataset": "PerLTQA",
                "sample_id": "fixture_sample",
                "fact_id": "f_tea",
                "source_type": "original_turn",
                "source_text": "A likes tea.",
                "source_id": "turn_1",
            }
        ],
    )
    write_jsonl(
        qa_audit,
        [
            {
                "qa_set": "locomo_style_main",
                "sample_id": "fixture_sample",
                "qa_idx": 0,
                "category": 1,
                "question_type": "single-hop",
                "difficulty": "easy",
                "whether_cross_session": False,
                "evidence": ["D1:1"],
                "evidence_detail": [{"dia_id": "D1:1", "source_origin": "original_turn"}],
                "answer_facts": [
                    {"fact": "A likes tea.", "source_fact_id": "f_tea", "supported_by": ["D1:1"]}
                ],
                "negative_evidence": [],
                "adversarial_reason": None,
                "verifier_status": "heuristic_seed_supported",
                "human_audit_status": "queued",
            },
            {
                "qa_set": "locomo_style_main",
                "sample_id": "fixture_sample",
                "qa_idx": 1,
                "category": 5,
                "question_type": "adversarial",
                "difficulty": "medium",
                "whether_cross_session": False,
                "evidence": [],
                "evidence_detail": [],
                "answer_facts": [],
                "negative_evidence": ["D1:1", "D1:2"],
                "adversarial_reason": "unsupported_fact",
                "verifier_status": "heuristic_adversarial_seed",
                "human_audit_status": "queued",
            },
        ],
    )
    return {
        "primary": primary,
        "provenance": provenance,
        "fact_ledger": fact_ledger,
        "qa_audit": qa_audit,
    }


def run_validator(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(paths["primary"]),
            "--provenance",
            str(paths["provenance"]),
            "--fact-ledger",
            str(paths["fact_ledger"]),
            "--qa-audit",
            str(paths["qa_audit"]),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_case(
    name: str,
    tempdir: Path,
    *,
    mutate: Any | None = None,
    expect_success: bool,
    expected_fragment: str | None = None,
) -> dict[str, Any]:
    case_dir = tempdir / name
    case_dir.mkdir()
    paths = fixture(case_dir)
    if mutate:
        mutate(paths)
    result = run_validator(paths)
    observed_success = result.returncode == 0
    if expect_success:
        passed = observed_success
    else:
        passed = (not observed_success) and (
            expected_fragment is None or expected_fragment in result.stdout
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "expected_fragment": expected_fragment,
        "returncode": result.returncode,
        "output_excerpt": result.stdout[:1000],
    }


def mutate_primary(paths: dict[str, Path], mutator: Any) -> None:
    data = json.loads(paths["primary"].read_text(encoding="utf-8"))
    mutator(data)
    write_json(paths["primary"], data)


def mutate_qa_audit(paths: dict[str, Path], mutator: Any) -> None:
    rows = [json.loads(line) for line in paths["qa_audit"].read_text(encoding="utf-8").splitlines() if line.strip()]
    mutator(rows)
    write_jsonl(paths["qa_audit"], rows)


def mutate_provenance(paths: dict[str, Path], mutator: Any) -> None:
    rows = [json.loads(line) for line in paths["provenance"].read_text(encoding="utf-8").splitlines() if line.strip()]
    mutator(rows)
    write_jsonl(paths["provenance"], rows)


def mutate_fact_ledger(paths: dict[str, Path]) -> None:
    write_jsonl(
        paths["fact_ledger"],
        [
            {
                "source_dataset": "PerLTQA",
                "sample_id": "fixture_sample",
                "fact_id": "f_tea",
                "source_type": "synthetic_continuation",
                "source_text": "A likes tea.",
                "source_id": "turn_1",
            }
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/locomo_style_validator_selftest.json"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="locomo_style_validator_selftest_") as tmp:
        tempdir = Path(tmp)
        cases = [
            run_case("valid_primary_and_sidecars_are_accepted", tempdir, expect_success=True),
            run_case(
                "cat5_missing_adversarial_answer_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_primary(paths, lambda data: data[0]["qa"][1].pop("adversarial_answer")),
                expect_success=False,
                expected_fragment="cat5 missing keys",
            ),
            run_case(
                "cat5_with_ordinary_answer_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_primary(paths, lambda data: data[0]["qa"][1].update({"answer": "medal"})),
                expect_success=False,
                expected_fragment="cat5 has non-loader extra keys",
            ),
            run_case(
                "answerable_primary_extra_field_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_primary(paths, lambda data: data[0]["qa"][0].update({"question_type": "single"})),
                expect_success=False,
                expected_fragment="answerable QA has non-loader extra keys",
            ),
            run_case(
                "missing_evidence_dia_id_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_primary(paths, lambda data: data[0]["qa"][0].update({"evidence": ["D9:9"]})),
                expect_success=False,
                expected_fragment="evidence dia_id not found",
            ),
            run_case(
                "non_original_fact_ledger_support_is_rejected",
                tempdir,
                mutate=mutate_fact_ledger,
                expect_success=False,
                expected_fragment="non-original source_type",
            ),
            run_case(
                "missing_provenance_source_file_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_provenance(paths, lambda rows: rows[0].pop("source_file")),
                expect_success=False,
                expected_fragment="missing provenance keys",
            ),
            run_case(
                "missing_fact_ledger_source_id_is_rejected",
                tempdir,
                mutate=lambda paths: write_jsonl(
                    paths["fact_ledger"],
                    [
                        {
                            "source_dataset": "PerLTQA",
                            "sample_id": "fixture_sample",
                            "fact_id": "f_tea",
                            "source_type": "original_turn",
                            "source_text": "A likes tea.",
                        }
                    ],
                ),
                expect_success=False,
                expected_fragment="missing fact-ledger keys",
            ),
            run_case(
                "missing_verifier_status_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_qa_audit(paths, lambda rows: rows[0].pop("verifier_status")),
                expect_success=False,
                expected_fragment="missing verifier_status",
            ),
            run_case(
                "invalid_cat5_adversarial_reason_is_rejected",
                tempdir,
                mutate=lambda paths: mutate_qa_audit(paths, lambda rows: rows[1].update({"adversarial_reason": "not_a_reason"})),
                expect_success=False,
                expected_fragment="cat5 adversarial_reason",
            ),
        ]

    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    report = {
        "status": status,
        "validator": str(VALIDATOR),
        "validator_sha256": sha256_file(VALIDATOR),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
