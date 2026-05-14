#!/usr/bin/env python3
"""Self-test OPELA temporal-policy checker."""

from __future__ import annotations

import argparse
import csv
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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_fixture() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "opela_fixture_0",
            "source_dataset": "OPELA",
            "language": "ko",
            "split": "eval",
            "conversation": {
                "speaker_a": "persona",
                "speaker_b": "user",
                "session_1_date_time": "OPELA virtual session 1; pause_hours=[5, 24]",
                "session_1": [{"speaker": "user", "dia_id": "D1:1", "text": "안녕"}],
                "session_2_date_time": "OPELA virtual session 2; pause_hours=[5, 24]",
                "session_2": [{"speaker": "persona", "dia_id": "D2:1", "text": "안녕!"}],
            },
            "observation": {"session_1_observation": [], "session_2_observation": []},
            "session_summary": {"session_1_summary": "", "session_2_summary": ""},
            "event_summary": {"events_session_1": [], "events_session_2": []},
            "qa": [{"question": "What did user say?", "answer": "안녕", "category": 1, "evidence": ["D1:1"]}],
        }
    ]


def provenance_fixture(source_record_id: str = "doc_1") -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture_0",
            "dia_id": "D1:1",
            "source_origin": "original_turn",
            "source_record_id": source_record_id,
        },
        {
            "source_dataset": "OPELA",
            "sample_id": "opela_fixture_0",
            "dia_id": "D2:1",
            "source_origin": "original_turn",
            "source_record_id": source_record_id,
        },
    ]


def write_json(path: Path, rows: Any) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_source_csv(path: Path, doc_id: str = "doc_1", pause_hour: str = "[5, 24]") -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "pause_hour"])
        writer.writeheader()
        writer.writerow({"doc_id": doc_id, "pause_hour": pause_hour})


def write_case(
    tempdir: Path,
    name: str,
    mutate_primary: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_provenance: Callable[[list[dict[str, Any]]], None] | None = None,
    source_doc_id: str = "doc_1",
    source_pause_hour: str = "[5, 24]",
) -> dict[str, Path]:
    case_dir = tempdir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    primary = deepcopy(primary_fixture())
    provenance = deepcopy(provenance_fixture(source_doc_id))
    if mutate_primary is not None:
        mutate_primary(primary)
    if mutate_provenance is not None:
        mutate_provenance(provenance)
    paths = {
        "primary_json": case_dir / "opela_primary.json",
        "provenance": case_dir / "opela_provenance.jsonl",
        "source_csv": case_dir / "oplea_open_data.csv",
        "output": case_dir / "report.json",
    }
    write_json(paths["primary_json"], primary)
    write_jsonl(paths["provenance"], provenance)
    write_source_csv(paths["source_csv"], doc_id=source_doc_id, pause_hour=source_pause_hour)
    return paths


def run_checker(checker: Path, paths: dict[str, Path]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--primary-json",
            str(paths["primary_json"]),
            "--provenance",
            str(paths["provenance"]),
            "--source-csv",
            str(paths["source_csv"]),
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
    parser.add_argument("--checker", type=Path, default=Path("scripts/check_locomo_opela_temporal_policy.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/opela_temporal_policy_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_opela_temporal_policy_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate_primary: Callable[[list[dict[str, Any]]], None] | None,
            expect_success: bool,
            expected_error_fragment: str | None = None,
            mutate_provenance: Callable[[list[dict[str, Any]]], None] | None = None,
            source_doc_id: str = "doc_1",
            source_pause_hour: str = "[5, 24]",
        ) -> None:
            paths = write_case(
                tempdir,
                name,
                mutate_primary=mutate_primary,
                mutate_provenance=mutate_provenance,
                source_doc_id=source_doc_id,
                source_pause_hour=source_pause_hour,
            )
            rc, summary = run_checker(args.checker, paths)
            cases.append(case_result(name, rc, summary, expect_success, expected_error_fragment))

        run_case("pause_hour_hint_format_is_accepted", None, True)

        def absolute_month_claim(primary: list[dict[str, Any]]) -> None:
            primary[0]["conversation"]["session_1_date_time"] = "2024-01, one month later"

        run_case("absolute_or_month_timeline_claim_is_rejected", absolute_month_claim, False, "forbidden timeline terms")

        def missing_pause_hours(primary: list[dict[str, Any]]) -> None:
            primary[0]["conversation"]["session_1_date_time"] = "OPELA virtual session 1"

        run_case("missing_pause_hours_hint_is_rejected", missing_pause_hours, False, "expected=")

        def pause_hour_mismatch(primary: list[dict[str, Any]]) -> None:
            primary[0]["conversation"]["session_2_date_time"] = "OPELA virtual session 2; pause_hours=[1]"

        run_case("pause_hour_mismatch_is_rejected", pause_hour_mismatch, False, "expected=")

        def missing_source_record(provenance: list[dict[str, Any]]) -> None:
            provenance[0]["source_record_id"] = "missing_doc"
            provenance[1]["source_record_id"] = "missing_doc"

        run_case(
            "missing_source_record_is_rejected",
            None,
            False,
            "source doc_id not found",
            mutate_provenance=missing_source_record,
        )

        def source_record_conflict(provenance: list[dict[str, Any]]) -> None:
            provenance[1]["source_record_id"] = "other_doc"

        run_case(
            "conflicting_source_record_ids_are_rejected",
            None,
            False,
            "source_record_id conflict",
            mutate_provenance=source_record_conflict,
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
