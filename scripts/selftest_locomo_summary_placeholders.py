#!/usr/bin/env python3
"""Self-test empty summary/observation placeholder checks."""

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


ARTIFACTS = [
    "PerLTQA-LoCoMo-style-eval",
    "OPELA-LoCoMo-style-eval",
    "JLongChat-LoCoMo-style-eval",
    "deL1L2IM-LoCoMo-style-eval",
    "multilingual_locomo_style_eval",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_sample(source_dataset: str) -> dict[str, Any]:
    return {
        "sample_id": f"{source_dataset.lower()}_fixture_0",
        "source_dataset": source_dataset,
        "language": "xx",
        "split": "eval",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1_date_time": "2024-01-01",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
            "session_2_date_time": "2024-01-02",
            "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "world"}],
        },
        "observation": {"session_1_observation": [], "session_2_observation": []},
        "session_summary": {"session_1_summary": "", "session_2_summary": ""},
        "event_summary": {"events_session_1": [], "events_session_2": []},
        "qa": [{"question": "What was said?", "answer": "hello", "category": 1, "evidence": ["D1:1"]}],
    }


def fixture_rows() -> dict[str, list[dict[str, Any]]]:
    source_map = {
        "PerLTQA-LoCoMo-style-eval": "PerLTQA",
        "OPELA-LoCoMo-style-eval": "OPELA",
        "JLongChat-LoCoMo-style-eval": "JLongChat",
        "deL1L2IM-LoCoMo-style-eval": "deL1L2IM",
        "multilingual_locomo_style_eval": "PerLTQA",
    }
    return {artifact: [base_sample(source)] for artifact, source in source_map.items()}


def write_fixture(
    primary_root: Path,
    mutate: Callable[[dict[str, list[dict[str, Any]]]], None] | None = None,
) -> None:
    rows = fixture_rows()
    if mutate is not None:
        mutate(rows)
    primary_root.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        (primary_root / f"{artifact}.json").write_text(
            json.dumps(rows[artifact], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def run_checker(checker: Path, primary_root: Path, output: Path) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--primary-root",
            str(primary_root),
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
    expect_success: bool,
    expected_fragment: str | None = None,
) -> dict[str, Any]:
    errors = [str(item) for item in summary.get("errors", [])]
    observed_success = returncode == 0 and summary.get("status") == "passed"
    fragment_found = (
        expected_fragment is None
        or any(expected_fragment in error for error in errors)
    )
    passed = observed_success if expect_success else (not observed_success and fragment_found)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "checker_status": summary.get("status"),
        "expected_fragment": expected_fragment,
        "errors": errors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default=Path("scripts/check_locomo_summary_placeholders.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/summary_placeholder_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_summary_placeholder_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate: Callable[[dict[str, list[dict[str, Any]]]], None] | None,
            expect_success: bool,
            expected_fragment: str | None = None,
        ) -> None:
            primary_root = tempdir / name / "primary"
            write_fixture(primary_root, mutate)
            rc, summary = run_checker(args.checker, primary_root, tempdir / name / "report.json")
            cases.append(case_result(name, rc, summary, expect_success, expected_fragment))

        run_case("empty_summary_placeholders_are_accepted", None, True)

        def nonempty_observation(rows: dict[str, list[dict[str, Any]]]) -> None:
            rows["PerLTQA-LoCoMo-style-eval"][0]["observation"]["session_1_observation"] = ["answer leak"]

        run_case("nonempty_observation_is_rejected", nonempty_observation, False, "observation.session_1_observation")

        def nonempty_session_summary(rows: dict[str, list[dict[str, Any]]]) -> None:
            rows["OPELA-LoCoMo-style-eval"][0]["session_summary"]["session_1_summary"] = "answer leak"

        run_case("nonempty_session_summary_is_rejected", nonempty_session_summary, False, "session_summary.session_1_summary")

        def nonempty_event_summary(rows: dict[str, list[dict[str, Any]]]) -> None:
            rows["JLongChat-LoCoMo-style-eval"][0]["event_summary"]["events_session_2"] = ["answer leak"]

        run_case("nonempty_event_summary_is_rejected", nonempty_event_summary, False, "event_summary.events_session_2")

        def missing_summary_key(rows: dict[str, list[dict[str, Any]]]) -> None:
            rows["deL1L2IM-LoCoMo-style-eval"][0]["session_summary"].pop("session_2_summary")

        run_case("missing_session_summary_key_is_rejected", missing_summary_key, False, "session_summary missing keys")

        def extra_summary_key(rows: dict[str, list[dict[str, Any]]]) -> None:
            rows["multilingual_locomo_style_eval"][0]["event_summary"]["events_session_99"] = []

        run_case("extra_event_summary_key_is_rejected", extra_summary_key, False, "event_summary has extra keys")

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
