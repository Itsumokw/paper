#!/usr/bin/env python3
"""Self-test session/turn order checks for LoCoMo-style artifacts."""

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


SOURCE_ARTIFACTS = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


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


def sample_fixture(source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_id = f"{source.lower()}_fixture_0"
    primary = [
        {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "en",
            "split": "eval",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:1", "text": f"{source} session one turn one."},
                    {"speaker": "B", "dia_id": "D1:2", "text": f"{source} session one turn two."},
                ],
                "session_2_date_time": "2024-01-02",
                "session_2": [
                    {"speaker": "A", "dia_id": "D2:1", "text": f"{source} session two turn one."},
                ],
            },
            "observation": {},
            "session_summary": {},
            "event_summary": {},
            "qa": [],
        }
    ]
    provenance = [
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D1:1",
            "session_id": "session_1",
            "turn_index": 1,
            "source_origin": "original_turn",
            "order_policy": "source_order",
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D1:2",
            "session_id": "session_1",
            "turn_index": 2,
            "source_origin": "original_turn",
            "order_policy": "source_order",
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D2:1",
            "session_id": "session_2",
            "turn_index": 1,
            "source_origin": "original_turn",
            "order_policy": "source_order",
        },
    ]
    return primary, provenance


def fixture_state() -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for source, artifact in SOURCE_ARTIFACTS.items():
        primary, provenance = sample_fixture(source)
        state[artifact] = {"primary": primary, "provenance": provenance}
    return state


def write_fixture(
    root: Path,
    mutate: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> None:
    state = fixture_state()
    if mutate is not None:
        mutate(state)
    primary_root = root / "primary"
    sidecar_root = root / "sidecars"
    for artifact, artifact_state in state.items():
        write_json(primary_root / f"{artifact}.json", artifact_state["primary"])
        write_jsonl(
            sidecar_root / artifact / f"{artifact}_provenance.jsonl",
            artifact_state["provenance"],
        )


def run_checker(checker: Path, root: Path) -> tuple[int, dict[str, Any]]:
    output = root / "session_order_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--primary-root",
            str(root / "primary"),
            "--sidecar-root",
            str(root / "sidecars"),
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
    summary["returncode"] = completed.returncode
    summary["_tempdir"] = str(root)
    return completed.returncode, summary


def case_result(
    name: str,
    returncode: int,
    summary: dict[str, Any],
    expect_success: bool,
    expected_error_fragments: list[str] | None = None,
) -> dict[str, Any]:
    tempdir = str(summary.get("_tempdir", ""))
    errors = [str(item).replace(tempdir + "/", "<tmp>/") for item in summary.get("errors", [])]
    fragments = expected_error_fragments or []
    missing_fragments = [
        fragment
        for fragment in fragments
        if not any(fragment in error for error in errors)
    ]
    observed_success = returncode == 0 and summary.get("status") == "passed"
    passed = observed_success if expect_success else (not observed_success and not missing_fragments)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "checker_status": summary.get("status"),
        "expected_error_fragments": fragments,
        "missing_expected_error_fragments": missing_fragments,
        "errors": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checker",
        type=Path,
        default=Path("scripts/check_locomo_session_order.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/session_order_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_session_order_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate: Callable[[dict[str, dict[str, Any]]], None] | None,
            expect_success: bool,
            expected_error_fragments: list[str] | None = None,
        ) -> None:
            root = tempdir / name
            write_fixture(root, mutate)
            rc, summary = run_checker(args.checker, root)
            cases.append(case_result(name, rc, summary, expect_success, expected_error_fragments))

        run_case("valid_session_order_is_accepted", None, True)

        def remove_session_1(state: dict[str, dict[str, Any]]) -> None:
            conversation = state["PerLTQA-LoCoMo-style-eval"]["primary"][0]["conversation"]
            conversation.pop("session_1")
            conversation.pop("session_1_date_time")

        run_case(
            "nonconsecutive_session_keys_are_rejected",
            remove_session_1,
            False,
            ["session keys=['session_2'] expected consecutive session_1..session_1"],
        )

        def remove_date(state: dict[str, dict[str, Any]]) -> None:
            state["OPELA-LoCoMo-style-eval"]["primary"][0]["conversation"].pop("session_2_date_time")

        run_case(
            "missing_session_datetime_is_rejected",
            remove_date,
            False,
            ["missing session_2_date_time"],
        )

        def corrupt_dia_id(state: dict[str, dict[str, Any]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["primary"][0]["conversation"]["session_1"][1][
                "dia_id"
            ] = "D1:9"

        run_case(
            "bad_dia_id_order_is_rejected",
            corrupt_dia_id,
            False,
            ["dia_id='D1:9' expected='D1:2'"],
        )

        def remove_provenance(state: dict[str, dict[str, Any]]) -> None:
            state["deL1L2IM-LoCoMo-style-eval"]["provenance"].pop()

        run_case(
            "primary_provenance_count_mismatch_is_rejected",
            remove_provenance,
            False,
            ["primary turn count=3 provenance rows=2"],
        )

        def swap_provenance_order(state: dict[str, dict[str, Any]]) -> None:
            rows = state["JLongChat-LoCoMo-style-eval"]["provenance"]
            rows[0], rows[1] = rows[1], rows[0]

        run_case(
            "provenance_order_mismatch_is_rejected",
            swap_provenance_order,
            False,
            ["provenance row 1 key="],
        )

        def corrupt_provenance_session(state: dict[str, dict[str, Any]]) -> None:
            state["OPELA-LoCoMo-style-eval"]["provenance"][0]["session_id"] = "session_2"

        run_case(
            "provenance_session_metadata_mismatch_is_rejected",
            corrupt_provenance_session,
            False,
            ["provenance session_id='session_2' expected='session_1'"],
        )

        def corrupt_provenance_turn_index(state: dict[str, dict[str, Any]]) -> None:
            state["PerLTQA-LoCoMo-style-eval"]["provenance"][0]["turn_index"] = 9

        run_case(
            "provenance_turn_index_mismatch_is_rejected",
            corrupt_provenance_turn_index,
            False,
            ["provenance turn_index=9 expected=1"],
        )

        def mix_synthetic_original(state: dict[str, dict[str, Any]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["provenance"][1][
                "source_origin"
            ] = "synthetic_bridge_turn"

        run_case(
            "synthetic_original_same_session_is_rejected",
            mix_synthetic_original,
            False,
            ["synthetic turns share a session with original_turn"],
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
