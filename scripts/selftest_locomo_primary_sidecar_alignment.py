#!/usr/bin/env python3
"""Self-test primary/provenance alignment checks for LoCoMo-style artifacts."""

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    first_text = f"{source} original turn one."
    second_text = f"{source} original turn two."
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
                    {"speaker": "A", "dia_id": "D1:1", "text": first_text},
                    {"speaker": "B", "dia_id": "D1:2", "text": second_text},
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
            "text": first_text,
            "raw_text_hash": sha256_text(first_text),
            "source_speaker": f"{source}_raw_a",
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D1:2",
            "session_id": "session_1",
            "turn_index": 2,
            "source_origin": "original_turn",
            "text": second_text,
            "raw_text_hash": sha256_text(second_text),
            "source_speaker": f"{source}_raw_b",
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
    output = root / "primary_sidecar_alignment_report.json"
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
        default=Path("scripts/check_locomo_primary_sidecar_alignment.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary_sidecar_alignment_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_primary_sidecar_alignment_selftest_") as tmp:
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

        run_case("valid_alignment_is_accepted", None, True)

        def remove_provenance(state: dict[str, dict[str, Any]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["provenance"].pop()

        run_case(
            "missing_provenance_row_is_rejected",
            remove_provenance,
            False,
            ["missing provenance row"],
        )

        def add_extra_provenance(state: dict[str, dict[str, Any]]) -> None:
            extra = deepcopy(state["OPELA-LoCoMo-style-eval"]["provenance"][0])
            extra["dia_id"] = "D9:9"
            state["OPELA-LoCoMo-style-eval"]["provenance"].append(extra)

        run_case(
            "extra_provenance_row_is_rejected",
            add_extra_provenance,
            False,
            ["provenance has 1 rows not present in primary JSON"],
        )

        def corrupt_primary_text(state: dict[str, dict[str, Any]]) -> None:
            state["PerLTQA-LoCoMo-style-eval"]["primary"][0]["conversation"]["session_1"][0]["text"] = (
                "changed primary text"
            )

        run_case(
            "primary_text_mismatch_is_rejected",
            corrupt_primary_text,
            False,
            ["primary/provenance text mismatch"],
        )

        def corrupt_raw_hash(state: dict[str, dict[str, Any]]) -> None:
            state["deL1L2IM-LoCoMo-style-eval"]["provenance"][0]["raw_text_hash"] = "bad_hash"

        run_case(
            "raw_text_hash_mismatch_is_rejected",
            corrupt_raw_hash,
            False,
            ["raw_text_hash does not match primary text"],
        )

        def corrupt_loader_speaker(state: dict[str, dict[str, Any]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["primary"][0]["conversation"]["session_1"][0]["speaker"] = "C"

        run_case(
            "loader_speaker_mismatch_is_rejected",
            corrupt_loader_speaker,
            False,
            ["speaker='C' not in"],
        )

        def remove_source_speaker(state: dict[str, dict[str, Any]]) -> None:
            state["OPELA-LoCoMo-style-eval"]["provenance"][0]["source_speaker"] = ""

        run_case(
            "original_turn_missing_source_speaker_is_rejected",
            remove_source_speaker,
            False,
            ["original_turn rows missing source_speaker"],
        )

        def corrupt_primary_source_dataset(state: dict[str, dict[str, Any]]) -> None:
            state["PerLTQA-LoCoMo-style-eval"]["primary"][0]["source_dataset"] = "OPELA"

        run_case(
            "primary_source_dataset_mismatch_is_rejected",
            corrupt_primary_source_dataset,
            False,
            ["primary source_dataset='OPELA' does not match expected 'PerLTQA'"],
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
