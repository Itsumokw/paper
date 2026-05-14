#!/usr/bin/env python3
"""Self-test original-turn hash coverage checks for LoCoMo-style sidecars."""

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
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def original_row(
    source_dataset: str,
    sample_id: str,
    dia_id: str,
    source_turn_id: str,
    source_file: str,
    text: str,
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "sample_id": sample_id,
        "dia_id": dia_id,
        "source_origin": "original_turn",
        "source_file": source_file,
        "source_turn_id": source_turn_id,
        "raw_text_hash": row_hash(text),
        "text": text,
    }


def hash_row(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": provenance["source_dataset"],
        "sample_id": provenance["sample_id"],
        "dia_id": provenance["dia_id"],
        "source_turn_id": provenance["source_turn_id"],
        "raw_text_hash": provenance["raw_text_hash"],
        "status": "captured_for_recheck",
    }


def fixture_state() -> dict[str, dict[str, list[dict[str, Any]]]]:
    jlong_lac = original_row(
        "JLongChat",
        "jlongchat_lac_fixture",
        "D1:1",
        "lac:day1:row1",
        "datasets/japanese-long-term-chat/utf8/lac-public-dialogue.tsv",
        "LAC fixture text",
    )
    jlong_jmsc = original_row(
        "JLongChat",
        "jlongchat_jmsc_fixture",
        "D1:1",
        "jmsc:pair1:sid1:tid1",
        "datasets/japanese-long-term-chat/utf8/jmsc-public-dialogue.tsv",
        "JMSC fixture text",
    )
    perltqa = original_row(
        "PerLTQA",
        "perltqa_fixture",
        "D1:1",
        "perltqa:row1",
        "datasets/PerLTQA/fixture.json",
        "PerLTQA fixture text",
    )
    opela = original_row(
        "OPELA",
        "opela_fixture",
        "D1:1",
        "opela:row1",
        "datasets/OPELA/fixture.csv",
        "OPELA fixture text",
    )
    del1l2im = original_row(
        "deL1L2IM",
        "del1l2im_fixture",
        "D1:1",
        "del1l2im:msg1",
        "datasets/deL1L2IM/fixture.xml",
        "deL1L2IM fixture text",
    )
    return {
        "PerLTQA-LoCoMo-style-eval": {
            "provenance": [perltqa],
            "hash_check": [hash_row(perltqa)],
        },
        "OPELA-LoCoMo-style-eval": {
            "provenance": [opela],
            "hash_check": [hash_row(opela)],
        },
        "JLongChat-LoCoMo-style-eval": {
            "provenance": [jlong_lac, jlong_jmsc],
            "hash_check": [hash_row(jlong_lac), hash_row(jlong_jmsc)],
        },
        "deL1L2IM-LoCoMo-style-eval": {
            "provenance": [del1l2im],
            "hash_check": [hash_row(del1l2im)],
        },
    }


def write_fixture(
    sidecar_root: Path,
    mutate: Callable[[dict[str, dict[str, list[dict[str, Any]]]]], None] | None = None,
) -> None:
    state = fixture_state()
    if mutate is not None:
        mutate(state)
    for artifact in ARTIFACTS:
        rows = state[artifact]
        sidecar_dir = sidecar_root / artifact
        write_jsonl(sidecar_dir / f"{artifact}_provenance.jsonl", rows["provenance"])
        write_jsonl(sidecar_dir / f"{artifact}_hash_check.jsonl", rows["hash_check"])


def run_checker(checker: Path, sidecar_root: Path, tempdir: Path, name: str) -> tuple[int, dict[str, Any]]:
    output = tempdir / f"{name}_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--sidecar-root",
            str(sidecar_root),
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
    summary["_tempdir"] = str(tempdir)
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
        default=Path("scripts/check_locomo_hash_coverage.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/hash_coverage_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_hash_coverage_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate: Callable[[dict[str, dict[str, list[dict[str, Any]]]]], None] | None,
            expect_success: bool,
            expected_error_fragments: list[str] | None = None,
        ) -> None:
            case_root = tempdir / name / "sidecars"
            write_fixture(case_root, mutate)
            rc, summary = run_checker(args.checker, case_root, tempdir / name, name)
            cases.append(case_result(name, rc, summary, expect_success, expected_error_fragments))

        run_case("valid_hash_coverage_is_accepted", None, True)

        def remove_hash_check(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["hash_check"].pop()

        run_case(
            "missing_hash_check_row_is_rejected",
            remove_hash_check,
            False,
            ["missing hash_check rows"],
        )

        def add_extra_hash_check(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            extra = deepcopy(state["OPELA-LoCoMo-style-eval"]["hash_check"][0])
            extra["dia_id"] = "D9:9"
            state["OPELA-LoCoMo-style-eval"]["hash_check"].append(extra)

        run_case(
            "extra_hash_check_row_is_rejected",
            add_extra_hash_check,
            False,
            ["hash_check has 1 rows not matched"],
        )

        def corrupt_hash(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            state["deL1L2IM-LoCoMo-style-eval"]["hash_check"][0]["raw_text_hash"] = "bad_hash"

        run_case(
            "raw_text_hash_mismatch_is_rejected",
            corrupt_hash,
            False,
            ["raw_text_hash mismatch"],
        )

        def corrupt_status(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            state["PerLTQA-LoCoMo-style-eval"]["hash_check"][0]["status"] = "stale"

        run_case(
            "bad_hash_check_status_is_rejected",
            corrupt_status,
            False,
            ["unexpected hash_check status"],
        )

        def remove_required_family(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            state["JLongChat-LoCoMo-style-eval"]["provenance"] = [
                row
                for row in state["JLongChat-LoCoMo-style-eval"]["provenance"]
                if "jmsc-public-dialogue" not in row["source_file"]
            ]
            state["JLongChat-LoCoMo-style-eval"]["hash_check"] = [
                row
                for row in state["JLongChat-LoCoMo-style-eval"]["hash_check"]
                if "jmsc" not in row["source_turn_id"]
            ]

        run_case(
            "missing_required_source_family_is_rejected",
            remove_required_family,
            False,
            ["JLongChat_JMSC: required source family has no original_turn hash coverage rows"],
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
