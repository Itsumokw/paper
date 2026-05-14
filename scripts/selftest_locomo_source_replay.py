#!/usr/bin/env python3
"""Self-test raw source replay checks for LoCoMo-style provenance sidecars."""

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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_rows(sidecar_root: Path) -> dict[str, list[dict[str, Any]]]:
    seeded: dict[str, list[dict[str, Any]]] = {}
    for artifact in SOURCE_ARTIFACTS.values():
        path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        for row in iter_jsonl(path):
            if row.get("source_origin") in {"original_turn", "memory_anchor_turn"}:
                seeded[artifact] = [row]
                break
        if artifact not in seeded:
            raise ValueError(f"no replayable provenance row found in {path}")
    return seeded


def write_fixture(
    sidecar_root: Path,
    rows: dict[str, list[dict[str, Any]]],
    mutate: Callable[[dict[str, list[dict[str, Any]]]], None] | None = None,
) -> None:
    state = deepcopy(rows)
    if mutate is not None:
        mutate(state)
    for artifact, artifact_rows in state.items():
        write_jsonl(sidecar_root / artifact / f"{artifact}_provenance.jsonl", artifact_rows)


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
        "--source-sidecar-root",
        type=Path,
        default=Path("datasets/locomo_style_eval/sidecars"),
    )
    parser.add_argument(
        "--checker",
        type=Path,
        default=Path("scripts/check_locomo_source_replay.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/source_replay_selftest.json"),
    )
    args = parser.parse_args()

    rows = seed_rows(args.source_sidecar_root)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_source_replay_selftest_") as tmp:
        tempdir = Path(tmp)

        def run_case(
            name: str,
            mutate: Callable[[dict[str, list[dict[str, Any]]]], None] | None,
            expect_success: bool,
            expected_error_fragments: list[str] | None = None,
        ) -> None:
            case_root = tempdir / name / "sidecars"
            write_fixture(case_root, rows, mutate)
            rc, summary = run_checker(args.checker, case_root, tempdir / name, name)
            cases.append(case_result(name, rc, summary, expect_success, expected_error_fragments))

        run_case("valid_replay_subset_is_accepted", None, True)

        def corrupt_text(state: dict[str, list[dict[str, Any]]]) -> None:
            state["JLongChat-LoCoMo-style-eval"][0]["text"] = "source replay self-test text mismatch"

        run_case(
            "text_mismatch_is_rejected",
            corrupt_text,
            False,
            ["replay text mismatch"],
        )

        def corrupt_speaker(state: dict[str, list[dict[str, Any]]]) -> None:
            state["deL1L2IM-LoCoMo-style-eval"][0]["source_speaker"] = "__wrong_speaker__"

        run_case(
            "source_speaker_mismatch_is_rejected",
            corrupt_speaker,
            False,
            ["source_speaker='__wrong_speaker__'"],
        )

        def corrupt_source_turn_id(state: dict[str, list[dict[str, Any]]]) -> None:
            state["JLongChat-LoCoMo-style-eval"][0]["source_turn_id"] = "__invalid_source_turn_id__"

        run_case(
            "invalid_source_turn_id_is_rejected",
            corrupt_source_turn_id,
            False,
            ["replay failed"],
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "source_sidecar_root": str(args.source_sidecar_root),
        "source_sidecar_root_sha256": sidecar_provenance_sha256(args.source_sidecar_root),
        "checker": str(args.checker),
        "checker_sha256": sha256_file(args.checker),
        "selftest_sha256": sha256_file(Path(__file__)),
        "seed_rows": {
            artifact: [
                {
                    "source_dataset": row.get("source_dataset"),
                    "sample_id": row.get("sample_id"),
                    "dia_id": row.get("dia_id"),
                    "source_origin": row.get("source_origin"),
                    "source_file": row.get("source_file"),
                    "source_turn_id": row.get("source_turn_id"),
                }
                for row in artifact_rows
            ]
            for artifact, artifact_rows in rows.items()
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def sidecar_provenance_sha256(sidecar_root: Path) -> str:
    digest = hashlib.sha256()
    for artifact in SOURCE_ARTIFACTS.values():
        path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
