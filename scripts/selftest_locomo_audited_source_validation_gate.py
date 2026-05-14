#!/usr/bin/env python3
"""Self-test release-gate validation for source-specific audited files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from check_locomo_style_release_gates import audited_source_validation_errors


SOURCES = {
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


def sample(source: str, idx: int) -> dict[str, Any]:
    return {
        "sample_id": f"{source.lower()}_{idx}",
        "source_dataset": source,
        "language": "en",
        "split": "eval",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1_date_time": "2026-01-01",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": f"{source} fact"}
            ],
        },
        "observation": {},
        "session_summary": {},
        "event_summary": {},
        "qa": [
            {
                "question": f"What source is {source}?",
                "answer": source,
                "category": 1,
                "evidence": ["D1:1"],
            }
        ],
    }


def build_fixture(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    combined_rows = [sample(source, 0) for source in SOURCES]
    combined_path = root / "primary" / "multilingual_locomo_style_eval_audited.json"
    write_json(combined_path, combined_rows)

    rows_by_source = {source: [row] for source, row in zip(SOURCES, combined_rows, strict=True)}
    output_source_files: dict[str, str] = {}
    for source, artifact in SOURCES.items():
        path = root / "primary" / "audited_sources" / f"{artifact}.json"
        write_json(path, rows_by_source[source])
        output_source_files[source] = str(path)

    audit_apply = {
        "status": "applied",
        "output_json": str(combined_path),
        "output_source_files": output_source_files,
    }
    return audit_apply, rows_by_source


def case_result(
    name: str,
    audit_apply: dict[str, Any],
    root: Path,
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = audited_source_validation_errors(audit_apply, root)
    observed_success = not errors
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None
            or any(expected_error_fragment in error for error in errors)
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "expected_error_fragment": expected_error_fragment,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/audited_source_validation_gate_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_audited_source_gate_selftest_") as tmp:
        root = Path(tmp)
        audit_apply, rows_by_source = build_fixture(root)

        cases.append(
            case_result("valid_audited_source_partition_is_accepted", audit_apply, root, expect_success=True)
        )

        bad = deepcopy(audit_apply)
        bad["output_source_files"].pop("OPELA")
        cases.append(
            case_result(
                "missing_source_file_entry_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="missing audited source output for OPELA",
            )
        )

        bad = deepcopy(audit_apply)
        bad["output_source_files"]["PerLTQA"] = str(root / "primary" / "PerLTQA-LoCoMo-style-eval.json")
        cases.append(
            case_result(
                "wrong_source_output_path_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="output path=",
            )
        )

        bad_path = root / "primary" / "audited_sources" / "JLongChat-LoCoMo-style-eval.json"
        wrong_source_rows = deepcopy(rows_by_source["JLongChat"])
        wrong_source_rows[0]["source_dataset"] = "OPELA"
        write_json(bad_path, wrong_source_rows)
        cases.append(
            case_result(
                "wrong_source_dataset_inside_file_is_rejected",
                audit_apply,
                root,
                expect_success=False,
                expected_error_fragment="contains other source_dataset values",
            )
        )
        write_json(bad_path, rows_by_source["JLongChat"])

        missing_row_path = root / "primary" / "audited_sources" / "deL1L2IM-LoCoMo-style-eval.json"
        write_json(missing_row_path, [])
        cases.append(
            case_result(
                "source_files_missing_combined_sample_is_rejected",
                audit_apply,
                root,
                expect_success=False,
                expected_error_fragment="audited source files missing combined sample_ids",
            )
        )

    gate_script = Path(__file__).with_name("check_locomo_style_release_gates.py")
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "gate_script": str(gate_script),
        "gate_script_sha256": sha256_file(gate_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
