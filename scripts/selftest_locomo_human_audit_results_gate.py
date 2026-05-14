#!/usr/bin/env python3
"""Self-test release-gate validation for completed human-audit summaries."""

from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import check_locomo_style_release_gates as gates


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def valid_summary(root: Path) -> dict[str, Any]:
    audit_packet = root / "human_audit_packet.jsonl"
    sidecar_root = root / "sidecars"
    validator = Path(gates.__file__).with_name("validate_locomo_human_audit_results.py")
    return {
        "status": "completed",
        "input_jsonl": str(audit_packet),
        "input_jsonl_sha256": gates.sha256_file(audit_packet),
        "validator": str(validator),
        "validator_sha256": gates.sha256_file(validator),
        "sidecar_root": str(sidecar_root),
        "sidecar_trace_files_sha256": gates.sidecar_trace_files_sha256(sidecar_root),
        "rows": 1,
        "decision_counts": {"pass": 1},
        "incomplete_count": 0,
        "errors": [],
    }


def write_audit_queue_and_packet(root: Path, packet_rows: list[dict[str, Any]], queue_rows: list[dict[str, Any]]) -> None:
    write_text(
        root / "human_audit_packet.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet_rows),
    )
    write_text(
        root / "human_audit_queue.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue_rows),
    )


def case_result(
    name: str,
    report: dict[str, Any],
    root: Path,
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = gates.human_audit_results_errors(report, root)
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


def packet_queue_case_result(
    name: str,
    root: Path,
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = gates.human_audit_packet_queue_errors(root)
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
        default=Path("datasets/locomo_style_eval/human_audit_results_gate_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_results_gate_selftest_") as tmp:
        root = Path(tmp)
        base_row = {
            "source_dataset": "Fixture",
            "sample_id": "fixture_0",
            "qa_idx": 0,
            "human_decision": "pass",
        }
        write_audit_queue_and_packet(root, [base_row], [base_row])
        (root / "sidecars").mkdir(parents=True, exist_ok=True)

        good = valid_summary(root)
        cases.append(case_result("valid_completed_summary_is_accepted", good, root, expect_success=True))
        cases.append(packet_queue_case_result("audit_packet_matches_queue_is_accepted", root, expect_success=True))

        bad = deepcopy(good)
        bad["status"] = "incomplete_or_failed"
        bad["incomplete_count"] = 1
        cases.append(
            case_result(
                "incomplete_summary_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="status=",
            )
        )

        bad = deepcopy(good)
        bad["input_jsonl_sha256"] = "0" * 64
        cases.append(
            case_result(
                "audit_packet_hash_mismatch_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="input_jsonl_sha256 mismatch",
            )
        )

        bad = deepcopy(good)
        bad["validator_sha256"] = "0" * 64
        cases.append(
            case_result(
                "validator_hash_mismatch_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="validator_sha256 mismatch",
            )
        )

        bad = deepcopy(good)
        bad["sidecar_trace_files_sha256"] = "0" * 64
        cases.append(
            case_result(
                "sidecar_trace_hash_mismatch_is_rejected",
                bad,
                root,
                expect_success=False,
                expected_error_fragment="sidecar_trace_files_sha256 mismatch",
            )
        )

        missing_packet_root = root / "missing_packet"
        missing_packet_root.mkdir()
        write_audit_queue_and_packet(missing_packet_root, [], [base_row])
        cases.append(
            packet_queue_case_result(
                "audit_packet_missing_queue_row_is_rejected",
                missing_packet_root,
                expect_success=False,
                expected_error_fragment="audit packet missing queue QA keys",
            )
        )

        extra_packet_root = root / "extra_packet"
        extra_packet_root.mkdir()
        extra_row = {**base_row, "qa_idx": 1}
        write_audit_queue_and_packet(extra_packet_root, [base_row, extra_row], [base_row])
        cases.append(
            packet_queue_case_result(
                "audit_packet_extra_row_is_rejected",
                extra_packet_root,
                expect_success=False,
                expected_error_fragment="audit packet has QA keys not in queue",
            )
        )

        duplicate_packet_root = root / "duplicate_packet"
        duplicate_packet_root.mkdir()
        write_audit_queue_and_packet(duplicate_packet_root, [base_row, base_row], [base_row])
        cases.append(
            packet_queue_case_result(
                "audit_packet_duplicate_key_is_rejected",
                duplicate_packet_root,
                expect_success=False,
                expected_error_fragment="duplicate audit keys",
            )
        )

    gate_script = Path(gates.__file__)
    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "gate_script": str(gate_script),
        "gate_script_sha256": gates.sha256_file(gate_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
