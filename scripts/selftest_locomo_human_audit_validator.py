#!/usr/bin/env python3
"""Self-test the human-audit validator's trace enforcement."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_trace_files_sha256(sidecar_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        {
            *sidecar_root.glob("*/*_fact_ledger.jsonl"),
            *sidecar_root.glob("*/*_provenance.jsonl"),
        }
    )
    for path in paths:
        digest.update(str(path.relative_to(sidecar_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_rows(audit_packet: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    answerable: dict[str, Any] | None = None
    cat5: dict[str, Any] | None = None
    for row in iter_jsonl(audit_packet):
        category = int(row.get("category", 0))
        if (
            category != 5
            and row.get("evidence")
            and row.get("answer_facts")
            and row.get("evidence_detail")
            and answerable is None
        ):
            answerable = row
        if category == 5 and cat5 is None:
            cat5 = row
        if answerable is not None and cat5 is not None:
            break
    if answerable is None:
        raise ValueError(f"no answerable traced seed row found in {audit_packet}")
    if cat5 is None:
        raise ValueError(f"no category-5 seed row found in {audit_packet}")
    return answerable, cat5


def base_completed(row: dict[str, Any], decision: str) -> dict[str, Any]:
    copied = deepcopy(row)
    copied["human_decision"] = decision
    copied["human_notes"] = "" if decision == "pass" else "validator self-test"
    for key in list(copied):
        if key.startswith("corrected_"):
            copied.pop(key)
    return copied


def run_validator(
    validator: Path,
    sidecar_root: Path,
    tempdir: Path,
    name: str,
    rows: list[dict[str, Any]],
    allow_incomplete: bool = False,
) -> tuple[int, dict[str, Any]]:
    input_jsonl = tempdir / f"{name}.jsonl"
    output_summary = tempdir / f"{name}_summary.json"
    write_jsonl(input_jsonl, rows)
    command = [
        sys.executable,
        str(validator),
        "--input-jsonl",
        str(input_jsonl),
        "--output-summary",
        str(output_summary),
        "--allow-failures",
        "--sidecar-root",
        str(sidecar_root),
    ]
    if allow_incomplete:
        command.append("--allow-incomplete")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    summary: dict[str, Any]
    if output_summary.exists():
        summary = json.loads(output_summary.read_text(encoding="utf-8"))
    else:
        summary = {
            "status": "validator_did_not_write_summary",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    summary["returncode"] = completed.returncode
    summary["_tempdir"] = str(tempdir)
    if completed.stderr.strip():
        summary["stderr"] = completed.stderr
    return completed.returncode, summary


def case_result(
    name: str,
    returncode: int,
    summary: dict[str, Any],
    expect_success: bool,
    expected_error_fragments: list[str] | None = None,
    expected_status: str = "completed",
) -> dict[str, Any]:
    tempdir = str(summary.get("_tempdir", ""))
    errors = [str(item).replace(tempdir + "/", "<tmp>/") for item in summary.get("errors", [])]
    fragments = expected_error_fragments or []
    missing_fragments = [
        fragment
        for fragment in fragments
        if not any(fragment in error for error in errors)
    ]
    observed_success = returncode == 0 and summary.get("status") == expected_status
    passed = observed_success if expect_success else (not observed_success and not missing_fragments)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": returncode,
        "validator_status": summary.get("status"),
        "expected_status": expected_status,
        "expected_error_fragments": fragments,
        "missing_expected_error_fragments": missing_fragments,
        "errors": errors[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-packet", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path("scripts/validate_locomo_human_audit_results.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_validator_selftest.json"),
    )
    args = parser.parse_args()

    answerable_seed, cat5_seed = seed_rows(args.audit_packet)

    pass_row = base_completed(answerable_seed, "pass")

    missing_trace_fix = base_completed(answerable_seed, "fix")
    missing_trace_fix["corrected_answer"] = str(answerable_seed.get("answer", "")) + " [self-test edit]"

    traced_fix = base_completed(answerable_seed, "fix")
    traced_fix["corrected_answer"] = answerable_seed.get("answer")
    traced_fix["corrected_evidence"] = answerable_seed.get("evidence", [])
    traced_fix["corrected_answer_facts"] = answerable_seed.get("answer_facts", [])
    traced_fix["corrected_evidence_detail"] = answerable_seed.get("evidence_detail", [])

    cat5_bad_trace = base_completed(cat5_seed, "fix")
    cat5_bad_trace["corrected_adversarial_answer"] = (
        cat5_seed.get("adversarial_answer") or "No supported answer is available."
    )
    cat5_bad_trace["corrected_negative_evidence"] = cat5_seed.get("negative_evidence", [])
    cat5_bad_trace["corrected_adversarial_reason"] = cat5_seed.get("adversarial_reason", "unsupported_fact")
    cat5_bad_trace["corrected_answer_facts"] = [{"source_fact_id": "selftest_should_not_be_allowed"}]

    cat5_good_fix = base_completed(cat5_seed, "fix")
    cat5_good_fix["corrected_adversarial_answer"] = (
        cat5_seed.get("adversarial_answer") or "No supported answer is available."
    )
    cat5_good_fix["corrected_negative_evidence"] = cat5_seed.get("negative_evidence", [])
    cat5_good_fix["corrected_adversarial_reason"] = cat5_seed.get("adversarial_reason", "unsupported_fact")

    cat5_invalid_reason = deepcopy(cat5_good_fix)
    cat5_invalid_reason["corrected_adversarial_reason"] = "not_a_reason"

    cat5_pass_invalid_original_reason = base_completed(cat5_seed, "pass")
    cat5_pass_invalid_original_reason["adversarial_reason"] = "not_a_reason"

    cat5_missing_negative_fields = base_completed(answerable_seed, "fix")
    cat5_missing_negative_fields["corrected_category"] = 5
    cat5_missing_negative_fields["corrected_adversarial_answer"] = "No supported answer is available."

    pass_with_corrected_field = base_completed(answerable_seed, "pass")
    pass_with_corrected_field["corrected_answer"] = answerable_seed.get("answer")

    fix_without_notes = base_completed(answerable_seed, "fix")
    fix_without_notes["human_notes"] = ""
    fix_without_notes["corrected_question"] = answerable_seed.get("question")

    fix_without_corrected_field = base_completed(answerable_seed, "fix")

    answerable_fix_with_cat5_fields = base_completed(answerable_seed, "fix")
    answerable_fix_with_cat5_fields["corrected_adversarial_answer"] = "No supported answer is available."
    answerable_fix_with_cat5_fields["corrected_negative_evidence"] = answerable_seed.get("evidence", [])
    answerable_fix_with_cat5_fields["corrected_adversarial_reason"] = "unsupported_fact"

    invalid_corrected_category = base_completed(answerable_seed, "fix")
    invalid_corrected_category["corrected_category"] = 6

    evidence_origin_mismatch = deepcopy(traced_fix)
    evidence_origin_mismatch["corrected_evidence_detail"] = deepcopy(
        evidence_origin_mismatch["corrected_evidence_detail"]
    )
    evidence_origin_mismatch["corrected_evidence_detail"][0]["source_origin"] = "__wrong_origin__"

    missing_fact_ledger_trace = deepcopy(traced_fix)
    missing_fact_ledger_trace["corrected_answer_facts"] = [
        {
            "fact": "validator self-test missing fact",
            "source_fact_id": "selftest_missing_fact_id",
            "supported_by": missing_fact_ledger_trace.get("corrected_evidence", []),
        }
    ]
    missing_fact_ledger_trace["corrected_evidence_detail"] = deepcopy(
        missing_fact_ledger_trace["corrected_evidence_detail"]
    )
    missing_fact_ledger_trace["corrected_evidence_detail"][0]["supports_answer_fact"] = [
        "selftest_missing_fact_id"
    ]

    todo_row = deepcopy(answerable_seed)
    todo_row["human_decision"] = "todo"
    todo_row["human_notes"] = ""
    for key in list(todo_row):
        if key.startswith("corrected_"):
            todo_row.pop(key)

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_audit_validator_selftest_") as tmp:
        tempdir = Path(tmp)
        rc, summary = run_validator(args.validator, args.sidecar_root, tempdir, "pass_row", [pass_row])
        cases.append(case_result("pass_row_is_accepted", rc, summary, True))

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "missing_trace_fix",
            [missing_trace_fix],
        )
        cases.append(
            case_result(
                "support_changing_fix_requires_trace",
                rc,
                summary,
                False,
                ["requires corrected_answer_facts", "requires corrected_evidence_detail"],
            )
        )

        rc, summary = run_validator(args.validator, args.sidecar_root, tempdir, "traced_fix", [traced_fix])
        cases.append(case_result("traced_support_changing_fix_is_accepted", rc, summary, True))

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "cat5_bad_trace",
            [cat5_bad_trace],
        )
        cases.append(
            case_result(
                "cat5_fix_rejects_answer_fact_trace",
                rc,
                summary,
                False,
                ["category 5 fix must not use corrected_answer_facts"],
            )
        )

        rc, summary = run_validator(args.validator, args.sidecar_root, tempdir, "cat5_good_fix", [cat5_good_fix])
        cases.append(case_result("cat5_fix_with_negative_fields_is_accepted", rc, summary, True))

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "cat5_invalid_reason",
            [cat5_invalid_reason],
        )
        cases.append(
            case_result(
                "cat5_fix_rejects_invalid_adversarial_reason",
                rc,
                summary,
                False,
                ["corrected_adversarial_reason='not_a_reason' must be one of"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "cat5_pass_invalid_original_reason",
            [cat5_pass_invalid_original_reason],
        )
        cases.append(
            case_result(
                "cat5_pass_rejects_invalid_original_adversarial_reason",
                rc,
                summary,
                False,
                ["adversarial_reason='not_a_reason' must be one of"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "cat5_missing_negative_fields",
            [cat5_missing_negative_fields],
        )
        cases.append(
            case_result(
                "cat5_fix_requires_negative_evidence_and_reason",
                rc,
                summary,
                False,
                [
                    "category 5 fix requires negative_evidence or corrected_negative_evidence",
                    "category 5 fix requires adversarial_reason or corrected_adversarial_reason",
                ],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "pass_with_corrected_field",
            [pass_with_corrected_field],
        )
        cases.append(
            case_result(
                "non_fix_decision_rejects_corrected_fields",
                rc,
                summary,
                False,
                ["corrected_* fields are only applied when human_decision='fix'"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "fix_without_notes",
            [fix_without_notes],
        )
        cases.append(
            case_result(
                "fix_decision_requires_human_notes",
                rc,
                summary,
                False,
                ["fix decision requires human_notes"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "fix_without_corrected_field",
            [fix_without_corrected_field],
        )
        cases.append(
            case_result(
                "fix_decision_requires_corrected_field",
                rc,
                summary,
                False,
                ["fix decision requires at least one corrected_* field"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "answerable_fix_with_cat5_fields",
            [answerable_fix_with_cat5_fields],
        )
        cases.append(
            case_result(
                "answerable_fix_rejects_cat5_fields",
                rc,
                summary,
                False,
                [
                    "answerable fix must not include corrected_adversarial_answer",
                    "answerable fix must not include corrected_negative_evidence",
                    "answerable fix must not include corrected_adversarial_reason",
                ],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "invalid_corrected_category",
            [invalid_corrected_category],
        )
        cases.append(
            case_result(
                "invalid_corrected_category_is_rejected",
                rc,
                summary,
                False,
                ["fix decision has invalid target category"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "evidence_origin_mismatch",
            [evidence_origin_mismatch],
        )
        cases.append(
            case_result(
                "corrected_evidence_detail_source_origin_mismatch_is_rejected",
                rc,
                summary,
                False,
                ["source_origin='__wrong_origin__' expected="],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "missing_fact_ledger_trace",
            [missing_fact_ledger_trace],
        )
        cases.append(
            case_result(
                "corrected_answer_fact_missing_source_fact_is_rejected",
                rc,
                summary,
                False,
                ["source_fact_id='selftest_missing_fact_id' missing fact ledger"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "todo_row_default",
            [todo_row],
        )
        cases.append(
            case_result(
                "todo_row_is_rejected_by_default",
                rc,
                summary,
                False,
                ["audit rows are incomplete"],
            )
        )

        rc, summary = run_validator(
            args.validator,
            args.sidecar_root,
            tempdir,
            "todo_row_allow_incomplete",
            [todo_row],
            allow_incomplete=True,
        )
        cases.append(
            case_result(
                "allow_incomplete_validates_completed_subset",
                rc,
                summary,
                True,
                expected_status="partial_valid",
            )
        )

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "audit_packet": str(args.audit_packet),
        "audit_packet_sha256": sha256_file(args.audit_packet),
        "sidecar_root": str(args.sidecar_root),
        "sidecar_trace_files_sha256": sidecar_trace_files_sha256(args.sidecar_root),
        "validator": str(args.validator),
        "validator_sha256": sha256_file(args.validator),
        "seed_answerable": {
            "source_dataset": answerable_seed.get("source_dataset"),
            "sample_id": answerable_seed.get("sample_id"),
            "qa_idx": answerable_seed.get("qa_idx"),
            "category": answerable_seed.get("category"),
        },
        "seed_cat5": {
            "source_dataset": cat5_seed.get("source_dataset"),
            "sample_id": cat5_seed.get("sample_id"),
            "qa_idx": cat5_seed.get("qa_idx"),
            "category": cat5_seed.get("category"),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
