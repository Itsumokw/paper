#!/usr/bin/env python3
"""Self-test audit application and audited-output replay integrity."""

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_fixture() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "fixture_sample_0",
            "source_dataset": "JLongChat",
            "language": "ja",
            "split": "eval",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "sid 1",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:1", "text": "old answer"},
                    {"speaker": "B", "dia_id": "D1:2", "text": "corrected answer"},
                ],
            },
            "observation": {},
            "session_summary": {},
            "event_summary": {},
            "qa": [
                {
                    "question": "What was said first?",
                    "answer": "old answer",
                    "category": 1,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "Question to delete",
                    "answer": "remove me",
                    "category": 1,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "Unsupported adversarial question",
                    "category": 5,
                    "evidence": [],
                    "adversarial_answer": "unsupported",
                },
                {
                    "question": "Unsupported adversarial question to pass",
                    "category": 5,
                    "evidence": [],
                    "adversarial_answer": "unsupported pass",
                },
            ],
        }
    ]


def audit_fixture() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "fixture_sample_0",
            "source_dataset": "JLongChat",
            "qa_idx": 0,
            "category": 1,
            "human_decision": "fix",
            "human_notes": "self-test corrected answer",
            "corrected_answer": "corrected answer",
            "corrected_evidence": ["D1:2"],
            "corrected_answer_facts": [
                {
                    "fact": "corrected answer",
                    "supported_by": ["D1:2"],
                    "source_fact_id": "fixture_fact_d1_2",
                }
            ],
            "corrected_evidence_detail": [
                {
                    "dia_id": "D1:2",
                    "source_origin": "original_turn",
                    "supports_answer_fact": ["fixture_fact_d1_2"],
                }
            ],
        },
        {
            "sample_id": "fixture_sample_0",
            "source_dataset": "JLongChat",
            "qa_idx": 1,
            "category": 1,
            "human_decision": "delete",
            "human_notes": "self-test deletion",
        },
        {
            "sample_id": "fixture_sample_0",
            "source_dataset": "JLongChat",
            "qa_idx": 2,
            "category": 5,
            "human_decision": "fix",
            "human_notes": "self-test cat5 correction keeps primary loader-only",
            "corrected_adversarial_answer": "unsupported corrected",
            "corrected_negative_evidence": ["D1:1"],
            "corrected_adversarial_reason": "unsupported_fact",
        },
        {
            "sample_id": "fixture_sample_0",
            "source_dataset": "JLongChat",
            "qa_idx": 3,
            "category": 5,
            "human_decision": "pass",
            "human_notes": "",
        },
    ]


def write_sidecars(sidecar_root: Path) -> None:
    artifact_dir = sidecar_root / "JLongChat-LoCoMo-style-eval"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        artifact_dir / "JLongChat-LoCoMo-style-eval_fact_ledger.jsonl",
        [
            {
                "fact_id": "fixture_fact_d1_1",
                "source_type": "original_turn",
                "source_text": "old answer",
                "source_id": "D1:1",
            },
            {
                "fact_id": "fixture_fact_d1_2",
                "source_type": "original_turn",
                "source_text": "corrected answer",
                "source_id": "D1:2",
            },
        ],
    )
    write_jsonl(
        artifact_dir / "JLongChat-LoCoMo-style-eval_provenance.jsonl",
        [
            {
                "sample_id": "fixture_sample_0",
                "dia_id": "D1:1",
                "source_origin": "original_turn",
                "text": "old answer",
            },
            {
                "sample_id": "fixture_sample_0",
                "dia_id": "D1:2",
                "source_origin": "original_turn",
                "text": "corrected answer",
            },
        ],
    )
    write_jsonl(
        artifact_dir / "JLongChat-LoCoMo-style-eval_qa_audit.jsonl",
        [
            {
                "source_dataset": "JLongChat",
                "sample_id": "fixture_sample_0",
                "qa_idx": 0,
                "category": 1,
                "answer_facts": [
                    {
                        "fact": "old answer",
                        "supported_by": ["D1:1"],
                        "source_fact_id": "fixture_fact_d1_1",
                    }
                ],
                "evidence_detail": [
                    {
                        "dia_id": "D1:1",
                        "source_origin": "original_turn",
                        "supports_answer_fact": ["fixture_fact_d1_1"],
                    }
                ],
            },
            {
                "source_dataset": "JLongChat",
                "sample_id": "fixture_sample_0",
                "qa_idx": 1,
                "category": 1,
                "answer_facts": [
                    {
                        "fact": "old answer",
                        "supported_by": ["D1:1"],
                        "source_fact_id": "fixture_fact_d1_1",
                    }
                ],
                "evidence_detail": [
                    {
                        "dia_id": "D1:1",
                        "source_origin": "original_turn",
                        "supports_answer_fact": ["fixture_fact_d1_1"],
                    }
                ],
            },
            {
                "source_dataset": "JLongChat",
                "sample_id": "fixture_sample_0",
                "qa_idx": 2,
                "category": 5,
                "negative_evidence": ["D1:1"],
                "adversarial_reason": "unsupported_fact",
            },
            {
                "source_dataset": "JLongChat",
                "sample_id": "fixture_sample_0",
                "qa_idx": 3,
                "category": 5,
                "negative_evidence": ["D1:1"],
                "adversarial_reason": "unsupported_fact",
            },
        ],
    )
    for artifact in ("PerLTQA-LoCoMo-style-eval", "OPELA-LoCoMo-style-eval", "deL1L2IM-LoCoMo-style-eval"):
        other_dir = sidecar_root / artifact
        other_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(other_dir / f"{artifact}_fact_ledger.jsonl", [])
        write_jsonl(other_dir / f"{artifact}_provenance.jsonl", [])
        write_jsonl(other_dir / f"{artifact}_qa_audit.jsonl", [])


def run_json_command(command: list[str], tempdir: Path) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    parsed["_tempdir"] = str(tempdir)
    if completed.stderr.strip() and completed.stdout.strip():
        parsed["stderr"] = completed.stderr
    return completed.returncode, parsed


def normalize_errors(result: dict[str, Any]) -> list[str]:
    tempdir = str(result.get("_tempdir", ""))
    return [str(item).replace(tempdir + "/", "<tmp>/") for item in result.get("errors", [])]


def case(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", **(details or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply-script",
        type=Path,
        default=Path("scripts/apply_locomo_human_audit_results.py"),
    )
    parser.add_argument(
        "--integrity-script",
        type=Path,
        default=Path("scripts/check_locomo_audited_apply_integrity.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/audit_apply_integrity_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_audit_apply_integrity_selftest_") as tmp:
        tempdir = Path(tmp)
        original_primary = tempdir / "primary.json"
        audit_jsonl = tempdir / "audit.jsonl"
        audited_primary = tempdir / "audited.json"
        apply_report = tempdir / "apply_report.json"
        output_source_dir = tempdir / "audited_sources"
        integrity_report = tempdir / "integrity_report.json"
        sidecar_root = tempdir / "sidecars"

        write_json(original_primary, primary_fixture())
        write_jsonl(audit_jsonl, audit_fixture())
        write_sidecars(sidecar_root)

        apply_command = [
            sys.executable,
            str(args.apply_script),
            "--primary-json",
            str(original_primary),
            "--audit-jsonl",
            str(audit_jsonl),
            "--output-json",
            str(audited_primary),
            "--output-report",
            str(apply_report),
            "--output-source-dir",
            str(output_source_dir),
        ]
        rc, apply_result = run_json_command(apply_command, tempdir)
        applied = rc == 0 and apply_result.get("status") == "applied"
        audited = load_json(audited_primary) if audited_primary.exists() else []
        source_file = output_source_dir / "JLongChat-LoCoMo-style-eval.json"
        apply_shape_ok = (
            applied
            and apply_result.get("input_qa") == 4
            and apply_result.get("output_qa") == 3
            and apply_result.get("fixed_count") == 2
            and apply_result.get("removed_count") == 1
            and source_file.exists()
            and audited[0]["qa"][0]["answer"] == "corrected answer"
            and audited[0]["qa"][0]["evidence"] == ["D1:2"]
            and audited[0]["qa"][1]["category"] == 5
            and audited[0]["qa"][1]["evidence"] == []
            and "answer" not in audited[0]["qa"][1]
            and audited[0]["qa"][1]["adversarial_answer"] == "unsupported corrected"
            and "negative_evidence" not in audited[0]["qa"][1]
            and "adversarial_reason" not in audited[0]["qa"][1]
            and audited[0]["qa"][2]["category"] == 5
        )
        cases.append(case("apply_fix_delete_pass_fixture", apply_shape_ok, {"errors": normalize_errors(apply_result)}))

        integrity_command = [
            sys.executable,
            str(args.integrity_script),
            "--original-primary",
            str(original_primary),
            "--audit-jsonl",
            str(audit_jsonl),
            "--audited-primary",
            str(audited_primary),
            "--sidecar-root",
            str(sidecar_root),
            "--output",
            str(integrity_report),
        ]
        rc, integrity_result = run_json_command(integrity_command, tempdir)
        cases.append(
            case(
                "integrity_accepts_exact_replay",
                rc == 0
                and integrity_result.get("status") == "passed"
                and integrity_result.get("trace_checked_qa") == 3
                and integrity_result.get("sidecar_root") == str(sidecar_root)
                and isinstance(integrity_result.get("sidecar_trace_files_sha256"), str),
                {"errors": normalize_errors(integrity_result)},
            )
        )

        bad_sidecar_root = tempdir / "bad_sidecars"
        write_sidecars(bad_sidecar_root)
        bad_artifact = bad_sidecar_root / "JLongChat-LoCoMo-style-eval"
        write_jsonl(
            bad_artifact / "JLongChat-LoCoMo-style-eval_fact_ledger.jsonl",
            [
                {
                    "fact_id": "fixture_fact_d1_1",
                    "source_type": "original_turn",
                    "source_text": "old answer",
                    "source_id": "D1:1",
                },
                {
                    "fact_id": "fixture_fact_d1_2",
                    "source_type": "synthetic_continuation_turn",
                    "source_text": "corrected answer",
                    "source_id": "D1:2",
                },
            ],
        )
        bad_trace_report = tempdir / "bad_trace_integrity_report.json"
        bad_trace_command = [
            sys.executable,
            str(args.integrity_script),
            "--original-primary",
            str(original_primary),
            "--audit-jsonl",
            str(audit_jsonl),
            "--audited-primary",
            str(audited_primary),
            "--sidecar-root",
            str(bad_sidecar_root),
            "--output",
            str(bad_trace_report),
        ]
        rc, bad_trace_result = run_json_command(bad_trace_command, tempdir)
        bad_trace_errors = normalize_errors(bad_trace_result)
        cases.append(
            case(
                "integrity_rejects_non_original_answer_fact_trace",
                rc != 0
                and any("source_type='synthetic_continuation_turn' is not original-backed" in error for error in bad_trace_errors),
                {"errors": bad_trace_errors},
            )
        )

        tampered_primary = tempdir / "audited_tampered.json"
        tampered = deepcopy(audited)
        tampered[0]["qa"][0]["answer"] = "tampered answer"
        write_json(tampered_primary, tampered)
        tampered_report = tempdir / "tampered_integrity_report.json"
        tampered_command = [
            sys.executable,
            str(args.integrity_script),
            "--original-primary",
            str(original_primary),
            "--audit-jsonl",
            str(audit_jsonl),
            "--audited-primary",
            str(tampered_primary),
            "--sidecar-root",
            str(sidecar_root),
            "--output",
            str(tampered_report),
        ]
        rc, tampered_result = run_json_command(tampered_command, tempdir)
        tampered_errors = normalize_errors(tampered_result)
        cases.append(
            case(
                "integrity_rejects_tampered_output",
                rc != 0 and any("differs from replayed expected output" in error for error in tampered_errors),
                {"errors": tampered_errors},
            )
        )

        incomplete_audit = tempdir / "audit_incomplete.jsonl"
        bad_rows = deepcopy(audit_fixture())
        bad_rows[0]["human_decision"] = "todo"
        write_jsonl(incomplete_audit, bad_rows)
        incomplete_report = tempdir / "apply_incomplete_report.json"
        rc, incomplete_result = run_json_command(
            [
                sys.executable,
                str(args.apply_script),
                "--primary-json",
                str(original_primary),
                "--audit-jsonl",
                str(incomplete_audit),
                "--output-json",
                str(tempdir / "should_not_write.json"),
                "--output-report",
                str(incomplete_report),
            ],
            tempdir,
        )
        incomplete_errors = normalize_errors(incomplete_result)
        cases.append(
            case(
                "apply_rejects_incomplete_audit",
                rc != 0 and any("incomplete" in error for error in incomplete_errors),
                {"errors": incomplete_errors},
            )
        )

    report = {
        "status": "passed" if all(item["status"] == "passed" for item in cases) else "failed",
        "apply_script": str(args.apply_script),
        "apply_script_sha256": sha256_file(args.apply_script),
        "integrity_script": str(args.integrity_script),
        "integrity_script_sha256": sha256_file(args.integrity_script),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
