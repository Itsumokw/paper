#!/usr/bin/env python3
"""Self-test QA trace integrity checks for LoCoMo-style eval sidecars."""

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


def sample_fixture(source: str, artifact: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_id = f"{source.lower()}_fixture_0"
    fact_1 = f"{sample_id}_fact_1"
    fact_2 = f"{sample_id}_fact_2"
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
                    {"speaker": "A", "dia_id": "D1:1", "text": f"{source} original fact one."},
                    {"speaker": "B", "dia_id": "D1:2", "text": f"{source} original fact two."},
                ],
                "session_2_date_time": "2024-01-02",
                "session_2": [
                    {"speaker": "A", "dia_id": "D2:1", "text": f"{source} later context."},
                ],
            },
            "observation": {},
            "session_summary": {},
            "event_summary": {},
            "qa": [
                {
                    "question": f"What did {source} say first?",
                    "answer": f"{source} original fact one.",
                    "category": 1,
                    "evidence": ["D1:1"],
                },
                {
                    "question": f"Did {source} claim an unsupported event?",
                    "category": 5,
                    "evidence": [],
                    "adversarial_answer": "No supported answer is present.",
                },
            ],
        }
    ]
    provenance = [
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D1:1",
            "source_origin": "original_turn",
            "text": f"{source} original fact one.",
            "source_fact_ids": [fact_1],
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D1:2",
            "source_origin": "original_turn",
            "text": f"{source} original fact two.",
            "source_fact_ids": [fact_2],
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "dia_id": "D2:1",
            "source_origin": "original_turn",
            "text": f"{source} later context.",
            "source_fact_ids": [fact_2],
        },
    ]
    facts = [
        {
            "source_dataset": source,
            "fact_id": fact_1,
            "source_type": "original_turn",
            "source_text": f"{source} original fact one.",
        },
        {
            "source_dataset": source,
            "fact_id": fact_2,
            "source_type": "original_turn",
            "source_text": f"{source} original fact two.",
        },
    ]
    qa_audit = [
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "qa_idx": 0,
            "qa_set": "locomo_style_main",
            "question": f"What did {source} say first?",
            "answer": f"{source} original fact one.",
            "category": 1,
            "question_type": "single-hop",
            "difficulty": "easy",
            "whether_cross_session": False,
            "evidence": ["D1:1"],
            "answer_facts": [
                {
                    "fact": f"{source} original fact one.",
                    "supported_by": ["D1:1"],
                    "source_fact_id": fact_1,
                }
            ],
            "evidence_detail": [
                {
                    "dia_id": "D1:1",
                    "source_origin": "original_turn",
                    "supports_answer_fact": [fact_1],
                }
            ],
            "negative_evidence": [],
            "adversarial_reason": None,
        },
        {
            "source_dataset": source,
            "sample_id": sample_id,
            "qa_idx": 1,
            "qa_set": "locomo_style_main",
            "question": f"Did {source} claim an unsupported event?",
            "category": 5,
            "question_type": "adversarial",
            "difficulty": "hard",
            "whether_cross_session": False,
            "evidence": [],
            "answer_facts": [],
            "evidence_detail": [],
            "negative_evidence": ["D2:1"],
            "adversarial_reason": "unsupported_fact",
            "adversarial_answer": "No supported answer is present.",
        },
    ]
    return primary, provenance, facts, qa_audit


def write_fixture(root: Path, mutate: Callable[[dict[str, Any]], None] | None = None) -> None:
    state: dict[str, Any] = {}
    for source, artifact in SOURCE_ARTIFACTS.items():
        primary, provenance, facts, qa_audit = sample_fixture(source, artifact)
        state[artifact] = {
            "primary": primary,
            "provenance": provenance,
            "facts": facts,
            "qa_audit": qa_audit,
        }
    if mutate:
        mutate(state)
    primary_root = root / "primary"
    sidecar_root = root / "sidecars"
    for artifact, artifact_state in state.items():
        write_json(primary_root / f"{artifact}.json", artifact_state["primary"])
        sidecar_dir = sidecar_root / artifact
        write_jsonl(sidecar_dir / f"{artifact}_provenance.jsonl", artifact_state["provenance"])
        write_jsonl(sidecar_dir / f"{artifact}_fact_ledger.jsonl", artifact_state["facts"])
        write_jsonl(sidecar_dir / f"{artifact}_qa_audit.jsonl", artifact_state["qa_audit"])


def run_checker(checker: Path, root: Path) -> tuple[int, dict[str, Any]]:
    output = root / "qa_trace_report.json"
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
        parsed = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    return completed.returncode, parsed


def case_result(
    name: str,
    rc: int,
    report: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = "; ".join(str(item) for item in report.get("errors", []))
    observed_success = rc == 0 and report.get("status") == "passed"
    if expect_success:
        passed = observed_success
    else:
        passed = not observed_success and (
            expected_error_fragment is None or expected_error_fragment in errors
        )
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "expect_success": expect_success,
        "returncode": rc,
        "observed_status": report.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": report.get("errors", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default=Path("scripts/check_locomo_qa_trace_integrity.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/qa_trace_integrity_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []

    test_cases: list[tuple[str, bool, str | None, Callable[[dict[str, Any]], None] | None]] = [
        ("valid_trace_fixture_is_accepted", True, None, None),
        (
            "cat5_missing_negative_evidence_is_rejected",
            False,
            "cat5 missing negative_evidence",
            lambda state: state["PerLTQA-LoCoMo-style-eval"]["qa_audit"][1].update({"negative_evidence": []}),
        ),
        (
            "non_original_answer_fact_is_rejected",
            False,
            "is not original-backed",
            lambda state: state["PerLTQA-LoCoMo-style-eval"]["facts"][0].update(
                {"source_type": "synthetic_continuation_turn"}
            ),
        ),
        (
            "evidence_detail_mismatch_is_rejected",
            False,
            "evidence_detail dia_ids do not match evidence",
            lambda state: state["PerLTQA-LoCoMo-style-eval"]["qa_audit"][0]["evidence_detail"][0].update(
                {"dia_id": "D1:2"}
            ),
        ),
        (
            "provenance_missing_fact_link_is_rejected",
            False,
            "no supported evidence provenance links source_fact_id",
            lambda state: state["PerLTQA-LoCoMo-style-eval"]["provenance"][0].update({"source_fact_ids": []}),
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="locomo_qa_trace_integrity_selftest_") as tmp:
        tempdir = Path(tmp)
        for name, expect_success, expected_error_fragment, mutate in test_cases:
            case_root = tempdir / name
            write_fixture(case_root, mutate=mutate)
            rc, report = run_checker(args.checker, case_root)
            case = case_result(
                name,
                rc,
                report,
                expect_success=expect_success,
                expected_error_fragment=expected_error_fragment,
            )
            if name == "valid_trace_fixture_is_accepted":
                case["per_artifact"] = report.get("per_artifact")
            cases.append(case)

    result = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "checker": str(args.checker),
        "checker_sha256": sha256_file(args.checker),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
