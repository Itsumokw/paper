#!/usr/bin/env python3
"""Self-test PlanMode provenance and synthetic-ratio checks."""

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


SOURCE_TO_ARTIFACT = {
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def provenance_row(source: str, sample_id: str, dia_id: str, origin: str) -> dict[str, Any]:
    return {
        "source_dataset": source,
        "sample_id": sample_id,
        "dia_id": dia_id,
        "session_id": dia_id.split(":", 1)[0].replace("D", "session_"),
        "turn_index": int(dia_id.split(":", 1)[1]),
        "source_origin": origin,
    }


def qa_row(
    source: str,
    sample_id: str,
    qa_idx: int,
    category: int,
    origins: list[str],
) -> dict[str, Any]:
    evidence = [f"D{idx + 1}:1" for idx, _ in enumerate(origins)]
    return {
        "source_dataset": source,
        "sample_id": sample_id,
        "qa_idx": qa_idx,
        "qa_set": "locomo_style_main",
        "question": f"{source} fixture question {qa_idx}?",
        "answer": f"{source} fixture answer {qa_idx}." if category != 5 else None,
        "category": category,
        "evidence": evidence if category != 5 else [],
        "answer_facts": [] if category == 5 else [{"source_fact_id": f"{sample_id}_fact_{qa_idx}"}],
        "evidence_detail": [
            {
                "dia_id": dia_id,
                "source_origin": origin,
                "supports_answer_fact": [f"{sample_id}_fact_{qa_idx}"],
            }
            for dia_id, origin in zip(evidence, origins)
        ]
        if category != 5
        else [],
        "negative_evidence": [] if category != 5 else ["D1:1"],
        "adversarial_reason": None if category != 5 else "unsupported_fact",
    }


def fixture_state() -> dict[str, dict[str, list[dict[str, Any]]]]:
    state: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source, artifact in SOURCE_TO_ARTIFACT.items():
        sample_id = f"{source.lower()}_fixture_0"
        if source == "PerLTQA":
            provenance = [
                provenance_row(source, sample_id, "D1:1", "memory_anchor_turn"),
                provenance_row(source, sample_id, "D2:1", "original_turn"),
            ]
            qa_audit = [qa_row(source, sample_id, 0, 1, ["memory_anchor_turn"])]
        else:
            provenance = [
                provenance_row(source, sample_id, "D1:1", "original_turn"),
                provenance_row(source, sample_id, "D2:1", "original_turn"),
            ]
            qa_audit = [qa_row(source, sample_id, 0, 1, ["original_turn"])]
        state[artifact] = {"provenance": provenance, "qa_audit": qa_audit}
    return state


def write_fixture(
    sidecar_root: Path,
    mutate: Callable[[dict[str, dict[str, list[dict[str, Any]]]]], None] | None = None,
) -> None:
    state = fixture_state()
    if mutate is not None:
        mutate(state)
    for artifact, rows in state.items():
        sidecar_dir = sidecar_root / artifact
        write_jsonl(sidecar_dir / f"{artifact}_provenance.jsonl", rows["provenance"])
        write_jsonl(sidecar_dir / f"{artifact}_qa_audit.jsonl", rows["qa_audit"])


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
        default=Path("scripts/check_locomo_planmode_provenance.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/planmode_provenance_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_planmode_provenance_selftest_") as tmp:
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

        run_case("valid_planmode_fixture_is_accepted", None, True)

        def exceed_synthetic_ratio(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            artifact = "deL1L2IM-LoCoMo-style-eval"
            state[artifact]["provenance"].append(
                provenance_row("deL1L2IM", "del1l2im_fixture_0", "D3:1", "synthetic_bridge_turn")
            )

        run_case(
            "synthetic_ratio_limit_is_enforced",
            exceed_synthetic_ratio,
            False,
            ["deL1L2IM: synthetic_turn_ratio="],
        )

        def add_forbidden_memory_anchor(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            artifact = "OPELA-LoCoMo-style-eval"
            state[artifact]["provenance"].append(
                provenance_row("OPELA", "opela_fixture_0", "D3:1", "memory_anchor_turn")
            )

        run_case(
            "memory_anchor_is_rejected_for_non_perltqa_sources",
            add_forbidden_memory_anchor,
            False,
            ["OPELA: memory_anchor_turn count=1 is not allowed"],
        )

        def add_synthetic_evidence_qa(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            artifact = "JLongChat-LoCoMo-style-eval"
            state[artifact]["qa_audit"][0]["evidence_detail"] = [
                {
                    "dia_id": "D1:1",
                    "source_origin": "synthetic_bridge_turn",
                    "supports_answer_fact": ["jlongchat_fixture_0_fact_0"],
                }
            ]

        run_case(
            "synthetic_evidence_qa_is_rejected_for_current_jlongchat",
            add_synthetic_evidence_qa,
            False,
            ["JLongChat: synthetic evidence QA count=1 should be 0 in current artifact"],
        )

        def remove_perltqa_memory_anchor_evidence(
            state: dict[str, dict[str, list[dict[str, Any]]]]
        ) -> None:
            artifact = "PerLTQA-LoCoMo-style-eval"
            state[artifact]["qa_audit"][0] = qa_row("PerLTQA", "perltqa_fixture_0", 0, 1, ["original_turn"])

        run_case(
            "perltqa_requires_memory_anchor_evidence_report",
            remove_perltqa_memory_anchor_evidence,
            False,
            ["PerLTQA: expected memory_anchor_turn evidence"],
        )

        def del1l2im_non_original_evidence(state: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
            artifact = "deL1L2IM-LoCoMo-style-eval"
            state[artifact]["qa_audit"][0]["evidence_detail"] = [
                {
                    "dia_id": "D1:1",
                    "source_origin": "memory_anchor_turn",
                    "supports_answer_fact": ["del1l2im_fixture_0_fact_0"],
                }
            ]

        run_case(
            "del1l2im_requires_all_answerable_evidence_original",
            del1l2im_non_original_evidence,
            False,
            ["deL1L2IM: expected all answerable QA to use original_turn evidence"],
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
