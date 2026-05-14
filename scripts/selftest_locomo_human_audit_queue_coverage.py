#!/usr/bin/env python3
"""Self-test human-audit queue coverage validation rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["source_dataset"]), str(row["sample_id"]), int(row["qa_idx"]))


def qa_row(
    source: str,
    sample_id: str,
    qa_idx: int,
    category: int,
    origin: str = "original_turn",
) -> dict[str, Any]:
    return {
        "source_dataset": source,
        "sample_id": sample_id,
        "qa_idx": qa_idx,
        "category": category,
        "question_type": "fixture",
        "difficulty": "easy",
        "whether_cross_session": False,
        "question": f"{sample_id} q{qa_idx}",
        "answer": "fixture",
        "evidence": ["D1:1"],
        "negative_evidence": [],
        "answer_facts": [
            {"fact": "fixture", "supported_by": ["D1:1"], "source_fact_id": f"{sample_id}_{qa_idx}"}
        ],
        "evidence_detail": [
            {
                "dia_id": "D1:1",
                "source_origin": origin,
                "supports_answer_fact": [f"{sample_id}_{qa_idx}"],
            }
        ],
    }


def build_fixture(tempdir: Path) -> dict[str, Any]:
    primary_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, int]] = set()

    for source in SOURCE_TO_ARTIFACT:
        prefix = source.lower().replace("-", "")
        for index in (0, 1):
            sample_id = f"{prefix}_{index}"
            primary_rows.append({"source_dataset": source, "sample_id": sample_id})
            row = qa_row(source, sample_id, 0, 1)
            qa_rows.append(row)
            selected_keys.add(audit_key(row))

    primary_rows.extend(
        [
            {"source_dataset": "OPELA", "sample_id": "opela_2"},
            {"source_dataset": "OPELA", "sample_id": "opela_3"},
            {"source_dataset": "JLongChat", "sample_id": "jlongchat_2"},
            {"source_dataset": "deL1L2IM", "sample_id": "del1l2im_2"},
            {"source_dataset": "PerLTQA", "sample_id": "perltqa_2"},
        ]
    )

    category_2_rows = [qa_row("OPELA", "opela_2", idx, 2) for idx in range(4)]
    category_4_rows = [qa_row("JLongChat", "jlongchat_2", idx, 4) for idx in range(4)]
    category_5_rows = [qa_row("deL1L2IM", "del1l2im_2", idx, 5) for idx in range(4)]
    synthetic_rows = [qa_row("OPELA", "opela_3", 0, 1, "synthetic_bridge_turn")]
    memory_rows = [qa_row("PerLTQA", "perltqa_2", idx, 1, "memory_anchor_turn") for idx in range(4)]
    qa_rows.extend(category_2_rows + category_4_rows + category_5_rows + synthetic_rows + memory_rows)

    for row in category_2_rows[:2] + category_4_rows[:2] + category_5_rows[:2] + synthetic_rows + memory_rows[:2]:
        selected_keys.add(audit_key(row))

    primary_json = tempdir / "primary.json"
    sidecar_root = tempdir / "sidecars"
    queue_jsonl = tempdir / "queue.jsonl"
    write_json(primary_json, primary_rows)
    rows_by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCE_TO_ARTIFACT}
    for row in qa_rows:
        rows_by_source[str(row["source_dataset"])].append(row)
    for source, artifact in SOURCE_TO_ARTIFACT.items():
        write_jsonl(sidecar_root / artifact / f"{artifact}_qa_audit.jsonl", rows_by_source[source])
    selected_rows = [row for row in qa_rows if audit_key(row) in selected_keys]
    write_jsonl(queue_jsonl, selected_rows)
    return {
        "primary_json": primary_json,
        "sidecar_root": sidecar_root,
        "queue_jsonl": queue_jsonl,
        "qa_rows": qa_rows,
        "selected_keys": selected_keys,
    }


def write_queue(path: Path, qa_rows: list[dict[str, Any]], selected_keys: set[tuple[str, str, int]]) -> None:
    write_jsonl(path, [row for row in qa_rows if audit_key(row) in selected_keys])


def run_validator(validator: Path, fixture: dict[str, Any], summary_path: Path) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        str(validator),
        "--primary-json",
        str(fixture["primary_json"]),
        "--sidecar-root",
        str(fixture["sidecar_root"]),
        "--queue-jsonl",
        str(fixture["queue_jsonl"]),
        "--output-summary",
        str(summary_path),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    return completed.returncode, parsed


def case_result(
    name: str,
    returncode: int,
    result: dict[str, Any],
    *,
    expect_success: bool,
    expected_error_fragment: str | None = None,
) -> dict[str, Any]:
    errors = "; ".join(str(error) for error in result.get("errors", []))
    observed_success = returncode == 0 and result.get("status") == "passed"
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
        "returncode": returncode,
        "validator_status": result.get("status"),
        "expected_error_fragment": expected_error_fragment,
        "errors": result.get("errors", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", type=Path, default=Path("scripts/validate_locomo_human_audit_queue.py"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_queue_coverage_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_queue_selftest_") as tmp:
        tempdir = Path(tmp)
        fixture = build_fixture(tempdir)
        summary_path = tempdir / "coverage.json"

        rc, result = run_validator(args.validator, fixture, summary_path)
        cases.append(case_result("valid_audit_queue_coverage_is_accepted", rc, result, expect_success=True))

        scenarios = [
            (
                "missing_full_sample_coverage_is_rejected",
                ("PerLTQA", "perltqa_0", 0),
                "PerLTQA: full-sample audit coverage=1 expected>=2",
            ),
            (
                "missing_category_2_coverage_is_rejected",
                ("OPELA", "opela_2", 1),
                "category 2: selected=1 expected>=2",
            ),
            (
                "missing_category_4_coverage_is_rejected",
                ("JLongChat", "jlongchat_2", 1),
                "category 4: selected=1 expected>=2",
            ),
            (
                "missing_category_5_coverage_is_rejected",
                ("deL1L2IM", "del1l2im_2", 1),
                "category 5: selected=1 expected>=2",
            ),
            (
                "missing_synthetic_adjacent_coverage_is_rejected",
                ("OPELA", "opela_3", 0),
                "synthetic-adjacent QA missing from audit queue",
            ),
            (
                "missing_perltqa_memory_anchor_coverage_is_rejected",
                ("PerLTQA", "perltqa_2", 1),
                "PerLTQA memory-anchor QA",
            ),
        ]
        for name, key, fragment in scenarios:
            selected = set(fixture["selected_keys"])
            selected.remove(key)
            write_queue(fixture["queue_jsonl"], fixture["qa_rows"], selected)
            rc, result = run_validator(args.validator, fixture, summary_path)
            cases.append(case_result(name, rc, result, expect_success=False, expected_error_fragment=fragment))

    report = {
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
        "validator": str(args.validator),
        "validator_sha256": sha256_file(args.validator),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
