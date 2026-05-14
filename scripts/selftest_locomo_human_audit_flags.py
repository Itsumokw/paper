#!/usr/bin/env python3
"""Self-test reviewer-facing human-audit flag summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from export_locomo_human_audit_csv import FIELDNAMES


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


def write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = {name: "" for name in FIELDNAMES}
            for name in FIELDNAMES:
                value = row.get(name, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                csv_row[name] = value
            writer.writerow(csv_row)


def fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_dataset": "JLongChat",
            "sample_id": "fixture_sample_0",
            "qa_idx": 0,
            "category": 2,
            "question": "What concrete information did A mention in session 1?",
            "answer": "The answer text may be shortened...",
            "evidence": ["D1:1", "D2:1"],
            "negative_evidence": [],
            "whether_cross_session": True,
            "answer_facts": [{"fact": "x...", "supported_by": ["D1:1"], "source_fact_id": "f0"}],
            "evidence_detail": [{"dia_id": "D1:1", "source_origin": "original_turn"}],
            "human_decision": "todo",
        },
        {
            "source_dataset": "PerLTQA",
            "sample_id": "fixture_sample_1",
            "qa_idx": 0,
            "category": 4,
            "question": "What general conversational act is shown by the cited turn?",
            "evidence": ["D3:2"],
            "negative_evidence": [],
            "whether_cross_session": False,
            "answer_facts": [{"fact": "y", "supported_by": ["D3:2"], "source_fact_id": "f1"}],
            "evidence_detail": [{"dia_id": "D3:2", "source_origin": "memory_anchor_turn"}],
            "human_decision": "todo",
        },
        {
            "source_dataset": "PerLTQA",
            "sample_id": "fixture_sample_1",
            "qa_idx": 1,
            "category": 5,
            "question": "What general conversational act is shown by the cited turn?",
            "evidence": [],
            "negative_evidence": ["D4:1"],
            "whether_cross_session": False,
            "answer_facts": [],
            "evidence_detail": [],
            "human_decision": "todo",
        },
    ]


def run_json(command: list[str], tempdir: Path) -> tuple[int, dict[str, Any]]:
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


def case(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", **(details or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_flags_selftest.json"),
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve().with_name("summarize_locomo_human_audit_flags.py")
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_flags_selftest_") as tmp:
        tempdir = Path(tmp)
        rows = fixture_rows()
        packet = tempdir / "packet.jsonl"
        batch_dir = tempdir / "batches"
        output_json = tempdir / "flags.json"
        output_md = tempdir / "flags.md"
        write_jsonl(packet, rows)
        write_batch_csv(batch_dir / "batch_001_flags_fixture.csv", rows)

        rc, result = run_json(
            [
                sys.executable,
                str(script_path),
                "--input-jsonl",
                str(packet),
                "--batch-dir",
                str(batch_dir),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
                "--top-n",
                "10",
            ],
            tempdir,
        )
        md = output_md.read_text(encoding="utf-8") if output_md.exists() else ""
        batch_stats = result.get("flagged_by_batch", {}).get("batch_001_flags_fixture.csv", {})
        top_rows = result.get("top_flagged_rows", [])
        cases.extend(
            [
                case(
                    "flags_summary_counts_expected_rows",
                    rc == 0
                    and result.get("status") == "completed"
                    and result.get("rows") == 3
                    and result.get("flagged_rows") == 3
                    and result.get("unflagged_rows") == 0,
                    {"errors": result.get("errors", [])},
                ),
                case(
                    "flags_summary_detects_expected_flags",
                    result.get("flag_counts", {}).get("cat2_multi_hop") == 1
                    and result.get("flag_counts", {}).get("cat4_reasoning") == 1
                    and result.get("flag_counts", {}).get("cat5_adversarial") == 1
                    and result.get("flag_counts", {}).get("memory_anchor_evidence") == 1
                    and result.get("flag_counts", {}).get("template_like_question") == 3
                    and result.get("flag_counts", {}).get("duplicate_question_text") == 2
                    and result.get("flag_counts", {}).get("literal_ascii_ellipsis_in_answer_or_fact") == 1,
                    {"flag_counts": result.get("flag_counts", {})},
                ),
                case(
                    "flags_summary_reports_batch_level_counts",
                    batch_stats.get("flagged_rows") == 3
                    and batch_stats.get("flag_counts", {}).get("template_like_question") == 3
                    and batch_stats.get("flag_counts", {}).get("memory_anchor_evidence") == 1,
                    {"flagged_by_batch": result.get("flagged_by_batch", {})},
                ),
                case(
                    "flags_summary_top_rows_link_review_material",
                    bool(top_rows)
                    and top_rows[0].get("csv", "").endswith("batch_001_flags_fixture.csv")
                    and top_rows[0].get("review_md", "").endswith("batch_001_flags_fixture.md"),
                    {"top_flagged_rows": top_rows[:1]},
                ),
                case(
                    "flags_markdown_contains_batch_table",
                    "## By Batch" in md
                    and "batch_001_flags_fixture.csv" in md
                    and "template_like_question=3" in md,
                    {},
                ),
            ]
        )

    report = {
        "status": "passed" if all(item["status"] == "passed" for item in cases) else "failed",
        "summary_script_sha256": sha256_file(script_path),
        "selftest_sha256": sha256_file(Path(__file__).resolve()),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
