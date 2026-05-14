#!/usr/bin/env python3
"""Self-test dataset-card summary generation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import summarize_locomo_dataset_card as card


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fixture_dataset() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "perltqa_0",
            "source_dataset": "PerLTQA",
            "language": "zh",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "d1",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "memory"}],
                "session_2_date_time": "d2",
                "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "reply"}],
            },
            "qa": [
                {"question": "q1", "answer": "a1", "category": 1, "evidence": ["D1:1"]},
                {"question": "q2", "answer": "a2", "category": 2, "evidence": ["D1:1", "D2:1"]},
                {"question": "q3", "category": 5, "evidence": [], "adversarial_answer": "no evidence"},
            ],
        }
    ]


def write_sidecars(root: Path) -> None:
    artifact = "PerLTQA-LoCoMo-style-eval"
    sidecar = root / artifact
    write_jsonl(
        sidecar / f"{artifact}_provenance.jsonl",
        [
            {
                "source_dataset": "PerLTQA",
                "sample_id": "perltqa_0",
                "dia_id": "D1:1",
                "source_origin": "memory_anchor_turn",
            },
            {
                "source_dataset": "PerLTQA",
                "sample_id": "perltqa_0",
                "dia_id": "D2:1",
                "source_origin": "original_turn",
            },
        ],
    )
    write_jsonl(
        sidecar / f"{artifact}_qa_audit.jsonl",
        [
            {
                "source_dataset": "PerLTQA",
                "sample_id": "perltqa_0",
                "qa_idx": 0,
                "category": 1,
                "whether_cross_session": False,
                "evidence_detail": [{"dia_id": "D1:1", "source_origin": "memory_anchor_turn"}],
            },
            {
                "source_dataset": "PerLTQA",
                "sample_id": "perltqa_0",
                "qa_idx": 1,
                "category": 2,
                "whether_cross_session": True,
                "evidence_detail": [
                    {"dia_id": "D1:1", "source_origin": "memory_anchor_turn"},
                    {"dia_id": "D2:1", "source_origin": "original_turn"},
                ],
            },
            {
                "source_dataset": "PerLTQA",
                "sample_id": "perltqa_0",
                "qa_idx": 2,
                "category": 5,
                "whether_cross_session": False,
                "evidence_detail": [],
            },
        ],
    )
    write_jsonl(
        sidecar / f"{artifact}_fact_ledger.jsonl",
        [
            {"source_dataset": "PerLTQA", "sample_id": "perltqa_0", "source_type": "original_event"},
            {"source_dataset": "PerLTQA", "sample_id": "perltqa_0", "source_type": "original_turn"},
        ],
    )
    for artifact in ("OPELA-LoCoMo-style-eval", "JLongChat-LoCoMo-style-eval", "deL1L2IM-LoCoMo-style-eval"):
        empty_sidecar = root / artifact
        write_jsonl(empty_sidecar / f"{artifact}_provenance.jsonl", [])
        write_jsonl(empty_sidecar / f"{artifact}_qa_audit.jsonl", [])
        write_jsonl(empty_sidecar / f"{artifact}_fact_ledger.jsonl", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/dataset_card_summary_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_dataset_card_selftest_") as tmp:
        tempdir = Path(tmp)
        primary = tempdir / "primary.json"
        sidecar_root = tempdir / "sidecars"
        manifest = tempdir / "manifest.json"
        release = tempdir / "release.json"
        output_md = tempdir / "card.md"
        write_json(primary, fixture_dataset())
        write_sidecars(sidecar_root)
        write_json(
            manifest,
            {
                "status": "bootstrap_harness_artifact_not_final_audited_release",
                "benchmark_claim": "fixture claim",
                "construction_mode": "fixture",
                "model_calls": 0,
            },
        )
        write_json(release, {"status": "blocked", "blocking_failed": ["human_audit_completed"]})
        summary = card.build_summary(primary, sidecar_root, manifest, release)
        card.write_markdown(output_md, summary)
        cases.extend(
            [
                {
                    "name": "overall_counts",
                    "status": "passed"
                    if summary["totals"] == {
                        "samples": 1,
                        "sessions": 2,
                        "turns": 2,
                        "qa": 3,
                        "answerable_qa": 2,
                        "cat5_qa": 1,
                    }
                    else "failed",
                    "observed": summary["totals"],
                },
                {
                    "name": "cross_session_and_evidence_buckets",
                    "status": "passed"
                    if summary["by_cross_session"] == {"false": 2, "true": 1}
                    and summary["by_evidence_provenance"] == {
                        "memory_anchor_turn": 1,
                        "memory_anchor_turn+original_turn": 1,
                        "negative_only": 1,
                    }
                    else "failed",
                    "observed_cross": summary["by_cross_session"],
                    "observed_evidence": summary["by_evidence_provenance"],
                },
                {
                    "name": "markdown_written",
                    "status": "passed" if "LoCoMo-style Eval Dataset Card" in output_md.read_text(encoding="utf-8") else "failed",
                },
                {
                    "name": "input_policy_is_explicit",
                    "status": "passed"
                    if summary["default_model_input_policy"] == {
                        "input_policy": "conversation_only",
                        "summary_visible": False,
                        "input_fields_included": ["conversation"],
                        "input_fields_excluded": ["observation", "session_summary", "event_summary", "sidecars"],
                        "summary_memory_setting": "must be reported separately from the main benchmark table",
                    }
                    and "Main evaluation is conversation-only" in output_md.read_text(encoding="utf-8")
                    and "`observation`, `session_summary`, `event_summary`, and sidecar metadata"
                    in output_md.read_text(encoding="utf-8")
                    else "failed",
                },
                {
                    "name": "perltqa_caveat_is_explicit",
                    "status": "passed"
                    if "PerLTQA" in summary["source_caveats"]
                    and "memory-anchored dialogue-style eval" in summary["source_caveats"]["PerLTQA"]
                    and "not a naturally occurring multi-session dialogue corpus"
                    in output_md.read_text(encoding="utf-8")
                    else "failed",
                },
            ]
        )

    failed = [row for row in cases if row["status"] != "passed"]
    result = {
        "status": "passed" if not failed else "failed",
        "summary_script": str(Path(card.__file__)),
        "summary_script_sha256": card.sha256_file(Path(card.__file__)),
        "selftest": str(Path(__file__)),
        "selftest_sha256": card.sha256_file(Path(__file__)),
        "cases": cases,
        "errors": [row["name"] for row in failed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
