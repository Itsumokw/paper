#!/usr/bin/env python3
"""Self-test metric metadata generation for fixed baseline reporting."""

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


ARTIFACTS = [
    "PerLTQA-LoCoMo-style-eval",
    "OPELA-LoCoMo-style-eval",
    "JLongChat-LoCoMo-style-eval",
    "deL1L2IM-LoCoMo-style-eval",
]


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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_primary(path: Path) -> None:
    write_json(
        path,
        [
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
                        {"speaker": "A", "dia_id": "D1:1", "text": "original fact"},
                        {"speaker": "B", "dia_id": "D1:2", "text": "negative context"},
                    ],
                    "session_2_date_time": "sid 2",
                    "session_2": [
                        {"speaker": "A", "dia_id": "D2:1", "text": "later memory anchor"}
                    ],
                },
                "observation": {},
                "session_summary": {},
                "event_summary": {},
                "qa": [
                    {
                        "question": "What original fact appears?",
                        "answer": "original fact",
                        "category": 1,
                        "evidence": ["D1:1"],
                    },
                    {
                        "question": "Which facts require two sessions?",
                        "answer": "original fact and later memory anchor",
                        "category": 2,
                        "evidence": ["D1:1", "D2:1"],
                    },
                    {
                        "question": "Is there evidence for a Mars move?",
                        "category": 5,
                        "evidence": [],
                        "adversarial_answer": "unsupported",
                    },
                ],
            }
        ],
    )


def build_sidecars(root: Path, *, include_d2: bool = True) -> None:
    for artifact in ARTIFACTS:
        rows: list[dict[str, Any]] = []
        if artifact == "JLongChat-LoCoMo-style-eval":
            rows.append(
                {
                    "sample_id": "fixture_sample_0",
                    "dia_id": "D1:1",
                    "source_origin": "original_turn",
                }
            )
            rows.append(
                {
                    "sample_id": "fixture_sample_0",
                    "dia_id": "D1:2",
                    "source_origin": "original_turn",
                }
            )
            if include_d2:
                rows.append(
                    {
                        "sample_id": "fixture_sample_0",
                        "dia_id": "D2:1",
                        "source_origin": "memory_anchor_turn",
                    }
                )
        write_jsonl(root / artifact / f"{artifact}_provenance.jsonl", rows)


def run_builder(
    builder: Path,
    primary: Path,
    sidecar_root: Path,
    output_jsonl: Path,
    summary_json: Path,
    tempdir: Path,
) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(builder),
            "--primary-json",
            str(primary),
            "--sidecar-root",
            str(sidecar_root),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    payload = completed.stdout.strip() or completed.stderr.strip() or "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"status": "unparseable_output", "output": payload}
    parsed["returncode"] = completed.returncode
    parsed["_tempdir"] = str(tempdir)
    return completed.returncode, parsed


def normalize_errors(result: dict[str, Any]) -> list[str]:
    tempdir = str(result.get("_tempdir", ""))
    return [str(item).replace(tempdir + "/", "<tmp>/") for item in result.get("errors", [])]


def case(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", **(details or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path("scripts/build_locomo_metric_metadata.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/metric_metadata_builder_selftest.json"),
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="locomo_metric_metadata_builder_selftest_") as tmp:
        tempdir = Path(tmp)
        primary = tempdir / "primary.json"
        sidecar_root = tempdir / "sidecars"
        output_jsonl = tempdir / "metric_metadata.jsonl"
        summary_json = tempdir / "metric_metadata_summary.json"
        build_primary(primary)
        build_sidecars(sidecar_root, include_d2=True)

        rc, result = run_builder(args.builder, primary, sidecar_root, output_jsonl, summary_json, tempdir)
        rows = list(iter_jsonl(output_jsonl)) if output_jsonl.exists() else []
        valid_rows = (
            rc == 0
            and result.get("status") == "passed"
            and result.get("rows") == 3
            and result.get("primary_sha256") == sha256_file(primary)
            and result.get("output_jsonl_sha256") == sha256_file(output_jsonl)
            and result.get("by_category") == {"1": 1, "2": 1, "5": 1}
            and result.get("by_cross_session") == {"false": 2, "true": 1}
            and result.get("by_evidence_provenance")
            == {
                "memory_anchor_turn+original_turn": 1,
                "negative_only": 1,
                "original_turn": 1,
            }
            and rows[0]["answerable"] is True
            and rows[1]["whether_cross_session"] is True
            and rows[1]["evidence_sessions"] == ["session_1", "session_2"]
            and rows[1]["evidence_provenance"] == "memory_anchor_turn+original_turn"
            and rows[2]["answerable"] is False
            and rows[2]["evidence_provenance"] == "negative_only"
        )
        cases.append(case("valid_metric_metadata_fixture", valid_rows, {"errors": normalize_errors(result)}))

        missing_sidecar_root = tempdir / "sidecars_missing_d2"
        missing_output = tempdir / "metric_metadata_missing.jsonl"
        missing_summary = tempdir / "metric_metadata_missing_summary.json"
        build_sidecars(missing_sidecar_root, include_d2=False)
        rc, result = run_builder(
            args.builder,
            primary,
            missing_sidecar_root,
            missing_output,
            missing_summary,
            tempdir,
        )
        missing_errors = normalize_errors(result)
        cases.append(
            case(
                "missing_provenance_is_rejected",
                rc != 0
                and result.get("status") == "failed"
                and any("evidence dia_id='D2:1' missing provenance" in error for error in missing_errors),
                {"errors": missing_errors},
            )
        )

        tampered_primary = tempdir / "primary_tampered.json"
        tampered = deepcopy(json.loads(primary.read_text(encoding="utf-8")))
        tampered[0]["qa"][0]["category"] = 3
        write_json(tampered_primary, tampered)
        tampered_output = tempdir / "metric_metadata_tampered.jsonl"
        tampered_summary = tempdir / "metric_metadata_tampered_summary.json"
        rc, result = run_builder(
            args.builder,
            tampered_primary,
            sidecar_root,
            tampered_output,
            tampered_summary,
            tempdir,
        )
        cases.append(
            case(
                "summary_tracks_primary_hash_and_counts",
                rc == 0
                and result.get("status") == "passed"
                and result.get("primary_sha256") == sha256_file(tampered_primary)
                and result.get("by_category") == {"2": 1, "3": 1, "5": 1},
                {"errors": normalize_errors(result)},
            )
        )

    report = {
        "status": "passed" if all(item["status"] == "passed" for item in cases) else "failed",
        "builder": str(args.builder),
        "builder_sha256": sha256_file(args.builder),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
