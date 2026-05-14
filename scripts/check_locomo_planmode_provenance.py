#!/usr/bin/env python3
"""Check PlanMode provenance and synthetic-ratio constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_RULES = {
    "PerLTQA": {
        "artifact": "PerLTQA-LoCoMo-style-eval",
        "plan_mode": "D_memory_anchored_dialogization",
        "max_synthetic_turn_ratio": 0.0,
        "allow_memory_anchor_turn": True,
    },
    "OPELA": {
        "artifact": "OPELA-LoCoMo-style-eval",
        "plan_mode": "C_persona_grounded_continuation",
        "max_synthetic_turn_ratio": 0.60,
        "allow_memory_anchor_turn": False,
    },
    "JLongChat": {
        "artifact": "JLongChat-LoCoMo-style-eval",
        "plan_mode": "A_or_B_raw_preserving_or_light_completion",
        "max_synthetic_turn_ratio": 0.40,
        "allow_memory_anchor_turn": False,
    },
    "deL1L2IM": {
        "artifact": "deL1L2IM-LoCoMo-style-eval",
        "plan_mode": "A_raw_preserving_native_conversion",
        "max_synthetic_turn_ratio": 0.0,
        "allow_memory_anchor_turn": False,
    },
}

SYNTHETIC_TURN_ORIGINS = {"synthetic_bridge_turn", "synthetic_continuation_turn"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def source_origins_from_qa(row: dict[str, Any]) -> set[str]:
    return {
        str(detail.get("source_origin"))
        for detail in row.get("evidence_detail", [])
        if isinstance(detail, dict) and detail.get("source_origin")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/planmode_provenance_summary.json"))
    args = parser.parse_args()

    errors: list[str] = []
    per_source: dict[str, Any] = {}
    provenance_files: list[Path] = []
    qa_audit_files: list[Path] = []

    for source, rule in SOURCE_RULES.items():
        artifact = rule["artifact"]
        provenance_path = args.sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        qa_audit_path = args.sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        provenance_files.append(provenance_path)
        qa_audit_files.append(qa_audit_path)

        provenance_counts: Counter[str] = Counter()
        for row in iter_jsonl(provenance_path):
            provenance_counts[str(row.get("source_origin"))] += 1

        total_turns = sum(provenance_counts.values())
        synthetic_turns = sum(provenance_counts[origin] for origin in SYNTHETIC_TURN_ORIGINS)
        memory_anchor_turns = provenance_counts["memory_anchor_turn"]
        synthetic_turn_ratio = synthetic_turns / total_turns if total_turns else 0.0
        constructed_non_original_ratio = (
            (total_turns - provenance_counts["original_turn"]) / total_turns if total_turns else 0.0
        )

        max_synthetic = float(rule["max_synthetic_turn_ratio"])
        if synthetic_turn_ratio > max_synthetic:
            errors.append(
                f"{source}: synthetic_turn_ratio={synthetic_turn_ratio:.4f} exceeds max={max_synthetic:.4f}"
            )
        if memory_anchor_turns and not rule["allow_memory_anchor_turn"]:
            errors.append(f"{source}: memory_anchor_turn count={memory_anchor_turns} is not allowed for {rule['plan_mode']}")

        qa_rows = 0
        answerable_qa_rows = 0
        qa_evidence_origin_counts: Counter[str] = Counter()
        synthetic_evidence_qa = 0
        memory_anchor_evidence_qa = 0
        original_evidence_qa = 0
        for row in iter_jsonl(qa_audit_path):
            qa_rows += 1
            if int(row.get("category") or 0) != 5:
                answerable_qa_rows += 1
            origins = source_origins_from_qa(row)
            for origin in origins:
                qa_evidence_origin_counts[origin] += 1
            if origins & SYNTHETIC_TURN_ORIGINS:
                synthetic_evidence_qa += 1
            if "memory_anchor_turn" in origins:
                memory_anchor_evidence_qa += 1
            if "original_turn" in origins:
                original_evidence_qa += 1

        if source == "deL1L2IM" and original_evidence_qa != answerable_qa_rows:
            errors.append(f"{source}: expected all answerable QA to use original_turn evidence")
        if source in {"OPELA", "JLongChat"} and synthetic_evidence_qa:
            errors.append(f"{source}: synthetic evidence QA count={synthetic_evidence_qa} should be 0 in current artifact")
        if source == "PerLTQA" and memory_anchor_evidence_qa == 0:
            errors.append("PerLTQA: expected memory_anchor_turn evidence for memory-anchored dialogization")

        per_source[source] = {
            "artifact": artifact,
            "plan_mode": rule["plan_mode"],
            "turns": total_turns,
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "synthetic_turns": synthetic_turns,
            "synthetic_turn_ratio": round(synthetic_turn_ratio, 6),
            "max_synthetic_turn_ratio": max_synthetic,
            "memory_anchor_turns": memory_anchor_turns,
            "constructed_non_original_ratio": round(constructed_non_original_ratio, 6),
            "qa_rows": qa_rows,
            "answerable_qa_rows": answerable_qa_rows,
            "qa_evidence_origin_counts": dict(sorted(qa_evidence_origin_counts.items())),
            "synthetic_evidence_qa": synthetic_evidence_qa,
            "memory_anchor_evidence_qa": memory_anchor_evidence_qa,
            "original_evidence_qa": original_evidence_qa,
        }

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "provenance_files": [file_record(path) for path in provenance_files],
            "qa_audit_files": [file_record(path) for path in qa_audit_files],
        },
        "sidecar_root": str(args.sidecar_root),
        "per_source": per_source,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
