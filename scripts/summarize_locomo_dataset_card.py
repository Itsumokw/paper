#!/usr/bin/env python3
"""Build a dataset-card style summary for the LoCoMo-style eval artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def session_keys(sample: dict[str, Any]) -> list[str]:
    conv = sample.get("conversation", {})
    return sorted(
        [
            key
            for key, value in conv.items()
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
        ],
        key=lambda item: int(item.split("_", 1)[1]) if item.split("_", 1)[1].isdigit() else 10**9,
    )


def session_from_dia_id(dia_id: str) -> str | None:
    if not dia_id.startswith("D") or ":" not in dia_id:
        return None
    raw = dia_id[1:].split(":", 1)[0]
    return raw if raw.isdigit() else None


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "total": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 2),
        "total": sum(values),
    }


def qa_audit_index(sidecar_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for artifact in SOURCE_ARTIFACTS.values():
        path = sidecar_root / artifact / f"{artifact}_qa_audit.jsonl"
        for row in iter_jsonl(path):
            rows[(str(row.get("sample_id")), int(row.get("qa_idx")))] = row
    return rows


def source_sidecar_counts(sidecar_root: Path, source: str) -> dict[str, Any]:
    artifact = SOURCE_ARTIFACTS[source]
    provenance_path = sidecar_root / artifact / f"{artifact}_provenance.jsonl"
    fact_path = sidecar_root / artifact / f"{artifact}_fact_ledger.jsonl"
    provenance = list(iter_jsonl(provenance_path))
    facts = list(iter_jsonl(fact_path))
    turn_origins = Counter(str(row.get("source_origin")) for row in provenance)
    fact_types = Counter(str(row.get("source_type")) for row in facts)
    return {
        "turn_origin_counts": dict(sorted(turn_origins.items())),
        "synthetic_turn_count": sum(
            count for origin, count in turn_origins.items() if origin.startswith("synthetic_")
        ),
        "synthetic_turn_ratio": pct(
            sum(count for origin, count in turn_origins.items() if origin.startswith("synthetic_")),
            sum(turn_origins.values()),
        ),
        "fact_source_type_counts": dict(sorted(fact_types.items())),
    }


def evidence_bucket(row: dict[str, Any], category: int) -> str:
    if category == 5:
        return "negative_only"
    origins = sorted(
        {
            str(item.get("source_origin"))
            for item in row.get("evidence_detail", [])
            if isinstance(item, dict) and item.get("source_origin")
        }
    )
    return "+".join(origins) if origins else "none"


def build_summary(primary_json: Path, sidecar_root: Path, manifest_path: Path | None, release_gate: Path | None) -> dict[str, Any]:
    dataset = load_json(primary_json)
    audit_rows = qa_audit_index(sidecar_root)
    manifest = load_json(manifest_path) if manifest_path and manifest_path.exists() else {}
    release = load_json(release_gate) if release_gate and release_gate.exists() else {}

    totals = {
        "samples": 0,
        "sessions": 0,
        "turns": 0,
        "qa": 0,
        "answerable_qa": 0,
        "cat5_qa": 0,
    }
    by_source: dict[str, Any] = {}
    by_language: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_cross_session: Counter[str] = Counter()
    by_evidence_provenance: Counter[str] = Counter()

    samples_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in dataset:
        samples_by_source[str(sample.get("source_dataset"))].append(sample)

    for source, samples in sorted(samples_by_source.items()):
        session_counts: list[int] = []
        turn_counts: list[int] = []
        qa_counts: list[int] = []
        source_categories: Counter[str] = Counter()
        source_cross_session: Counter[str] = Counter()
        source_evidence: Counter[str] = Counter()
        source_languages: Counter[str] = Counter()

        for sample in samples:
            keys = session_keys(sample)
            session_counts.append(len(keys))
            turn_counts.append(sum(len(sample.get("conversation", {}).get(key, [])) for key in keys))
            qa_rows = sample.get("qa", [])
            qa_counts.append(len(qa_rows))
            source_languages[str(sample.get("language"))] += 1
            by_language[str(sample.get("language"))] += 1

            for qa_idx, qa in enumerate(qa_rows):
                category = int(qa.get("category"))
                key = (str(sample.get("sample_id")), qa_idx)
                audit = audit_rows.get(key, {})
                if audit.get("whether_cross_session") is None:
                    evidence = qa.get("evidence", [])
                    sessions = {session_from_dia_id(str(item)) for item in evidence}
                    cross_session = len({item for item in sessions if item}) > 1
                else:
                    cross_session = bool(audit.get("whether_cross_session"))
                bucket = evidence_bucket(audit, category)

                source_categories[str(category)] += 1
                source_cross_session[str(cross_session).lower()] += 1
                source_evidence[bucket] += 1
                by_category[str(category)] += 1
                by_cross_session[str(cross_session).lower()] += 1
                by_evidence_provenance[bucket] += 1

        sidecar_counts = (
            source_sidecar_counts(sidecar_root, source)
            if source in SOURCE_ARTIFACTS
            else {"turn_origin_counts": {}, "synthetic_turn_count": 0, "synthetic_turn_ratio": 0, "fact_source_type_counts": {}}
        )
        by_source[source] = {
            "languages": dict(sorted(source_languages.items())),
            "samples": len(samples),
            "sessions": numeric_summary(session_counts),
            "turns": numeric_summary(turn_counts),
            "qa": numeric_summary(qa_counts),
            "categories": dict(sorted(source_categories.items())),
            "cross_session": dict(sorted(source_cross_session.items())),
            "cross_session_ratio": pct(source_cross_session.get("true", 0), sum(source_cross_session.values())),
            "evidence_provenance": dict(sorted(source_evidence.items())),
            **sidecar_counts,
        }
        totals["samples"] += len(samples)
        totals["sessions"] += sum(session_counts)
        totals["turns"] += sum(turn_counts)
        totals["qa"] += sum(qa_counts)
        totals["cat5_qa"] += source_categories.get("5", 0)

    totals["answerable_qa"] = totals["qa"] - totals["cat5_qa"]

    return {
        "status": "bootstrap_not_final" if release.get("status") == "blocked" else str(manifest.get("status") or "unknown"),
        "benchmark_claim": manifest.get("benchmark_claim"),
        "construction_mode": manifest.get("construction_mode"),
        "model_calls": manifest.get("model_calls"),
        "default_model_input_policy": {
            "input_policy": "conversation_only",
            "summary_visible": False,
            "input_fields_included": ["conversation"],
            "input_fields_excluded": ["observation", "session_summary", "event_summary", "sidecars"],
            "summary_memory_setting": "must be reported separately from the main benchmark table",
        },
        "source_caveats": {
            "PerLTQA": (
                "PerLTQA is a memory-anchored dialogue-style eval, not a naturally occurring "
                "multi-session dialogue corpus. Interpret its memory_anchor_turn evidence and "
                "PerLTQA-specific ratios separately from native-dialogue sources."
            )
        },
        "primary_json": str(primary_json),
        "primary_sha256": sha256_file(primary_json),
        "sidecar_root": str(sidecar_root),
        "manifest": str(manifest_path) if manifest_path else None,
        "release_gate_report": str(release_gate) if release_gate else None,
        "release_gate_status": release.get("status"),
        "blocking_failed": release.get("blocking_failed", []),
        "totals": totals,
        "by_source_dataset": by_source,
        "by_language": dict(sorted(by_language.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_cross_session": dict(sorted(by_cross_session.items())),
        "cross_session_ratio": pct(by_cross_session.get("true", 0), sum(by_cross_session.values())),
        "by_evidence_provenance": dict(sorted(by_evidence_provenance.items())),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    lines = [
        "# LoCoMo-style Eval Dataset Card",
        "",
        f"Status: `{summary['status']}`",
        f"Claim: {summary.get('benchmark_claim')}",
        f"Construction mode: `{summary.get('construction_mode')}`; model calls: `{summary.get('model_calls')}`",
        "",
        "This is a bootstrap harness artifact, not a final audited release, while release gates remain blocked.",
        "",
        "## Default Model Input Policy",
        "",
        "Main evaluation is conversation-only: models receive chronological `conversation` turns.",
        "",
        "The default context excludes `observation`, `session_summary`, `event_summary`, and sidecar metadata. "
        "Any summary-visible or summary-memory method must be reported as a separate setting, not mixed into the main table.",
        "",
        "## Source Caveats",
        "",
        (
            "PerLTQA is a memory-anchored dialogue-style eval, not a naturally occurring multi-session "
            "dialogue corpus. Interpret its `memory_anchor_turn` evidence and PerLTQA-specific ratios "
            "separately from native-dialogue sources."
        ),
        "",
        "## Overall",
        "",
        "| Samples | Sessions | Turns | QA | Answerable QA | Cat5 QA | Cross-session ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {totals['samples']} | {totals['sessions']} | {totals['turns']} | {totals['qa']} | "
            f"{totals['answerable_qa']} | {totals['cat5_qa']} | {summary['cross_session_ratio']:.3f} |"
        ),
        "",
        "## By Source",
        "",
        "| Source | Lang | Samples | Sessions | Turns | QA | Cross-session QA | Evidence provenance | Synthetic turn ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for source, row in summary["by_source_dataset"].items():
        lang = ", ".join(f"{key}:{value}" for key, value in row["languages"].items())
        ev = ", ".join(f"{key}:{value}" for key, value in row["evidence_provenance"].items())
        lines.append(
            f"| {source} | {lang} | {row['samples']} | {row['sessions']['total']} | "
            f"{row['turns']['total']} | {row['qa']['total']} | "
            f"{row['cross_session'].get('true', 0)} | {ev} | {row['synthetic_turn_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## QA Categories",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
    )
    for category, count in summary["by_category"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "## Release Blockers",
            "",
        ]
    )
    blockers = summary.get("blocking_failed", [])
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("No blocking release gates recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary/multilingual_locomo_style_eval.json"),
    )
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/locomo_style_eval/manifest.json"))
    parser.add_argument("--release-gate-report", type=Path, default=Path("datasets/locomo_style_eval/release_gate_report.json"))
    parser.add_argument("--output-json", type=Path, default=Path("datasets/locomo_style_eval/dataset_card_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("datasets/locomo_style_eval/DATASET_CARD.md"))
    args = parser.parse_args()

    summary = build_summary(args.primary_json, args.sidecar_root, args.manifest, args.release_gate_report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
