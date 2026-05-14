#!/usr/bin/env python3
"""Validate reviewer-facing human-audit handoff docs use the safe finalizer path."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_DOCS = [
    "NEXT_STEPS.md",
    "RUNBOOK.md",
    "HUMAN_AUDIT_HANDOFF.md",
    "HUMAN_AUDIT_QUICKSTART_ZH.md",
    "HUMAN_AUDIT_WORKPLAN.md",
    "HUMAN_AUDIT_ASSIGNMENTS.md",
    "human_audit_review_guide.md",
    "human_audit_batches/README.md",
]

REQUIRED_REFERENCES = [
    "scripts/finalize_locomo_human_audit_batches.py",
    "datasets/locomo_style_eval/human_audit_batches",
    "datasets/locomo_style_eval/human_audit_packet.jsonl",
]
PROGRESS_REFRESH_SCRIPT = "scripts/summarize_locomo_human_audit_batches.py"
REQUIRED_REVIEW_REFRESH_REFERENCES = [
    "--output-review-md-dir",
    "datasets/locomo_style_eval/human_audit_batch_reviews",
    "--sidecar-root",
    "datasets/locomo_style_eval/sidecars",
]
ASSIGNMENT_DOC = "HUMAN_AUDIT_ASSIGNMENTS.md"
NEXT_STEPS_DOC = "NEXT_STEPS.md"
RUNBOOK_DOC = "RUNBOOK.md"
FULL_CSV_FINALIZER_DOCS = {
    NEXT_STEPS_DOC,
    "HUMAN_AUDIT_HANDOFF.md",
    "HUMAN_AUDIT_QUICKSTART_ZH.md",
    "HUMAN_AUDIT_WORKPLAN.md",
    RUNBOOK_DOC,
}
BASELINE_RESULTS_DOC = "baseline_results/README.md"
PRIMARY_README_DOC = "primary/README.md"
RECENT_SESSION_README_DOC = "recent_session_ablation/README.md"
REQUIRED_NEXT_STEPS_REFERENCES = [
    "If Human Audit Is Skipped",
    "scripts/check_locomo_skipped_audit_stop_point.py",
    "skipped_audit_stop_point_report.json",
    "skipped_audit_stop_point_report.md",
    "human_audit_batches_finalize_dry_run.json",
    "human_audit_csv_finalize_dry_run.json",
    "primary/multilingual_locomo_style_eval.json",
    "primary/multilingual_locomo_style_eval_audited.json",
    "primary/audited_sources/",
    "baseline_results/metric_metadata.jsonl",
    "baseline_results/summary.json",
    "baseline_results/predictions/",
    "baseline_results/normalized/",
    "baseline_results/normalization_summaries/",
    "recent_session_ablation/model_results_summary.json",
    "recent_session_ablation/model_prediction_records.jsonl",
]
REQUIRED_RUNBOOK_SKIPPED_REFERENCES = [
    "Skipped Human-audit Stop Point",
    "scripts/check_locomo_skipped_audit_stop_point.py",
    "skipped_audit_stop_point_report.json",
    "skipped_audit_stop_point_report.md",
    "human_audit_batches_finalize_dry_run.json",
    "human_audit_csv_finalize_dry_run.json",
]
REQUIRED_FULL_CSV_FINALIZER_REFERENCES = [
    "scripts/finalize_locomo_human_audit_csv.py",
    "datasets/locomo_style_eval/human_audit_sheet.csv",
]
REQUIRED_BASELINE_RESULTS_REFERENCES = [
    "primary/multilingual_locomo_style_eval_audited.json",
    "primary/multilingual_locomo_style_eval.json",
    "baseline_results/metric_metadata.jsonl",
    "baseline_results/summary.json",
    "predictions/",
    "normalized/",
    "normalization_summaries/",
    "MemGAS",
    "clean audited metrics",
]
REQUIRED_PRIMARY_README_REFERENCES = [
    "bootstrap primary JSON files",
    "not a final",
    "multilingual_locomo_style_eval.json",
    "multilingual_locomo_style_eval_audited.json",
    "conversation-only",
    "observation",
    "session_summary",
    "event_summary",
    "sidecar metadata",
]
REQUIRED_RECENT_SESSION_README_REFERENCES = [
    "bootstrap diagnostic inputs",
    "not final model-result evidence",
    "primary/multilingual_locomo_style_eval.json",
    "primary/multilingual_locomo_style_eval_audited.json",
    "fixed_eval_settings.json",
    "model_results_summary.json",
    "model_prediction_records.jsonl",
]
REVIEWER_TODO_EXPORT_SCRIPT = "scripts/export_locomo_human_audit_reviewer_todos.py"
REQUIRED_REVIEWER_TODO_EXPORT_REFERENCES = [
    "--audit-packet-jsonl",
    "datasets/locomo_style_eval/human_audit_packet.jsonl",
]
REQUIRED_ASSIGNMENT_REFERENCES = [
    "datasets/locomo_style_eval/human_audit_assignment_risk_summary.json",
    "datasets/locomo_style_eval/human_audit_assignment_risk_summary.md",
    "scripts/summarize_locomo_human_audit_assignments.py",
    REVIEWER_TODO_EXPORT_SCRIPT,
    "scripts/check_locomo_human_audit_reviewer_todos.py",
    "datasets/locomo_style_eval/human_audit_reviewer_todos",
    "next_todo_rows",
]

FORBIDDEN_REAL_PACKET_MERGE_RE = re.compile(
    r"scripts/merge_locomo_human_audit_batches\.py[\s\S]{0,800}"
    r"--output-jsonl\s+datasets/locomo_style_eval/human_audit_packet\.jsonl",
    re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def markdown_links_existing_errors(root: Path, doc_path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for match in re.finditer(r"`(datasets/locomo_style_eval/[^`]+|scripts/[^`]+)`", text):
        raw = match.group(1)
        if "*" in raw or "<" in raw or ">" in raw:
            continue
        candidate = root.parent.parent / raw if raw.startswith("datasets/") else root.parent.parent / raw
        if not candidate.exists():
            errors.append(f"{doc_path.name}: referenced path missing: {raw}")
    return errors


def validate_docs(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    doc_records: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        path = root / rel
        if not path.exists():
            errors.append(f"missing required doc: {path}")
            continue
        doc_records.append(file_record(path))
        text = path.read_text(encoding="utf-8")
        for reference in REQUIRED_REFERENCES:
            if reference not in text:
                errors.append(f"{path.name}: missing required reference {reference!r}")
        if FORBIDDEN_REAL_PACKET_MERGE_RE.search(text):
            errors.append(f"{path.name}: lower-level merge command writes directly to real human_audit_packet.jsonl")
        if PROGRESS_REFRESH_SCRIPT in text:
            for reference in REQUIRED_REVIEW_REFRESH_REFERENCES:
                if reference not in text:
                    errors.append(
                        f"{path.name}: progress refresh command missing required reference {reference!r}"
                    )
        if REVIEWER_TODO_EXPORT_SCRIPT in text:
            for reference in REQUIRED_REVIEWER_TODO_EXPORT_REFERENCES:
                if reference not in text:
                    errors.append(
                        f"{path.name}: reviewer todo export command missing required reference {reference!r}"
                    )
        if rel == ASSIGNMENT_DOC:
            for reference in REQUIRED_ASSIGNMENT_REFERENCES:
                if reference not in text:
                    errors.append(f"{path.name}: assignment doc missing required reference {reference!r}")
        if rel == NEXT_STEPS_DOC:
            for reference in REQUIRED_NEXT_STEPS_REFERENCES:
                if reference not in text:
                    errors.append(f"{path.name}: next-steps doc missing required reference {reference!r}")
        if rel == RUNBOOK_DOC:
            for reference in REQUIRED_RUNBOOK_SKIPPED_REFERENCES:
                if reference not in text:
                    errors.append(f"{path.name}: runbook missing skipped-audit reference {reference!r}")
        if rel in FULL_CSV_FINALIZER_DOCS:
            for reference in REQUIRED_FULL_CSV_FINALIZER_REFERENCES:
                if reference not in text:
                    errors.append(f"{path.name}: full-CSV finalizer doc missing reference {reference!r}")
        errors.extend(markdown_links_existing_errors(root, path, text))

    boundary_docs = {
        "baseline_results_doc": (BASELINE_RESULTS_DOC, REQUIRED_BASELINE_RESULTS_REFERENCES),
        "primary_readme_doc": (PRIMARY_README_DOC, REQUIRED_PRIMARY_README_REFERENCES),
        "recent_session_readme_doc": (RECENT_SESSION_README_DOC, REQUIRED_RECENT_SESSION_README_REFERENCES),
    }
    boundary_doc_records: dict[str, dict[str, str] | None] = {}
    for report_key, (rel, required_references) in boundary_docs.items():
        path = root / rel
        record: dict[str, str] | None = None
        if not path.exists():
            errors.append(f"missing required doc: {path}")
        else:
            record = file_record(path)
            text = path.read_text(encoding="utf-8")
            for reference in required_references:
                if reference not in text:
                    errors.append(f"{path.name}: boundary doc missing required reference {reference!r}")
        boundary_doc_records[report_key] = record

    batch_dir = root / "human_audit_batches"
    review_dir = root / "human_audit_batch_reviews"
    batch_csvs = sorted(batch_dir.glob("batch_*.csv"))
    review_mds = sorted(review_dir.glob("batch_*.md"))
    review_doc_records: list[dict[str, Any]] = []
    if len(batch_csvs) != 10:
        errors.append(f"expected 10 batch CSVs, found {len(batch_csvs)}")
    if len(review_mds) != len(batch_csvs):
        errors.append(f"review markdown count={len(review_mds)} does not match batch CSV count={len(batch_csvs)}")
    missing_reviews = [
        str(review_dir / f"{path.stem}.md")
        for path in batch_csvs
        if not (review_dir / f"{path.stem}.md").exists()
    ]
    if missing_reviews:
        errors.append(f"missing batch review markdown files: {missing_reviews[:10]}")
    review_mds_with_fact_ledger_support = 0
    for path in review_mds:
        text = path.read_text(encoding="utf-8")
        has_fact_ledger_support = "Source fact ledger support:" in text
        if has_fact_ledger_support:
            review_mds_with_fact_ledger_support += 1
        record: dict[str, Any] = file_record(path)
        record["has_fact_ledger_support"] = has_fact_ledger_support
        review_doc_records.append(record)
    if review_mds and review_mds_with_fact_ledger_support != len(review_mds):
        errors.append(
            "review markdown fact-ledger support count="
            f"{review_mds_with_fact_ledger_support} expected={len(review_mds)}"
        )

    return {
        "status": "passed" if not errors else "failed",
        "root": str(root),
        "required_docs": doc_records,
        **boundary_doc_records,
        "batch_csv_count": len(batch_csvs),
        "batch_review_md_count": len(review_mds),
        "batch_review_md_with_fact_ledger_support": review_mds_with_fact_ledger_support,
        "batch_review_docs": review_doc_records,
        "required_references": REQUIRED_REFERENCES,
        "progress_refresh_script": PROGRESS_REFRESH_SCRIPT,
        "required_review_refresh_references": REQUIRED_REVIEW_REFRESH_REFERENCES,
        "reviewer_todo_export_script": REVIEWER_TODO_EXPORT_SCRIPT,
        "required_reviewer_todo_export_references": REQUIRED_REVIEWER_TODO_EXPORT_REFERENCES,
        "required_next_steps_references": REQUIRED_NEXT_STEPS_REFERENCES,
        "required_runbook_skipped_references": REQUIRED_RUNBOOK_SKIPPED_REFERENCES,
        "full_csv_finalizer_docs": sorted(FULL_CSV_FINALIZER_DOCS),
        "required_full_csv_finalizer_references": REQUIRED_FULL_CSV_FINALIZER_REFERENCES,
        "required_baseline_results_references": REQUIRED_BASELINE_RESULTS_REFERENCES,
        "required_primary_readme_references": REQUIRED_PRIMARY_README_REFERENCES,
        "required_recent_session_readme_references": REQUIRED_RECENT_SESSION_README_REFERENCES,
        "required_assignment_references": REQUIRED_ASSIGNMENT_REFERENCES,
        "forbidden_real_packet_merge_pattern": FORBIDDEN_REAL_PACKET_MERGE_RE.pattern,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/locomo_style_eval"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/human_audit_handoff_docs_report.json"),
    )
    args = parser.parse_args()

    report = validate_docs(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
