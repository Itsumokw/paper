#!/usr/bin/env python3
"""Summarize LoCoMo-style release-gate blockers into actionable files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKER_ACTIONS: dict[str, dict[str, str]] = {
    "human_audit_completed": {
        "group": "human_audit",
        "action": (
            "Complete every row in human_audit_packet.jsonl, the full CSV, or the batch CSVs; "
            "then validate with the safe full-CSV or batch finalizer dry-run before committing."
        ),
        "command": "\n".join(
            [
                ".venv/bin/python scripts/finalize_locomo_human_audit_batches.py \\",
                "  --base-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl \\",
                "  --input-dir datasets/locomo_style_eval/human_audit_batches \\",
                "  --sidecar-root datasets/locomo_style_eval/sidecars \\",
                "  --summary-json datasets/locomo_style_eval/human_audit_batches_finalize_dry_run.json \\",
                "  --validation-summary datasets/locomo_style_eval/human_audit_batches_finalize_validation.json \\",
                "  --dry-run",
                "",
                ".venv/bin/python scripts/finalize_locomo_human_audit_csv.py \\",
                "  --base-jsonl datasets/locomo_style_eval/human_audit_packet.jsonl \\",
                "  --decisions-csv datasets/locomo_style_eval/human_audit_sheet.csv \\",
                "  --sidecar-root datasets/locomo_style_eval/sidecars \\",
                "  --summary-json datasets/locomo_style_eval/human_audit_csv_finalize_dry_run.json \\",
                "  --import-summary datasets/locomo_style_eval/human_audit_csv_finalize_import.json \\",
                "  --validation-summary datasets/locomo_style_eval/human_audit_csv_finalize_validation.json \\",
                "  --dry-run",
            ]
        ),
    },
    "human_audit_applied": {
        "group": "post_audit",
        "action": "Apply completed audit decisions to create audited primary/source files.",
        "command": ".venv/bin/python scripts/run_locomo_post_audit_pipeline.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/post_audit_pipeline_report.json",
    },
    "audited_primary_validation_passed": {
        "group": "post_audit",
        "action": "Validate the audited merged primary file after audit application.",
        "command": ".venv/bin/python scripts/run_locomo_post_audit_pipeline.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/post_audit_pipeline_report.json",
    },
    "audited_source_files_validation_passed": {
        "group": "post_audit",
        "action": "Validate audited source-specific files and partitioning against the merged audited file.",
        "command": ".venv/bin/python scripts/run_locomo_post_audit_pipeline.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/post_audit_pipeline_report.json",
    },
    "audited_apply_integrity_passed": {
        "group": "post_audit",
        "action": "Check audited output exactly replays completed audit decisions and preserves trace integrity.",
        "command": ".venv/bin/python scripts/run_locomo_post_audit_pipeline.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/post_audit_pipeline_report.json",
    },
    "metric_metadata_created": {
        "group": "post_audit",
        "action": "Build metric metadata from the audited eval file after audit application.",
        "command": ".venv/bin/python scripts/run_locomo_post_audit_pipeline.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/post_audit_pipeline_report.json",
    },
    "recent_session_model_results_exist": {
        "group": "model_results",
        "action": "Run the full conversation/last-session/last-3-session model diagnostic on the audited eval using fixed settings.",
        "command": "\n".join(
            [
                ".venv/bin/python scripts/preflight_locomo_style_experiment.py \\",
                "  --dataset datasets/locomo_style_eval/primary/multilingual_locomo_style_eval_audited.json \\",
                "  --model Qwen/Qwen3-8B \\",
                "  --fail-if-gpu-busy \\",
                "  --fail-if-busy-process \\",
                "  --output datasets/locomo_style_eval/experiment_preflight.json",
                "",
                ".venv/bin/python scripts/run_locomo_recent_session_model_diagnostic.py \\",
                "  --ablation-root datasets/locomo_style_eval/recent_session_ablation \\",
                "  --output datasets/locomo_style_eval/recent_session_ablation/model_results_summary.json \\",
                "  --records-output datasets/locomo_style_eval/recent_session_ablation/model_prediction_records.jsonl \\",
                "  --max-context-chars 24000 \\",
                "  --max-answer-tokens 96 \\",
                "  --request-timeout 90 \\",
                "  --workers 3 \\",
                "  --model Qwen/Qwen3-8B \\",
                "  --settings-file datasets/locomo_style_eval/fixed_eval_settings.json \\",
                "  --enforce-settings",
            ]
        ),
    },
    "fixed_baseline_results_exist": {
        "group": "model_results",
        "action": "Run required fixed baseline methods and build/validate baseline_results/summary.json.",
        "command": "\n".join(
            [
                ".venv/bin/python scripts/build_locomo_baseline_summary.py \\",
                "  --dataset datasets/locomo_style_eval/primary/multilingual_locomo_style_eval_audited.json \\",
                "  --metric-metadata datasets/locomo_style_eval/baseline_results/metric_metadata.jsonl \\",
                "  --settings-file datasets/locomo_style_eval/fixed_eval_settings.json \\",
                "  --output datasets/locomo_style_eval/baseline_results/summary.json \\",
                "  --prediction-jsonl 'Full Context=<path-to-full-context-predictions.jsonl>' \\",
                "  --prediction-jsonl 'A-MEM=<path-to-amem-predictions.jsonl>' \\",
                "  --prediction-jsonl 'Mem0=<path-to-mem0-predictions.jsonl>' \\",
                "  --prediction-jsonl 'SimpleMem=<path-to-simplemem-predictions.jsonl>' \\",
                "  --prediction-jsonl 'HiGMem=<path-to-higmem-predictions.jsonl>'",
                "",
                ".venv/bin/python scripts/validate_locomo_baseline_results.py \\",
                "  datasets/locomo_style_eval/baseline_results/summary.json",
            ]
        ),
    },
}

GROUP_ORDER = ["human_audit", "post_audit", "model_results", "other"]
MAX_EVIDENCE_CHARS = 900


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gate_evidence(report: dict[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for check in report.get("checks", []):
        if isinstance(check, dict) and "name" in check:
            evidence[str(check.get("name"))] = compact_evidence(str(check.get("evidence", "")))
    return evidence


def compact_evidence(text: str) -> str:
    if len(text) <= MAX_EVIDENCE_CHARS:
        return text
    omitted = len(text) - MAX_EVIDENCE_CHARS
    return f"{text[:MAX_EVIDENCE_CHARS].rstrip()}... [truncated {omitted} chars; see release_gate_report.json for full evidence]"


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    blockers = [str(item) for item in report.get("blocking_failed", [])]
    evidence_by_name = gate_evidence(report)
    grouped: dict[str, list[dict[str, str]]] = {group: [] for group in GROUP_ORDER}
    for blocker in blockers:
        meta = BLOCKER_ACTIONS.get(
            blocker,
            {
                "group": "other",
                "action": "Inspect the release-gate evidence and resolve the failed blocking check.",
                "command": ".venv/bin/python scripts/check_locomo_style_release_gates.py --root datasets/locomo_style_eval --output datasets/locomo_style_eval/release_gate_report.json",
            },
        )
        group = meta["group"] if meta["group"] in grouped else "other"
        grouped[group].append(
            {
                "name": blocker,
                "action": meta["action"],
                "command": meta["command"],
                "evidence": evidence_by_name.get(blocker, ""),
            }
        )
    grouped = {group: rows for group, rows in grouped.items() if rows}
    return {
        "status": "passed" if not blockers else "blocked",
        "release_gate_status": report.get("status"),
        "blocking_failed_count": len(blockers),
        "blocking_failed": blockers,
        "groups": grouped,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Release Blockers Summary",
        "",
        f"Status: `{summary['status']}`",
        f"Blocking failed count: `{summary['blocking_failed_count']}`",
        "",
    ]
    if not summary["blocking_failed"]:
        lines.append("No blocking release gates remain.")
    else:
        for group in GROUP_ORDER:
            rows = summary.get("groups", {}).get(group, [])
            if not rows:
                continue
            title = {
                "human_audit": "Human Audit",
                "post_audit": "Post-Audit Pipeline",
                "model_results": "Model Results",
                "other": "Other",
            }[group]
            lines.extend([f"## {title}", ""])
            for row in rows:
                lines.extend(
                    [
                        f"### `{row['name']}`",
                        "",
                        row["action"],
                        "",
                        "Evidence:",
                        "",
                        f"```text\n{row['evidence']}\n```",
                        "",
                        "Suggested command:",
                        "",
                        f"```bash\n{row['command']}\n```",
                        "",
                    ]
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-gate-report",
        type=Path,
        default=Path("datasets/locomo_style_eval/release_gate_report.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/release_blockers_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("datasets/locomo_style_eval/release_blockers_summary.md"),
    )
    args = parser.parse_args()

    report = load_json(args.release_gate_report)
    summary = summarize(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
