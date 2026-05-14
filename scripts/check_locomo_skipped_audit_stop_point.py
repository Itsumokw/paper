#!/usr/bin/env python3
"""Check that a skipped-human-audit run is safely stopped at bootstrap state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INPUT_FILES = [
    "primary/multilingual_locomo_style_eval.json",
    "manifest.json",
    "release_gate_report.json",
    "human_audit_results_summary.json",
    "human_audit_batches_finalize_dry_run.json",
    "human_audit_csv_finalize_dry_run.json",
    "post_audit_pipeline_report.json",
]

EXPECTED_BLOCKERS = [
    "human_audit_completed",
    "human_audit_applied",
    "audited_primary_validation_passed",
    "audited_source_files_validation_passed",
    "audited_apply_integrity_passed",
    "metric_metadata_created",
    "recent_session_model_results_exist",
    "fixed_baseline_results_exist",
]
TRANSIENT_SELF_FRESHNESS_BLOCKERS = {"skipped_audit_stop_point_report_fresh"}

FINAL_OUTPUTS = [
    "primary/multilingual_locomo_style_eval_audited.json",
    "primary/audited_sources",
    "baseline_results/metric_metadata.jsonl",
    "baseline_results/summary.json",
    "baseline_results/predictions",
    "baseline_results/normalized",
    "baseline_results/normalization_summaries",
    "recent_session_ablation/model_results_summary.json",
    "recent_session_ablation/model_prediction_records.jsonl",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        state["sha256"] = sha256_file(path)
    return state


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_report_errors(report: dict[str, Any]) -> list[str]:
    errors = [str(item) for item in report.get("errors", [])]
    for key in ("import", "validation"):
        nested = report.get(key)
        if isinstance(nested, dict):
            errors.extend(str(item) for item in nested.get("errors", []))
    return errors


def changed_rows(report: dict[str, Any]) -> Any:
    if "changed_rows" in report:
        return report.get("changed_rows")
    imported = report.get("import")
    if isinstance(imported, dict):
        return imported.get("changed_rows")
    return None


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Skipped Audit Stop Point",
        "",
        f"Status: `{report['status']}`",
        "",
        "This report is valid only for the user-directed case where manual human audit is skipped.",
        "It confirms that the repository remains at bootstrap-harness state and has not created final audited/model-result artifacts.",
        "",
        "## Freshness",
        "",
        f"- Checker: `{report.get('checker')}`",
        f"- Checker SHA256: `{report.get('checker_sha256')}`",
        "",
        "### Input Files",
        "",
    ]
    for rel, state in report.get("input_files", {}).items():
        suffix = f", sha256={state.get('sha256')}" if state.get("sha256") else ""
        lines.append(f"- `{rel}`: exists={state.get('exists')}{suffix}")
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for check in report["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
        if check.get("detail"):
            lines.append(f"  - {check['detail']}")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def add_check(checks: list[dict[str, str]], errors: list[str], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "status": "passed" if ok else "failed", "detail": detail})
    if not ok:
        errors.append(f"{name}: {detail or 'failed'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/locomo_style_eval"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/skipped_audit_stop_point_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("datasets/locomo_style_eval/skipped_audit_stop_point_report.md"),
    )
    args = parser.parse_args()

    root = args.root
    errors: list[str] = []
    checks: list[dict[str, str]] = []

    bootstrap_primary = root / "primary" / "multilingual_locomo_style_eval.json"
    add_check(
        checks,
        errors,
        "bootstrap_primary_exists",
        bootstrap_primary.is_file(),
        str(bootstrap_primary),
    )

    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    manifest_status = str(manifest.get("status"))
    add_check(
        checks,
        errors,
        "manifest_not_final_release",
        manifest_path.is_file() and manifest_status != "final_audited_release",
        f"exists={manifest_path.is_file()} status={manifest_status!r}",
    )

    release_gate = load_json(root / "release_gate_report.json")
    blockers = [str(item) for item in release_gate.get("blocking_failed", [])]
    expected_blocker_set = set(EXPECTED_BLOCKERS)
    blocker_set = set(blockers)
    duplicate_blockers = sorted({item for item in blockers if blockers.count(item) > 1})
    missing_blockers = sorted(expected_blocker_set - blocker_set)
    transient_blockers = sorted((blocker_set - expected_blocker_set) & TRANSIENT_SELF_FRESHNESS_BLOCKERS)
    unexpected_blockers = sorted(blocker_set - expected_blocker_set - TRANSIENT_SELF_FRESHNESS_BLOCKERS)
    add_check(
        checks,
        errors,
        "release_gate_expected_blockers_only",
        release_gate.get("status") == "blocked"
        and not missing_blockers
        and not unexpected_blockers
        and not duplicate_blockers
        and len(blockers) == len(EXPECTED_BLOCKERS) + len(transient_blockers),
        (
            f"status={release_gate.get('status')!r} blockers={blockers} "
            f"missing={missing_blockers} unexpected={unexpected_blockers} "
            f"transient={transient_blockers} duplicates={duplicate_blockers}"
        ),
    )

    audit_summary = load_json(root / "human_audit_results_summary.json")
    incomplete_count = int(audit_summary.get("incomplete_count") or 0)
    todo_count = int((audit_summary.get("decision_counts") or {}).get("todo") or 0)
    add_check(
        checks,
        errors,
        "strict_human_audit_incomplete",
        audit_summary.get("status") == "incomplete_or_failed"
        and audit_summary.get("allow_incomplete") is False
        and incomplete_count > 0
        and todo_count > 0,
        (
            f"status={audit_summary.get('status')!r} allow_incomplete={audit_summary.get('allow_incomplete')!r} "
            f"incomplete_count={incomplete_count} todo={todo_count}"
        ),
    )

    finalizer = load_json(root / "human_audit_batches_finalize_dry_run.json")
    finalizer_errors = collect_report_errors(finalizer)
    add_check(
        checks,
        errors,
        "batch_finalizer_dry_run_failed_safely",
        finalizer.get("status") == "failed"
        and finalizer.get("dry_run") is True
        and finalizer.get("committed") is False
        and changed_rows(finalizer) == 0
        and any("human_decision=todo" in item for item in finalizer_errors),
        (
            f"status={finalizer.get('status')!r} dry_run={finalizer.get('dry_run')!r} "
            f"committed={finalizer.get('committed')!r} changed_rows={changed_rows(finalizer)!r}"
        ),
    )

    csv_finalizer = load_json(root / "human_audit_csv_finalize_dry_run.json")
    csv_finalizer_errors = collect_report_errors(csv_finalizer)
    add_check(
        checks,
        errors,
        "csv_finalizer_dry_run_failed_safely",
        csv_finalizer.get("status") == "failed"
        and csv_finalizer.get("dry_run") is True
        and csv_finalizer.get("committed") is False
        and changed_rows(csv_finalizer) == 0
        and any("human_decision=todo" in item for item in csv_finalizer_errors),
        (
            f"status={csv_finalizer.get('status')!r} dry_run={csv_finalizer.get('dry_run')!r} "
            f"committed={csv_finalizer.get('committed')!r} changed_rows={changed_rows(csv_finalizer)!r}"
        ),
    )

    post_audit = load_json(root / "post_audit_pipeline_report.json")
    add_check(
        checks,
        errors,
        "post_audit_pipeline_stopped_before_apply",
        post_audit.get("status") == "failed" and post_audit.get("failed_step") == "validate_human_audit_results",
        f"status={post_audit.get('status')!r} failed_step={post_audit.get('failed_step')!r}",
    )

    stale_outputs = [str(root / rel) for rel in FINAL_OUTPUTS if (root / rel).exists()]
    add_check(
        checks,
        errors,
        "final_outputs_absent",
        not stale_outputs,
        f"stale_outputs={stale_outputs}",
    )

    report = {
        "status": "passed" if not errors else "failed",
        "root": str(root),
        "purpose": "safe_stop_point_when_manual_human_audit_is_skipped",
        "checker": str(Path(__file__)),
        "checker_sha256": sha256_file(Path(__file__)),
        "input_files": {rel: file_state(root / rel) for rel in INPUT_FILES},
        "expected_blockers": EXPECTED_BLOCKERS,
        "final_outputs_that_must_be_absent": [str(root / rel) for rel in FINAL_OUTPUTS],
        "final_outputs_state": {rel: file_state(root / rel) for rel in FINAL_OUTPUTS},
        "checks": checks,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
