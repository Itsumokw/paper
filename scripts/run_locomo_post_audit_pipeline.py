#!/usr/bin/env python3
"""Run the no-model post-human-audit pipeline for LoCoMo-style eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_FINAL_BLOCKERS = {
    "recent_session_model_results_exist",
    "fixed_baseline_results_exist",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_step(name: str, cmd: list[str], report: dict[str, Any]) -> bool:
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    row = {
        "name": name,
        "cmd": cmd,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    report["steps"].append(row)
    return proc.returncode == 0


def release_gate_has_only_allowed_blockers(path: Path) -> tuple[bool, list[str], str]:
    if not path.is_file():
        return False, ["release gate report missing"], "missing"
    data = load_json(path)
    blockers = [str(item) for item in data.get("blocking_failed", [])]
    unexpected = sorted(set(blockers) - ALLOWED_FINAL_BLOCKERS)
    status = str(data.get("status"))
    return not unexpected, blockers, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/locomo_style_eval"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/post_audit_pipeline_report.json"))
    args = parser.parse_args()

    root = args.root
    py = sys.executable
    primary_json = root / "primary" / "multilingual_locomo_style_eval.json"
    audit_jsonl = root / "human_audit_packet.jsonl"
    audited_json = root / "primary" / "multilingual_locomo_style_eval_audited.json"
    audited_source_dir = root / "primary" / "audited_sources"
    audit_results_summary = root / "human_audit_results_summary.json"
    audit_apply_report = root / "human_audit_apply_report.json"
    audited_apply_integrity = root / "audited_apply_integrity_report.json"
    metric_metadata = root / "baseline_results" / "metric_metadata.jsonl"
    metric_metadata_summary = root / "baseline_results" / "metric_metadata_summary.json"
    ablation_root = root / "recent_session_ablation"
    release_gate_report = root / "release_gate_report.json"

    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "steps": [],
        "paths": {
            "primary_json": str(primary_json),
            "audit_jsonl": str(audit_jsonl),
            "audited_json": str(audited_json),
            "audit_results_summary": str(audit_results_summary),
            "audit_apply_report": str(audit_apply_report),
            "audited_apply_integrity": str(audited_apply_integrity),
            "metric_metadata": str(metric_metadata),
            "metric_metadata_summary": str(metric_metadata_summary),
            "ablation_root": str(ablation_root),
            "release_gate_report": str(release_gate_report),
        },
    }

    steps = [
        (
            "validate_human_audit_results",
            [
                py,
                "scripts/validate_locomo_human_audit_results.py",
                "--input-jsonl",
                str(audit_jsonl),
                "--output-summary",
                str(audit_results_summary),
                "--allow-failures",
                "--sidecar-root",
                str(root / "sidecars"),
            ],
        ),
        (
            "apply_human_audit",
            [
                py,
                "scripts/apply_locomo_human_audit_results.py",
                "--primary-json",
                str(primary_json),
                "--audit-jsonl",
                str(audit_jsonl),
                "--output-json",
                str(audited_json),
                "--output-report",
                str(audit_apply_report),
                "--output-source-dir",
                str(audited_source_dir),
            ],
        ),
        (
            "validate_audited_primary",
            [
                py,
                "scripts/validate_locomo_style_eval.py",
                str(audited_json),
            ],
        ),
        (
            "check_audited_apply_integrity",
            [
                py,
                "scripts/check_locomo_audited_apply_integrity.py",
                "--original-primary",
                str(primary_json),
                "--audit-jsonl",
                str(audit_jsonl),
                "--audited-primary",
                str(audited_json),
                "--sidecar-root",
                str(root / "sidecars"),
                "--output",
                str(audited_apply_integrity),
            ],
        ),
        (
            "build_metric_metadata",
            [
                py,
                "scripts/build_locomo_metric_metadata.py",
                "--primary-json",
                str(audited_json),
                "--sidecar-root",
                str(root / "sidecars"),
                "--output-jsonl",
                str(metric_metadata),
                "--summary-json",
                str(metric_metadata_summary),
            ],
        ),
        (
            "rebuild_recent_session_ablation",
            [
                py,
                "scripts/make_locomo_recent_session_ablation.py",
                "--input",
                str(audited_json),
                "--output-root",
                str(ablation_root),
            ],
        ),
        (
            "check_release_gates",
            [
                py,
                "scripts/check_locomo_style_release_gates.py",
                "--root",
                str(root),
                "--output",
                str(release_gate_report),
            ],
        ),
    ]

    for name, cmd in steps:
        ok = run_step(name, cmd, report)
        if not ok and name != "check_release_gates":
            report["status"] = "failed"
            report["failed_step"] = name
            report["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1

    release_ok, blockers, gate_status = release_gate_has_only_allowed_blockers(release_gate_report)
    report["release_gate_status"] = gate_status
    report["release_gate_blocking_failed"] = blockers
    report["allowed_remaining_blockers"] = sorted(ALLOWED_FINAL_BLOCKERS)
    report["status"] = "post_audit_ready_for_model_runs" if release_ok else "failed"
    if not release_ok:
        report["failed_step"] = "check_release_gates"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if release_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
