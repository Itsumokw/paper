#!/usr/bin/env python3
"""Safely commit completed human-audit batch CSV decisions to the audit packet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from import_locomo_human_audit_csv import iter_jsonl, write_jsonl
from merge_locomo_human_audit_batches import load_batch_rows, merge_rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Audit packet to write after validation. Defaults to --base-jsonl.",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--pattern", default="batch_*.csv")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate completed batch edits but do not replace the output packet.",
    )
    args = parser.parse_args()

    output_jsonl = args.output_jsonl or args.base_jsonl
    if args.validation_summary.exists():
        args.validation_summary.unlink()
    base_rows = list(iter_jsonl(args.base_jsonl))
    csv_rows, batches, load_errors = load_batch_rows(args.input_dir, args.pattern)
    output_rows, decision_counts, changed_rows, merge_errors = merge_rows(
        base_rows,
        csv_rows,
        f"{args.input_dir}/{args.pattern}",
    )
    errors = load_errors + merge_errors
    if decision_counts.get("todo", 0):
        errors.append(f"{decision_counts['todo']} audit rows still have human_decision=todo")

    validation_report: dict[str, Any] | None = None
    validation_returncode: int | None = None
    committed = False
    staged_path_record: str | None = None

    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_batch_finalize_") as tmp:
        tempdir = Path(tmp)
        staged_path = tempdir / "completed_audit_packet.jsonl"
        if not errors:
            write_jsonl(staged_path, output_rows)
            command = [
                sys.executable,
                str(Path(__file__).with_name("validate_locomo_human_audit_results.py")),
                "--input-jsonl",
                str(staged_path),
                "--output-summary",
                str(args.validation_summary),
                "--allow-failures",
                "--sidecar-root",
                str(args.sidecar_root),
            ]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            validation_returncode = completed.returncode
            if args.validation_summary.exists():
                validation_report = load_json(args.validation_summary)
                validation_report["input_jsonl"] = "staged_temporary_packet"
                args.validation_summary.write_text(
                    json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                errors.append(
                    "validator did not write summary: "
                    f"returncode={completed.returncode} stderr={completed.stderr.strip()!r}"
                )
            if completed.returncode != 0 and validation_report is not None:
                errors.extend(str(item) for item in validation_report.get("errors", []))
            if validation_report is not None and validation_report.get("status") != "completed":
                errors.append(f"validator status={validation_report.get('status')!r} expected='completed'")

        if not errors:
            staged_path_record = str(staged_path) if args.dry_run else "committed_via_atomic_replace"
            if not args.dry_run:
                output_jsonl.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{output_jsonl.name}.",
                    suffix=".tmp",
                    dir=str(output_jsonl.parent),
                )
                os.close(fd)
                tmp_output = Path(tmp_name)
                try:
                    write_jsonl(tmp_output, output_rows)
                    os.replace(tmp_output, output_jsonl)
                    committed = True
                finally:
                    if tmp_output.exists():
                        tmp_output.unlink()

    status = "dry_run_valid" if args.dry_run and not errors else "committed" if committed else "failed"
    summary = {
        "status": status,
        "purpose": "safe_completed_batch_commit",
        "dry_run": args.dry_run,
        "committed": committed,
        "base_jsonl": str(args.base_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "sidecar_root": str(args.sidecar_root),
        "staged_jsonl": staged_path_record,
        "validation_summary": str(args.validation_summary),
        "base_rows": len(base_rows),
        "batch_rows": len(csv_rows),
        "batches": batches,
        "changed_rows": changed_rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "validation_returncode": validation_returncode,
        "validation": validation_report,
        "errors": errors,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status in {"dry_run_valid", "committed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
