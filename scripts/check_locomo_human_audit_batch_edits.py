#!/usr/bin/env python3
"""Safely check in-progress human-audit batch CSV edits.

This command merges batch CSV edits into a temporary packet, validates completed
rows with the normal human-audit validator, and leaves the real audit packet
untouched.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--pattern", default="batch_*.csv")
    parser.add_argument(
        "--merged-jsonl",
        type=Path,
        default=None,
        help="Optional path for the merged temporary packet. Omit to discard it after validation.",
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=None,
        help="Optional path for the underlying validator summary. Omit to embed only in output-summary.",
    )
    args = parser.parse_args()

    base_rows = list(iter_jsonl(args.base_jsonl))
    if args.validation_summary and args.validation_summary.exists():
        args.validation_summary.unlink()
    csv_rows, batches, load_errors = load_batch_rows(args.input_dir, args.pattern)
    output_rows, decision_counts, changed_rows, merge_errors = merge_rows(
        base_rows,
        csv_rows,
        f"{args.input_dir}/{args.pattern}",
    )
    errors = load_errors + merge_errors

    validation_report: dict[str, Any] | None = None
    validation_returncode: int | None = None
    merged_path_record: str | None = None
    validation_path_record: str | None = None

    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_batch_check_") as tmp:
        tempdir = Path(tmp)
        merged_path = args.merged_jsonl or (tempdir / "merged_audit_packet.jsonl")
        validation_path = args.validation_summary or (tempdir / "validation_summary.json")

        if not errors:
            write_jsonl(merged_path, output_rows)
            merged_path_record = str(merged_path) if args.merged_jsonl else "temporary_discarded"
            command = [
                sys.executable,
                str(Path(__file__).with_name("validate_locomo_human_audit_results.py")),
                "--input-jsonl",
                str(merged_path),
                "--output-summary",
                str(validation_path),
                "--allow-failures",
                "--allow-incomplete",
                "--sidecar-root",
                str(args.sidecar_root),
            ]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            validation_returncode = completed.returncode
            validation_path_record = str(validation_path) if args.validation_summary else "embedded_only"
            if validation_path.exists():
                validation_report = load_json(validation_path)
                if args.validation_summary and not args.merged_jsonl:
                    validation_report["input_jsonl"] = "temporary_discarded"
                    validation_path.write_text(
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

    if errors:
        status = "failed"
    else:
        status = str(validation_report.get("status")) if validation_report else "failed"

    summary = {
        "status": status,
        "purpose": "safe_in_progress_batch_check_not_release_gate",
        "satisfies_release_gate": False,
        "base_jsonl": str(args.base_jsonl),
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "sidecar_root": str(args.sidecar_root),
        "merged_jsonl": merged_path_record,
        "validation_summary": validation_path_record,
        "base_rows": len(base_rows),
        "batch_rows": len(csv_rows),
        "batches": batches,
        "changed_rows": changed_rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "validation_returncode": validation_returncode,
        "validation": validation_report,
        "errors": errors,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status in {"completed", "partial_valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
