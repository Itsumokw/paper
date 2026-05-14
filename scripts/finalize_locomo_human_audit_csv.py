#!/usr/bin/env python3
"""Safely commit completed full human-audit CSV decisions to the audit packet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_output = Path(tmp_name)
    try:
        tmp_output.write_bytes(source.read_bytes())
        os.replace(tmp_output, target)
    finally:
        if tmp_output.exists():
            tmp_output.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Audit packet to write after validation. Defaults to --base-jsonl.",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--import-summary", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate completed CSV edits but do not replace the output packet.",
    )
    args = parser.parse_args()

    output_jsonl = args.output_jsonl or args.base_jsonl
    for stale_path in (args.import_summary, args.validation_summary):
        if stale_path.exists():
            stale_path.unlink()

    errors: list[str] = []
    import_report: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    import_returncode: int | None = None
    validation_returncode: int | None = None
    committed = False
    staged_path_record: str | None = None

    with tempfile.TemporaryDirectory(prefix="locomo_human_audit_csv_finalize_") as tmp:
        tempdir = Path(tmp)
        staged_path = tempdir / "completed_audit_packet.jsonl"
        import_command = [
            sys.executable,
            str(Path(__file__).with_name("import_locomo_human_audit_csv.py")),
            "--base-jsonl",
            str(args.base_jsonl),
            "--decisions-csv",
            str(args.decisions_csv),
            "--output-jsonl",
            str(staged_path),
            "--summary-json",
            str(args.import_summary),
            "--require-complete",
        ]
        imported = subprocess.run(import_command, check=False, text=True, capture_output=True)
        import_returncode = imported.returncode
        if args.import_summary.exists():
            import_report = load_json(args.import_summary)
            if import_report.get("output_jsonl"):
                import_report["output_jsonl"] = "staged_temporary_packet"
            args.import_summary.write_text(
                json.dumps(import_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            errors.append(
                "CSV import did not write summary: "
                f"returncode={imported.returncode} stderr={imported.stderr.strip()!r}"
            )
        if imported.returncode != 0:
            if import_report is not None:
                errors.extend(str(item) for item in import_report.get("errors", []))
            else:
                errors.append(imported.stderr.strip() or imported.stdout.strip() or "CSV import failed")

        if not errors:
            validation_command = [
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
            validated = subprocess.run(validation_command, check=False, text=True, capture_output=True)
            validation_returncode = validated.returncode
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
                    f"returncode={validated.returncode} stderr={validated.stderr.strip()!r}"
                )
            if validated.returncode != 0 and validation_report is not None:
                errors.extend(str(item) for item in validation_report.get("errors", []))
            if validation_report is not None and validation_report.get("status") != "completed":
                errors.append(f"validator status={validation_report.get('status')!r} expected='completed'")

        if not errors:
            staged_path_record = str(staged_path) if args.dry_run else "committed_via_atomic_replace"
            if not args.dry_run:
                write_bytes_atomic(staged_path, output_jsonl)
                committed = True

    status = "dry_run_valid" if args.dry_run and not errors else "committed" if committed else "failed"
    summary = {
        "status": status,
        "purpose": "safe_completed_full_csv_commit",
        "dry_run": args.dry_run,
        "committed": committed,
        "base_jsonl": str(args.base_jsonl),
        "decisions_csv": str(args.decisions_csv),
        "output_jsonl": str(output_jsonl),
        "sidecar_root": str(args.sidecar_root),
        "staged_jsonl": staged_path_record,
        "import_summary": str(args.import_summary),
        "validation_summary": str(args.validation_summary),
        "import_returncode": import_returncode,
        "validation_returncode": validation_returncode,
        "import": import_report,
        "validation": validation_report,
        "errors": errors,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status in {"dry_run_valid", "committed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
