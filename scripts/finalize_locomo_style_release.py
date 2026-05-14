#!/usr/bin/env python3
"""Finalize the LoCoMo-style eval artifact only after release gates pass.

This script is intentionally conservative. It reruns the release-gate checker,
refuses to update the manifest while any blocking gate is failed, and only then
marks the artifact as a final audited release.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STATUS = "final_audited_locomo_style_eval_release"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_release_gate(root: Path, output: Path) -> dict[str, Any]:
    gate_script = Path(__file__).with_name("check_locomo_style_release_gates.py")
    command = [
        sys.executable,
        str(gate_script),
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if output.exists():
        return load_json(output)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "release gate checker did not produce JSON; "
            f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/locomo_style_eval"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--gate-output", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run gates and print the manifest update that would be written.",
    )
    args = parser.parse_args()

    root = args.root
    manifest_path = args.manifest or root / "manifest.json"
    gate_output = args.gate_output or root / "release_gate_report.json"

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    gate_report = run_release_gate(root, gate_output)
    if gate_report.get("status") != "release_ready":
        blocking = gate_report.get("blocking_failed", [])
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "release gates are not all passed",
                    "blocking_failed": blocking,
                    "release_gate_report": str(gate_output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    manifest = load_json(manifest_path)
    updated = dict(manifest)
    updated.update(
        {
            "status": FINAL_STATUS,
            "finalized_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "release_gate_report": str(gate_output),
            "release_root": str(root),
        }
    )

    if args.dry_run:
        print(json.dumps({"status": "dry_run_release_ready", "manifest": updated}, ensure_ascii=False, indent=2))
        return 0

    write_json(manifest_path, updated)
    print(
        json.dumps(
            {
                "status": "finalized",
                "manifest": str(manifest_path),
                "final_status": FINAL_STATUS,
                "release_gate_report": str(gate_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
