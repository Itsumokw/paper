#!/usr/bin/env python3
"""Self-test release-blocker summary grouping."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import summarize_locomo_release_blockers as summary_script


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case(name: str, report: dict[str, Any], predicate, expected: str) -> dict[str, Any]:
    summary = summary_script.summarize(report)
    ok = bool(predicate(summary))
    return {
        "name": name,
        "status": "passed" if ok else "failed",
        "expected": expected,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/release_blocker_summary_selftest.json"),
    )
    args = parser.parse_args()

    blocked_report = {
        "status": "blocked",
        "blocking_failed": [
            "human_audit_completed",
            "audited_primary_validation_passed",
            "recent_session_model_results_exist",
            "unknown_gate",
        ],
        "checks": [
            {"name": "human_audit_completed", "evidence": "373 todo"},
            {"name": "audited_primary_validation_passed", "evidence": "audit not applied"},
            {"name": "recent_session_model_results_exist", "evidence": "summary missing"},
            {"name": "unknown_gate", "evidence": "custom failure"},
        ],
    }
    passed_report = {"status": "passed", "blocking_failed": [], "checks": []}

    cases = [
        case(
            "known_blockers_are_grouped",
            blocked_report,
            lambda s: {
                "human_audit",
                "post_audit",
                "model_results",
                "other",
            }
            == set(s["groups"]),
            "all expected groups exist",
        ),
        case(
            "evidence_is_preserved",
            blocked_report,
            lambda s: s["groups"]["human_audit"][0]["evidence"] == "373 todo"
            and s["groups"]["other"][0]["evidence"] == "custom failure",
            "gate evidence copied into summary rows",
        ),
        case(
            "human_audit_command_includes_batch_and_full_csv_finalizers",
            blocked_report,
            lambda s: "finalize_locomo_human_audit_batches.py" in s["groups"]["human_audit"][0]["command"]
            and "finalize_locomo_human_audit_csv.py" in s["groups"]["human_audit"][0]["command"]
            and "human_audit_csv_finalize_dry_run.json" in s["groups"]["human_audit"][0]["command"],
            "human audit remediation command exposes both safe finalizer dry-runs",
        ),
        case(
            "long_evidence_is_compacted",
            {
                "status": "blocked",
                "blocking_failed": ["human_audit_completed"],
                "checks": [{"name": "human_audit_completed", "evidence": "x" * 1200}],
            },
            lambda s: len(s["groups"]["human_audit"][0]["evidence"]) < 1000
            and "truncated" in s["groups"]["human_audit"][0]["evidence"],
            "long gate evidence is shortened in blocker summaries",
        ),
        case(
            "passed_report_has_no_blockers",
            passed_report,
            lambda s: s["status"] == "passed" and s["blocking_failed_count"] == 0 and not s["groups"],
            "passed release report produces empty blocker groups",
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="locomo_release_blockers_selftest_") as tmp:
        tempdir = Path(tmp)
        md_path = tempdir / "summary.md"
        summary_script.write_markdown(md_path, summary_script.summarize(blocked_report))
        md_text = md_path.read_text(encoding="utf-8")
        cases.append(
            {
                "name": "markdown_contains_actions",
                "status": "passed"
                if "Human Audit" in md_text
                and "Post-Audit Pipeline" in md_text
                and "Model Results" in md_text
                else "failed",
                "expected": "markdown contains grouped sections",
            }
        )

    failed = [row for row in cases if row["status"] != "passed"]
    result = {
        "status": "passed" if not failed else "failed",
        "summary_script": str(Path(summary_script.__file__)),
        "summary_script_sha256": sha256_file(Path(summary_script.__file__)),
        "selftest": str(Path(__file__)),
        "selftest_sha256": sha256_file(Path(__file__)),
        "cases": cases,
        "errors": [row["name"] for row in failed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
