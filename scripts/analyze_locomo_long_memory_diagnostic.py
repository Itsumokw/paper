#!/usr/bin/env python3
"""Evidence-locality diagnostic for LoCoMo-style long-memory datasets.

This is a no-model construction audit. It does not replace the model-side
recent-session baselines, but it flags whether answerable QA can be supported
using only the last session or last three sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIA_RE = re.compile(r"^D(\d+):(\d+)$")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
            suffix = key.rsplit("_", 1)[-1]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return sorted(numbers)


def evidence_sessions(evidence: list[Any]) -> set[int]:
    sessions: set[int] = set()
    for ev in evidence:
        match = DIA_RE.match(str(ev))
        if match:
            sessions.add(int(match.group(1)))
    return sessions


def ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def analyze(path: Path) -> dict[str, Any]:
    data = load_json(path)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    weak_samples: list[dict[str, Any]] = []

    for sample in data:
        source = str(sample.get("source_dataset", "unknown"))
        sample_id = str(sample.get("sample_id", "unknown"))
        sessions = session_numbers(sample.get("conversation", {}))
        if not sessions:
            continue
        last_session = {sessions[-1]}
        last_three = set(sessions[-3:])
        sample_counts: Counter[str] = Counter()
        for qa in sample.get("qa", []):
            category = str(qa.get("category"))
            if category == "5":
                continue
            ev_sessions = evidence_sessions(qa.get("evidence", []))
            if not ev_sessions:
                continue
            for bucket in (overall, by_source[source], sample_counts):
                bucket["answerable_qa"] += 1
                bucket[f"cat{category}"] += 1
                if len(ev_sessions) > 1:
                    bucket["cross_session_qa"] += 1
                if ev_sessions <= last_session:
                    bucket["last_session_sufficient"] += 1
                if ev_sessions <= last_three:
                    bucket["last_three_sessions_sufficient"] += 1
                if min(ev_sessions) < sessions[-1]:
                    bucket["requires_prior_to_last_session"] += 1
                if min(ev_sessions) < sessions[-3] if len(sessions) >= 3 else False:
                    bucket["requires_prior_to_last_three"] += 1
        if sample_counts["answerable_qa"]:
            last3_ratio = ratio(sample_counts["last_three_sessions_sufficient"], sample_counts["answerable_qa"])
            if last3_ratio >= 0.8:
                weak_samples.append(
                    {
                        "sample_id": sample_id,
                        "source_dataset": source,
                        "answerable_qa": sample_counts["answerable_qa"],
                        "last_three_sessions_sufficient_ratio": last3_ratio,
                    }
                )

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        total = counter["answerable_qa"]
        return {
            "answerable_qa": total,
            "cross_session_qa": counter["cross_session_qa"],
            "cross_session_ratio": ratio(counter["cross_session_qa"], total),
            "last_session_sufficient": counter["last_session_sufficient"],
            "last_session_sufficient_ratio": ratio(counter["last_session_sufficient"], total),
            "last_three_sessions_sufficient": counter["last_three_sessions_sufficient"],
            "last_three_sessions_sufficient_ratio": ratio(counter["last_three_sessions_sufficient"], total),
            "requires_prior_to_last_session": counter["requires_prior_to_last_session"],
            "requires_prior_to_last_session_ratio": ratio(counter["requires_prior_to_last_session"], total),
            "requires_prior_to_last_three": counter["requires_prior_to_last_three"],
            "requires_prior_to_last_three_ratio": ratio(counter["requires_prior_to_last_three"], total),
            "categories": {
                key[3:]: value
                for key, value in sorted(counter.items())
                if key.startswith("cat")
            },
        }

    return {
        "status": "passed",
        "input_files": {
            "primary_json": file_record(path),
        },
        "path": str(path),
        "diagnostic_type": "no_model_evidence_locality",
        "interpretation": (
            "A high last-session or last-three ratio means many QA may not require long-range memory. "
            "This is a construction audit, not a replacement for model recent-session baselines."
        ),
        "overall": summarize(overall),
        "by_source_dataset": {source: summarize(counter) for source, counter in sorted(by_source.items())},
        "weak_samples_last_three_ratio_ge_0_8": weak_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = analyze(args.primary_json)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
