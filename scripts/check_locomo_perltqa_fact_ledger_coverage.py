#!/usr/bin/env python3
"""Check PerLTQA PlanMode-D fact-ledger coverage against source fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


PAPER_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_speaker_line(raw_line: str, default_speaker: str) -> tuple[str, str]:
    line = normalize_text(raw_line).strip()
    if ":" in line:
        speaker, text = line.split(":", 1)
        return speaker.strip() or default_speaker, text.strip()
    if "：" in line:
        speaker, text = line.split("：", 1)
        return speaker.strip() or default_speaker, text.strip()
    return default_speaker, line


def perltqa_by_person(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.exists():
        return result
    for row in load_json(path):
        if isinstance(row, dict):
            for person, value in row.items():
                result[str(person)] = value
    return result


def iter_perltqa_source_qas(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if "Question" in node and "Answer" in node:
            yield node
        for value in node.values():
            yield from iter_perltqa_source_qas(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_perltqa_source_qas(item)


def sample_raw_index(sample_id: str) -> int | None:
    prefix = "perltqa_"
    if not sample_id.startswith(prefix):
        return None
    try:
        return int(sample_id[len(prefix) :])
    except ValueError:
        return None


def fact_id_for_dia(sample_id: str, dia_id: str) -> str:
    return f"{sample_id}_turn_{dia_id.replace(':', '_')}"


def expected_perltqa_facts(
    sample_id: str,
    source_item: dict[str, Any],
    source_qas_for_person: Any,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    profile = source_item.get("profile") or {}
    protagonist = str(profile.get("Protagonist") or sample_id)

    for field_name, value in profile.items():
        source_text = normalize_text(f"{field_name}: {value}")
        expected.append(
            {
                "fact_id": f"{sample_id}_profile_{field_name}",
                "source_type": "original_persona",
                "source_text": source_text,
                "source_id": f"profile.{field_name}",
                "field": field_name,
                "source_hash": sha256_text(source_text),
            }
        )

    for rel_id, relationship in sorted((source_item.get("social_relationship") or {}).items()):
        if not isinstance(relationship, dict):
            continue
        source_text = normalize_text(json.dumps(relationship, ensure_ascii=False, sort_keys=True))
        expected.append(
            {
                "fact_id": f"{sample_id}_relationship_{rel_id}",
                "source_type": "original_relationship",
                "source_text": source_text,
                "source_id": f"social_relationship.{rel_id}",
                "field": "social_relationship",
                "source_hash": sha256_text(source_text),
            }
        )

    for qa_idx, row in enumerate(iter_perltqa_source_qas(source_qas_for_person)):
        question = normalize_text(row.get("Question")).strip()
        answer = normalize_text(row.get("Answer")).strip()
        if not question or not answer:
            continue
        reference = normalize_text(row.get("Reference Memory")).strip()
        source_text = f"Question: {question}\nAnswer: {answer}"
        if reference:
            source_text += f"\nReference Memory: {reference}"
        source_text = normalize_text(source_text)
        expected.append(
            {
                "fact_id": f"{sample_id}_original_qa_{qa_idx}",
                "source_type": "original_qa",
                "source_text": source_text,
                "source_id": f"{protagonist}.original_qa.{qa_idx}",
                "field": "perltqa.original_qa",
                "source_hash": sha256_text(source_text),
            }
        )

    dialogues = source_item.get("dialogues") or {}
    events = source_item.get("events") or {}
    sorted_dialogues = sorted(
        dialogues.items(),
        key=lambda kv: (
            int(str(kv[0]).rsplit("#", 1)[-1])
            if "#" in str(kv[0]) and str(kv[0]).rsplit("#", 1)[-1].isdigit()
            else 0
        ),
    )
    session_num = 0
    seen_event_fact_ids: set[str] = set()
    for dialogue_key, dialogue in sorted_dialogues:
        date_to_lines = dialogue.get("contents") or {}
        if not isinstance(date_to_lines, dict) or not date_to_lines:
            continue
        event_id = str(dialogue.get("events") or str(dialogue_key).split("#", 1)[0])
        event = events.get(event_id, {})
        event_text = normalize_text(event.get("content") or event.get("summary") or "")
        if event_text:
            fact_id = f"{sample_id}_event_{event_id}"
            if fact_id not in seen_event_fact_ids:
                expected.append(
                    {
                        "fact_id": fact_id,
                        "source_type": "original_event",
                        "source_text": event_text,
                        "source_id": event_id,
                        "field": "events.content",
                        "source_hash": sha256_text(event_text),
                    }
                )
                seen_event_fact_ids.add(fact_id)
        session_num += 1
        turn_counter = 1 if event_text else 0
        for date_label, lines in date_to_lines.items():
            iterable = lines if isinstance(lines, list) else [lines]
            for line_idx, raw_line in enumerate(iterable, start=1):
                _, text = parse_speaker_line(str(raw_line), protagonist)
                if not text:
                    continue
                turn_counter += 1
                dia_id = f"D{session_num}:{turn_counter}"
                source_text = normalize_text(text)
                expected.append(
                    {
                        "fact_id": fact_id_for_dia(sample_id, dia_id),
                        "source_type": "original_turn",
                        "source_text": source_text,
                        "source_id": f"{dialogue_key}:{date_label}:{line_idx}",
                        "field": "turn.text",
                        "source_hash": sha256_text(source_text),
                        "dia_id": dia_id,
                    }
                )
    return expected


def fact_matches(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for field in ("source_type", "source_text", "source_id", "field", "source_hash"):
        if observed.get(field) != expected.get(field):
            mismatches.append(f"{field}={observed.get(field)!r} expected={expected.get(field)!r}")
    if expected.get("source_type") == "original_turn" and observed.get("dia_id") != expected.get("dia_id"):
        mismatches.append(f"dia_id={observed.get('dia_id')!r} expected={expected.get('dia_id')!r}")
    return mismatches


def build_report(
    *,
    primary_json: Path,
    source_memory: Path,
    source_qa: Path,
    fact_ledger: Path,
    provenance: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    primary = load_json(primary_json)
    memories = load_json(source_memory)
    qas_by_person = perltqa_by_person(source_qa)
    fact_rows = list(iter_jsonl(fact_ledger))
    provenance_rows = list(iter_jsonl(provenance))

    fact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    for row in fact_rows:
        if row.get("source_dataset") != "PerLTQA":
            continue
        sample_id = str(row.get("sample_id"))
        fact_index[str(row.get("fact_id"))].append(row)
        observed_by_sample[sample_id][str(row.get("source_type"))] += 1

    perltqa_samples = [sample for sample in primary if sample.get("source_dataset") == "PerLTQA"]
    if not perltqa_samples:
        errors.append("primary JSON contains no PerLTQA samples")

    allowed_source_types = {
        "original_persona",
        "original_relationship",
        "original_event",
        "original_turn",
        "original_qa",
    }
    unexpected_types = sorted(
        {
            str(row.get("source_type"))
            for row in fact_rows
            if row.get("source_dataset") == "PerLTQA" and str(row.get("source_type")) not in allowed_source_types
        }
    )
    if unexpected_types:
        errors.append(f"PerLTQA fact ledger has unexpected source_type values: {unexpected_types}")

    provenance_original_turns = {
        str(row.get("source_fact_id"))
        for row in provenance_rows
        if row.get("source_dataset") == "PerLTQA" and row.get("source_origin") == "original_turn"
    }

    per_sample: dict[str, Any] = {}
    total_expected = Counter()
    total_observed = Counter()
    mismatch_examples: list[str] = []
    missing_examples: list[str] = []
    for sample in perltqa_samples:
        sample_id = str(sample.get("sample_id"))
        raw_idx = sample_raw_index(sample_id)
        if raw_idx is None or raw_idx < 0 or raw_idx >= len(memories):
            errors.append(f"{sample_id}: cannot resolve source memory index")
            continue
        source_item = memories[raw_idx]
        profile = source_item.get("profile") or {}
        protagonist = str(profile.get("Protagonist") or sample_id)
        expected = expected_perltqa_facts(sample_id, source_item, qas_by_person.get(protagonist, {}))
        expected_counts = Counter(str(row["source_type"]) for row in expected)
        observed_counts = observed_by_sample.get(sample_id, Counter())
        total_expected.update(expected_counts)
        total_observed.update(observed_counts)

        required_types = {
            "original_persona",
            "original_relationship",
            "original_event",
            "original_turn",
        }
        if expected_counts.get("original_qa", 0) > 0:
            required_types.add("original_qa")
        for source_type in sorted(required_types):
            if expected_counts.get(source_type, 0) > 0 and observed_counts.get(source_type, 0) == 0:
                errors.append(f"{sample_id}: missing required fact source_type={source_type}")

        missing = 0
        mismatched = 0
        for expected_row in expected:
            fact_id = str(expected_row["fact_id"])
            candidates = fact_index.get(fact_id, [])
            if not candidates:
                missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append(f"{sample_id}: missing fact_id={fact_id}")
                continue
            if not any(not fact_matches(expected_row, candidate) for candidate in candidates):
                mismatched += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        f"{sample_id}: fact_id={fact_id} mismatch: {fact_matches(expected_row, candidates[0])}"
                    )

        expected_turn_fact_ids = {
            str(row["fact_id"]) for row in expected if row.get("source_type") == "original_turn"
        }
        provenance_missing = sorted(expected_turn_fact_ids - provenance_original_turns)
        if provenance_missing:
            errors.append(
                f"{sample_id}: {len(provenance_missing)} original_turn facts missing provenance; "
                f"first={provenance_missing[:5]}"
            )

        per_sample[sample_id] = {
            "source_record_index": raw_idx,
            "protagonist": protagonist,
            "source_qa_available": expected_counts.get("original_qa", 0) > 0,
            "expected_counts": dict(sorted(expected_counts.items())),
            "observed_counts": dict(sorted(observed_counts.items())),
            "missing_expected_fact_count": missing,
            "mismatched_expected_fact_count": mismatched,
            "expected_original_turn_facts_missing_provenance": len(provenance_missing),
        }
        if missing:
            errors.append(f"{sample_id}: missing {missing} expected fact-ledger rows")
        if mismatched:
            errors.append(f"{sample_id}: {mismatched} expected fact-ledger rows have mismatched content")

    if not any(row.get("source_qa_available") for row in per_sample.values()):
        errors.append("no selected PerLTQA sample has original QA coverage in source perltqa.json")
    for source_type in sorted(allowed_source_types):
        if total_observed.get(source_type, 0) == 0:
            errors.append(f"PerLTQA fact ledger has zero rows for required source_type={source_type}")

    return {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "primary_json": file_record(primary_json),
            "source_memory": file_record(source_memory),
            "source_qa": file_record(source_qa),
            "fact_ledger": file_record(fact_ledger),
            "provenance": file_record(provenance),
        },
        "sample_count": len(perltqa_samples),
        "total_expected_counts": dict(sorted(total_expected.items())),
        "total_observed_counts": dict(sorted(total_observed.items())),
        "per_sample": per_sample,
        "missing_examples": missing_examples,
        "mismatch_examples": mismatch_examples,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=Path("datasets/locomo_style_eval/primary/PerLTQA-LoCoMo-style-eval.json"),
    )
    parser.add_argument(
        "--source-memory",
        type=Path,
        default=Path("datasets/PerLTQA/Dataset/zh/perltmem.json"),
    )
    parser.add_argument(
        "--source-qa",
        type=Path,
        default=Path("datasets/PerLTQA/Dataset/zh/perltqa.json"),
    )
    parser.add_argument(
        "--fact-ledger",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/PerLTQA-LoCoMo-style-eval/"
            "PerLTQA-LoCoMo-style-eval_fact_ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(
            "datasets/locomo_style_eval/sidecars/PerLTQA-LoCoMo-style-eval/"
            "PerLTQA-LoCoMo-style-eval_provenance.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/locomo_style_eval/perltqa_fact_ledger_coverage_report.json"),
    )
    args = parser.parse_args()

    report = build_report(
        primary_json=args.primary_json,
        source_memory=args.source_memory,
        source_qa=args.source_qa,
        fact_ledger=args.fact_ledger,
        provenance=args.provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
