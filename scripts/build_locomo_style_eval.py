#!/usr/bin/env python3
"""Build source-faithful LoCoMo-style eval drafts from local source datasets.

This script implements the first concrete construction harness for
docs/locomo_style_eval_goal.md:

- primary JSON files stay close to LoCoMo loader-facing fields;
- provenance, fact ledger, hash checks, and QA audit live in sidecars;
- original text is preserved except minimal control-character cleanup;
- answerable seed QA is only created when evidence can point to a concrete
  LoCoMo-style dia_id.

The output is a bootstrap eval artifact, not a claim that LLM expansion or
human audit has been completed.

This construction script intentionally makes zero local-model, remote-model,
or API calls. Any service chat preflight belongs to baseline experiment setup,
not dataset generation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PAPER_ROOT = Path("/home/stu0032/paper")
DATASETS_ROOT = PAPER_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = DATASETS_ROOT / "locomo_style_eval"

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Sidecars:
    fact_ledger: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    qa_audit: list[dict[str, Any]] = field(default_factory=list)
    hash_check: list[dict[str, Any]] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)


@dataclass
class BuildResult:
    name: str
    samples: list[dict[str, Any]]
    sidecars: Sidecars


def dia_fact_id(sample_id: str, dia_id: str) -> str:
    return f"{sample_id}_turn_{dia_id.replace(':', '_')}"


def minimal_normalize(text: Any) -> str:
    """Preserve source text while removing illegal controls and normalizing EOL."""
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = CTRL_RE.sub("", value)
    return unicodedata.normalize("NFC", value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_speaker_line(raw_line: str, default_speaker: str) -> tuple[str, str]:
    line = minimal_normalize(raw_line).strip()
    if ":" in line:
        speaker, text = line.split(":", 1)
        return speaker.strip() or default_speaker, text.strip()
    if "：" in line:
        speaker, text = line.split("：", 1)
        return speaker.strip() or default_speaker, text.strip()
    return default_speaker, line


def turn_record(speaker: str, dia_id: str, text: str) -> dict[str, str]:
    return {"speaker": speaker, "dia_id": dia_id, "text": text}


def locomo_session_datetime(session_num: int, sample_idx: int = 0) -> str:
    dt = datetime(2023, 5, 1, 1, 0) + timedelta(days=(session_num - 1) * 7 + sample_idx, hours=session_num % 11)
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{dt.strftime('%M')} {dt.strftime('%p')} on {dt.day} {dt.strftime('%B')}, {dt.year}"


def add_provenance(
    sidecars: Sidecars,
    *,
    source_dataset: str,
    sample_id: str,
    dia_id: str,
    text: str,
    source_origin: str,
    source_file: str,
    source_record_id: str,
    source_turn_id: str,
    session_id: str,
    turn_index: int,
    order_policy: str = "source_order",
    grounded_in_fact_ids: list[str] | None = None,
    source_speaker: str | None = None,
) -> None:
    text_hash = sha256_text(text)
    source_fact_ids = list(grounded_in_fact_ids or [])
    if source_origin == "original_turn" and text:
        turn_fact_id = dia_fact_id(sample_id, dia_id)
        source_fact_ids.append(turn_fact_id)
        sidecars.fact_ledger.append(
            {
                "source_dataset": source_dataset,
                "sample_id": sample_id,
                "fact_id": turn_fact_id,
                "source_type": "original_turn",
                "source_text": text,
                "source_id": source_turn_id,
                "field": "turn.text",
                "source_hash": text_hash,
                "dia_id": dia_id,
            }
        )
    sidecars.provenance.append(
        {
            "source_dataset": source_dataset,
            "sample_id": sample_id,
            "dia_id": dia_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "source_origin": source_origin,
            "source_file": source_file,
            "source_record_id": source_record_id,
            "source_turn_id": source_turn_id,
            "source_fact_id": source_fact_ids[0] if source_fact_ids else None,
            "source_fact_ids": source_fact_ids,
            "raw_text_hash": text_hash,
            "text": text,
            "order_policy": order_policy,
            "grounded_in_fact_ids": source_fact_ids,
            "source_speaker": source_speaker,
        }
    )
    if source_origin == "original_turn":
        sidecars.hash_check.append(
            {
                "source_dataset": source_dataset,
                "sample_id": sample_id,
                "dia_id": dia_id,
                "source_turn_id": source_turn_id,
                "raw_text_hash": text_hash,
                "status": "captured_for_recheck",
            }
        )


def add_fact(
    sidecars: Sidecars,
    *,
    source_dataset: str,
    sample_id: str,
    fact_id: str,
    source_type: str,
    source_text: str,
    source_id: str,
    field: str | None = None,
) -> None:
    sidecars.fact_ledger.append(
        {
            "source_dataset": source_dataset,
            "sample_id": sample_id,
            "fact_id": fact_id,
            "source_type": source_type,
            "source_text": minimal_normalize(source_text),
            "source_id": source_id,
            "field": field,
            "source_hash": sha256_text(minimal_normalize(source_text)),
        }
    )


def qa_record(
    *,
    question: str,
    answer: str | None,
    category: int,
    evidence: list[str],
    adversarial_answer: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question": question,
        "category": category,
        "evidence": evidence,
    }
    if category == 5:
        row["adversarial_answer"] = adversarial_answer or "No supporting evidence in the conversation."
    else:
        row["answer"] = answer
    return row


def add_qa_audit(
    sidecars: Sidecars,
    *,
    source_dataset: str,
    sample_id: str,
    qa_idx: int,
    qa: dict[str, Any],
    answer_facts: list[dict[str, Any]],
    evidence_detail: list[dict[str, Any]],
    qa_set: str = "locomo_style_main",
    negative_evidence: list[str] | None = None,
    adversarial_reason: str | None = None,
) -> None:
    category = int(qa.get("category") or 0)
    evidence = list(qa.get("evidence", []))
    evidence_sessions = {
        str(ev).split(":", 1)[0]
        for ev in evidence
        if isinstance(ev, str) and ":" in ev
    }
    whether_cross_session = len(evidence_sessions) > 1
    if category == 5:
        question_type = "adversarial"
    elif category == 3:
        question_type = "temporal"
    elif category == 2:
        question_type = "multi-hop"
    elif category == 4:
        question_type = "commonsense"
    else:
        question_type = "single-hop"
    difficulty = "hard" if whether_cross_session and category in {2, 3, 4} else "medium" if category in {2, 3, 5} else "easy"
    sidecars.qa_audit.append(
        {
            "source_dataset": source_dataset,
            "sample_id": sample_id,
            "qa_idx": qa_idx,
            "qa_set": qa_set,
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "category": category,
            "question_type": question_type,
            "difficulty": difficulty,
            "whether_cross_session": whether_cross_session,
            "evidence": evidence,
            "answer_facts": answer_facts,
            "evidence_detail": evidence_detail,
            "negative_evidence": negative_evidence or [],
            "adversarial_reason": adversarial_reason,
            "verifier_status": "heuristic_seed_supported" if qa.get("category") != 5 else "heuristic_adversarial_seed",
            "human_audit_status": "not_started",
        }
    )


def empty_summary_dict(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    conv = sample["conversation"]
    observation: dict[str, Any] = {}
    session_summary: dict[str, str] = {}
    event_summary: dict[str, Any] = {}
    for key in conv:
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(conv[key], list):
            observation[f"{key}_observation"] = []
            session_summary[f"{key}_summary"] = ""
            event_summary[f"events_{key}"] = []
    return observation, session_summary, event_summary


def perltqa_by_person(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.exists():
        return result
    for row in load_json(path):
        if isinstance(row, dict):
            for person, value in row.items():
                result[person] = value
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


def build_perltqa(max_samples: int) -> BuildResult:
    source = "PerLTQA"
    sidecars = Sidecars()
    mem_path = DATASETS_ROOT / "PerLTQA" / "Dataset" / "zh" / "perltmem.json"
    qa_path = DATASETS_ROOT / "PerLTQA" / "Dataset" / "zh" / "perltqa.json"
    memories = load_json(mem_path)
    qas_by_person = perltqa_by_person(qa_path)
    samples: list[dict[str, Any]] = []

    def perltqa_record_strength(index_and_item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        raw_idx, source_item = index_and_item
        usable_dialogues = 0
        turn_count = 0
        for dialogue in (source_item.get("dialogues") or {}).values():
            contents = dialogue.get("contents") or {}
            if contents:
                usable_dialogues += 1
                for lines in contents.values():
                    turn_count += len(lines if isinstance(lines, list) else [lines])
        return (usable_dialogues, turn_count, -raw_idx)

    selected_memories = sorted(enumerate(memories), key=perltqa_record_strength, reverse=True)[:max_samples]

    for sample_idx, (raw_idx, item) in enumerate(selected_memories):
        profile = item.get("profile", {})
        protagonist = str(profile.get("Protagonist") or f"PerLTQA_{sample_idx:04d}")
        sample_id = f"perltqa_{raw_idx:04d}"
        other_speaker = "其他参与者"
        conv: dict[str, Any] = {"speaker_a": protagonist, "speaker_b": other_speaker}
        qa: list[dict[str, Any]] = []
        session_anchor: dict[str, str] = {}
        session_fact: dict[str, str] = {}

        for field_name, value in profile.items():
            fact_id = f"{sample_id}_profile_{field_name}"
            add_fact(
                sidecars,
                source_dataset=source,
                sample_id=sample_id,
                fact_id=fact_id,
                source_type="original_persona",
                source_text=f"{field_name}: {value}",
                source_id=f"profile.{field_name}",
                field=field_name,
            )

        for rel_id, relationship in sorted((item.get("social_relationship") or {}).items()):
            if not isinstance(relationship, dict):
                continue
            rel_text = json.dumps(relationship, ensure_ascii=False, sort_keys=True)
            add_fact(
                sidecars,
                source_dataset=source,
                sample_id=sample_id,
                fact_id=f"{sample_id}_relationship_{rel_id}",
                source_type="original_relationship",
                source_text=rel_text,
                source_id=f"social_relationship.{rel_id}",
                field="social_relationship",
            )

        for qa_idx, row in enumerate(iter_perltqa_source_qas(qas_by_person.get(protagonist, {}))):
            question = minimal_normalize(row.get("Question", "")).strip()
            answer = minimal_normalize(row.get("Answer", "")).strip()
            if not question or not answer:
                continue
            reference = minimal_normalize(row.get("Reference Memory", "")).strip()
            source_text = f"Question: {question}\nAnswer: {answer}"
            if reference:
                source_text += f"\nReference Memory: {reference}"
            add_fact(
                sidecars,
                source_dataset=source,
                sample_id=sample_id,
                fact_id=f"{sample_id}_original_qa_{qa_idx}",
                source_type="original_qa",
                source_text=source_text,
                source_id=f"{protagonist}.original_qa.{qa_idx}",
                field="perltqa.original_qa",
            )

        dialogues = item.get("dialogues", {})
        sorted_dialogues = sorted(
            dialogues.items(),
            key=lambda kv: int(str(kv[0]).rsplit("#", 1)[-1]) if "#" in str(kv[0]) and str(kv[0]).rsplit("#", 1)[-1].isdigit() else 0,
        )
        session_num = 0
        for dialogue_key, dialogue in sorted_dialogues:
            event_id = str(dialogue.get("events") or dialogue_key.split("#", 1)[0])
            event = (item.get("events") or {}).get(event_id, {})
            fact_id = f"{sample_id}_event_{event_id}"
            event_text = minimal_normalize(event.get("content") or event.get("summary") or "")
            if event_text:
                add_fact(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    fact_id=fact_id,
                    source_type="original_event",
                    source_text=event_text,
                    source_id=event_id,
                    field="events.content",
                )
            date_to_lines = dialogue.get("contents") or {}
            if not isinstance(date_to_lines, dict) or not date_to_lines:
                continue
            session_num += 1
            session_key = f"session_{session_num}"
            conv[f"{session_key}_date_time"] = locomo_session_datetime(session_num, sample_idx)
            turns: list[dict[str, str]] = []

            turn_counter = 0
            for date_label, lines in date_to_lines.items():
                for line_idx, raw_line in enumerate(lines if isinstance(lines, list) else [lines], start=1):
                    speaker, text = parse_speaker_line(str(raw_line), protagonist)
                    if not text:
                        continue
                    raw_speaker = speaker
                    loader_speaker = protagonist if speaker == protagonist else other_speaker
                    turn_counter += 1
                    dia_id = f"D{session_num}:{turn_counter}"
                    turns.append(turn_record(loader_speaker, dia_id, text))
                    add_provenance(
                        sidecars,
                        source_dataset=source,
                        sample_id=sample_id,
                        dia_id=dia_id,
                        text=text,
                        source_origin="original_turn",
                        source_file=str(mem_path.relative_to(PAPER_ROOT)),
                        source_record_id=protagonist,
                        source_turn_id=f"{dialogue_key}:{date_label}:{line_idx}",
                        session_id=session_key,
                        turn_index=turn_counter,
                        source_speaker=raw_speaker,
                    )
            if turns:
                conv[session_key] = turns

        qa = add_basic_turn_qas(
            sidecars,
            source_dataset=source,
            sample_id=sample_id,
            conv=conv,
            language_label="Chinese PerLTQA",
            max_qas=34,
        )

        sample = {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "zh",
            "split": "eval",
            "conversation": conv,
            "qa": qa[:40],
        }
        observation, session_summary, event_summary = empty_summary_dict(sample)
        sample["observation"] = observation
        sample["session_summary"] = session_summary
        sample["event_summary"] = event_summary
        samples.append(sample)

    sidecars.reports.append(
        f"PerLTQA PlanMode D bootstrap: selected {len(samples)} perltmem records with the most usable dialogue/event anchors; "
        "memory_anchor_turns are used for original event facts; profile, social_relationship, events, dialogues, and original QA are recorded in the fact ledger; original PerLTQA QA is not copied into final eval; "
        "non-protagonist source speakers are mapped to speaker_b in primary JSON and preserved as source_speaker in provenance."
    )
    return BuildResult("PerLTQA-LoCoMo-style-eval", samples, sidecars)


def interleave_speaker_lines(a_lines: list[str], b_lines: list[str]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    max_len = max(len(a_lines), len(b_lines))
    for idx in range(max_len):
        if idx < len(a_lines):
            rows.append(("persona", idx + 1, a_lines[idx]))
        if idx < len(b_lines):
            rows.append(("user", idx + 1, b_lines[idx]))
    return rows


def build_opela(max_samples: int) -> BuildResult:
    source = "OPELA"
    sidecars = Sidecars()
    csv_path = DATASETS_ROOT / "OPELA" / "data" / "oplea_open_data.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("total_turn") or 0), reverse=True)
    samples: list[dict[str, Any]] = []

    for sample_idx, row in enumerate(rows[:max_samples]):
        doc_id = row["doc_id"]
        sample_id = f"opela_{sample_idx:04d}"
        persona_name = row.get("persona_name_original") or "persona"
        user_name = f"user_{row.get('user_id', sample_idx)}"
        persona_lines = [minimal_normalize(x).strip() for x in row.get("persona_text_all", "").splitlines() if x.strip()]
        user_lines = [minimal_normalize(x).strip() for x in row.get("user_text_all", "").splitlines() if x.strip()]
        turns_source = interleave_speaker_lines(persona_lines, user_lines)
        try:
            expected_turns = int(row.get("total_turn") or 0)
        except ValueError:
            expected_turns = 0
        if expected_turns > 0 and len(turns_source) > expected_turns:
            turns_source = turns_source[:expected_turns]
        session_count = max(1, min(8, int(row.get("pause_count") or 0) + 1))
        chunk_size = max(8, (len(turns_source) + session_count - 1) // session_count)
        conv: dict[str, Any] = {"speaker_a": persona_name, "speaker_b": user_name}
        first_turns_by_speaker: dict[str, str] = {}

        for session_num, offset in enumerate(range(0, len(turns_source), chunk_size), start=1):
            session_key = f"session_{session_num}"
            conv[f"{session_key}_date_time"] = locomo_session_datetime(session_num, sample_idx)
            session_turns: list[dict[str, str]] = []
            for local_idx, (speaker_type, source_idx, text) in enumerate(turns_source[offset : offset + chunk_size], start=1):
                speaker = persona_name if speaker_type == "persona" else user_name
                dia_id = f"D{session_num}:{local_idx}"
                session_turns.append(turn_record(speaker, dia_id, text))
                first_turns_by_speaker.setdefault(speaker, dia_id)
                add_provenance(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    dia_id=dia_id,
                    text=text,
                    source_origin="original_turn",
                    source_file=str(csv_path.relative_to(PAPER_ROOT)),
                    source_record_id=doc_id,
                    source_turn_id=f"{speaker_type}_text_all:{source_idx}",
                    session_id=session_key,
                    turn_index=local_idx,
                    order_policy="alternating_reconstruction_from_aggregated_columns",
                    source_speaker=speaker,
                )
            if session_turns:
                conv[session_key] = session_turns

        for field_name in ("persona_summary", "user_summary"):
            if row.get(field_name):
                add_fact(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    fact_id=f"{sample_id}_{field_name}",
                    source_type="original_memory",
                    source_text=row[field_name],
                    source_id=f"{doc_id}.{field_name}",
                    field=field_name,
                )

        qa = add_basic_turn_qas(
            sidecars,
            source_dataset=source,
            sample_id=sample_id,
            conv=conv,
            language_label="Korean OPELA",
        )

        sample = {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "ko",
            "split": "eval",
            "conversation": conv,
            "qa": qa,
        }
        observation, session_summary, event_summary = empty_summary_dict(sample)
        sample["observation"] = observation
        sample["session_summary"] = session_summary
        sample["event_summary"] = event_summary
        samples.append(sample)

    sidecars.reports.append(
        f"OPELA PlanMode C bootstrap: selected top {len(samples)} rows by total_turn; "
        "turn order is reconstructed from aggregated per-speaker text columns and recorded in provenance."
    )
    return BuildResult("OPELA-LoCoMo-style-eval", samples, sidecars)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sorted_session_keys(conv: dict[str, Any]) -> list[str]:
    def key_num(key: str) -> int:
        suffix = key.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    return sorted(
        [key for key, value in conv.items() if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)],
        key=key_num,
    )


def compact_fragment(text: str, limit: int = 180) -> str:
    value = " ".join(minimal_normalize(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def qa_answer_fragment(text: str, limit: int = 96) -> str:
    return compact_fragment(text, limit)


def turn_hint(text: str, limit: int = 40) -> str:
    value = compact_fragment(text, limit)
    return value.replace('"', "'")


def is_contentful_turn(text: str) -> bool:
    value = " ".join(minimal_normalize(text).split())
    if len(value) < 8:
        return False
    if value.startswith("记忆锚点："):
        return False
    lowered = value.lower()
    trivial = {
        "hello",
        "hi",
        "안녕",
        "안녕하세요",
        "こんばんは",
        "こんにちは",
        "おはようございます",
        "hallo",
        "guten tag",
        "네",
        "응",
        "はい",
        "ㅋㅋㅋ",
        "ㅎㅎㅎ",
    }
    return lowered not in trivial and value not in trivial


def dia_session_label(dia_id: str) -> str:
    session = str(dia_id).split(":", 1)[0].removeprefix("D")
    return session if session.isdigit() else "?"


def conversation_act_answer(speaker: str, text: str) -> str:
    lowered = text.lower()
    question_markers = ("?", "？", "吗", "呢", "어", "까", "か")
    greeting_markers = ("hello", "hi", "안녕", "こんばんは", "おはよう", "こんにちは", "hallo", "guten")
    if any(marker in text for marker in question_markers):
        return "asking a question"
    elif any(marker in lowered or marker in text for marker in greeting_markers):
        return "greeting or opening the conversation"
    return "sharing information in the conversation"


def flatten_turns(conv: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_key in sorted_session_keys(conv):
        session_num = int(session_key.rsplit("_", 1)[-1])
        date = conv.get(f"{session_key}_date_time", "")
        for turn in conv[session_key]:
            text = compact_fragment(turn.get("text", ""), 240)
            if not text:
                continue
            rows.append(
                {
                    "session_key": session_key,
                    "session_num": session_num,
                    "date": date,
                    "speaker": turn.get("speaker", ""),
                    "dia_id": turn.get("dia_id", ""),
                    "text": text,
                }
            )
    return rows


def diverse_turns(turns: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(turns) <= limit:
        return turns
    picked: list[dict[str, Any]] = []
    seen_dia: set[str] = set()
    session_firsts: dict[int, dict[str, Any]] = {}
    session_lasts: dict[int, dict[str, Any]] = {}
    for turn in turns:
        session_firsts.setdefault(int(turn["session_num"]), turn)
        session_lasts[int(turn["session_num"])] = turn
    for candidate in list(session_firsts.values()) + list(session_lasts.values()):
        if candidate["dia_id"] not in seen_dia:
            picked.append(candidate)
            seen_dia.add(candidate["dia_id"])
        if len(picked) >= limit:
            return picked
    step = max(1, len(turns) // max(1, limit))
    for idx in range(0, len(turns), step):
        candidate = turns[idx]
        if candidate["dia_id"] not in seen_dia:
            picked.append(candidate)
            seen_dia.add(candidate["dia_id"])
        if len(picked) >= limit:
            break
    return picked[:limit]


def first_content_turns_by_session(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for turn in turns:
        session_num = int(turn["session_num"])
        if session_num not in seen:
            picked.append(turn)
            seen.add(session_num)
    return picked


def last_content_turns_by_session(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked_reversed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for turn in reversed(turns):
        session_num = int(turn["session_num"])
        if session_num not in seen:
            picked_reversed.append(turn)
            seen.add(session_num)
    return list(reversed(picked_reversed))


def add_seed_qa(
    sidecars: Sidecars,
    qas: list[dict[str, Any]],
    *,
    source_dataset: str,
    sample_id: str,
    qa: dict[str, Any],
    source_origin: str = "original_turn",
    source_fact_ids: list[str | None] | None = None,
    negative_evidence: list[str] | None = None,
    adversarial_reason: str | None = None,
) -> None:
    evidence = list(qa.get("evidence", []))
    fact_ids = source_fact_ids or (
        [dia_fact_id(sample_id, ev) for ev in evidence] if source_origin == "original_turn" else [None for _ in evidence]
    )
    qas.append(qa)
    answer_facts = []
    if qa.get("category") != 5:
        answer_facts = [
            {
                "fact": qa.get("answer", ""),
                "supported_by": evidence,
                "source_fact_id": fact_id,
            }
            for fact_id in fact_ids
        ]
    add_qa_audit(
        sidecars,
        source_dataset=source_dataset,
        sample_id=sample_id,
        qa_idx=len(qas) - 1,
        qa=qa,
        answer_facts=answer_facts,
        evidence_detail=[
            {
                "dia_id": ev,
                "source_origin": source_origin,
                "supports_answer_fact": [fact_id for fact_id in fact_ids if fact_id],
            }
            for ev in evidence
        ],
        negative_evidence=negative_evidence,
        adversarial_reason=adversarial_reason,
    )


def add_basic_turn_qas(
    sidecars: Sidecars,
    *,
    source_dataset: str,
    sample_id: str,
    conv: dict[str, Any],
    language_label: str,
    max_qas: int = 22,
) -> list[dict[str, Any]]:
    qas: list[dict[str, Any]] = []
    all_turns = flatten_turns(conv)
    content_turns = [turn for turn in all_turns if is_contentful_turn(turn["text"])] or all_turns
    selected = diverse_turns(content_turns, min(10, max_qas))
    used_questions: set[str] = set()
    speaker_a = str(conv.get("speaker_a", "speaker A"))
    speaker_b = str(conv.get("speaker_b", "speaker B"))

    for turn in selected:
        question = (
            f"In session {turn['session_num']} of the {language_label} conversation, "
            f"what did {turn['speaker']} say in the turn about \"{turn_hint(turn['text'])}\"?"
        )
        if question in used_questions:
            continue
        used_questions.add(question)
        answer = qa_answer_fragment(turn["text"])
        qa = qa_record(question=question, answer=answer, category=1, evidence=[turn["dia_id"]])
        add_seed_qa(
            sidecars,
            qas,
            source_dataset=source_dataset,
            sample_id=sample_id,
            qa=qa,
        )
        if len(qas) >= max_qas:
            return qas

    session_firsts = first_content_turns_by_session(content_turns)
    session_lasts = last_content_turns_by_session(content_turns)

    for left, right in zip(session_firsts, session_lasts[1:]):
        if left["session_num"] == right["session_num"]:
            continue
        qa = qa_record(
            question=(
                f"In the {language_label} conversation, what did {left['speaker']} say in session {left['session_num']} "
                f"and what did {right['speaker']} say in session {right['session_num']}?"
            ),
            answer=f"Session {left['session_num']}: {qa_answer_fragment(left['text'], 72)} Session {right['session_num']}: {qa_answer_fragment(right['text'], 72)}",
            category=2,
            evidence=[left["dia_id"], right["dia_id"]],
        )
        add_seed_qa(
            sidecars,
            qas,
            source_dataset=source_dataset,
            sample_id=sample_id,
            qa=qa,
        )
        if len(qas) >= max_qas - 6:
            break

    for left, right in zip(session_firsts, session_firsts[1:]):
        qa = qa_record(
            question=(
                f"Which statement came earlier in the {language_label} conversation: "
                f"{left['speaker']}'s session {left['session_num']} statement or {right['speaker']}'s session {right['session_num']} statement?"
            ),
            answer=f"Earlier session {left['session_num']}: {qa_answer_fragment(left['text'], 72)}",
            category=3,
            evidence=[left["dia_id"], right["dia_id"]],
        )
        add_seed_qa(
            sidecars,
            qas,
            source_dataset=source_dataset,
            sample_id=sample_id,
            qa=qa,
        )
        if len(qas) >= max_qas - 4:
            break

    for turn in diverse_turns(content_turns, 2):
        if len(qas) >= max_qas - 2:
            break
        qa = qa_record(
            question=(
                f"What general conversational act is shown by {turn['speaker']}'s cited turn "
                f"in the {language_label} conversation?"
            ),
            answer=conversation_act_answer(turn["speaker"], turn["text"]),
            category=4,
            evidence=[turn["dia_id"]],
        )
        add_seed_qa(
            sidecars,
            qas,
            source_dataset=source_dataset,
            sample_id=sample_id,
            qa=qa,
        )

    if all_turns:
        adversarial_templates = [
            (
                f"Does the {language_label} conversation provide evidence that {speaker_a} or {speaker_b} won an Olympic gold medal?",
                "unsupported_fact",
            ),
            (
                f"Does the {language_label} conversation explicitly say {speaker_a} or {speaker_b} moved to Mars?",
                "unsupported_fact",
            ),
        ]
        for question, reason in adversarial_templates:
            if len(qas) >= max_qas:
                break
            qa = qa_record(
                question=question,
                answer=None,
                category=5,
                evidence=[],
            )
            add_seed_qa(
                sidecars,
                qas,
                source_dataset=source_dataset,
                sample_id=sample_id,
                qa=qa,
                negative_evidence=[all_turns[0]["dia_id"], all_turns[-1]["dia_id"]],
                adversarial_reason=reason,
            )

    for turn in content_turns:
        if len(qas) >= max_qas:
            break
        question = (
            f"What was stated by {turn['speaker']} at evidence turn {turn['dia_id']} "
            f"in the {language_label} conversation?"
        )
        if question in used_questions:
            continue
        used_questions.add(question)
        qa = qa_record(question=question, answer=qa_answer_fragment(turn["text"]), category=1, evidence=[turn["dia_id"]])
        add_seed_qa(
            sidecars,
            qas,
            source_dataset=source_dataset,
            sample_id=sample_id,
            qa=qa,
        )
    return qas


def build_jlongchat(lac_samples: int, jmsc_samples: int) -> BuildResult:
    source = "JLongChat"
    sidecars = Sidecars()
    samples: list[dict[str, Any]] = []

    lac_path = DATASETS_ROOT / "japanese-long-term-chat" / "utf8" / "lac-public-dialogue.tsv"
    lac_rows = read_tsv(lac_path)
    by_room: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in lac_rows:
        by_room[row["room"]].append(row)
    for room_idx, (room, rows) in enumerate(sorted(by_room.items())[:lac_samples]):
        sample_id = f"jlongchat_lac_{room_idx:04d}"
        raw_speakers = [
            row["speaker"].strip("[]").replace("SPK--", "")
            for row in rows
            if row.get("speaker")
        ]
        speaker_counts = Counter(raw_speakers)
        ranked_speakers = [speaker for speaker, _ in speaker_counts.most_common()]
        if len(ranked_speakers) <= 2:
            speaker_a = ranked_speakers[0] if ranked_speakers else "TEXTCHAT01"
            speaker_b = ranked_speakers[1] if len(ranked_speakers) > 1 else "TEXTCHAT02"
            speaker_map = {speaker_a: speaker_a, speaker_b: speaker_b}
        else:
            speaker_a = ranked_speakers[0]
            speaker_b = "Other participants"
            speaker_map = {speaker_a: speaker_a}
        conv: dict[str, Any] = {"speaker_a": speaker_a, "speaker_b": speaker_b}
        by_day: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_day[int(row["dayid"])].append(row)
        for session_num, day in enumerate(sorted(by_day), start=1):
            session_key = f"session_{session_num}"
            conv[f"{session_key}_date_time"] = locomo_session_datetime(session_num, room_idx)
            session_turns: list[dict[str, str]] = []
            for local_idx, row in enumerate(by_day[day], start=1):
                text = minimal_normalize(row["utterance"]).strip()
                raw_speaker = row["speaker"].strip("[]").replace("SPK--", "")
                speaker = speaker_map.get(raw_speaker, speaker_b)
                dia_id = f"D{session_num}:{local_idx}"
                session_turns.append(turn_record(speaker, dia_id, text))
                add_provenance(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    dia_id=dia_id,
                    text=text,
                    source_origin="original_turn",
                    source_file=str(lac_path.relative_to(PAPER_ROOT)),
                    source_record_id=room,
                    source_turn_id=f"{room}:day{day}:row{local_idx}",
                    session_id=session_key,
                    turn_index=local_idx,
                    source_speaker=raw_speaker,
                )
            conv[session_key] = session_turns
        sample = {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "ja",
            "split": "eval",
            "conversation": conv,
            "qa": [],
        }
        sample["qa"] = add_basic_turn_qas(sidecars, source_dataset=source, sample_id=sample_id, conv=conv, language_label="Japanese LAC")
        observation, session_summary, event_summary = empty_summary_dict(sample)
        sample["observation"] = observation
        sample["session_summary"] = session_summary
        sample["event_summary"] = event_summary
        samples.append(sample)

    jmsc_path = DATASETS_ROOT / "japanese-long-term-chat" / "utf8" / "jmsc-public-dialogue.tsv"
    jmsc_rows = read_tsv(jmsc_path)
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in jmsc_rows:
        by_pair[row["pair_id"]].append(row)
    ranked_pairs = sorted(by_pair.items(), key=lambda kv: len({row["sid"] for row in kv[1]}), reverse=True)
    for pair_idx, (pair_id, rows) in enumerate(ranked_pairs[:jmsc_samples]):
        sample_id = f"jlongchat_jmsc_{pair_idx:04d}"
        speakers = sorted({row["speaker"] for row in rows})
        conv = {
            "speaker_a": speakers[0] if speakers else f"{pair_id}-A",
            "speaker_b": speakers[1] if len(speakers) > 1 else f"{pair_id}-B",
        }
        by_sid: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_sid[int(row["sid"])].append(row)
        for session_num, sid in enumerate(sorted(by_sid), start=1):
            session_key = f"session_{session_num}"
            conv[f"{session_key}_date_time"] = locomo_session_datetime(session_num, pair_idx + lac_samples)
            session_turns = []
            for row in sorted(by_sid[sid], key=lambda r: int(r["tid"])):
                local_idx = int(row["tid"])
                text = minimal_normalize(row["utt"]).strip()
                dia_id = f"D{session_num}:{local_idx}"
                session_turns.append(turn_record(row["speaker"], dia_id, text))
                add_provenance(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    dia_id=dia_id,
                    text=text,
                    source_origin="original_turn",
                    source_file=str(jmsc_path.relative_to(PAPER_ROOT)),
                    source_record_id=pair_id,
                    source_turn_id=f"{pair_id}:sid{sid}:tid{local_idx}",
                    session_id=session_key,
                    turn_index=local_idx,
                    source_speaker=row["speaker"],
                )
                if row.get("sum"):
                    add_fact(
                        sidecars,
                        source_dataset=source,
                        sample_id=sample_id,
                        fact_id=f"{sample_id}_summary_sid{sid}_tid{local_idx}",
                        source_type="original_memory",
                        source_text=row["sum"],
                        source_id=f"{pair_id}:sid{sid}:tid{local_idx}:sum",
                        field="sum",
                    )
            conv[session_key] = session_turns
        sample = {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "ja",
            "split": "eval",
            "conversation": conv,
            "qa": [],
        }
        sample["qa"] = add_basic_turn_qas(sidecars, source_dataset=source, sample_id=sample_id, conv=conv, language_label="Japanese JMSC")
        observation, session_summary, event_summary = empty_summary_dict(sample)
        sample["observation"] = observation
        sample["session_summary"] = session_summary
        sample["event_summary"] = event_summary
        samples.append(sample)

    sidecars.reports.append(
        f"JLongChat PlanMode A/B bootstrap: {lac_samples} LAC rooms plus {jmsc_samples} JMSC pairs selected; "
        "no Japanese source text was translated or polished; multi-party LAC rooms are reduced to two loader speakers "
        "in primary JSON while raw source speakers are preserved as source_speaker in provenance."
    )
    return BuildResult("JLongChat-LoCoMo-style-eval", samples, sidecars)


def xml_attr(element: ET.Element, name: str) -> str:
    if name in element.attrib:
        return element.attrib[name]
    xml_name = f"{{http://www.w3.org/XML/1998/namespace}}{name}"
    return element.attrib.get(xml_name, "")


def build_del1l2im() -> BuildResult:
    source = "deL1L2IM"
    sidecars = Sidecars()
    base = DATASETS_ROOT / "deL1L2IM" / "extracted" / "transformation" / "Tei-P5" / "teip5-chat"
    xml_paths = sorted(base.glob("Chat-*.xml"))
    samples: list[dict[str, Any]] = []
    msg_ns = "{http://www.example.org/ns/nonTEI}message"

    for sample_idx, xml_path in enumerate(xml_paths):
        sample_id = f"del1l2im_{sample_idx:04d}"
        root = ET.parse(xml_path).getroot()
        messages = [el for el in root.iter() if el.tag == msg_ns or el.tag.split("}")[-1] == "message"]
        speakers = []
        for msg in messages:
            who = msg.attrib.get("who", "speaker")
            if who not in speakers:
                speakers.append(who)
        conv: dict[str, Any] = {
            "speaker_a": speakers[0].split("#")[-1] if speakers else "speaker_a",
            "speaker_b": speakers[1].split("#")[-1] if len(speakers) > 1 else "speaker_b",
        }
        by_date: dict[str, list[ET.Element]] = defaultdict(list)
        for msg in messages:
            timestamp = msg.attrib.get("timestamp", "")
            date = timestamp.split("T", 1)[0] if "T" in timestamp else f"session_{len(by_date) + 1}"
            by_date[date].append(msg)
        for session_num, date in enumerate(sorted(by_date), start=1):
            session_key = f"session_{session_num}"
            conv[f"{session_key}_date_time"] = locomo_session_datetime(session_num, sample_idx)
            session_turns = []
            for local_idx, msg in enumerate(by_date[date], start=1):
                text = minimal_normalize(" ".join("".join(msg.itertext()).split())).strip()
                if not text:
                    continue
                speaker = msg.attrib.get("who", "speaker").split("#")[-1]
                dia_id = f"D{session_num}:{local_idx}"
                session_turns.append(turn_record(speaker, dia_id, text))
                add_provenance(
                    sidecars,
                    source_dataset=source,
                    sample_id=sample_id,
                    dia_id=dia_id,
                    text=text,
                    source_origin="original_turn",
                    source_file=str(xml_path.relative_to(PAPER_ROOT)),
                    source_record_id=xml_path.stem,
                    source_turn_id=xml_attr(msg, "id") or f"{xml_path.stem}:{date}:{local_idx}",
                    session_id=session_key,
                    turn_index=local_idx,
                    source_speaker=speaker,
                )
            if session_turns:
                conv[session_key] = session_turns
        sample = {
            "sample_id": sample_id,
            "source_dataset": source,
            "language": "de",
            "split": "eval",
            "conversation": conv,
            "qa": [],
        }
        sample["qa"] = add_basic_turn_qas(sidecars, source_dataset=source, sample_id=sample_id, conv=conv, language_label="German deL1L2IM")
        observation, session_summary, event_summary = empty_summary_dict(sample)
        sample["observation"] = observation
        sample["session_summary"] = session_summary
        sample["event_summary"] = event_summary
        samples.append(sample)

    sidecars.reports.append(
        f"deL1L2IM PlanMode A bootstrap: converted {len(samples)} TEI chat XML files without synthetic turns."
    )
    return BuildResult("deL1L2IM-LoCoMo-style-eval", samples, sidecars)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    session_counts = []
    turn_counts = []
    qa_counts = []
    categories: Counter[str] = Counter()
    for sample in samples:
        conv = sample.get("conversation", {})
        sessions = [key for key, value in conv.items() if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)]
        session_counts.append(len(sessions))
        turn_counts.append(sum(len(conv[key]) for key in sessions))
        qa_counts.append(len(sample.get("qa", [])))
        categories.update(str(qa.get("category")) for qa in sample.get("qa", []))

    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
            "total": sum(values),
        }

    return {
        "samples": len(samples),
        "sessions": stats(session_counts),
        "turns": stats(turn_counts),
        "qa": stats(qa_counts),
        "categories": dict(sorted(categories.items())),
    }


def write_result(output_root: Path, result: BuildResult) -> Path:
    primary_dir = output_root / "primary"
    sidecar_dir = output_root / "sidecars" / result.name
    primary_path = primary_dir / f"{result.name}.json"
    dump_json(primary_path, result.samples)
    dump_jsonl(sidecar_dir / f"{result.name}_fact_ledger.jsonl", result.sidecars.fact_ledger)
    dump_jsonl(sidecar_dir / f"{result.name}_provenance.jsonl", result.sidecars.provenance)
    dump_jsonl(sidecar_dir / f"{result.name}_qa_audit.jsonl", result.sidecars.qa_audit)
    dump_jsonl(sidecar_dir / f"{result.name}_hash_check.jsonl", result.sidecars.hash_check)
    report = [
        f"# Construction Report: {result.name}",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Status",
        "",
        "Bootstrap harness artifact. Model-based expansion, source-entailment verification beyond heuristic seed checks, and human audit are not complete.",
        "",
        "Construction mode: no-model deterministic conversion/seed QA. Model calls: 0.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summarize_samples(result.samples), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    report.extend(f"- {line}" for line in result.sidecars.reports)
    (sidecar_dir / f"{result.name}_construction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return primary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--perltqa-samples", type=int, default=10)
    parser.add_argument("--opela-samples", type=int, default=10)
    parser.add_argument("--lac-samples", type=int, default=4)
    parser.add_argument("--jmsc-samples", type=int, default=6)
    parser.add_argument("--skip-combined", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        build_perltqa(args.perltqa_samples),
        build_opela(args.opela_samples),
        build_jlongchat(args.lac_samples, args.jmsc_samples),
        build_del1l2im(),
    ]
    primary_paths = []
    for result in results:
        path = write_result(args.output_root, result)
        primary_paths.append(path)
        print(f"[build] wrote {path} samples={len(result.samples)}")
    if not args.skip_combined:
        combined = []
        for result in results:
            combined.extend(result.samples)
        combined_path = args.output_root / "primary" / "multilingual_locomo_style_eval.json"
        dump_json(combined_path, combined)
        print(f"[build] wrote {combined_path} samples={len(combined)}")
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "bootstrap_harness_artifact_not_final_audited_release",
        "construction_mode": "no_model_deterministic_conversion_seed_qa",
        "model_calls": 0,
        "local_model_required_for_construction": False,
        "service_preflight_policy": "optional_tiny_chat_for_baseline_readiness_only_not_dataset_generation",
        "benchmark_claim": (
            "LoCoMo-loader-compatible source-faithful eval; not claimed to be LoCoMo-equivalent "
            "in data generation process or naturalness"
        ),
        "primary_files": [str(path.relative_to(PAPER_ROOT)) for path in primary_paths],
        "combined_file": None if args.skip_combined else str((args.output_root / "primary" / "multilingual_locomo_style_eval.json").relative_to(PAPER_ROOT)),
        "summaries": {result.name: summarize_samples(result.samples) for result in results},
    }
    dump_json(args.output_root / "manifest.json", manifest)
    print(f"[build] wrote {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
