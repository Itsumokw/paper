#!/usr/bin/env python3
"""Export a readable human-audit packet with evidence text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_keys(conversation: dict[str, Any]) -> list[str]:
    def key_num(key: str) -> int:
        suffix = key.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    return sorted(
        [
            key
            for key, value in conversation.items()
            if key.startswith("session_")
            and not key.endswith("_date_time")
            and isinstance(value, list)
        ],
        key=key_num,
    )


def build_turn_index(samples: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        conversation = sample.get("conversation", {})
        for session_key in session_keys(conversation):
            date = conversation.get(f"{session_key}_date_time", "")
            for turn in conversation[session_key]:
                dia_id = str(turn.get("dia_id"))
                index[(sample_id, dia_id)] = {
                    "dia_id": dia_id,
                    "session_id": session_key,
                    "date_time": date,
                    "speaker": turn.get("speaker"),
                    "text": turn.get("text"),
                }
    return index


def evidence_texts(row: dict[str, Any], turn_index: dict[tuple[str, str], dict[str, Any]], key: str) -> list[dict[str, Any]]:
    sample_id = str(row.get("sample_id"))
    output = []
    for dia_id in row.get(key, []) or []:
        output.append(turn_index.get((sample_id, str(dia_id)), {"dia_id": dia_id, "missing": True}))
    return output


def markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").strip()


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LoCoMo-style Human Audit Packet",
        "",
        "Decision values: `pass`, `fail`, `fix`, `delete`.",
        "",
        "Review criteria:",
        "",
        "- answer is supported by evidence;",
        "- evidence IDs point to the right turn(s);",
        "- category and cross-session/difficulty labels are reasonable;",
        "- cat5 has no answerable evidence and should be refused;",
        "- no answer-critical fact comes only from unsupported synthetic content.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row.get('source_dataset')} / {row.get('sample_id')} / QA {row.get('qa_idx')}",
                "",
                f"- Audit reasons: {', '.join(row.get('audit_reasons', []))}",
                f"- Category: {row.get('category')} ({row.get('question_type')})",
                f"- Difficulty: {row.get('difficulty')}",
                f"- Cross-session: {row.get('whether_cross_session')}",
                f"- Decision: `{row.get('human_decision', 'todo')}`",
                f"- Notes: {markdown_escape(row.get('human_notes', ''))}",
                "",
                f"Question: {markdown_escape(row.get('question'))}",
                "",
            ]
        )
        if row.get("category") == 5:
            lines.append(f"Adversarial unsupported answer: {markdown_escape(row.get('adversarial_answer'))}")
        else:
            lines.append(f"Answer: {markdown_escape(row.get('answer'))}")
        lines.append("")
        if row.get("answer_facts"):
            lines.append("Answer facts:")
            for fact in row["answer_facts"]:
                lines.append(f"- `{fact.get('source_fact_id')}`: {markdown_escape(fact.get('fact'))}")
            lines.append("")
        if row.get("evidence_detail"):
            lines.append("Evidence detail:")
            for detail in row["evidence_detail"]:
                lines.append(
                    f"- `{detail.get('dia_id')}` origin={markdown_escape(detail.get('source_origin'))} "
                    f"supports={markdown_escape(detail.get('supports_answer_fact', []))}"
                )
            lines.append("")
        if row.get("evidence_text"):
            lines.append("Evidence:")
            for ev in row["evidence_text"]:
                lines.append(
                    f"- `{ev.get('dia_id')}` {ev.get('session_id', '')} {ev.get('speaker', '')}: {markdown_escape(ev.get('text', ev))}"
                )
            lines.append("")
        if row.get("negative_evidence_text"):
            lines.append("Negative evidence:")
            for ev in row["negative_evidence_text"]:
                lines.append(
                    f"- `{ev.get('dia_id')}` {ev.get('session_id', '')} {ev.get('speaker', '')}: {markdown_escape(ev.get('text', ev))}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-json", type=Path, required=True)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    samples = load_json(args.primary_json)
    turn_index = build_turn_index(samples)
    rows = []
    for row in iter_jsonl(args.queue_jsonl):
        enriched = dict(row)
        enriched["evidence_text"] = evidence_texts(row, turn_index, "evidence")
        enriched["negative_evidence_text"] = evidence_texts(row, turn_index, "negative_evidence")
        rows.append(enriched)

    write_jsonl(args.output_jsonl, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "rows": len(rows),
                "output_jsonl": str(args.output_jsonl),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
