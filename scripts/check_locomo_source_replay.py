#!/usr/bin/env python3
"""Replay provenance pointers against raw source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAPER_ROOT = Path("/home/stu0032/paper")
DATASETS_ROOT = PAPER_ROOT / "datasets"

SOURCE_ARTIFACTS = {
    "PerLTQA": "PerLTQA-LoCoMo-style-eval",
    "OPELA": "OPELA-LoCoMo-style-eval",
    "JLongChat": "JLongChat-LoCoMo-style-eval",
    "deL1L2IM": "deL1L2IM-LoCoMo-style-eval",
}


def minimal_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return text


def parse_speaker_line(line: str, default_speaker: str) -> tuple[str, str]:
    if ":" in line:
        speaker, text = line.split(":", 1)
        return speaker.strip() or default_speaker, text.strip()
    if "：" in line:
        speaker, text = line.split("：", 1)
        return speaker.strip() or default_speaker, text.strip()
    return default_speaker, line


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


def resolve_source_file(source_file: str) -> Path | None:
    if not source_file:
        return None
    path = Path(source_file)
    if path.is_absolute():
        return path
    paper_relative = PAPER_ROOT / source_file
    if paper_relative.exists():
        return paper_relative
    dataset_relative = DATASETS_ROOT / source_file
    if dataset_relative.exists():
        return dataset_relative
    return paper_relative


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def xml_attr(element: ET.Element, name: str) -> str:
    if name in element.attrib:
        return element.attrib[name]
    xml_name = f"{{http://www.w3.org/XML/1998/namespace}}{name}"
    return element.attrib.get(xml_name, "")


class SourceReplay:
    def __init__(self) -> None:
        self._perltqa_by_protagonist: dict[str, dict[str, Any]] | None = None
        self._opela_by_doc_id: dict[str, dict[str, str]] | None = None
        self._lac_by_room_day: dict[tuple[str, int], list[dict[str, str]]] | None = None
        self._jmsc_by_pair_sid_tid: dict[tuple[str, int, int], dict[str, str]] | None = None
        self._del1l2im_messages: dict[str, dict[str, ET.Element]] = {}
        self._del1l2im_fallback: dict[str, dict[str, ET.Element]] = {}

    def replay(self, row: dict[str, Any]) -> tuple[str, str | None] | None:
        source = row.get("source_dataset")
        if source == "PerLTQA":
            return self.replay_perltqa(row)
        if source == "OPELA":
            return self.replay_opela(row)
        if source == "JLongChat":
            return self.replay_jlongchat(row)
        if source == "deL1L2IM":
            return self.replay_del1l2im(row)
        raise KeyError(f"unsupported source_dataset={source!r}")

    def perltqa_by_protagonist(self) -> dict[str, dict[str, Any]]:
        if self._perltqa_by_protagonist is None:
            path = DATASETS_ROOT / "PerLTQA" / "Dataset" / "zh" / "perltmem.json"
            self._perltqa_by_protagonist = {
                str(item.get("profile", {}).get("Protagonist")): item
                for item in load_json(path)
                if item.get("profile", {}).get("Protagonist")
            }
        return self._perltqa_by_protagonist

    def replay_perltqa(self, row: dict[str, Any]) -> tuple[str, str | None] | None:
        item = self.perltqa_by_protagonist()[str(row["source_record_id"])]
        source_turn_id = str(row["source_turn_id"])
        source_origin = str(row.get("source_origin"))
        if source_origin == "memory_anchor_turn":
            event = (item.get("events") or {}).get(source_turn_id)
            if event is None:
                raise KeyError(f"PerLTQA event not found: {source_turn_id}")
            event_text = minimal_normalize(event.get("content") or event.get("summary") or "")
            return f"记忆锚点：{event_text}", str(row.get("source_record_id"))

        raw_without_line, _, raw_line_idx = source_turn_id.rpartition(":")
        if not raw_without_line or not raw_line_idx.isdigit():
            raise ValueError(f"invalid PerLTQA source_turn_id={source_turn_id!r}")
        dialogue_key, sep, date_label = raw_without_line.partition(":")
        if not sep:
            raise ValueError(f"invalid PerLTQA source_turn_id={source_turn_id!r}")
        line_idx = int(raw_line_idx)
        dialogue = (item.get("dialogues") or {})[dialogue_key]
        lines = (dialogue.get("contents") or {})[date_label]
        raw_line = (lines if isinstance(lines, list) else [lines])[line_idx - 1]
        speaker, text = parse_speaker_line(str(raw_line), str(row.get("source_record_id")))
        return text, speaker

    def opela_by_doc_id(self) -> dict[str, dict[str, str]]:
        if self._opela_by_doc_id is None:
            path = DATASETS_ROOT / "OPELA" / "data" / "oplea_open_data.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                self._opela_by_doc_id = {row["doc_id"]: row for row in csv.DictReader(f)}
        return self._opela_by_doc_id

    def replay_opela(self, row: dict[str, Any]) -> tuple[str, str | None]:
        source_row = self.opela_by_doc_id()[str(row["source_record_id"])]
        field, _, raw_idx = str(row["source_turn_id"]).partition(":")
        if field not in {"persona_text_all", "user_text_all"} or not raw_idx.isdigit():
            raise ValueError(f"invalid OPELA source_turn_id={row['source_turn_id']!r}")
        lines = [minimal_normalize(x).strip() for x in source_row.get(field, "").splitlines() if x.strip()]
        text = lines[int(raw_idx) - 1]
        if field == "persona_text_all":
            speaker = source_row.get("persona_name_original") or "persona"
        else:
            speaker = f"user_{source_row.get('user_id')}"
        return text, speaker

    def lac_by_room_day(self) -> dict[tuple[str, int], list[dict[str, str]]]:
        if self._lac_by_room_day is None:
            path = DATASETS_ROOT / "japanese-long-term-chat" / "utf8" / "lac-public-dialogue.tsv"
            grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
            for source_row in read_tsv(path):
                grouped[(source_row["room"], int(source_row["dayid"]))].append(source_row)
            self._lac_by_room_day = grouped
        return self._lac_by_room_day

    def jmsc_by_pair_sid_tid(self) -> dict[tuple[str, int, int], dict[str, str]]:
        if self._jmsc_by_pair_sid_tid is None:
            path = DATASETS_ROOT / "japanese-long-term-chat" / "utf8" / "jmsc-public-dialogue.tsv"
            self._jmsc_by_pair_sid_tid = {
                (row["pair_id"], int(row["sid"]), int(row["tid"])): row
                for row in read_tsv(path)
            }
        return self._jmsc_by_pair_sid_tid

    def replay_jlongchat(self, row: dict[str, Any]) -> tuple[str, str | None]:
        source_file = str(row.get("source_file", ""))
        source_turn_id = str(row["source_turn_id"])
        if source_file.endswith("lac-public-dialogue.tsv"):
            match = re.fullmatch(r"(.+):day(\d+):row(\d+)", source_turn_id)
            if not match:
                raise ValueError(f"invalid LAC source_turn_id={source_turn_id!r}")
            room, day_raw, row_raw = match.groups()
            source_row = self.lac_by_room_day()[(room, int(day_raw))][int(row_raw) - 1]
            text = minimal_normalize(source_row["utterance"]).strip()
            speaker = source_row["speaker"].strip("[]").replace("SPK--", "")
            return text, speaker
        if source_file.endswith("jmsc-public-dialogue.tsv"):
            match = re.fullmatch(r"(.+):sid(\d+):tid(\d+)", source_turn_id)
            if not match:
                raise ValueError(f"invalid JMSC source_turn_id={source_turn_id!r}")
            pair_id, sid_raw, tid_raw = match.groups()
            source_row = self.jmsc_by_pair_sid_tid()[(pair_id, int(sid_raw), int(tid_raw))]
            text = minimal_normalize(source_row["utt"]).strip()
            return text, source_row["speaker"]
        raise ValueError(f"unsupported JLongChat source_file={source_file!r}")

    def load_del1l2im_xml(self, source_file: str) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
        if source_file in self._del1l2im_messages:
            return self._del1l2im_messages[source_file], self._del1l2im_fallback[source_file]
        path = PAPER_ROOT / source_file
        root = ET.parse(path).getroot()
        msg_ns = "{http://www.example.org/ns/nonTEI}message"
        messages = [el for el in root.iter() if el.tag == msg_ns or el.tag.split("}")[-1] == "message"]
        by_id: dict[str, ET.Element] = {}
        fallback: dict[str, ET.Element] = {}
        by_date: dict[str, list[ET.Element]] = defaultdict(list)
        for msg in messages:
            msg_id = xml_attr(msg, "id")
            if msg_id:
                by_id[msg_id] = msg
            timestamp = msg.attrib.get("timestamp", "")
            date = timestamp.split("T", 1)[0] if "T" in timestamp else f"session_{len(by_date) + 1}"
            by_date[date].append(msg)
        stem = path.stem
        for date in sorted(by_date):
            for local_idx, msg in enumerate(by_date[date], start=1):
                fallback[f"{stem}:{date}:{local_idx}"] = msg
        self._del1l2im_messages[source_file] = by_id
        self._del1l2im_fallback[source_file] = fallback
        return by_id, fallback

    def replay_del1l2im(self, row: dict[str, Any]) -> tuple[str, str | None]:
        by_id, fallback = self.load_del1l2im_xml(str(row["source_file"]))
        source_turn_id = str(row["source_turn_id"])
        msg = by_id[source_turn_id] if source_turn_id in by_id else fallback[source_turn_id]
        text = minimal_normalize(" ".join("".join(msg.itertext()).split())).strip()
        speaker = msg.attrib.get("who", "speaker").split("#")[-1]
        return text, speaker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, default=Path("datasets/locomo_style_eval/sidecars"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/source_replay_report.json"))
    args = parser.parse_args()

    replay = SourceReplay()
    errors: list[str] = []
    warnings: list[str] = []
    per_artifact: dict[str, Any] = {}
    provenance_files: list[Path] = []
    raw_source_files: set[Path] = set()

    for source, artifact in SOURCE_ARTIFACTS.items():
        path = args.sidecar_root / artifact / f"{artifact}_provenance.jsonl"
        provenance_files.append(path)
        rows = 0
        replayed_rows = 0
        source_origin_counts: Counter[str] = Counter()
        text_mismatches = 0
        speaker_mismatches = 0
        replay_errors = 0
        for line_idx, row in enumerate(iter_jsonl(path), start=1):
            rows += 1
            raw_source = resolve_source_file(str(row.get("source_file", "")))
            if raw_source is not None:
                raw_source_files.add(raw_source)
            origin = str(row.get("source_origin"))
            source_origin_counts[origin] += 1
            if origin not in {"original_turn", "memory_anchor_turn"}:
                continue
            try:
                expected = replay.replay(row)
            except Exception as exc:  # noqa: BLE001
                replay_errors += 1
                if replay_errors <= 10:
                    errors.append(f"{path}:{line_idx}: replay failed: {exc}")
                continue
            if expected is None:
                continue
            replayed_rows += 1
            expected_text, expected_speaker = expected
            if str(row.get("text", "")) != expected_text:
                text_mismatches += 1
                if text_mismatches <= 10:
                    errors.append(f"{path}:{line_idx}: replay text mismatch for {row.get('dia_id')}")
            if origin == "original_turn" and str(row.get("source_speaker", "")) != str(expected_speaker):
                speaker_mismatches += 1
                if speaker_mismatches <= 10:
                    errors.append(
                        f"{path}:{line_idx}: source_speaker={row.get('source_speaker')!r} "
                        f"expected={expected_speaker!r}"
                    )
            if origin == "memory_anchor_turn" and expected_speaker is None:
                warnings.append(f"{path}:{line_idx}: memory anchor has no source speaker context")
        if replay_errors > 10:
            errors.append(f"{artifact}: total replay errors={replay_errors}")
        if text_mismatches > 10:
            errors.append(f"{artifact}: total replay text mismatches={text_mismatches}")
        if speaker_mismatches > 10:
            errors.append(f"{artifact}: total replay speaker mismatches={speaker_mismatches}")

        per_artifact[artifact] = {
            "source_dataset": source,
            "provenance_rows": rows,
            "replayed_rows": replayed_rows,
            "source_origin_counts": dict(sorted(source_origin_counts.items())),
            "text_mismatches": text_mismatches,
            "speaker_mismatches": speaker_mismatches,
            "replay_errors": replay_errors,
        }

    report = {
        "status": "passed" if not errors else "failed",
        "input_files": {
            "provenance_files": [file_record(path) for path in provenance_files],
            "raw_source_files": [file_record(path) for path in sorted(raw_source_files)],
        },
        "sidecar_root": str(args.sidecar_root),
        "per_artifact": per_artifact,
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
