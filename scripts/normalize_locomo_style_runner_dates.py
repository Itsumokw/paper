#!/usr/bin/env python3
"""Normalize LoCoMo-style session date strings for baseline runners.

Some upstream multilingual sources carry dates such as "2019年5月5日",
"dayid 1", or "OPELA virtual session 1". The LightMem LoCoMo loader accepts
only strings like "1:56 PM on 8 May, 2023". This script rewrites only
conversation session_*_date_time fields into that parseable format and keeps
the original value in session_*_date_time_original.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


LOCOMO_FORMAT = "%I:%M %p on %d %B, %Y"


def format_locomo(dt: datetime) -> str:
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{dt.strftime('%M')} {dt.strftime('%p')} on {dt.day} {dt.strftime('%B')}, {dt.year}"


def parse_time_suffix(value: str) -> tuple[int, int]:
    hour = 9
    minute = 0
    time_match = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", value)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
    else:
        hour_match = re.search(r"(\d{1,2})\s*点", value)
        if hour_match:
            hour = int(hour_match.group(1))
    if any(marker in value for marker in ("下午", "晚上")) and hour < 12:
        hour += 12
    if "中午" in value:
        hour = 12 if hour in (9, 0) else hour
    if hour > 23:
        hour = 9
    return hour, minute


def parse_datetime(value: Any, fallback_year: int) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, LOCOMO_FORMAT)
    except ValueError:
        pass

    iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?", text)
    if iso:
        year, month, day = map(int, iso.group(1, 2, 3))
        hour = int(iso.group(4) or 9)
        minute = int(iso.group(5) or 0)
        second = int(iso.group(6) or 0)
        return datetime(year, month, day, hour, minute, second)

    zh = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if zh:
        year, month, day = map(int, zh.group(1, 2, 3))
        hour, minute = parse_time_suffix(text)
        return datetime(year, month, day, hour, minute)

    range_year = re.search(r"(\d{4})\s*-\s*\d{4}年", text)
    if range_year:
        return datetime(int(range_year.group(1)), 1, 1, 9, 0)

    zh_year = re.search(r"(\d{4})年", text)
    if zh_year:
        return datetime(int(zh_year.group(1)), 1, 1, 9, 0)

    zh_no_year = re.search(r"(\d{1,2})月\s*(\d{1,2})日", text)
    if zh_no_year:
        month, day = map(int, zh_no_year.group(1, 2))
        hour, minute = parse_time_suffix(text)
        return datetime(fallback_year, month, day, hour, minute)

    return None


def session_index(key: str) -> int | None:
    match = re.fullmatch(r"session_(\d+)_date_time", key)
    return int(match.group(1)) if match else None


def normalize(data: list[dict[str, Any]]) -> dict[str, Any]:
    changed = 0
    fallback = 0
    examples: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(data):
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue
        date_keys = sorted(
            (key for key in conversation if session_index(key) is not None),
            key=lambda key: session_index(key) or 0,
        )
        parsed_years = [
            dt.year
            for key in date_keys
            if (dt := parse_datetime(conversation.get(key), 2023)) is not None
        ]
        fallback_year = parsed_years[0] if parsed_years else 2023
        base = datetime(fallback_year, 1, 1, 9, 0) + timedelta(days=sample_idx * 100)

        for key in date_keys:
            idx = session_index(key) or 1
            original = conversation.get(key)
            parsed = parse_datetime(original, fallback_year)
            used_fallback = False
            if parsed is None:
                parsed = base + timedelta(days=idx - 1)
                used_fallback = True
                fallback += 1
            normalized = format_locomo(parsed)
            if original != normalized:
                conversation[f"{key}_original"] = original
                conversation[key] = normalized
                changed += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "sample_id": sample.get("sample_id"),
                            "key": key,
                            "original": original,
                            "normalized": normalized,
                            "fallback": used_fallback,
                        }
                    )
    return {"changed": changed, "fallback": fallback, "examples": examples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected top-level list")
    summary = normalize(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
