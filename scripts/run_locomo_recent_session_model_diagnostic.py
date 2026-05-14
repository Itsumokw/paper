#!/usr/bin/env python3
"""Run model-side recent-session diagnostics on LoCoMo-style ablation files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ABLATION_FILES = {
    "full_conversation": "full_conversation.json",
    "last_session_only": "last_session_only.json",
    "last_3_sessions_only": "last_3_sessions_only.json",
}

INPUT_POLICY_REPORT = {
    "input_policy": "conversation_only",
    "summary_visible": False,
    "input_fields_rendered": ["conversation"],
    "input_fields_excluded": ["observation", "session_summary", "event_summary", "sidecars"],
    "prompt_policy": "conversation_history_only_direct_answer",
    "context_renderer": "conversation.session_i_date_time_and_turn_dia_id_speaker_text_only",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def render_context(sample: dict[str, Any], max_chars: int) -> str:
    lines: list[str] = []
    conversation = sample.get("conversation", {})
    for session_key in session_keys(conversation):
        lines.append(f"[{session_key} {conversation.get(f'{session_key}_date_time', '')}]")
        for turn in conversation[session_key]:
            lines.append(f"{turn.get('dia_id')} {turn.get('speaker')}: {turn.get('text')}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n...[context truncated]...\n{text[-half:]}"


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def metric_tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    cjk_chars = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    if cjk_chars:
        ascii_words = re.findall(r"[a-z0-9]+", text)
        return cjk_chars + ascii_words
    return text.split()


def token_f1(prediction: Any, reference: Any) -> float:
    pred = set(metric_tokens(prediction))
    ref = set(metric_tokens(reference))
    if not pred or not ref:
        return 0.0
    common = pred & ref
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def answerable_qas(sample: dict[str, Any], categories: set[str]) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for idx, qa in enumerate(sample.get("qa", [])):
        category = str(qa.get("category"))
        if category == "5":
            continue
        if category in categories:
            rows.append((idx, qa))
    return rows


def build_prompt(context: str, question: str) -> str:
    return f"""/no_think
You answer questions from conversation history only.

Rules:
- Use only the provided conversation.
- If the answer is unsupported by the visible conversation, say: Not enough information.
- Answer directly in one short phrase or sentence.
- Do not reason step by step.

Conversation:
{context}

Question:
{question}

Answer:"""


def call_model(client: Any, model: str, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    token_usage = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
    }
    return content.strip(), token_usage


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_context: dict[str, list[float]] = defaultdict(list)
    by_context_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    error_count = 0
    for record in records:
        if record.get("error"):
            error_count += 1
            continue
        context_name = str(record["context_name"])
        category = str(record["category"])
        score = float(record.get("token_f1", 0.0))
        by_context[context_name].append(score)
        by_context_category[context_name][category].append(score)
    summary = {}
    for context_name, scores in sorted(by_context.items()):
        summary[context_name] = {
            "count": len(scores),
            "mean_token_f1": mean(scores) if scores else 0.0,
            "by_category": {
                category: {
                    "count": len(values),
                    "mean_token_f1": mean(values) if values else 0.0,
                }
                for category, values in sorted(by_context_category[context_name].items())
            },
        }
    if "full_conversation" in summary:
        full = summary["full_conversation"]["mean_token_f1"]
        for context_name in ("last_session_only", "last_3_sessions_only"):
            if context_name in summary:
                summary[context_name]["delta_vs_full_token_f1"] = summary[context_name]["mean_token_f1"] - full
    return {
        "contexts": summary,
        "records": len(records),
        "errors": error_count,
    }


def parse_categories(value: str) -> set[str]:
    categories = {item.strip() for item in value.split(",") if item.strip()}
    if not categories:
        raise argparse.ArgumentTypeError("at least one category is required")
    return categories


def settings_validation_errors(args: argparse.Namespace, settings: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if settings.get("status") != "predeclared":
        errors.append(f"settings.status={settings.get('status')!r} expected='predeclared'")

    model = settings.get("model", {})
    if model.get("served_model") != args.model:
        errors.append(f"settings.model.served_model={model.get('served_model')!r} expected={args.model!r}")

    recent = settings.get("recent_session_diagnostic", {})
    if recent.get("ablation_root") != str(args.ablation_root):
        errors.append(f"settings.recent_session_diagnostic.ablation_root={recent.get('ablation_root')!r}")
    if {str(item) for item in recent.get("categories", [])} != set(args.categories):
        errors.append(
            f"settings.recent_session_diagnostic.categories={recent.get('categories')!r} "
            f"expected={sorted(args.categories)!r}"
        )
    if int(recent.get("max_context_chars", -1)) != args.max_context_chars:
        errors.append(f"settings.max_context_chars={recent.get('max_context_chars')!r} expected={args.max_context_chars}")
    if int(recent.get("max_answer_tokens", -1)) != args.max_answer_tokens:
        errors.append(f"settings.max_answer_tokens={recent.get('max_answer_tokens')!r} expected={args.max_answer_tokens}")
    if float(recent.get("request_timeout_seconds", -1)) != float(args.request_timeout):
        errors.append(
            f"settings.request_timeout_seconds={recent.get('request_timeout_seconds')!r} "
            f"expected={args.request_timeout}"
        )
    if int(recent.get("workers", -1)) != args.workers:
        errors.append(f"settings.workers={recent.get('workers')!r} expected={args.workers}")
    return errors


def ablation_manifest_validation_errors(
    manifest_path: Path,
    ablation_manifest: dict[str, Any],
    settings: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"ablation manifest not found: {manifest_path}"]
    if settings is None:
        return errors
    audited_primary = str(settings.get("dataset", {}).get("audited_primary") or "")
    if not audited_primary:
        errors.append("settings.dataset.audited_primary missing")
        return errors
    observed_input = str(ablation_manifest.get("input") or "")
    if observed_input != audited_primary:
        errors.append(f"ablation_manifest.input={observed_input!r} expected audited_primary={audited_primary!r}")
    audited_path = Path(audited_primary)
    if not audited_path.is_file():
        errors.append(f"audited primary not found: {audited_path}")
    elif ablation_manifest.get("input_sha256") != sha256_file(audited_path):
        errors.append("ablation_manifest.input_sha256 does not match audited primary")
    return errors


def run_record(client: Any, model: str, max_answer_tokens: int, record: dict[str, Any], prompt: str) -> dict[str, Any]:
    started = time.time()
    try:
        prediction, token_usage = call_model(client, model, prompt, max_answer_tokens)
        record["prediction"] = prediction
        record["token_usage"] = token_usage
        record["token_f1"] = token_f1(prediction, record["reference"])
    except Exception as exc:  # noqa: BLE001 - record failure and keep resumable output.
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_seconds"] = round(time.time() - started, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("datasets/locomo_style_eval/recent_session_ablation"))
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/recent_session_ablation/model_results_summary.json"))
    parser.add_argument("--records-output", type=Path, default=Path("datasets/locomo_style_eval/recent_session_ablation/model_prediction_records.jsonl"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--categories", type=parse_categories, default=parse_categories("1,2,3,4"))
    parser.add_argument("--limit-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--limit-qa-per-sample", type=int, default=0, help="0 means all QA")
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--settings-file", type=Path, default=None)
    parser.add_argument("--enforce-settings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings: dict[str, Any] | None = None
    settings_hash: str | None = None
    settings_errors: list[str] = []
    if args.settings_file:
        if not args.settings_file.is_file():
            settings_errors.append(f"settings file not found: {args.settings_file}")
        else:
            settings = load_json(args.settings_file)
            settings_hash = sha256_file(args.settings_file)
            settings_errors = settings_validation_errors(args, settings)
    elif args.enforce_settings:
        settings_errors.append("--enforce-settings requires --settings-file")

    manifest_path = args.ablation_root / "recent_session_ablation_manifest.json"
    ablation_manifest = load_json(manifest_path) if manifest_path.exists() else {}
    if args.enforce_settings:
        settings_errors.extend(ablation_manifest_validation_errors(manifest_path, ablation_manifest, settings))

    if args.enforce_settings and settings_errors:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "partial_or_failed",
            "model": args.model,
            "runner_script": str(Path(__file__)),
            "runner_script_sha256": sha256_file(Path(__file__)),
            **INPUT_POLICY_REPORT,
            "settings_file": str(args.settings_file) if args.settings_file else None,
            "settings_sha256": settings_hash,
            "settings_source": settings.get("fixed_baselines", {}).get("settings_source") if settings else None,
            "settings_errors": settings_errors,
            "ablation_manifest": str(manifest_path),
            "ablation_input": ablation_manifest.get("input"),
            "ablation_input_sha256": ablation_manifest.get("input_sha256"),
        }
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    if args.dry_run:
        client = None
    else:
        from openai import OpenAI

        client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/"), timeout=args.request_timeout, max_retries=0)

    tasks: list[tuple[dict[str, Any], str]] = []
    expected_records = 0
    context_counts: Counter[str] = Counter()

    for context_name, filename in ABLATION_FILES.items():
        data = load_json(args.ablation_root / filename)
        if args.limit_samples:
            data = data[: args.limit_samples]
        for sample in data:
            qa_items = answerable_qas(sample, args.categories)
            if args.limit_qa_per_sample:
                qa_items = qa_items[: args.limit_qa_per_sample]
            expected_records += len(qa_items)
            context = render_context(sample, args.max_context_chars)
            for qa_idx, qa in qa_items:
                context_counts[context_name] += 1
                prompt = build_prompt(context, str(qa.get("question", "")))
                record = {
                    "context_name": context_name,
                    "sample_id": sample.get("sample_id"),
                    "source_dataset": sample.get("source_dataset"),
                    "qa_idx": qa_idx,
                    "category": qa.get("category"),
                    "model": args.model,
                    "ablation_input_sha256": ablation_manifest.get("input_sha256"),
                    "question": qa.get("question"),
                    "reference": qa.get("answer", ""),
                    "prediction": "",
                    "token_f1": 0.0,
                    "latency_seconds": None,
                    "token_usage": {},
                    "error": None,
                }
                if args.dry_run:
                    record["error"] = "dry_run_no_model_call"
                    tasks.append((record, prompt))
                else:
                    tasks.append((record, prompt))

    records: list[dict[str, Any]] = []
    if args.dry_run:
        records = [record for record, _ in tasks]
    elif args.workers <= 1:
        for record, prompt in tasks:
            records.append(run_record(client, args.model, args.max_answer_tokens, record, prompt))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_record, client, args.model, args.max_answer_tokens, record, prompt)
                for record, prompt in tasks
            ]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(
            key=lambda row: (
                str(row.get("context_name")),
                str(row.get("sample_id")),
                int(row.get("qa_idx") or 0),
            )
        )

    args.records_output.parent.mkdir(parents=True, exist_ok=True)
    with args.records_output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    complete = (
        not args.dry_run
        and not args.limit_samples
        and not args.limit_qa_per_sample
        and len(records) == expected_records
        and not any(record.get("error") for record in records)
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed" if complete else "partial_or_failed",
        "model": args.model,
        "base_url": args.base_url,
        "runner_script": str(Path(__file__)),
        "runner_script_sha256": sha256_file(Path(__file__)),
        **INPUT_POLICY_REPORT,
        "settings_file": str(args.settings_file) if args.settings_file else None,
        "settings_sha256": settings_hash,
        "settings_source": settings.get("fixed_baselines", {}).get("settings_source") if settings else None,
        "settings_errors": settings_errors,
        "ablation_manifest": str(manifest_path),
        "ablation_input": ablation_manifest.get("input"),
        "ablation_input_sha256": ablation_manifest.get("input_sha256"),
        "categories": sorted(args.categories),
        "limit_samples": args.limit_samples,
        "limit_qa_per_sample": args.limit_qa_per_sample,
        "max_context_chars": args.max_context_chars,
        "max_answer_tokens": args.max_answer_tokens,
        "request_timeout": args.request_timeout,
        "workers": args.workers,
        "expected_records": expected_records,
        "written_records": len(records),
        "context_counts": dict(sorted(context_counts.items())),
        "records_output": str(args.records_output),
        "records_output_sha256": sha256_file(args.records_output) if args.records_output.is_file() else None,
        "summary": summarize_records(records),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
