#!/usr/bin/env python3
"""Preflight model-service and GPU state before LoCoMo-style eval experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


BUSY_PATTERNS = (
    "run_memgas",
    "run_mem0",
    "run_amem",
    "run_higmem",
    "run_simplemem",
    "run_core_baselines",
    "run_locomo_recent_session_model_diagnostic",
    "locomo_2026_sota.py run-",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_models(base_url: str, timeout: float) -> tuple[list[str], str | None]:
    url = base_url.rstrip("/") + "/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/predeclared endpoint preflight.
            payload = json.loads(response.read().decode("utf-8"))
        return [str(item.get("id")) for item in payload.get("data", [])], None
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def chat_check(base_url: str, api_key: str, model: str, timeout: float) -> tuple[str | None, str | None]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply OK."}],
            temperature=0,
            max_tokens=4,
        )
        return (response.choices[0].message.content or "").strip(), None
    except Exception as exc:  # noqa: BLE001 - preflight should record exact client failure.
        return None, f"{type(exc).__name__}: {exc}"


def gpu_rows() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return rows, None


def busy_processes() -> list[dict[str, str]]:
    try:
        raw = subprocess.check_output(
            ["ps", "-eo", "pid,ppid,stat,pcpu,pmem,etime,cmd"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return []
    rows = []
    current_pid = str(os.getpid())
    for line in raw.splitlines()[1:]:
        if current_pid in line[:16]:
            continue
        lower = line.lower()
        if any(pattern.lower() in lower for pattern in BUSY_PATTERNS):
            fields = line.split(None, 6)
            if len(fields) == 7:
                rows.append(
                    {
                        "pid": fields[0],
                        "ppid": fields[1],
                        "stat": fields[2],
                        "pcpu": fields[3],
                        "pmem": fields[4],
                        "etime": fields[5],
                        "cmd": fields[6],
                    }
                )
    return rows


def dataset_counts(path: Path) -> dict[str, int]:
    data = load_json(path)
    qa_count = 0
    cat5_count = 0
    for sample in data:
        for qa in sample.get("qa", []):
            qa_count += 1
            if qa.get("category") == 5:
                cat5_count += 1
    return {
        "samples": len(data),
        "qa_count": qa_count,
        "answerable_qa_count": qa_count - cat5_count,
        "cat5_qa_count": cat5_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/locomo_style_eval/primary/multilingual_locomo_style_eval_audited.json"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-gpu-util", type=int, default=80)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.90)
    parser.add_argument("--fail-if-gpu-busy", action="store_true")
    parser.add_argument("--fail-if-busy-process", action="store_true")
    parser.add_argument("--chat-check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("datasets/locomo_style_eval/experiment_preflight.json"))
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    dataset_info: dict[str, Any] = {"path": str(args.dataset), "exists": args.dataset.exists()}
    if args.dataset.exists():
        dataset_info.update(dataset_counts(args.dataset))
    else:
        errors.append(f"dataset not found yet: {args.dataset}")

    models, model_error = fetch_models(args.base_url, args.timeout)
    if model_error:
        errors.append(f"model service not reachable: {model_error}")
    elif args.model not in models:
        errors.append(f"model {args.model!r} not served; available={models}")

    chat_response = None
    chat_error = None
    service_preflight_policy = "service_preflight_only" if args.chat_check else "not_requested"
    if args.chat_check and not model_error and args.model in models:
        chat_response, chat_error = chat_check(args.base_url, args.api_key, args.model, args.timeout)
        if chat_error:
            errors.append(f"chat preflight failed: {chat_error}")

    gpus, gpu_error = gpu_rows()
    if gpu_error:
        warnings.append(f"gpu status unavailable: {gpu_error}")
    busy_gpus = []
    for gpu in gpus:
        memory_ratio = gpu["memory_used_mib"] / max(1, gpu["memory_total_mib"])
        gpu["memory_used_ratio"] = round(memory_ratio, 4)
        if gpu["utilization_gpu_percent"] > args.max_gpu_util or memory_ratio > args.max_gpu_memory_ratio:
            busy_gpus.append(gpu)
    if busy_gpus:
        message = f"gpu busy above thresholds: {busy_gpus}"
        if args.fail_if_gpu_busy:
            errors.append(message)
        else:
            warnings.append(message)

    processes = busy_processes()
    if processes:
        message = f"busy experiment processes detected: {len(processes)}"
        if args.fail_if_busy_process:
            errors.append(message)
        else:
            warnings.append(message)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not errors else "failed",
        "dataset": dataset_info,
        "base_url": args.base_url,
        "model": args.model,
        "served_models": models,
        "chat_check_requested": args.chat_check,
        "service_preflight_policy": service_preflight_policy,
        "chat_response_used_for_dataset_or_metrics": False,
        "chat_response": chat_response,
        "chat_error": chat_error,
        "gpu_thresholds": {
            "max_gpu_util": args.max_gpu_util,
            "max_gpu_memory_ratio": args.max_gpu_memory_ratio,
            "fail_if_gpu_busy": args.fail_if_gpu_busy,
        },
        "gpus": gpus,
        "busy_process_patterns": list(BUSY_PATTERNS),
        "busy_processes": processes,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
