#!/usr/bin/env python3
"""Utilities for the 2026 LoCoMo memory-system reproduction runs."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


PAPER_ROOT = Path("/home/stu0032/paper")
DATASET = PAPER_ROOT / "baseline" / "MAGMA" / "data" / "locomo10.json"
MEMGAS_ROOT = PAPER_ROOT / "baseline" / "MemGAS"
OMNI_ROOT = PAPER_ROOT / "baseline" / "SimpleMem" / "OmniSimpleMem"
REME_ROOT = PAPER_ROOT / "baseline" / "ReMe"
PYTHON = PAPER_ROOT / ".venv" / "bin" / "python"


def apply_cpu_thread_limits() -> None:
    cpu_threads = int(os.environ.get("EXPERIMENT_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
    interop_threads = int(os.environ.get("EXPERIMENT_CPU_INTEROP_THREADS", "1"))
    try:
        import torch

        torch.set_num_threads(max(1, cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, interop_threads))
        except RuntimeError:
            pass
    except Exception:
        pass


apply_cpu_thread_limits()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    value = str(text)
    if not value:
        return 0
    return max(1, len(value) // 4)


def guard_estimate_tokens(text: Any) -> int:
    """Conservative token estimate used before vLLM calls.

    vLLM enforces prompt + completion <= max_model_len. The normal metrics
    estimate above intentionally mirrors previous reporting; this guard uses a
    stricter ratio and extra overhead to avoid context-limit failures.
    """
    if text is None:
        return 0
    value = str(text)
    if not value:
        return 0
    return max(1, (len(value) + 2) // 3)


def trim_text_by_guard_tokens(text: str, budget_tokens: int) -> str:
    if guard_estimate_tokens(text) <= budget_tokens:
        return text
    char_budget = max(64, budget_tokens * 3)
    if len(text) <= char_budget:
        return text
    head = max(32, char_budget // 2)
    tail = max(32, char_budget - head - 64)
    return f"{text[:head]}\n...[truncated for context budget]...\n{text[-tail:]}"


def guarded_completion_budget(
    prompt: str,
    requested_max_tokens: int,
    *,
    max_model_tokens: int,
    buffer_tokens: int,
    min_output_tokens: int,
) -> int:
    prompt_tokens = guard_estimate_tokens(prompt) + 256
    available = max_model_tokens - prompt_tokens - buffer_tokens
    return max(1, min(requested_max_tokens, max(1, available)))


def summarize_records_runtime(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        float(record["latency_seconds"])
        for record in records
        if record.get("latency_seconds") is not None
    ]
    token_usages = [
        record.get("token_usage")
        for record in records
        if isinstance(record.get("token_usage"), dict)
    ]

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "latency": {
            "count": len(latencies),
            "avg_seconds": avg(latencies),
            "total_seconds": sum(latencies),
            "max_seconds": max(latencies) if latencies else 0.0,
        },
        "tokens": {
            "count": len(token_usages),
            "avg_prompt_tokens": avg([float(t.get("prompt_tokens") or 0) for t in token_usages]),
            "avg_completion_tokens": avg([float(t.get("completion_tokens") or 0) for t in token_usages]),
            "avg_total_tokens": avg([float(t.get("total_tokens") or 0) for t in token_usages]),
            "total_prompt_tokens": sum(int(t.get("prompt_tokens") or 0) for t in token_usages),
            "total_completion_tokens": sum(int(t.get("completion_tokens") or 0) for t in token_usages),
            "total_tokens": sum(int(t.get("total_tokens") or 0) for t in token_usages),
        },
    }


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                f.flush()
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()


def load_locomo(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"LoCoMo JSON must be a list: {path}")
    return data


def sample_id(sample: dict[str, Any], index: int) -> str:
    if sample.get("sample_id"):
        return str(sample["sample_id"])
    conv = sample.get("conversation", {})
    left = conv.get("speaker_a", f"sample_{index}")
    right = conv.get("speaker_b", "")
    return f"{left}_{right}".strip("_").replace(" ", "_")


def session_numbers(sample: dict[str, Any]) -> list[int]:
    conv = sample.get("conversation", {})
    nums = []
    for key in conv:
        if key.startswith("session_") and not key.endswith("_date_time"):
            suffix = key.rsplit("_", 1)[-1]
            if suffix.isdigit():
                nums.append(int(suffix))
    return sorted(set(nums))


def session_lines(sample: dict[str, Any], session_num: int) -> list[str]:
    conv = sample.get("conversation", {})
    turns = conv.get(f"session_{session_num}", []) or []
    timestamp = conv.get(f"session_{session_num}_date_time", "")
    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or "").strip()
        if turn.get("img_url") and turn.get("blip_caption"):
            caption = f"[Image: {turn['blip_caption']}]"
            text = f"{caption} {text}".strip() if text else caption
        if not text:
            continue
        speaker = turn.get("speaker", "Speaker")
        prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(f"{prefix}{speaker}: {text}")
    return lines


def qa_reference(qa: dict[str, Any]) -> str:
    if str(qa.get("category")) == "5" and qa.get("adversarial_answer") is not None:
        return str(qa.get("adversarial_answer", ""))
    return str(qa.get("answer", qa.get("golden_answer", qa.get("reference", ""))))


def dataset_count(path: Path) -> tuple[int, int, Counter[str]]:
    samples = load_locomo(path)
    counts: Counter[str] = Counter()
    total = 0
    for sample in samples:
        for qa in sample.get("qa", []):
            total += 1
            counts[str(qa.get("category", "unknown"))] += 1
    return len(samples), total, counts


def normalize_prediction_record(
    *,
    sample_id_value: str,
    qa_idx: int,
    qa: dict[str, Any],
    prediction: str,
    retrieval: Any = None,
    latency_seconds: float | None = None,
    error: str | None = None,
    source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "sample_id": sample_id_value,
        "qa_idx": qa_idx,
        "category": qa.get("category"),
        "question": qa.get("question", ""),
        "prediction": prediction or "",
        "reference": qa_reference(qa),
        "retrieval": retrieval,
        "latency_seconds": latency_seconds,
        "error": error,
    }


def openai_client(api_key: str, base_url: str):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=float(os.environ.get("LOCOMO_OPENAI_TIMEOUT", "180")),
        max_retries=0,
    )


def is_fatal_vllm_error(message: str) -> bool:
    lowered = message.lower()
    fatal_markers = (
        "connection error",
        "connection refused",
        "server disconnected",
        "internal server error",
        "service unavailable",
        "engine core",
        "cuda out of memory",
    )
    return any(marker in lowered for marker in fatal_markers)


def check_vllm_endpoint(base_url: str) -> None:
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=5) as response:
        response.read(256)


def normalize_short_answer(answer: str) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text, flags=re.I).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "answer" in parsed:
                text = str(parsed["answer"]).strip()
        except json.JSONDecodeError:
            pass

    def repl(match: re.Match) -> str:
        raw = match.group(0)
        try:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            return raw
        return f"{dt.day} {dt.strftime('%B')} {dt.year}"

    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:]{5,8})?\b", repl, text)
    return text.strip()


def answer_with_context(
    client: Any,
    model: str,
    question: str,
    memories: list[str],
    max_tokens: int = 256,
) -> tuple[str, dict[str, int]]:
    max_model_tokens = int(os.environ.get("LOCOMO_MAX_MODEL_TOKENS", os.environ.get("VLLM_MAX_MODEL_LEN", "32768")))
    buffer_tokens = int(os.environ.get("LOCOMO_TOKEN_GUARD_BUFFER", "1536"))
    min_output_tokens = int(os.environ.get("LOCOMO_MIN_ANSWER_TOKENS", "512"))
    requested_max_tokens = max(min_output_tokens, int(max_tokens))

    prefix = """You are an intelligent memory assistant tasked with answering LoCoMo questions from retrieved conversation memories.

# Instructions
1. Answer based ONLY on the provided memories.
2. Output one precise, concise answer; usually less than 5-6 words.
3. Do not include reasoning, explanations, citations, markdown, or copied memory blocks.
4. If the answer is a date, convert ISO dates such as 2023-05-01 to natural text such as "1 May 2023".
5. If a memory contains relative time such as "last year" or "two months ago", resolve it using that memory's timestamp.
6. If multiple memories conflict, prefer the most recent relevant memory.

# Memories
"""
    suffix = f"""

# Question
{question}

# Answer
"""
    output_reserve = min(requested_max_tokens, int(os.environ.get("LOCOMO_MAX_ANSWER_TOKENS", "8192")))
    prompt_budget = max(
        256,
        max_model_tokens - output_reserve - buffer_tokens - guard_estimate_tokens(prefix + suffix) - 256,
    )
    prompt_budget = min(prompt_budget, int(os.environ.get("LOCOMO_MAX_PROMPT_TOKENS", "18000")))
    selected_memories: list[str] = []
    used_tokens = 0
    for memory in memories[:20]:
        memory_tokens = guard_estimate_tokens(memory)
        if used_tokens + memory_tokens <= prompt_budget:
            selected_memories.append(memory)
            used_tokens += memory_tokens
            continue
        remaining = prompt_budget - used_tokens
        if remaining > 128:
            selected_memories.append(trim_text_by_guard_tokens(memory, remaining))
        break

    context = "\n\n".join(selected_memories)
    prompt = f"{prefix}{context}{suffix}"
    safe_max_tokens = guarded_completion_budget(
        prompt,
        requested_max_tokens,
        max_model_tokens=max_model_tokens,
        buffer_tokens=buffer_tokens,
        min_output_tokens=min_output_tokens,
    )
    last_error = None
    max_retries = int(os.environ.get("LOCOMO_QA_MAX_RETRIES", "2"))
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=safe_max_tokens,
                timeout=float(os.environ.get("LOCOMO_OPENAI_TIMEOUT", "180")),
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            answer = normalize_short_answer(content)
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if prompt_tokens is None:
                prompt_tokens = estimate_tokens(prompt)
            if completion_tokens is None:
                completion_tokens = estimate_tokens(answer)
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            return answer, {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if "maximum context length" in str(exc) and len(selected_memories) > 1:
                selected_memories = selected_memories[: max(1, len(selected_memories) // 2)]
                context = "\n\n".join(selected_memories)
                prompt = f"{prefix}{context}{suffix}"
                safe_max_tokens = guarded_completion_budget(
                    prompt,
                    requested_max_tokens,
                    max_model_tokens=max_model_tokens,
                    buffer_tokens=buffer_tokens,
                    min_output_tokens=min_output_tokens,
                )
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"QA generation failed after retries: {last_error}")


def run_memgas(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(MEMGAS_ROOT))
    from quickstart import MemGASMemory, MemoryConfig

    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_locomo(data_path)
    if args.max_conversations:
        samples = samples[: args.max_conversations]

    prediction_log = out_dir / "prediction_log.jsonl"
    normalized_path = out_dir / "normalized_predictions.json"
    if prediction_log.exists():
        prediction_log.unlink()

    client = openai_client(args.api_key, args.base_url)
    records: list[dict[str, Any]] = []
    ingest_errors: list[dict[str, str]] = []
    records_lock = threading.Lock()
    jsonl_lock = threading.Lock()
    abort_event = threading.Event()
    abort_on_vllm_failure = os.environ.get("LOCOMO_ABORT_ON_VLLM_FAILURE", "1") != "0"
    started = time.time()

    for sample_index, sample in enumerate(samples):
        sid = sample_id(sample, sample_index)
        storage_dir = out_dir / "memory_data" / sid
        mem = MemGASMemory(
            MemoryConfig(
                storage_dir=str(storage_dir),
                embedder=args.embedder,
                device=args.device,
                llm_provider="vllm",
                llm_model=args.model,
                llm_api_key=args.api_key,
                llm_base_url=args.base_url,
                llm_max_tokens=args.summary_max_tokens,
                llm_max_retries=int(os.environ.get("MEMGAS_LLM_MAX_RETRIES", "3")),
                llm_retry_wait_sec=float(os.environ.get("MEMGAS_LLM_RETRY_WAIT_SEC", "5")),
                default_mode=args.method,
            )
        )

        for sess in session_numbers(sample):
            lines = session_lines(sample, sess)
            if not lines:
                continue
            try:
                mem.add(
                    session=lines,
                    conversation_id=sid,
                    metadata={"sample_id": sid, "session": sess},
                )
            except Exception as exc:  # noqa: BLE001
                ingest_errors.append({"sample_id": sid, "session": str(sess), "error": str(exc)})
                append_jsonl(
                    out_dir / "ingest_errors.jsonl",
                    {"sample_id": sid, "session": sess, "error": str(exc)},
                )
        mem.save()

        qas = list(enumerate(sample.get("qa", [])))

        def run_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            qa_idx, qa = item
            if abort_event.is_set():
                raise RuntimeError("MemGAS QA aborted after a fatal vLLM failure")
            question = str(qa.get("question", ""))
            t0 = time.time()
            retrieval = []
            error = None
            prediction = ""
            token_usage = None
            try:
                hits = mem.retrieve(
                    query=question,
                    topk=args.topk,
                    conversation_id=sid,
                    mode=args.method,
                )
                retrieval = hits
                memories = []
                for hit in hits:
                    session_text = "\n".join(hit.get("session") or [])
                    parts = [
                        f"summary: {hit.get('summary', '')}",
                        f"keywords: {', '.join(hit.get('keywords') or [])}",
                        f"session: {session_text}",
                    ]
                    memories.append("\n".join(part for part in parts if part.strip()))
                prediction, token_usage = answer_with_context(
                    client,
                    args.model,
                    question,
                    memories,
                    max_tokens=args.answer_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if abort_on_vllm_failure and is_fatal_vllm_error(error):
                    abort_event.set()
                    try:
                        check_vllm_endpoint(args.base_url)
                        health = "vLLM /models endpoint still reachable"
                    except Exception as health_exc:  # noqa: BLE001
                        health = f"vLLM health check failed: {health_exc}"
                    raise RuntimeError(
                        f"Fatal vLLM error during MemGAS QA sample={sid} qa_idx={qa_idx}: {error}; {health}"
                    ) from exc
            return normalize_prediction_record(
                sample_id_value=sid,
                qa_idx=qa_idx,
                qa=qa,
                prediction=prediction,
                retrieval=retrieval,
                latency_seconds=time.time() - t0,
                error=error,
                source="memgas",
            ) | {"token_usage": token_usage}

        workers = max(1, min(args.qa_workers, len(qas) or 1))
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(run_one, item) for item in qas]
        try:
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception:
                    abort_event.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                append_jsonl(prediction_log, record, jsonl_lock)
                with records_lock:
                    records.append(record)
        finally:
            executor.shutdown(wait=not abort_event.is_set(), cancel_futures=True)

    records.sort(key=lambda r: (str(r["sample_id"]), int(r["qa_idx"])))
    result = {
        "records": records,
        "summary": {
            "method": "MemGAS",
            "model": args.model,
            "data_path": str(data_path),
            "num_records": len(records),
            "runtime_seconds": time.time() - started,
            "runtime_stats": summarize_records_runtime(records),
            "repo_commit": git_commit(MEMGAS_ROOT),
            "notes": "MemGAS quickstart API with local vLLM answer generation.",
        },
    }
    dump_json(normalized_path, result)
    dump_json(out_dir / "summary.json", result["summary"])
    print(f"[memgas] wrote {len(records)} predictions to {normalized_path}")


def normalize_omni(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    result_path = out_dir / "locomo_omnimem_results.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    data = load_json(result_path)
    records = []
    for sample_index, sample in enumerate(data):
        sid = str(sample.get("sample_id") or f"sample_{sample_index}")
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            pred_key = next((key for key in qa if key.endswith("_prediction")), "prediction")
            records.append(
                normalize_prediction_record(
                    sample_id_value=sid,
                    qa_idx=qa_idx,
                    qa=qa,
                    prediction=str(qa.get(pred_key, "")),
                    retrieval=None,
                    latency_seconds=None,
                    error=None,
                    source="omnisimplemem",
                )
            )
    dump_json(
        out_dir / "normalized_predictions.json",
        {
            "records": records,
            "summary": {
                "method": "Omni-SimpleMem",
                "num_records": len(records),
                "repo_commit": git_commit(PAPER_ROOT / "baseline" / "SimpleMem"),
            },
        },
    )
    print(f"[omni] wrote {len(records)} normalized predictions")


def run_reme(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = REME_ROOT / "benchmark" / "locomo"
    spec = importlib.util.spec_from_file_location("reme_locomo_eval", bench_dir / "eval_reme.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ReMe LoCoMo evaluator")
    module = importlib.util.module_from_spec(spec)

    old_cwd = Path.cwd()
    try:
        os.chdir(bench_dir)
        sys.path.insert(0, str(bench_dir))
        spec.loader.exec_module(module)
        register_reme_local_embedding()
        from reme.reme import ReMe as InstalledReMe

        def local_reme_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
            factory_kwargs.setdefault("llm_api_key", os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")))
            factory_kwargs.setdefault("llm_base_url", os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")))
            factory_kwargs.setdefault(
                "default_embedding_model_config",
                {
                    "backend": "local_sentence_transformer",
                    "model_name": os.environ.get("REME_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                    "dimensions": int(os.environ.get("REME_EMBEDDING_DIM", "384")),
                    "max_batch_size": int(os.environ.get("REME_EMBEDDING_BATCH_SIZE", "32")),
                    "enable_cache": True,
                },
            )
            factory_kwargs.setdefault(
                "default_vector_store_config",
                {
                    "backend": "chroma",
                    "collection_name": os.environ.get("REME_COLLECTION_NAME", "reme_locomo_qwen25_3b"),
                    "embedding_model": "default",
                },
            )
            factory_kwargs.setdefault("working_dir", str(out_dir / ".reme_working"))
            factory_kwargs.setdefault("enable_logo", False)
            return InstalledReMe(*factory_args, **factory_kwargs)

        module.ReMe = local_reme_factory
        config = module.EvalConfig(
            data_path=str(Path(args.data_path).resolve()),
            top_k=args.topk,
            user_num=args.user_num,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            output_dir=str(out_dir),
            reme_model_name=args.model,
            eval_model_name=args.eval_model,
            algo_version=args.algo_version,
            enable_thinking_params=args.enable_thinking_params,
        )
        (out_dir / "eval_results.jsonl").touch()

        async def _run() -> None:
            evaluator = module.LocomoEvaluator(config)
            os.chdir(out_dir)
            async with evaluator:
                await evaluator.run_evaluation()

        asyncio.run(_run())
    finally:
        os.chdir(old_cwd)


def register_reme_local_embedding() -> None:
    from reme.core.embedding.base_embedding_model import BaseEmbeddingModel
    from reme.core.registry_factory import R

    class LocalSentenceTransformerEmbeddingModel(BaseEmbeddingModel):
        _model = None
        _lock = threading.Lock()
        _encode_lock = threading.Lock()

        def _get_model(self):
            if self.__class__._model is None:
                with self.__class__._lock:
                    if self.__class__._model is None:
                        from sentence_transformers import SentenceTransformer

                        self.__class__._model = SentenceTransformer(self.model_name, device="cpu")
            return self.__class__._model

        async def _get_embeddings(self, input_text: list[str], **kwargs: Any) -> list[list[float]]:
            return self._get_embeddings_sync(input_text, **kwargs)

        def _get_embeddings_sync(self, input_text: list[str], **kwargs: Any) -> list[list[float]]:
            model = self._get_model()
            with self.__class__._encode_lock:
                vectors = model.encode(
                    input_text,
                    batch_size=self.max_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return [list(map(float, vector)) for vector in vectors]

    R.embedding_models.register("local_sentence_transformer")(LocalSentenceTransformerEmbeddingModel)


def normalize_reme(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    eval_path = out_dir / "eval_results.jsonl"
    dataset = load_locomo(Path(args.data_path))
    by_uuid: dict[str, dict[str, Any]] = {}
    for idx, sample in enumerate(dataset):
        conv = sample.get("conversation", {})
        uuid = f"{conv.get('speaker_a', '')}_{conv.get('speaker_b', '')}".strip("_")
        by_uuid[uuid] = {"sample": sample, "sample_id": sample_id(sample, idx)}

    generated: dict[tuple[str, str, int], dict[str, Any]] = {}
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                uuid = str(user_data.get("uuid", ""))
                records = user_data.get("evaluation_results", {}).get("question_answering_records", [])
                for record in records:
                    generated[(uuid, str(record.get("question", "")), int(record.get("category", -1)))] = record

    records = []
    for idx, sample in enumerate(dataset):
        sid = sample_id(sample, idx)
        conv = sample.get("conversation", {})
        uuid = f"{conv.get('speaker_a', '')}_{conv.get('speaker_b', '')}".strip("_")
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            key = (uuid, str(qa.get("question", "")), int(qa.get("category", -1)))
            rec = generated.get(key)
            if rec is None:
                prediction = ""
                error = "missing_from_reme_official_output"
                retrieval = None
                latency = None
                if int(qa.get("category", -1)) == 5:
                    prediction = "The information is not mentioned in the provided memories."
                    error = "reme_official_runner_skips_category_5; standardized refusal inserted by adapter"
            else:
                prediction = str(rec.get("system_response", ""))
                error = None
                retrieval = rec.get("retrieved_nodes") or rec.get("retrieved_memories")
                latency = (rec.get("search_duration_ms") or 0) / 1000
            records.append(
                normalize_prediction_record(
                    sample_id_value=sid,
                    qa_idx=qa_idx,
                    qa=qa,
                    prediction=prediction,
                    retrieval=retrieval,
                    latency_seconds=latency,
                    error=error,
                    source="reme",
                )
            )

    dump_json(
        out_dir / "normalized_predictions.json",
        {
            "records": records,
            "summary": {
                "method": "ReMe",
                "num_records": len(records),
                "official_eval_results": str(eval_path),
                "repo_commit": git_commit(REME_ROOT),
                "runtime_package": reme_version(),
                "notes": "Official ReMe LoCoMo evaluator skips category 5; adapter inserts marked cat5 refusal records.",
            },
        },
    )
    print(f"[reme] wrote {len(records)} normalized predictions")


def git_commit(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def reme_version() -> str | None:
    try:
        import reme

        return getattr(reme, "__version__", None)
    except Exception:
        return None


def check_vllm(base_url: str, api_key: str, model: str) -> None:
    check_vllm_endpoint(base_url)
    client = openai_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply OK."}],
        temperature=0,
        max_tokens=4,
        timeout=float(os.environ.get("LOCOMO_OPENAI_TIMEOUT", "180")),
    )
    if not response.choices:
        raise RuntimeError("vLLM chat completion returned no choices")


def preflight(args: argparse.Namespace) -> None:
    samples, total, counts = dataset_count(Path(args.data_path))
    expected_samples = int(os.environ.get("LOCOMO_PREFLIGHT_EXPECTED_SAMPLES", "10"))
    expected_qa = int(os.environ.get("LOCOMO_PREFLIGHT_EXPECTED_QA", "1986"))
    if samples != expected_samples or total != expected_qa:
        raise RuntimeError(f"unexpected LoCoMo10 shape: samples={samples}, qa={total}")
    print(f"[preflight] dataset ok: samples={samples}, qa={total}, categories={dict(counts)}")

    if not args.skip_vllm:
        check_vllm(args.base_url, args.api_key, args.model)
        print("[preflight] vLLM ok")

    targets = set(args.targets)
    if "memgas" in targets:
        if not (MEMGAS_ROOT / "quickstart" / "memory.py").exists():
            raise FileNotFoundError("MemGAS quickstart missing")
        sys.path.insert(0, str(MEMGAS_ROOT))
        import quickstart  # noqa: F401
        import igraph  # noqa: F401
        from quickstart.embedder import TextEmbedder

        embedder = TextEmbedder(
            backend=os.environ.get("MEMGAS_EMBEDDER", "minilm"),
            device=os.environ.get("MEMGAS_DEVICE", "cpu"),
        )
        vector = embedder.encode(["embedding preflight"])
        if vector.numel() == 0:
            raise RuntimeError("MemGAS embedding preflight returned an empty vector")
        print(f"[preflight] MemGAS ok: {git_commit(MEMGAS_ROOT)}")

    if "omni" in targets:
        sys.path.insert(0, str(OMNI_ROOT))
        from omni_memory import OmniMemoryConfig  # noqa: F401
        print(f"[preflight] Omni-SimpleMem ok: {git_commit(PAPER_ROOT / 'baseline' / 'SimpleMem')}")

    if "reme" in targets:
        if not (REME_ROOT / "benchmark" / "locomo" / "eval_reme.py").exists():
            raise FileNotFoundError("ReMe LoCoMo evaluator missing")
        import reme  # noqa: F401
        print(f"[preflight] ReMe ok: repo={git_commit(REME_ROOT)}, package={reme_version()}")


def complete_prediction_file(path: Path, expected: int = 1986) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
        records = data.get("records", data if isinstance(data, list) else [])
        return isinstance(records, list) and len(records) == expected
    except Exception:
        return False


def summarize(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    rows = []
    for name, directory in (
        ("MemGAS", Path(args.memgas_dir)),
        ("Omni-SimpleMem", Path(args.omni_dir)),
        ("ReMe", Path(args.reme_dir)),
    ):
        metrics_path = directory / "normalized_metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        overall = metrics.get("overall", {})
        rows.append(
            {
                "model": name,
                "dir": str(directory),
                "count": metrics.get("count"),
                "f1": overall.get("f1", {}).get("mean"),
                "bleu1": overall.get("bleu1", {}).get("mean"),
                "rouge_l": overall.get("rouge_l", {}).get("mean"),
                "bertscore_f1": overall.get("bertscore_f1", {}).get("mean"),
                "bertscore_error": metrics.get("bertscore", {}).get("error"),
            }
        )

    dump_json(run_root / "summary.json", {"runs": rows})
    lines = [
        "# 2026 SOTA Memory Reproduction Summary",
        "",
        "| Model | QA | F1 | BLEU1 | ROUGE-L | BERTScore-F1 | Dir |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fmt = lambda v: "" if v is None else f"{float(v):.4f}"
        lines.append(
            f"| {row['model']} | {row['count'] or ''} | {fmt(row['f1'])} | "
            f"{fmt(row['bleu1'])} | {fmt(row['rouge_l'])} | {fmt(row['bertscore_f1'])} | "
            f"{row['dir']} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {run_root / 'summary.md'}")


def validate_predictions(args: argparse.Namespace) -> None:
    path = Path(args.input)
    data = load_json(path)
    records = data.get("records", data if isinstance(data, list) else [])
    if len(records) != args.expected_count:
        raise RuntimeError(f"{path} has {len(records)} records, expected {args.expected_count}")
    counts = Counter(str(record.get("category", "unknown")) for record in records)
    expected_counts = {"1": 282, "2": 321, "3": 96, "4": 841, "5": 446}
    if args.expected_count == 1986 and dict(counts) != expected_counts:
        raise RuntimeError(f"{path} category counts {dict(counts)} != {expected_counts}")
    bad_errors = []
    empty_predictions = []
    for record in records:
        if not str(record.get("prediction") or "").strip():
            empty_predictions.append(
                {
                    "sample_id": record.get("sample_id"),
                    "qa_idx": record.get("qa_idx"),
                    "category": record.get("category"),
                }
            )
        error = record.get("error")
        if not error:
            continue
        if (
            args.allow_reme_cat5_placeholder
            and str(record.get("category")) == "5"
            and "reme_official_runner_skips_category_5" in str(error)
        ):
            continue
        bad_errors.append(
            {
                "sample_id": record.get("sample_id"),
                "qa_idx": record.get("qa_idx"),
                "category": record.get("category"),
                "error": error,
            }
        )
    if bad_errors:
        raise RuntimeError(f"{path} contains non-allowed errors: {bad_errors[:10]}")
    if empty_predictions:
        raise RuntimeError(f"{path} contains empty predictions: {empty_predictions[:10]}")
    print(f"[validate] ok: {path} records={len(records)} categories={dict(counts)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight")
    pre.add_argument("--data-path", default=str(DATASET))
    pre.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    pre.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    pre.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    pre.add_argument("--skip-vllm", action="store_true")
    pre.add_argument("--targets", nargs="+", default=["memgas", "omni", "reme"])
    pre.set_defaults(func=preflight)

    memgas = sub.add_parser("run-memgas")
    memgas.add_argument("--data-path", default=str(DATASET))
    memgas.add_argument("--output-dir", required=True)
    memgas.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    memgas.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    memgas.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    memgas.add_argument("--embedder", default=os.environ.get("MEMGAS_EMBEDDER", "minilm"))
    memgas.add_argument("--device", default=os.environ.get("MEMGAS_DEVICE", "cpu"))
    memgas.add_argument("--method", default=os.environ.get("MEMGAS_METHOD", "memgas"))
    memgas.add_argument("--topk", type=int, default=int(os.environ.get("MEMGAS_TOPK", "20")))
    memgas.add_argument("--qa-workers", type=int, default=int(os.environ.get("MEMGAS_QA_WORKERS", "12")))
    memgas.add_argument("--answer-max-tokens", type=int, default=int(os.environ.get("MEMGAS_ANSWER_MAX_TOKENS", "8192")))
    memgas.add_argument("--summary-max-tokens", type=int, default=int(os.environ.get("MEMGAS_SUMMARY_MAX_TOKENS", "8192")))
    memgas.add_argument("--max-conversations", type=int, default=None)
    memgas.add_argument("--fresh", action="store_true")
    memgas.set_defaults(func=run_memgas)

    omni = sub.add_parser("normalize-omni")
    omni.add_argument("--output-dir", required=True)
    omni.set_defaults(func=normalize_omni)

    reme = sub.add_parser("run-reme")
    reme.add_argument("--data-path", default=str(DATASET))
    reme.add_argument("--output-dir", required=True)
    reme.add_argument("--model", default=os.environ.get("REME_MODEL", os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct")))
    reme.add_argument("--eval-model", default=os.environ.get("REME_EVAL_MODEL", os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct")))
    reme.add_argument("--topk", type=int, default=int(os.environ.get("REME_TOPK", "20")))
    reme.add_argument("--user-num", type=int, default=int(os.environ.get("REME_USER_NUM", "10")))
    reme.add_argument("--max-concurrency", type=int, default=int(os.environ.get("REME_MAX_CONCURRENCY", "1")))
    reme.add_argument("--batch-size", type=int, default=int(os.environ.get("REME_BATCH_SIZE", "20")))
    reme.add_argument("--algo-version", default=os.environ.get("REME_ALGO_VERSION", "locomo"))
    reme.add_argument("--enable-thinking-params", action="store_true")
    reme.add_argument("--fresh", action="store_true")
    reme.set_defaults(func=run_reme)

    nreme = sub.add_parser("normalize-reme")
    nreme.add_argument("--data-path", default=str(DATASET))
    nreme.add_argument("--output-dir", required=True)
    nreme.set_defaults(func=normalize_reme)

    summ = sub.add_parser("summarize")
    summ.add_argument("--run-root", required=True)
    summ.add_argument("--memgas-dir", required=True)
    summ.add_argument("--omni-dir", required=True)
    summ.add_argument("--reme-dir", required=True)
    summ.set_defaults(func=summarize)

    val = sub.add_parser("validate")
    val.add_argument("--input", required=True)
    val.add_argument("--expected-count", type=int, default=1986)
    val.add_argument("--allow-reme-cat5-placeholder", action="store_true")
    val.set_defaults(func=validate_predictions)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
