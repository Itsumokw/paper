#!/usr/bin/env python3
"""Fast LoCoMo HiGMemPlus evaluation using existing HiGMem checkpoints.

This runner avoids rebuilding HiGMem memories. It loads the per-sample
checkpoint created by an earlier full LoCoMo HiGMem run, evaluates only the
manifest-selected QA rows, and optionally adds the non-invasive HiGMemPlus
evidence layer at retrieval time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
HIGMEM_DIR = ROOT / "baseline" / "HiGMem"
for path in [str(ROOT), str(ROOT / "scripts"), str(HIGMEM_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NLTK_DATA", "/home/stu0032/nltk_data")
os.environ.setdefault("HIGMEM_USE_STREAMING", "1")
os.environ.setdefault("HIGMEM_MAX_TOKENS", "512")
os.environ.setdefault("HIGMEM_OPENAI_TIMEOUT", "180")

from baseline.HiGMemPlus import METHODS, HiGMemPlusEnhancer, RawTurn  # noqa: E402
from fphm_core import FPHMSystem  # noqa: E402
from fphm_logger import FPHMLogger  # noqa: E402
from load_dataset import LoCoMoSample, QA, load_locomo_dataset  # noqa: E402
from memory_layer import LLMController  # noqa: E402
from run_fphm_evaluation import build_category_prompt, generate_keyword_query  # noqa: E402


METHOD_CHOICES = METHODS
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("datasets/subsets/locomo10_10pct_seed20260517.json"))
    parser.add_argument("--subset-manifest", type=Path, default=Path("datasets/subsets/locomo10_10pct_seed20260517_manifest.json"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem"),
    )
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reproductions/higmem_plus/locomo10_10pct_fast"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--k-profile", type=int, default=0)
    parser.add_argument("--k-event", type=int, default=10)
    parser.add_argument("--k-turn", type=int, default=10)
    parser.add_argument("--component-k", type=int, default=10)
    parser.add_argument("--edge-k", type=int, default=8)
    parser.add_argument("--episode-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--answer-max-tokens", type=int, default=64)
    parser.add_argument("--answer-timeout", type=float, default=120.0)
    parser.add_argument("--disable-query-rewriting", action="store_true", default=True)
    parser.add_argument("--ablation-no-filter", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def normalize_tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def token_f1(prediction: Any, reference: Any) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    common = sum((pred_counts & ref_counts).values())
    if not common:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: Any, reference: Any) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if not pred or not ref:
        return 0.0
    ref_counts = Counter(ref)
    overlap = 0
    for tok in pred:
        if ref_counts[tok] > 0:
            overlap += 1
            ref_counts[tok] -= 1
    precision = overlap / len(pred)
    brevity = 1.0 if len(pred) > len(ref) else pow(2.718281828, 1 - len(ref) / max(1, len(pred)))
    return precision * brevity


def is_answer_supported(prediction: str, reference: str, category: int) -> bool:
    if category == 5:
        lowered = prediction.lower()
        return "not mentioned" in lowered or "don't know" in lowered or "insufficient" in lowered or "cannot answer" in lowered
    return token_f1(prediction, reference) >= 0.5 or str(prediction).strip().lower() == str(reference).strip().lower()


def qa_reference(qa: QA) -> str:
    value = qa.final_answer
    return "" if value is None else str(value)


def turn_index_from_id(turn_id: str) -> int:
    if ":" not in turn_id:
        return 0
    try:
        return int(turn_id.split(":")[-1])
    except ValueError:
        return 0


def iter_sample_turns(sample: LoCoMoSample) -> list[tuple[Any, str, int]]:
    turns = []
    for session_id in sorted(sample.conversation.sessions.keys()):
        session = sample.conversation.sessions[session_id]
        for turn in sorted(session.turns, key=lambda item: turn_index_from_id(item.dia_id)):
            turns.append((turn, session.date_time, int(session_id)))
    return turns


def raw_turns_for_sample(sample: LoCoMoSample) -> tuple[list[RawTurn], dict[str, RawTurn]]:
    raw_turns: list[RawTurn] = []
    raw_by_id: dict[str, RawTurn] = {}
    for turn, date_time, session_id in iter_sample_turns(sample):
        raw = RawTurn(
            turn_id=turn.dia_id,
            text=turn.text,
            speaker=turn.speaker,
            timestamp=date_time,
            dataset="locomo",
            episode_id=f"sample_{sample.sample_id}",
            session_or_scene_id=f"sample_{sample.sample_id}_session_{session_id}",
            chronological_order=session_id,
            turn_index=turn_index_from_id(turn.dia_id),
            metadata={"sample_id": sample.sample_id, "session_id": session_id},
        )
        raw_turns.append(raw)
        raw_by_id[raw.turn_id] = raw
    return raw_turns, raw_by_id


def checkpoint_path(root: Path, sample_id: str) -> Path:
    sample_dir = root / f"sample_{sample_id}" / "checkpoints"
    paths = sorted(sample_dir.glob("*_final.pkl"))
    if not paths:
        raise FileNotFoundError(f"No final HiGMem checkpoint under {sample_dir}")
    return paths[-1]


def load_higmem_from_checkpoint(args: argparse.Namespace, sample_id: str, out_dir: Path) -> FPHMSystem:
    llm_controller = LLMController("openai", model=args.model, api_key=args.api_key, api_base=args.api_base)
    system = FPHMSystem(
        llm_controller=llm_controller,
        run_name=f"locomo_{args.method}_sample_{sample_id}",
        use_character_profile=False,
        use_event_metadata_mode=True,
        ablation_no_link=True,
        ablation_no_filter=args.ablation_no_filter,
        k_event_affiliation=args.k_event,
        log_dir=str(out_dir / "fphm_logs"),
    )
    with checkpoint_path(args.checkpoint_root, sample_id).open("rb") as f:
        state = pickle.load(f)
    system.__dict__.update(state)
    system.llm = llm_controller
    system.logger = FPHMLogger(log_dir=str(out_dir / "fphm_logs"), run_name=f"locomo_{args.method}_sample_{sample_id}")
    system.executor = ThreadPoolExecutor(max_workers=4)
    system.ablation_no_filter = args.ablation_no_filter
    return system


def build_prompt(category: int, context: str, question: str, qa: QA) -> str:
    seed_material = f"{category}\0{question}".encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
    state = random.getstate()
    try:
        random.seed(seed)
        return build_category_prompt(category=category, context=context, question=question, qa=qa)
    finally:
        random.setstate(state)


def answer_question(client: OpenAI, args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, int], str | None]:
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=args.answer_max_tokens,
            timeout=args.answer_timeout,
        )
        usage = getattr(response, "usage", None)
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return (response.choices[0].message.content or "").strip(), token_usage, None
    except Exception as exc:  # noqa: BLE001
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, f"{type(exc).__name__}: {exc}"


def evidence_ids_from_records(records: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for record in records:
        metadata = record.get("metadata") or {}
        turn_id = metadata.get("turn_id")
        if turn_id:
            ids.add(str(turn_id))
        for key in ["source_turn_ids", "evidence_turn_ids"]:
            for item in metadata.get(key) or []:
                ids.add(str(item))
    return ids


def evidence_text_support(context: str, evidence_ids: list[str], raw_by_id: dict[str, RawTurn]) -> bool:
    if not evidence_ids:
        return False
    lowered_context = (context or "").lower()
    for eid in evidence_ids:
        raw = raw_by_id.get(str(eid))
        if raw and raw.text and raw.text[:120].lower() in lowered_context:
            return True
    return False


def records_from_higmem_trace(trace: dict[str, Any], system: FPHMSystem) -> list[dict[str, Any]]:
    records = []
    for turn_id in trace.get("relevant_turn_ids") or []:
        note = system.turn_notes.get(turn_id)
        if not note:
            continue
        records.append(
            {
                "used_content": note.content,
                "metadata": {
                    "source": "higmem_relevant_turn",
                    "turn_note_id": turn_id,
                    "speaker": note.speaker,
                    "timestamp": note.timestamp,
                },
            }
        )
    for event_id in trace.get("relevant_event_ids") or []:
        event = system.events.get(event_id)
        if not event:
            continue
        records.append(
            {
                "used_content": getattr(event, "summary_content", "") or getattr(event, "title", ""),
                "metadata": {
                    "source": "higmem_event",
                    "event_id": event_id,
                    "turn_note_ids": list(getattr(event, "turn_note_ids", []) or []),
                },
            }
        )
    return records


def category_metadata(category: int) -> dict[str, Any]:
    name = CATEGORY_NAMES.get(category, "unknown")
    return {
        "category": str(category),
        "question_type": name,
        "answerability_label": "unanswerable" if category == 5 else "answerable",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if sha256_file(args.dataset_path) != manifest.get("subset_sha256"):
        raise ValueError(f"Subset hash mismatch for {args.dataset_path}")

    out_dir = args.output_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "command_env": out_dir / "command.env",
        "dataset_manifest": out_dir / "dataset_manifest.json",
        "raw_predictions": out_dir / "raw_predictions.jsonl",
        "retrieved_evidence": out_dir / "retrieved_evidence.jsonl",
        "component_traces": out_dir / "component_traces.jsonl",
        "graph_traces": out_dir / "graph_traces.jsonl",
        "repair_traces": out_dir / "repair_traces.jsonl",
        "episode_traces": out_dir / "episode_traces.jsonl",
        "route_traces": out_dir / "route_traces.jsonl",
        "metrics": out_dir / "metrics.json",
        "stats": out_dir / "stats.json",
        "summary": out_dir / "summary.md",
        "run_log": out_dir / "run.log",
    }
    if not args.resume:
        for key in [
            "raw_predictions",
            "retrieved_evidence",
            "component_traces",
            "graph_traces",
            "repair_traces",
            "episode_traces",
            "route_traces",
        ]:
            if paths[key].exists():
                paths[key].unlink()
    for key, path in paths.items():
        if path.suffix == ".jsonl":
            path.touch(exist_ok=True)

    command_lines = [
        f"METHOD={args.method}",
        f"MODEL={args.model}",
        f"API_BASE={args.api_base}",
        f"DATASET_PATH={args.dataset_path}",
        f"DATASET_SHA256={sha256_file(args.dataset_path)}",
        f"SUBSET_MANIFEST={args.subset_manifest}",
        f"SUBSET_MANIFEST_SHA256={sha256_file(args.subset_manifest)}",
        f"CHECKPOINT_ROOT={args.checkpoint_root}",
        f"K_PROFILE={args.k_profile}",
        f"K_EVENT={args.k_event}",
        f"K_TURN={args.k_turn}",
        f"COMPONENT_K={args.component_k}",
        f"EDGE_K={args.edge_k}",
        f"EPISODE_K={args.episode_k}",
        f"MAX_CONTEXT_CHARS={args.max_context_chars}",
        f"ANSWER_MAX_TOKENS={args.answer_max_tokens}",
        f"DISABLE_QUERY_REWRITING={args.disable_query_rewriting}",
        f"ABLATION_NO_FILTER={args.ablation_no_filter}",
        "COMMAND=" + " ".join(sys.argv),
    ]
    paths["command_env"].write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    dataset_manifest = {
        "dataset": "LoCoMo10",
        "method": args.method,
        "dataset_path": str(args.dataset_path),
        "dataset_sha256": sha256_file(args.dataset_path),
        "subset_manifest": str(args.subset_manifest),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "checkpoint_root": str(args.checkpoint_root),
        "source_manifest": manifest,
    }
    write_json(paths["dataset_manifest"], dataset_manifest)

    completed_ids = {row.get("row_id") for row in read_jsonl(paths["raw_predictions"])} if args.resume else set()
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, max_retries=0)
    samples = load_locomo_dataset(args.dataset_path)
    if args.sample_ids:
        sample_filter = {str(item) for item in args.sample_ids}
        samples = [sample for sample in samples if sample.sample_id in sample_filter]

    predictions: list[dict[str, Any]] = read_jsonl(paths["raw_predictions"]) if args.resume else []
    token_usage: Counter[str] = Counter()
    started = time.time()
    run_log = [f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", f"method={args.method}", f"samples={len(samples)}"]
    processed = 0

    for sample in samples:
        sample_start = time.time()
        system = load_higmem_from_checkpoint(args, sample.sample_id, out_dir)
        raw_turns, raw_by_id = raw_turns_for_sample(sample)
        enhancer = None
        if args.method != "baseline_higmem":
            enhancer = HiGMemPlusEnhancer(
                dataset="locomo",
                method=args.method,
                component_k=args.component_k,
                edge_k=args.edge_k,
                episode_k=args.episode_k,
                max_context_chars=args.max_context_chars,
            )
            enhancer.add_turns(raw_turns)
        sample_trace = {
            "sample_id": sample.sample_id,
            "checkpoint": str(checkpoint_path(args.checkpoint_root, sample.sample_id)),
            "turn_count": len(raw_turns),
            "qa_count": len(sample.qa),
        }
        run_log.append(f"sample_start={json.dumps(sample_trace, sort_keys=True)}")

        for qa_index, qa in enumerate(sample.qa):
            if args.max_questions is not None and processed >= args.max_questions:
                break
            row_id = f"{sample.sample_id}:{qa_index}"
            if row_id in completed_ids:
                continue
            qa_started = time.time()
            category = int(qa.category or 0)
            metadata = category_metadata(category)
            if args.disable_query_rewriting:
                keyword_query = qa.question
                profile_keys: list[str] = []
            else:
                query_data = generate_keyword_query(system.llm, qa.question)
                keyword_query = query_data.get("keyword_query") or qa.question
                profile_keys = query_data.get("profile_retrieval_keys") or []

            retrieval_started = time.time()
            base_context, base_trace = system.retrieve_for_query(
                original_query=qa.question,
                keyword_query=keyword_query,
                profile_retrieval_keys=profile_keys,
                k_profile=args.k_profile,
                k_event=args.k_event,
                k_turn=args.k_turn,
                return_trace=True,
            )
            base_records = records_from_higmem_trace(base_trace, system)
            final_context = base_context
            evidence_records = list(base_records)
            component_trace: list[dict[str, Any]] = []
            graph_trace: list[dict[str, Any]] = []
            repair_trace: list[dict[str, Any]] = []
            episode_trace: list[dict[str, Any]] = []
            route_trace: list[dict[str, Any]] = []
            enhance_stats: dict[str, Any] = {"method": "baseline_higmem"}
            if enhancer is not None:
                result = enhancer.retrieve(
                    question=qa.question,
                    base_context=base_context,
                    base_records=base_records,
                    metadata=metadata,
                )
                final_context = result.context
                evidence_records = result.evidence_records
                component_trace = result.component_trace
                graph_trace = result.graph_trace
                repair_trace = result.repair_trace
                episode_trace = result.episode_trace
                route_trace = result.route_trace
                enhance_stats = result.stats
            retrieval_seconds = time.time() - retrieval_started

            prompt = build_prompt(category, final_context, qa.question, qa)
            answer_started = time.time()
            prediction, usage, error = answer_question(client, args, prompt)
            answer_seconds = time.time() - answer_started
            token_usage.update(usage)
            reference = qa_reference(qa)
            f1 = token_f1(prediction, reference)
            b1 = bleu1(prediction, reference)
            evidence_ids = [str(item) for item in qa.evidence or []]
            retrieved_turn_ids = evidence_ids_from_records(evidence_records)
            support = bool(set(evidence_ids) & retrieved_turn_ids) or evidence_text_support(final_context, evidence_ids, raw_by_id)
            supported_answer = is_answer_supported(prediction, reference, category)
            error_type = "correct" if supported_answer else ("retrieval_miss" if not support else "generation_miss")
            if enhance_stats.get("repair_needed") and not support:
                error_type = "repair_failed"
            row = {
                "row_id": row_id,
                "method": args.method,
                "sample_id": sample.sample_id,
                "qa_index": qa_index,
                "question": qa.question,
                "prediction": prediction,
                "reference": reference,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
                "evidence_ids": evidence_ids,
                "retrieved_turn_ids": sorted(retrieved_turn_ids),
                "evidence_support": support,
                "answer_correct_proxy": supported_answer,
                "error_type": error_type,
                "f1": f1,
                "bleu1": b1,
                "latency_seconds": time.time() - qa_started,
                "retrieval_seconds": retrieval_seconds,
                "answer_seconds": answer_seconds,
                "retrieved_context_chars": len(final_context or ""),
                "retrieved_context_tokens_approx": max(1, len(final_context or "") // 4) if final_context else 0,
                "base_context_tokens_approx": max(1, len(base_context or "") // 4) if base_context else 0,
                "answer_token_usage": usage,
                "answer_error": error,
                "higmem_trace": base_trace,
                "enhance_stats": enhance_stats,
                "sufficiency_status": enhance_stats.get("sufficiency_status"),
                "drill_down": bool(enhance_stats.get("repaired_turns", 0)),
            }
            predictions.append(row)
            append_jsonl(paths["raw_predictions"], row)
            append_jsonl(
                paths["retrieved_evidence"],
                {
                    "row_id": row_id,
                    "method": args.method,
                    "question": qa.question,
                    "context_sha256": hashlib.sha256((final_context or "").encode("utf-8")).hexdigest(),
                    "context_chars": len(final_context or ""),
                    "context_head": (final_context or "")[:4000],
                    "context_tail": (final_context or "")[-4000:] if len(final_context or "") > 4000 else "",
                    "evidence_records": evidence_records[:200],
                    "evidence_ids": evidence_ids,
                    "retrieved_turn_ids": sorted(retrieved_turn_ids),
                    "evidence_support": support,
                },
            )
            if component_trace:
                append_jsonl(paths["component_traces"], {"row_id": row_id, "trace": component_trace})
            if graph_trace:
                append_jsonl(paths["graph_traces"], {"row_id": row_id, "trace": graph_trace})
            if repair_trace:
                append_jsonl(paths["repair_traces"], {"row_id": row_id, "trace": repair_trace})
            if episode_trace:
                append_jsonl(paths["episode_traces"], {"row_id": row_id, "trace": episode_trace})
            if route_trace:
                append_jsonl(paths["route_traces"], {"row_id": row_id, "trace": route_trace})
            processed += 1
        contextlib_suppress_shutdown(system)
        run_log.append(f"sample_done={sample.sample_id} seconds={time.time() - sample_start:.3f}")
        if args.max_questions is not None and processed >= args.max_questions:
            break

    metrics = build_metrics(predictions)
    metrics["dataset"] = dataset_manifest
    metrics["artifact_paths"] = {key: str(value) for key, value in paths.items()}
    stats = {
        "method": args.method,
        "started_at": run_log[0].split("=", 1)[1],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": time.time() - started,
        "processed_this_run": processed,
        "total_rows": len(predictions),
        "token_usage": dict(token_usage),
    }
    write_json(paths["metrics"], metrics)
    write_json(paths["stats"], stats)
    paths["run_log"].write_text("\n".join(run_log + [f"overall={json.dumps(metrics['overall'], sort_keys=True)}"]) + "\n", encoding="utf-8")
    paths["summary"].write_text(build_summary(args, metrics, stats), encoding="utf-8")
    return {"metrics": metrics, "stats": stats}


def contextlib_suppress_shutdown(system: FPHMSystem) -> None:
    try:
        system.executor.shutdown(wait=True)
    except Exception:
        pass


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "f1": statistics.mean(float(row.get("f1", 0.0)) for row in rows),
        "bleu1": statistics.mean(float(row.get("bleu1", 0.0)) for row in rows),
        "judge_accuracy_proxy": sum(1 for row in rows if row.get("answer_correct_proxy")) / len(rows),
        "evidence_support_rate": sum(1 for row in rows if row.get("evidence_support")) / len(rows),
        "drill_down_rate": sum(1 for row in rows if row.get("drill_down")) / len(rows),
        "avg_context_tokens": statistics.mean(int(row.get("retrieved_context_tokens_approx", 0)) for row in rows),
        "avg_latency_seconds": statistics.mean(float(row.get("latency_seconds", 0.0)) for row in rows),
        "avg_retrieval_seconds": statistics.mean(float(row.get("retrieval_seconds", 0.0)) for row in rows),
        "avg_answer_seconds": statistics.mean(float(row.get("answer_seconds", 0.0)) for row in rows),
        "avg_total_tokens": statistics.mean(int((row.get("answer_token_usage") or {}).get("total_tokens", 0)) for row in rows),
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"overall": summarize_rows(rows), "by_category": {}, "by_error_type": {}}
    grouped_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_category[str(row.get("category_name") or row.get("category"))].append(row)
        grouped_error[str(row.get("error_type") or "unknown")].append(row)
    metrics["by_category"] = {key: summarize_rows(value) for key, value in sorted(grouped_category.items())}
    metrics["by_error_type"] = {key: summarize_rows(value) for key, value in sorted(grouped_error.items())}
    metrics["bad_cases"] = [
        {
            "row_id": row.get("row_id"),
            "question": row.get("question"),
            "prediction": row.get("prediction"),
            "reference": row.get("reference"),
            "category_name": row.get("category_name"),
            "error_type": row.get("error_type"),
            "evidence_support": row.get("evidence_support"),
            "sufficiency_status": row.get("sufficiency_status"),
            "f1": row.get("f1"),
        }
        for row in rows
        if not row.get("answer_correct_proxy")
    ][:80]
    return metrics


def build_summary(args: argparse.Namespace, metrics: dict[str, Any], stats: dict[str, Any]) -> str:
    overall = metrics["overall"]
    lines = [
        f"# LoCoMo10 10% {args.method} Fast Run",
        "",
        f"- Count: {overall.get('count')}",
        f"- F1: {overall.get('f1')}",
        f"- BLEU1: {overall.get('bleu1')}",
        f"- Judge accuracy proxy: {overall.get('judge_accuracy_proxy')}",
        f"- Evidence support rate: {overall.get('evidence_support_rate')}",
        f"- Drill-down rate: {overall.get('drill_down_rate')}",
        f"- Avg context tokens: {overall.get('avg_context_tokens')}",
        f"- Avg latency seconds: {overall.get('avg_latency_seconds')}",
        f"- Runtime seconds: {stats.get('runtime_seconds')}",
        "",
        "## By Category",
        "",
    ]
    for key, value in metrics.get("by_category", {}).items():
        lines.append(
            f"- {key}: count={value.get('count')} f1={value.get('f1')} "
            f"support={value.get('evidence_support_rate')} latency={value.get('avg_latency_seconds')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps({"method": args.method, "overall": result["metrics"]["overall"], "output_dir": str(args.output_dir / args.method)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
