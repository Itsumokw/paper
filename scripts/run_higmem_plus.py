#!/usr/bin/env python3
"""Run HiGMemPlus methods on manifest-controlled LongDialQA/DialSim smoke sets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from baseline.HiGMemPlus import METHODS, HiGMemPlusEnhancer, RawTurn  # noqa: E402
from run_longdialqa_baseline import (  # noqa: E402
    HiGMemAdapter,
    append_jsonl,
    answer_with_openai,
    approx_tokens,
    build_metrics,
    build_prompt,
    context_artifact,
    distill_option,
    flatten_retrieved_scene_ids,
    load_sessions,
    load_subset_from_manifest,
    predicted_option_text,
    read_jsonl,
    retrieved_records_artifact,
    sha256_file,
    token_f1,
    turn_records_for_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["longdialqa"], default="longdialqa")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--normalized-dir", type=Path, default=Path("datasets/DialSim/longdialqa_normalized_v1.1_seed0"))
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/higmem_plus"))
    parser.add_argument("--shows", nargs="+", default=["friends", "bigbang", "theoffice"])
    parser.add_argument("--row-shard-index", type=int, default=0, help="0-based contiguous QA row shard index after show filtering.")
    parser.add_argument("--row-shard-count", type=int, default=1, help="Number of contiguous QA row shards after show filtering.")
    parser.add_argument("--max-qa-per-show", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-per-show", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--history-scene-limit",
        type=int,
        default=int(os.environ.get("HIGMEM_PLUS_HISTORY_SCENE_LIMIT", "0")),
        help="Engineering-smoke only: feed only the latest N scenes before each question. 0 preserves full allowed history.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--component-k", type=int, default=10)
    parser.add_argument("--edge-k", type=int, default=8)
    parser.add_argument("--episode-k", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--answer-max-tokens", type=int, default=int(os.environ.get("LONGDIALQA_ANSWER_MAX_TOKENS", "32")))
    parser.add_argument("--answer-timeout", type=float, default=float(os.environ.get("LONGDIALQA_ANSWER_TIMEOUT", "120")))
    parser.add_argument("--max-context-chars", type=int, default=int(os.environ.get("HIGMEM_PLUS_MAX_CONTEXT_CHARS", "60000")))
    parser.add_argument("--max-saved-context-chars", type=int, default=int(os.environ.get("LONGDIALQA_MAX_SAVED_CONTEXT_CHARS", "20000")))
    parser.add_argument("--max-saved-records", type=int, default=int(os.environ.get("LONGDIALQA_MAX_SAVED_RECORDS", "200")))
    parser.add_argument("--qa-workers", type=int, default=int(os.environ.get("HIGMEM_PLUS_QA_WORKERS", "1")))
    parser.add_argument("--qa-view-executor-workers", type=int, default=int(os.environ.get("HIGMEM_PLUS_QA_VIEW_EXECUTOR_WORKERS", "4")))
    parser.add_argument("--resume", action="store_true", help="Reuse existing JSONL artifacts and skip completed QA ids.")
    parser.add_argument("--disable-higmem-query-rewrite", action="store_true", default=os.environ.get("LONGDIALQA_HIGMEM_DISABLE_QUERY_REWRITE", "1") == "1")
    parser.add_argument("--higmem-scene-unit", action="store_true", default=os.environ.get("LONGDIALQA_HIGMEM_SCENE_UNIT", "1") == "1")
    parser.add_argument("--higmem-scene-max-chars", type=int, default=int(os.environ.get("LONGDIALQA_HIGMEM_SCENE_MAX_CHARS", "8000")))
    parser.add_argument("--higmem-link-window", type=int, default=int(os.environ.get("LONGDIALQA_HIGMEM_LINK_WINDOW", "1")))
    parser.add_argument("--higmem-k-event-affiliation", type=int, default=int(os.environ.get("LONGDIALQA_HIGMEM_K_EVENT_AFFILIATION", "10")))
    return parser.parse_args()


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.row_shard_count < 1:
        raise ValueError("--row-shard-count must be >= 1")
    if args.row_shard_index < 0 or args.row_shard_index >= args.row_shard_count:
        raise ValueError("--row-shard-index must be in [0, row_shard_count)")
    rows = [row for row in rows if row.get("show") in set(args.shows)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["show"])].append(row)
    selected = []
    for show in sorted(grouped):
        show_rows = sorted(grouped[show], key=lambda row: (int(row["session_ordinal"]), int(row["ask_turn_index"]), str(row["id"])))
        if args.smoke_per_show is not None:
            show_rows = show_rows[: args.smoke_per_show]
        selected.extend(show_rows)
    selected.sort(key=lambda row: (str(row["show"]), int(row["session_ordinal"]), int(row["ask_turn_index"]), str(row["id"])))
    if args.row_shard_count > 1:
        shard_size = (len(selected) + args.row_shard_count - 1) // args.row_shard_count
        start = args.row_shard_index * shard_size
        end = min(len(selected), start + shard_size)
        selected = selected[start:end]
    if args.max_questions is not None:
        selected = selected[: args.max_questions]
    return selected


def to_raw_turn(turn: Any) -> RawTurn:
    return RawTurn(
        turn_id=turn.turn_id,
        text=turn.text,
        speaker=turn.speaker,
        timestamp=turn.date,
        dataset="longdialqa",
        show=turn.show,
        show_name=turn.show_name,
        episode_id="",
        session_or_scene_id=turn.scene_id,
        chronological_order=turn.session_ordinal,
        turn_index=turn.turn_index,
        metadata={
            "raw": turn.raw,
            "scene_id": turn.scene_id,
            "session_ordinal": turn.session_ordinal,
        },
    )


def prepare_paths(out_dir: Path, resume: bool = False) -> dict[str, Path]:
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
    for key, path in paths.items():
        if (
            not resume
            and key not in {"command_env", "dataset_manifest", "metrics", "stats", "summary", "run_log"}
            and path.exists()
        ):
            path.unlink()
        if path.suffix == ".jsonl":
            path.touch()
    return paths


def save_command_env(path: Path, args: argparse.Namespace, subset_path: Path, normalized_manifest: Path) -> None:
    lines = [
        f"DATASET={args.dataset}",
        f"METHOD={args.method}",
        f"OPENAI_MODEL={args.model}",
        f"OPENAI_BASE_URL={args.api_base}",
        f"NORMALIZED_DIR={args.normalized_dir}",
        f"NORMALIZED_MANIFEST={normalized_manifest}",
        f"NORMALIZED_MANIFEST_SHA256={sha256_file(normalized_manifest)}",
        f"SUBSET_MANIFEST={args.subset_manifest}",
        f"SUBSET_MANIFEST_SHA256={sha256_file(args.subset_manifest)}",
        f"SUBSET_PATH={subset_path}",
        f"SUBSET_SHA256={sha256_file(subset_path)}",
        f"SHOWS={','.join(args.shows)}",
        f"ROW_SHARD_INDEX={args.row_shard_index}",
        f"ROW_SHARD_COUNT={args.row_shard_count}",
        f"SMOKE_PER_SHOW={args.smoke_per_show}",
        f"MAX_QUESTIONS={args.max_questions}",
        f"HISTORY_SCENE_LIMIT={args.history_scene_limit}",
        f"TOP_K={args.top_k}",
        f"COMPONENT_K={args.component_k}",
        f"EDGE_K={args.edge_k}",
        f"EPISODE_K={args.episode_k}",
        f"ANSWER_MAX_TOKENS={args.answer_max_tokens}",
        f"ANSWER_TIMEOUT={args.answer_timeout}",
        f"MAX_CONTEXT_CHARS={args.max_context_chars}",
        f"QA_WORKERS={args.qa_workers}",
        f"QA_VIEW_EXECUTOR_WORKERS={args.qa_view_executor_workers}",
        f"HIGMEM_SCENE_UNIT={args.higmem_scene_unit}",
        f"HIGMEM_SCENE_MAX_CHARS={args.higmem_scene_max_chars}",
        f"HIGMEM_LINK_WINDOW={args.higmem_link_window}",
        f"HIGMEM_DISABLE_QUERY_REWRITE={args.disable_higmem_query_rewrite}",
        f"HIGMEM_MAX_TOKENS={os.environ.get('HIGMEM_MAX_TOKENS', '2048')}",
        "COMMAND=" + " ".join(sys.argv),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clone_embedding_retriever(retriever: Any) -> Any:
    clone = object.__new__(retriever.__class__)
    clone.model = retriever.model
    clone.corpus = list(getattr(retriever, "corpus", []) or [])
    clone.document_ids = list(getattr(retriever, "document_ids", []) or [])
    clone.id_to_index = dict(getattr(retriever, "id_to_index", {}) or {})
    embeddings = getattr(retriever, "embeddings", None)
    clone.embeddings = embeddings.copy() if embeddings is not None else None
    return clone


def clone_adapter_for_qa(adapter: HiGMemAdapter, args: argparse.Namespace) -> HiGMemAdapter:
    """Freeze the current HiGMem state for one QA without changing visible history."""
    if adapter.scene_unit:
        adapter._flush_scene()
    from memory_layer import LLMController

    llm = LLMController("openai", model=args.model, api_key=args.api_key, api_base=args.api_base)
    qa_system = adapter.system.spawn_qa_view(llm_controller=llm, executor_workers=args.qa_view_executor_workers)
    qa_system.turn_notes = copy.deepcopy(adapter.system.turn_notes)
    qa_system.events = copy.deepcopy(adapter.system.events)
    qa_system.profiles = copy.deepcopy(adapter.system.profiles)
    qa_system.recent_turns_window = copy.deepcopy(adapter.system.recent_turns_window)
    qa_system.turn_retriever = clone_embedding_retriever(adapter.system.turn_retriever)
    qa_system.event_retriever = clone_embedding_retriever(adapter.system.event_retriever)
    if getattr(adapter.system, "use_character_profile", False) and hasattr(adapter.system, "profile_retriever"):
        qa_system.profile_retriever = clone_embedding_retriever(adapter.system.profile_retriever)

    qa_adapter = copy.copy(adapter)
    qa_adapter.system = qa_system
    qa_adapter.scene_key = None
    qa_adapter.scene_turns = []
    return qa_adapter


def evaluate_one_row(
    *,
    row: dict[str, Any],
    adapter: HiGMemAdapter,
    enhancer: HiGMemPlusEnhancer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, max_retries=0)
    qa_started = time.time()
    retrieval_started = time.time()
    base_context, base_records, base_stats = adapter.retrieve(row, args.top_k)
    if args.method == "baseline_higmem":
        plus_result = enhancer.retrieve(
            question=row["question"],
            base_context=base_context,
            base_records=base_records,
            metadata={**row, "dataset": "longdialqa"},
        )
        context = base_context[: args.max_context_chars] if args.max_context_chars > 0 else base_context
        retrieved_records = base_records
        method_stats = {"base_stats": base_stats, "plus_stats": plus_result.stats}
    else:
        plus_result = enhancer.retrieve(
            question=row["question"],
            base_context=base_context,
            base_records=base_records,
            metadata={**row, "dataset": "longdialqa"},
        )
        context = plus_result.context
        retrieved_records = plus_result.evidence_records
        method_stats = {"base_stats": base_stats, "plus_stats": plus_result.stats}
    retrieval_seconds = time.time() - retrieval_started

    prompt = build_prompt(row, context)
    answer_started = time.time()
    answer, usage, error = answer_with_openai(client, args, prompt)
    answer_seconds = time.time() - answer_started
    if error:
        failures.append({"qa_id": row["id"], "stage": "answer", "error": error})

    predicted_option, ambiguous = distill_option(answer)
    strict_correct = bool(predicted_option and predicted_option == row["gold_option"] and not ambiguous)
    correct = strict_correct
    if not predicted_option and row["gold_option"] == "(E)":
        lowered = answer.lower()
        if "don't know" in lowered or "cannot answer" in lowered or "insufficient" in lowered or "not enough" in lowered:
            predicted_option = "(E)"
            correct = True
    pred_text = predicted_option_text(predicted_option, row["options"], answer)
    retrieved_scene_ids = flatten_retrieved_scene_ids(retrieved_records)
    evidence_scene_ids = [str(item) for item in row.get("evidence_scene_ids") or []]
    has_evidence = bool(evidence_scene_ids)
    evidence_recall_any = bool(has_evidence and set(evidence_scene_ids).intersection(retrieved_scene_ids))
    evidence_recall_all = bool(has_evidence and set(evidence_scene_ids).issubset(set(retrieved_scene_ids)))

    common = {
        "method": args.method,
        "baseline": "higmem",
        "qa_id": row["id"],
        "show": row["show"],
        "show_name": row["show_name"],
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "session_ordinal": row["session_ordinal"],
        "question": row["question"],
        "question_prompt": row["question_prompt"],
        "question_source": row["question_source"],
        "question_type": row["question_type"],
        "answerability_label": "answerable" if row["answerable"] else "unanswerable",
        "hop_type": row["hop_type"],
        "gold_option": row["gold_option"],
        "gold_answer": row["answer"],
        "prediction": answer,
        "answer_error": error,
        "predicted_option": predicted_option,
        "predicted_answer_text": pred_text,
        "ambiguous": ambiguous,
        "strict_correct": strict_correct,
        "correct": correct,
        "token_f1": token_f1(pred_text, row["answer"]),
        "retrieved_k": len(retrieved_records),
        "retrieved_context_chars": len(context or ""),
        "retrieved_context_tokens_approx": approx_tokens(context or ""),
        "retrieved_scene_ids": retrieved_scene_ids,
        "evidence_scene_ids": evidence_scene_ids,
        "has_evidence_scene": has_evidence,
        "evidence_recall_any": evidence_recall_any,
        "evidence_recall_all": evidence_recall_all,
        "latency_seconds": time.time() - qa_started,
        "retrieval_seconds": retrieval_seconds,
        "answer_seconds": answer_seconds,
        "answer_token_usage": usage,
        "method_stats": method_stats,
    }
    retrieved_row = {
        **{key: common[key] for key in ["method", "qa_id", "show", "show_name", "question", "question_type"]},
        **context_artifact(context, args.max_saved_context_chars),
        **retrieved_records_artifact(retrieved_records, args.max_saved_records),
        "retrieved_scene_ids": retrieved_scene_ids,
        "evidence_scene_ids": evidence_scene_ids,
        "evidence_recall_any": evidence_recall_any,
        "evidence_recall_all": evidence_recall_all,
    }
    return {
        "prediction": common,
        "retrieved_evidence": retrieved_row,
        "component_trace": plus_result.component_trace,
        "graph_trace": plus_result.graph_trace,
        "repair_trace": plus_result.repair_trace,
        "episode_trace": plus_result.episode_trace,
        "route_trace": plus_result.route_trace,
        "failures": failures,
        "usage": usage,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.openai_base_url = args.api_base
    args.openai_api_key = args.api_key
    normalized_manifest = args.normalized_dir / "manifest.json"
    sessions_path = args.normalized_dir / "sessions.jsonl"
    all_rows, subset_path, subset_manifest = load_subset_from_manifest(args)
    selected_rows = select_rows(all_rows, args)
    sessions = load_sessions(sessions_path)

    out_dir = args.output_dir / args.method
    paths = prepare_paths(out_dir, resume=args.resume)
    save_command_env(paths["command_env"], args, subset_path, normalized_manifest)
    dataset_manifest = {
        "dataset": "LongDialQA/DialSim",
        "method": args.method,
        "normalized_manifest": str(normalized_manifest),
        "normalized_manifest_sha256": sha256_file(normalized_manifest),
        "subset_manifest": str(args.subset_manifest),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "subset_path": str(subset_path),
        "subset_sha256": sha256_file(subset_path),
        "source_subset_rows": len(all_rows),
        "selected_rows": len(selected_rows),
        "shows_filter": list(args.shows),
        "row_shard_index": args.row_shard_index,
        "row_shard_count": args.row_shard_count,
        "smoke_per_show": args.smoke_per_show,
        "max_questions": args.max_questions,
        "history_scene_limit": args.history_scene_limit,
        "selected_qa_ids": [row["id"] for row in selected_rows],
        "source_manifest": subset_manifest,
    }
    paths["dataset_manifest"].write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    run_log = [
        f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"method={args.method}",
        f"selected_rows={len(selected_rows)}",
        f"subset_manifest={args.subset_manifest}",
    ]
    client = OpenAI(api_key=args.api_key, base_url=args.api_base, max_retries=0)
    predictions: list[dict[str, Any]] = read_jsonl(paths["raw_predictions"]) if args.resume else []
    completed_ids = {
        str(row.get("qa_id"))
        for row in predictions
        if row.get("qa_id") and not row.get("answer_error") and str(row.get("prediction") or "").strip()
    }
    failures: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "method": args.method,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_rows": len(selected_rows),
        "source_subset_rows": len(all_rows),
        "token_usage": Counter(),
        "shows": {},
    }
    started = time.time()

    subset_by_show: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        subset_by_show[str(row["show"])].append(row)
    for show_rows in subset_by_show.values():
        show_rows.sort(key=lambda row: (int(row["session_ordinal"]), int(row["ask_turn_index"]), str(row["id"])))

    for show, qa_rows in sorted(subset_by_show.items()):
        show_started = time.time()
        adapter = HiGMemAdapter(args, out_dir, show)
        enhancer = HiGMemPlusEnhancer(
            dataset="longdialqa",
            method=args.method,
            component_k=args.component_k,
            edge_k=args.edge_k,
            episode_k=args.episode_k,
            max_context_chars=args.max_context_chars,
        )
        show_sessions = sessions[show]
        next_session_idx = 0
        next_turn_idx_in_session = 0

        def feed_until(target_session_ordinal: int, target_turn_index: int) -> None:
            nonlocal next_session_idx, next_turn_idx_in_session
            if args.history_scene_limit > 0:
                start_session_ordinal = max(1, target_session_ordinal - args.history_scene_limit + 1)
                for session in show_sessions:
                    session_ordinal = int(session["session_ordinal"])
                    if session_ordinal < start_session_ordinal or session_ordinal > target_session_ordinal:
                        continue
                    turns = turn_records_for_session(session)
                    limit = target_turn_index if session_ordinal == target_session_ordinal else len(turns)
                    for turn in turns[: min(limit, len(turns))]:
                        adapter.add_turn(turn)
                        enhancer.add_turn(to_raw_turn(turn))
                return
            while next_session_idx < len(show_sessions):
                session = show_sessions[next_session_idx]
                session_ordinal = int(session["session_ordinal"])
                turns = turn_records_for_session(session)
                if session_ordinal < target_session_ordinal:
                    while next_turn_idx_in_session < len(turns):
                        turn = turns[next_turn_idx_in_session]
                        adapter.add_turn(turn)
                        enhancer.add_turn(to_raw_turn(turn))
                        next_turn_idx_in_session += 1
                    next_session_idx += 1
                    next_turn_idx_in_session = 0
                    continue
                if session_ordinal == target_session_ordinal:
                    limit = min(target_turn_index, len(turns))
                    while next_turn_idx_in_session < limit:
                        turn = turns[next_turn_idx_in_session]
                        adapter.add_turn(turn)
                        enhancer.add_turn(to_raw_turn(turn))
                        next_turn_idx_in_session += 1
                    return
                return

        def emit_result(result: dict[str, Any]) -> None:
            common = result["prediction"]
            predictions.append(common)
            append_jsonl(paths["raw_predictions"], common)
            append_jsonl(paths["retrieved_evidence"], result["retrieved_evidence"])
            stats["token_usage"].update(result.get("usage") or {})
            for failure in result.get("failures") or []:
                failures.append(failure)
                append_jsonl(paths["run_log"], {"failure": failure})
            for trace_key, trace_path in [
                ("component_trace", paths["component_traces"]),
                ("graph_trace", paths["graph_traces"]),
                ("repair_trace", paths["repair_traces"]),
                ("episode_trace", paths["episode_traces"]),
                ("route_trace", paths["route_traces"]),
            ]:
                append_jsonl(trace_path, {"qa_id": common["qa_id"], "method": args.method, "trace": result[trace_key]})

        def emit_failure(row: dict[str, Any], exc: BaseException, qa_started: float) -> None:
            failure = {
                "qa_id": row.get("id"),
                "stage": "qa",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            append_jsonl(paths["run_log"], {"failure": failure})
            failed_prediction = {
                "method": args.method,
                "baseline": "higmem",
                "qa_id": row.get("id"),
                "show": row.get("show"),
                "show_name": row.get("show_name"),
                "question": row.get("question"),
                "question_source": row.get("question_source"),
                "question_type": row.get("question_type"),
                "answerability_label": "answerable" if row.get("answerable") else "unanswerable",
                "hop_type": row.get("hop_type"),
                "gold_option": row.get("gold_option"),
                "gold_answer": row.get("answer"),
                "prediction": "",
                "predicted_option": "",
                "predicted_answer_text": "",
                "ambiguous": False,
                "strict_correct": False,
                "correct": False,
                "token_f1": 0.0,
                "retrieved_k": 0,
                "retrieved_context_chars": 0,
                "retrieved_context_tokens_approx": 0,
                "retrieved_scene_ids": [],
                "evidence_scene_ids": row.get("evidence_scene_ids") or [],
                "has_evidence_scene": bool(row.get("evidence_scene_ids")),
                "evidence_recall_any": False,
                "evidence_recall_all": False,
                "latency_seconds": time.time() - qa_started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            predictions.append(failed_prediction)
            append_jsonl(paths["raw_predictions"], failed_prediction)

        if args.qa_workers > 1 and args.history_scene_limit == 0:
            pending = {}
            max_pending = max(1, args.qa_workers * 2)
            with ThreadPoolExecutor(max_workers=args.qa_workers) as qa_pool:
                def drain(block: bool = False) -> None:
                    nonlocal pending
                    if not pending:
                        return
                    done, _not_done = wait(
                        pending,
                        return_when=FIRST_COMPLETED if block else FIRST_COMPLETED,
                        timeout=None if block else 0,
                    )
                    for future in done:
                        row_for_future = pending.pop(future)
                        try:
                            emit_result(future.result())
                        except Exception as exc:  # noqa: BLE001
                            emit_failure(row_for_future, exc, time.time())

                for row in qa_rows:
                    qa_started = time.time()
                    try:
                        feed_until(int(row["session_ordinal"]), int(row["ask_turn_index"]))
                        if str(row["id"]) in completed_ids:
                            continue
                        adapter_snapshot = clone_adapter_for_qa(adapter, args)
                        enhancer_snapshot = copy.deepcopy(enhancer)
                        future = qa_pool.submit(
                            evaluate_one_row,
                            row=row,
                            adapter=adapter_snapshot,
                            enhancer=enhancer_snapshot,
                            args=args,
                        )
                        pending[future] = row
                        if len(pending) >= max_pending:
                            drain(block=True)
                        else:
                            drain(block=False)
                    except Exception as exc:  # noqa: BLE001
                        emit_failure(row, exc, qa_started)
                while pending:
                    drain(block=True)
        else:
            for row in qa_rows:
                qa_started = time.time()
                try:
                    feed_until(int(row["session_ordinal"]), int(row["ask_turn_index"]))
                    if str(row["id"]) in completed_ids:
                        continue
                    emit_result(evaluate_one_row(row=row, adapter=adapter, enhancer=enhancer, args=args))
                except Exception as exc:  # noqa: BLE001
                    emit_failure(row, exc, qa_started)
                    continue

        show_stats = adapter.finalize()
        stats["shows"][show] = {**show_stats, "qa_rows": len(qa_rows), "seconds": time.time() - show_started}
        run_log.append(f"show_done={show} stats={json.dumps(stats['shows'][show], ensure_ascii=False, sort_keys=True)}")

    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stats["runtime_seconds"] = time.time() - started
    stats["token_usage"] = dict(stats["token_usage"])
    stats["failures"] = failures
    metrics = build_metrics(predictions)
    metrics["artifact_paths"] = {key: str(path) for key, path in paths.items()}
    metrics["dataset"] = dataset_manifest
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["stats"].write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    run_log.append(f"finished_at={stats['finished_at']}")
    run_log.append(f"runtime_seconds={stats['runtime_seconds']:.3f}")
    run_log.append(f"overall={json.dumps(metrics['overall'], ensure_ascii=False, sort_keys=True)}")
    paths["run_log"].write_text("\n".join(str(line) for line in run_log) + "\n", encoding="utf-8")
    paths["summary"].write_text(
        "\n".join(
            [
                f"# HiGMemPlus {args.method} LongDialQA/DialSim Smoke",
                "",
                f"- Method: `{args.method}`",
                f"- Selected rows: {len(selected_rows)}",
                f"- Accuracy: {metrics['overall'].get('accuracy')}",
                f"- Mean retrieved K: {metrics['overall'].get('mean_retrieved_k')}",
                f"- Mean retrieved tokens approx: {metrics['overall'].get('mean_context_tokens_approx')}",
                f"- Runtime seconds: {stats['runtime_seconds']}",
                f"- Dataset manifest: `{paths['dataset_manifest']}`",
                f"- Raw predictions: `{paths['raw_predictions']}`",
                f"- Retrieved evidence: `{paths['retrieved_evidence']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics, "stats": stats}


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps({"method": args.method, "overall": result["metrics"]["overall"], "output_dir": str(args.output_dir / args.method)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
