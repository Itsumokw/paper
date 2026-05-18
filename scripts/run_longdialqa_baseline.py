#!/usr/bin/env python3
"""Run one memory baseline on normalized LongDialQA/DialSim rows.

The runner uses the normalized adapter artifacts from
`normalize_longdialqa_dialsim.py`.  It evaluates one baseline per process so
baseline repositories with conflicting module names stay isolated.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
from collections import Counter
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


BASELINES = ("full_context", "a_mem", "mem0", "simplemem", "higmem", "memgas")
LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)"]
ROOT = Path(__file__).resolve().parents[1]
RAG_PROMPT_TEMPLATE = (ROOT / "baseline/DialSim/prompt/RAG_qa_prompt_multi_choice_structured.txt").read_text(
    encoding="utf-8"
)


@dataclass
class TurnRecord:
    show: str
    show_name: str
    scene_id: str
    session_ordinal: int
    date: str
    turn_id: str
    turn_index: int
    speaker: str
    text: str
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("datasets/DialSim/longdialqa_normalized_v1.1_seed0"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/longdialqa_baselines/qwen25_3b_seed0_smoke"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output directory name under --run-root. Use it for safe show-sharded parallel runs.",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        required=True,
        help="Required deterministic subset manifest. The runner refuses to infer full-data defaults.",
    )
    parser.add_argument("--shows", nargs="+", default=["friends", "bigbang", "theoffice"])
    parser.add_argument("--max-qa-per-show", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--answer-max-tokens", type=int, default=int(os.environ.get("LONGDIALQA_ANSWER_MAX_TOKENS", "32")))
    parser.add_argument("--answer-timeout", type=float, default=float(os.environ.get("LONGDIALQA_ANSWER_TIMEOUT", "120")))
    parser.add_argument("--full-context-max-chars", type=int, default=int(os.environ.get("LONGDIALQA_FULL_CONTEXT_MAX_CHARS", "60000")))
    parser.add_argument("--retrieved-context-max-chars", type=int, default=int(os.environ.get("LONGDIALQA_RETRIEVED_CONTEXT_MAX_CHARS", "60000")))
    parser.add_argument("--max-saved-context-chars", type=int, default=int(os.environ.get("LONGDIALQA_MAX_SAVED_CONTEXT_CHARS", "20000")))
    parser.add_argument("--max-saved-records", type=int, default=int(os.environ.get("LONGDIALQA_MAX_SAVED_RECORDS", "200")))
    parser.add_argument(
        "--resume",
        action="store_true",
        default=os.environ.get("LONGDIALQA_RESUME", "0") == "1",
        help="Resume an interrupted run by reusing existing artifact rows and skipping completed QA ids.",
    )
    parser.add_argument(
        "--amem-evo-threshold",
        type=int,
        default=int(os.environ.get("LONGDIALQA_AMEM_EVO_THRESHOLD", "100000000")),
        help="A-MEM memory evolution threshold. Default avoids repeated JSON-evolution failures with the local 3B judge model.",
    )
    parser.add_argument(
        "--amem-disable-evolution",
        action="store_true",
        default=os.environ.get("LONGDIALQA_AMEM_DISABLE_EVOLUTION", "1") == "1",
        help="Skip A-MEM evolution LLM calls while preserving note storage, embeddings, retrieval, and answer evaluation.",
    )
    parser.add_argument(
        "--amem-disable-analysis",
        action="store_true",
        default=os.environ.get("LONGDIALQA_AMEM_DISABLE_ANALYSIS", "1") == "1",
        help="Skip A-MEM per-note LLM keyword/context/tag analysis for reproducible local-model runs.",
    )
    parser.add_argument(
        "--memgas-force-heuristic-granularity",
        action="store_true",
        default=os.environ.get("LONGDIALQA_MEMGAS_FORCE_HEURISTIC_GRANULARITY", "1") == "1",
        help="Use deterministic local summary/keyword extraction for MemGAS instead of LLM-based granularity generation.",
    )
    parser.add_argument("--embedding-model", default=os.environ.get("LONGDIALQA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--model-path", default=os.environ.get("QWEN25_3B_MODEL_PATH", "/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean"))
    parser.add_argument("--disable-higmem-query-rewrite", action="store_true", default=os.environ.get("LONGDIALQA_HIGMEM_DISABLE_QUERY_REWRITE", "0") == "1")
    parser.add_argument(
        "--higmem-scene-unit",
        action="store_true",
        default=os.environ.get("LONGDIALQA_HIGMEM_SCENE_UNIT", "1") == "1",
        help="Feed LongDialQA scenes as HiGMem turns for tractable DialSim reproduction.",
    )
    parser.add_argument(
        "--higmem-scene-max-chars",
        type=int,
        default=int(os.environ.get("LONGDIALQA_HIGMEM_SCENE_MAX_CHARS", "10000")),
        help="Maximum characters from a LongDialQA scene passed into HiGMem construction; 0 disables truncation.",
    )
    parser.add_argument(
        "--higmem-link-window",
        type=int,
        default=int(os.environ.get("LONGDIALQA_HIGMEM_LINK_WINDOW", "2")),
        help="Recent scene-unit window used by HiGMem event-affiliation prompts for DialSim reproduction.",
    )
    parser.add_argument(
        "--higmem-k-event-affiliation",
        type=int,
        default=int(os.environ.get("LONGDIALQA_HIGMEM_K_EVENT_AFFILIATION", "10")),
        help="Candidate event count used by HiGMem event-affiliation prompts.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list records in {path}")
    return data


def load_subset_from_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    subset_path = Path(manifest["subset_path"])
    if not subset_path.is_absolute():
        subset_path = ROOT / subset_path
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file from manifest does not exist: {subset_path}")
    rows = read_records(subset_path)
    if not rows:
        raise ValueError(f"Subset is empty: {subset_path}")
    if args.max_qa_per_show is not None:
        raise ValueError("--max-qa-per-show is disabled for manifest-controlled runs")
    expected_sha = manifest.get("subset_sha256")
    if expected_sha and sha256_file(subset_path) != expected_sha:
        raise ValueError(f"Subset hash mismatch for {subset_path}")
    return rows, subset_path, manifest


def load_sessions(path: Path) -> dict[str, list[dict[str, Any]]]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        sessions[row["show"]].append(row)
    for show_rows in sessions.values():
        show_rows.sort(key=lambda row: row["session_ordinal"])
    return dict(sessions)


def turn_records_for_session(session: dict[str, Any]) -> list[TurnRecord]:
    rows = []
    for turn in session["turns"]:
        rows.append(
            TurnRecord(
                show=session["show"],
                show_name=session["show_name"],
                scene_id=session["scene_id"],
                session_ordinal=int(session["session_ordinal"]),
                date=session["date"],
                turn_id=turn["turn_id"],
                turn_index=int(turn["turn_index"]),
                speaker=turn["speaker"],
                text=turn["text"],
                raw=turn["raw"],
            )
        )
    return rows


def turn_content(turn: TurnRecord) -> str:
    return f"[Date: {turn.date}; Scene: {turn.scene_id}; Turn: {turn.turn_id}] Speaker {turn.speaker} says: {turn.text}"


def context_block_from_turn(turn: TurnRecord) -> dict[str, Any]:
    content = turn_content(turn)
    return {
        "content": content,
        "used_content": content,
        "metadata": {
            "show": turn.show,
            "show_name": turn.show_name,
            "scene_id": turn.scene_id,
            "session_ordinal": turn.session_ordinal,
            "date": turn.date,
            "turn_id": turn.turn_id,
            "turn_index": turn.turn_index,
            "speaker": turn.speaker,
        },
    }


def extract_scene_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"Scene:\s*([A-Za-z0-9_]+)", text)))


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def context_artifact(context: str, max_chars: int) -> dict[str, Any]:
    context = context or ""
    artifact = {
        "context_chars": len(context),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "context_truncated_for_artifact": False,
        "context": context,
    }
    if max_chars >= 0 and len(context) > max_chars:
        half = max(0, max_chars // 2)
        artifact = {
            "context_chars": len(context),
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "context_truncated_for_artifact": True,
            "context_head": context[:half],
            "context_tail": context[-half:] if half else "",
        }
    return artifact


def retrieved_records_artifact(records: list[dict[str, Any]], max_records: int) -> dict[str, Any]:
    def shrink(record: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(record.get("metadata") or {})
        used_content = str(record.get("used_content") or "")
        return {
            "content_sha256": hashlib.sha256(used_content.encode("utf-8")).hexdigest(),
            "used_content_preview": used_content[:500],
            "metadata": metadata,
        }

    if max_records < 0 or len(records) <= max_records:
        return {"retrieved_records": records, "retrieved_records_truncated": False}
    half = max(1, max_records // 2)
    saved = [shrink(record) for record in records[:half]] + [shrink(record) for record in records[-half:]]
    return {
        "retrieved_records": saved,
        "retrieved_records_truncated": True,
        "retrieved_records_original_count": len(records),
    }


def normalize_answer_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return [tok for tok in text.split() if tok]


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer_text(pred)
    gold_toks = normalize_answer_text(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def distill_option(answer: str) -> tuple[str, bool]:
    upper = answer.upper()
    present = [label for label in LABELS if label in upper]
    if len(present) == 1:
        return present[0], False
    if len(present) > 1:
        return "", True
    return "", False


def predicted_option_text(option: str, options: list[str], raw_answer: str) -> str:
    if option in LABELS:
        idx = LABELS.index(option)
        if idx < len(options):
            return options[idx]
    return raw_answer


def build_prompt(row: dict[str, Any], context: str) -> str:
    context = context.strip() or "No history.\n"
    return (
        RAG_PROMPT_TEMPLATE.replace("<<<Chatbot>>>", row.get("chatbot") or "the agent")
        .replace("<<<Date>>>", row.get("date") or "")
        .replace("<<<Dialog_History>>>", context)
        .replace("<<<Question>>>", row["question_prompt"])
    )


def answer_with_openai(client: OpenAI, args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, int], str | None]:
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
        return response.choices[0].message.content or "", token_usage, None
    except Exception as exc:  # noqa: BLE001
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, f"{type(exc).__name__}: {exc}"


class BaselineAdapter:
    name: str

    def add_turn(self, turn: TurnRecord) -> None:
        raise NotImplementedError

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:
        return {}


class FullContextAdapter(BaselineAdapter):
    name = "full_context"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.memories: list[dict[str, Any]] = []

    def add_turn(self, turn: TurnRecord) -> None:
        self.memories.append(context_block_from_turn(turn))

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        selected = list(self.memories)
        context = "\n".join(item["used_content"] for item in selected)
        truncated = False
        if self.max_chars > 0 and len(context) > self.max_chars:
            truncated = True
            context = context[-self.max_chars :]
        return context, selected, {"truncated": truncated, "stored_turns": len(self.memories)}


class LightMemAdapter(BaselineAdapter):
    def __init__(self, baseline: str, args: argparse.Namespace, out_dir: Path, show: str) -> None:
        toolkit = ROOT / "baseline/LightMem/src/lightmem/memory_toolkits"
        vendor = toolkit / "memories/layers/baselines"
        lightmem_src = ROOT / "baseline/LightMem/src"
        for path in [str(vendor), str(toolkit), str(lightmem_src)]:
            if path not in sys.path:
                sys.path.insert(0, path)
        self.baseline = baseline
        self.show = show
        self.out_dir = out_dir
        safe_show = re.sub(r"[^a-zA-Z0-9_]+", "_", show)
        if baseline == "a_mem":
            from memories.layers.amem import AMEMConfig, AMEMLayer

            cfg = AMEMConfig(
                user_id=f"longdialqa_{safe_show}",
                embedder_provider="sentence-transformers",
                retriever_name_or_path=args.embedding_model,
                base_url=args.openai_base_url,
                llm_backend="openai",
                llm_model=args.model,
                evo_threshold=args.amem_evo_threshold,
                api_key=args.openai_api_key,
                save_dir=str(out_dir / "memory" / safe_show),
            )
            self.layer = AMEMLayer(cfg)
            if args.amem_disable_analysis:
                self.layer.memory_layer.analyze_content = lambda _content: {"keywords": [], "context": "General", "tags": []}
            if args.amem_disable_evolution:
                self.layer.memory_layer.process_memory = lambda note: (False, note)
        elif baseline == "mem0":
            from memories.layers.memzero import MemZeroConfig, MemZeroLayer

            cfg = MemZeroConfig(
                user_id=f"longdialqa_{safe_show}",
                save_dir=str(out_dir / "memory" / safe_show),
                retriever_name_or_path=args.embedding_model,
                embedding_model_dims=384,
                use_gpu="cpu",
                llm_backend="openai",
                llm_model=args.model,
                llm_max_tokens=int(os.environ.get("MEM0_LLM_MAX_TOKENS", "4096")),
                embedder_provider="huggingface",
                vector_store_provider="qdrant",
                qdrant_on_disk=True,
                collection_name=f"longdialqa_{safe_show}_{int(time.time())}",
            )
            self.layer = MemZeroLayer(cfg)
        else:
            raise ValueError(baseline)
        self.add_count = 0

    def add_turn(self, turn: TurnRecord) -> None:
        self.layer.add_message(
            {
                "role": "user",
                "name": turn.speaker or "unknown",
                "content": turn_content(turn),
            },
            timestamp=f"{turn.date} turn {turn.turn_index}",
        )
        self.add_count += 1

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        memories = self.layer.retrieve(row["question"], k=top_k)
        records = []
        parts = []
        for idx, memory in enumerate(memories, start=1):
            used = str(memory.get("used_content") or memory.get("content") or "")
            parts.append(f"### Memory {idx}:\n{used}")
            metadata = dict(memory.get("metadata") or {})
            metadata["scene_ids_in_text"] = extract_scene_ids(used)
            records.append({"used_content": used, "content": memory.get("content"), "metadata": metadata})
        return "\n\n".join(parts), records, {"stored_turns": self.add_count}

    def finalize(self) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            self.layer.save_memory()
        close = getattr(self.layer, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        return {"stored_turns": self.add_count}


class HiGMemAdapter(BaselineAdapter):
    name = "higmem"

    def __init__(self, args: argparse.Namespace, out_dir: Path, show: str) -> None:
        path = str(ROOT / "baseline/HiGMem")
        if path not in sys.path:
            sys.path.insert(0, path)
        os.environ.setdefault("HIGMEM_MAX_TOKENS", "2048")
        from fphm_core import FPHMSystem
        from memory_layer import LLMController

        self.args = args
        self.show = show
        llm = LLMController("openai", model=args.model, api_key=args.openai_api_key, api_base=args.openai_base_url)
        self.system = FPHMSystem(
            llm_controller=llm,
            run_name=f"longdialqa_{show}",
            use_character_profile=False,
            use_event_metadata_mode=True,
            ablation_no_link=True,
            k_event_affiliation=args.higmem_k_event_affiliation,
            immediate_link_window=args.higmem_link_window,
            log_dir=str(out_dir / "fphm_logs"),
        )
        self.add_count = 0
        self.scene_unit = args.higmem_scene_unit
        self.scene_max_chars = args.higmem_scene_max_chars
        self.scene_key: tuple[str, int] | None = None
        self.scene_turns: list[TurnRecord] = []
        self.scene_chunks = 0
        self.scene_truncations = 0
        self.scene_truncated_chars = 0

    def _flush_scene(self) -> None:
        if not self.scene_turns:
            return
        first = self.scene_turns[0]
        scene_text = "\n".join(turn_content(turn) for turn in self.scene_turns)
        if self.scene_max_chars > 0 and len(scene_text) > self.scene_max_chars:
            original_chars = len(scene_text)
            keep_head = self.scene_max_chars // 2
            keep_tail = self.scene_max_chars - keep_head
            scene_text = (
                scene_text[:keep_head]
                + "\n...[scene truncated for HiGMem construction budget; full source remains in normalized data]...\n"
                + scene_text[-keep_tail:]
            )
            self.scene_truncations += 1
            self.scene_truncated_chars += original_chars - len(scene_text)
        speakers = []
        for turn in self.scene_turns:
            if turn.speaker and turn.speaker not in speakers:
                speakers.append(turn.speaker)
        self.system.add_turn(
            turn_id=f"{first.scene_id}_sceneunit",
            turn_content=scene_text,
            speaker=", ".join(speakers[:8]) or "scene",
            timestamp=f"{first.date} scene {first.session_ordinal}",
        )
        self.scene_chunks += 1
        self.scene_turns = []
        self.scene_key = None

    def add_turn(self, turn: TurnRecord) -> None:
        if self.scene_unit:
            key = (turn.scene_id, turn.session_ordinal)
            if self.scene_key is not None and key != self.scene_key:
                self._flush_scene()
            if self.scene_key is None:
                self.scene_key = key
            self.scene_turns.append(turn)
            self.add_count += 1
            return
        self.system.add_turn(
            turn_id=turn.turn_id,
            turn_content=turn.text,
            speaker=turn.speaker or "unknown",
            timestamp=f"{turn.date} turn {turn.turn_index}",
        )
        self.add_count += 1

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if self.scene_unit:
            self._flush_scene()
        if self.args.disable_higmem_query_rewrite:
            keyword_query = row["question"]
            profile_keys: list[str] = []
        else:
            from run_fphm_evaluation import generate_keyword_query

            query_data = generate_keyword_query(self.system.llm, row["question"])
            keyword_query = query_data.get("keyword_query") or row["question"]
            profile_keys = query_data.get("profile_retrieval_keys") or []
        context, trace = self.system.retrieve_for_query(
            original_query=row["question"],
            keyword_query=keyword_query,
            profile_retrieval_keys=profile_keys,
            k_profile=0,
            k_event=top_k,
            k_turn=top_k,
            return_trace=True,
        )
        records = [{"used_content": context, "metadata": {"trace": trace, "scene_ids_in_text": extract_scene_ids(context)}}]
        return context, records, {
            "stored_turns": self.add_count,
            "scene_chunks": self.scene_chunks,
            "scene_truncations": self.scene_truncations,
            "scene_truncated_chars": self.scene_truncated_chars,
            "trace": trace,
        }

    def finalize(self) -> dict[str, Any]:
        if self.scene_unit:
            with contextlib.suppress(Exception):
                self._flush_scene()
        with contextlib.suppress(Exception):
            self.system.executor.shutdown(wait=True)
        return {
            "stored_turns": self.add_count,
            "scene_chunks": self.scene_chunks,
            "scene_truncations": self.scene_truncations,
            "scene_truncated_chars": self.scene_truncated_chars,
            "events": len(getattr(self.system, "events", {}) or {}),
            "turn_notes": len(getattr(self.system, "turn_notes", {}) or {}),
        }


class SimpleMemAdapter(BaselineAdapter):
    name = "simplemem"

    def __init__(self, args: argparse.Namespace, out_dir: Path, show: str) -> None:
        path = str(ROOT / "baseline/SimpleMem")
        if path not in sys.path:
            sys.path.insert(0, path)
        os.environ.setdefault("OPENAI_API_KEY", args.openai_api_key)
        os.environ.setdefault("OPENAI_BASE_URL", args.openai_base_url)
        os.environ.setdefault("OPENAI_MODEL", args.model)
        os.environ.setdefault("SIMPLEMEM_BUILD_WORKERS", "1")
        os.environ.setdefault("SIMPLEMEM_RETRIEVAL_WORKERS", "1")
        from main import SimpleMemSystem

        self.system = SimpleMemSystem(
            api_key=args.openai_api_key,
            model=args.model,
            base_url=args.openai_base_url,
            db_path=str(out_dir / "lancedb" / show),
            table_name=f"memory_entries_{show}",
            clear_db=True,
            enable_thinking=False,
            use_streaming=True,
            enable_planning=True,
            enable_reflection=False,
            enable_parallel_processing=False,
            max_parallel_workers=1,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
        )
        self.add_count = 0

    def add_turn(self, turn: TurnRecord) -> None:
        self.system.add_dialogue(turn.speaker or "unknown", turn_content(turn), f"{turn.date} turn {turn.turn_index}")
        self.add_count += 1

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.system.finalize()
        contexts = self.system.hybrid_retriever.retrieve(row["question"], enable_reflection=False)
        contexts = contexts[:top_k]
        records = []
        parts = []
        for idx, entry in enumerate(contexts, start=1):
            content = getattr(entry, "lossless_restatement", "")
            used = f"Content: {content}"
            if getattr(entry, "timestamp", None):
                used += f"\nTime: {entry.timestamp}"
            parts.append(f"### Memory {idx}:\n{used}")
            records.append(
                {
                    "used_content": used,
                    "content": content,
                    "metadata": {
                        "entry_id": getattr(entry, "entry_id", None),
                        "keywords": getattr(entry, "keywords", []),
                        "scene_ids_in_text": extract_scene_ids(used),
                    },
                }
            )
        return "\n\n".join(parts), records, {"stored_turns": self.add_count, "memory_entries": len(self.system.get_all_memories())}

    def finalize(self) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            self.system.finalize()
        return {"stored_turns": self.add_count, "memory_entries": len(self.system.get_all_memories())}


class MemGASAdapter(BaselineAdapter):
    name = "memgas"

    def __init__(self, args: argparse.Namespace, out_dir: Path, show: str) -> None:
        path = str(ROOT / "baseline/MemGAS")
        if path not in sys.path:
            sys.path.insert(0, path)
        os.environ.setdefault("OPENAI_API_KEY", args.openai_api_key)
        os.environ.setdefault("OPENAI_BASE_URL", args.openai_base_url)
        os.environ.setdefault("OPENAI_MODEL", args.model)
        os.environ.setdefault("MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH", "1")
        from quickstart.config import MemoryConfig
        from quickstart.memory import MemGASMemory

        cfg = MemoryConfig(
            storage_dir=str(out_dir / "store" / show),
            embedder=os.environ.get("MEMGAS_EMBEDDER", "minilm"),
            llm_model=args.model,
            llm_provider="vllm",
            llm_api_key=args.openai_api_key,
            llm_base_url=args.openai_base_url,
            llm_max_tokens=int(os.environ.get("MEMGAS_LLM_MAX_TOKENS", "256")),
            llm_temperature=0.0,
            device=os.environ.get("MEMGAS_DEVICE", "cpu"),
            auto_save=True,
        )
        self.memory = MemGASMemory(cfg)
        if args.memgas_force_heuristic_granularity:
            self.memory.llm.summarize_and_keywords = self._heuristic_summarize_and_keywords  # type: ignore[assignment]
        self.show = show
        self.add_count = 0
        self.scene_key: tuple[str, int] | None = None
        self.scene_turns: list[str] = []
        self.scene_metadata: dict[str, Any] = {}
        self.scene_chunks = 0

    @staticmethod
    def _heuristic_summarize_and_keywords(session_text: str) -> tuple[str, list[str]]:
        text = " ".join(str(session_text or "").split())
        words = text.split()
        summary = " ".join(words[:120])
        if len(words) > 120:
            summary += " ..."
        tokens = re.findall(r"[A-Za-z][A-Za-z']+", text.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "who",
            "when",
            "where",
            "which",
            "about",
            "into",
            "your",
            "their",
            "there",
            "just",
            "could",
            "would",
            "should",
            "maybe",
            "like",
            "them",
            "then",
            "than",
            "into",
            "been",
            "were",
            "here",
            "some",
            "more",
            "most",
            "over",
            "also",
            "after",
            "before",
            "because",
            "said",
            "says",
            "saying",
            "speaker",
            "scene",
            "turn",
            "date",
        }
        counts = Counter(tok for tok in tokens if len(tok) >= 4 and tok not in stopwords)
        keywords: list[str] = []
        for tok, _ in counts.most_common(30):
            keywords.append(tok)
        if not keywords:
            keywords = [word.lower() for word in words[:15] if word]
        return summary or text[:200], keywords

    def _flush_scene(self) -> None:
        if not self.scene_turns:
            return
        self.memory.add(
            list(self.scene_turns),
            conversation_id=self.show,
            metadata=dict(self.scene_metadata),
        )
        self.scene_chunks += 1
        self.scene_turns = []
        self.scene_metadata = {}
        self.scene_key = None

    def add_turn(self, turn: TurnRecord) -> None:
        key = (turn.scene_id, turn.session_ordinal)
        if self.scene_key is not None and key != self.scene_key:
            self._flush_scene()
        if self.scene_key is None:
            self.scene_key = key
            self.scene_metadata = {
                "scene_id": turn.scene_id,
                "session_ordinal": turn.session_ordinal,
                "date": turn.date,
                "turn_ids": [],
                "speakers": [],
            }
        self.scene_turns.append(turn_content(turn))
        self.scene_metadata.setdefault("turn_ids", []).append(turn.turn_id)
        speakers = self.scene_metadata.setdefault("speakers", [])
        if turn.speaker and turn.speaker not in speakers:
            speakers.append(turn.speaker)
        self.add_count += 1

    def retrieve(self, row: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self._flush_scene()
        hits = self.memory.retrieve(row["question"], topk=top_k, conversation_id=self.show)
        records = []
        parts = []
        for idx, hit in enumerate(hits, start=1):
            session_text = "\n".join(hit.get("session") or [])
            used = (
                f"Summary: {hit.get('summary', '')}\n"
                f"Keywords: {', '.join(hit.get('keywords') or [])}\n"
                f"Session:\n{session_text}"
            )
            parts.append(f"### Memory {idx}:\n{used}")
            metadata = dict(hit.get("metadata") or {})
            metadata["memory_id"] = hit.get("memory_id")
            metadata["score"] = hit.get("score")
            metadata["scene_ids_in_text"] = extract_scene_ids(used)
            records.append({"used_content": used, "content": session_text, "metadata": metadata})
        return "\n\n".join(parts), records, {"stored_turns": self.add_count, "scene_chunks": self.scene_chunks, "memory_records": len(self.memory)}

    def finalize(self) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            self._flush_scene()
        with contextlib.suppress(Exception):
            self.memory.save()
        return {"stored_turns": self.add_count, "scene_chunks": self.scene_chunks, "memory_records": len(self.memory)}


def make_adapter(name: str, args: argparse.Namespace, out_dir: Path, show: str) -> BaselineAdapter:
    if name == "full_context":
        return FullContextAdapter(max_chars=args.full_context_max_chars)
    if name in {"a_mem", "mem0"}:
        return LightMemAdapter(name, args, out_dir, show)
    if name == "higmem":
        return HiGMemAdapter(args, out_dir, show)
    if name == "simplemem":
        return SimpleMemAdapter(args, out_dir, show)
    if name == "memgas":
        return MemGASAdapter(args, out_dir, show)
    raise ValueError(name)


def flatten_retrieved_scene_ids(records: list[dict[str, Any]]) -> list[str]:
    scene_ids: set[str] = set()
    for record in records:
        metadata = record.get("metadata") or {}
        if metadata.get("scene_id"):
            scene_ids.add(str(metadata["scene_id"]))
        for sid in metadata.get("scene_ids_in_text") or []:
            scene_ids.add(str(sid))
        used = record.get("used_content")
        if isinstance(used, str):
            scene_ids.update(extract_scene_ids(used))
    return sorted(scene_ids)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
        "strict_accuracy": sum(1 for row in rows if row["strict_correct"]) / len(rows),
        "option_parse_rate": sum(1 for row in rows if row.get("predicted_option")) / len(rows),
        "mean_token_f1": sum(float(row.get("token_f1", 0.0)) for row in rows) / len(rows),
        "mean_context_chars": sum(int(row.get("retrieved_context_chars", 0)) for row in rows) / len(rows),
        "mean_context_tokens_approx": sum(int(row.get("retrieved_context_tokens_approx", 0)) for row in rows) / len(rows),
        "mean_retrieved_k": sum(int(row.get("retrieved_k", 0)) for row in rows) / len(rows),
        "evidence_recall_any": (
            sum(1 for row in rows if row.get("evidence_recall_any")) / sum(1 for row in rows if row.get("has_evidence_scene"))
            if any(row.get("has_evidence_scene") for row in rows)
            else None
        ),
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {"overall": summarize_group(rows), "by_show": {}, "by_question_source": {}, "by_question_type": {}, "by_answerability": {}, "by_hop": {}}
    group_specs = {
        "by_show": "show_name",
        "by_question_source": "question_source",
        "by_question_type": "question_type",
        "by_answerability": "answerability_label",
        "by_hop": "hop_type",
    }
    for out_key, row_key in group_specs.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(row_key, "unknown"))].append(row)
        metrics[out_key] = {key: summarize_group(items) for key, items in sorted(grouped.items())}
    return metrics


def save_command_env(path: Path, args: argparse.Namespace, subset_path: Path, normalized_manifest: Path) -> None:
    lines = [
        f"BASELINE={args.baseline}",
        f"OPENAI_MODEL={args.model}",
        f"OPENAI_BASE_URL={args.openai_base_url}",
        f"NORMALIZED_DIR={args.normalized_dir}",
        f"NORMALIZED_MANIFEST={normalized_manifest}",
        f"NORMALIZED_MANIFEST_SHA256={sha256_file(normalized_manifest)}",
        f"SUBSET_MANIFEST={args.subset_manifest}",
        f"SUBSET_MANIFEST_SHA256={sha256_file(args.subset_manifest)}",
        f"SUBSET_PATH={subset_path}",
        f"SUBSET_SHA256={sha256_file(subset_path)}",
        f"MAX_QA_PER_SHOW={args.max_qa_per_show}",
        f"SHOWS={','.join(args.shows)}",
        f"RUN_NAME={args.run_name}",
        f"TOP_K={args.top_k}",
        f"ANSWER_MAX_TOKENS={args.answer_max_tokens}",
        f"ANSWER_TIMEOUT={args.answer_timeout}",
        f"FULL_CONTEXT_MAX_CHARS={args.full_context_max_chars}",
        f"RETRIEVED_CONTEXT_MAX_CHARS={args.retrieved_context_max_chars}",
        f"MAX_SAVED_CONTEXT_CHARS={args.max_saved_context_chars}",
        f"MAX_SAVED_RECORDS={args.max_saved_records}",
        f"RESUME={args.resume}",
        f"AMEM_EVO_THRESHOLD={args.amem_evo_threshold}",
        f"AMEM_DISABLE_EVOLUTION={args.amem_disable_evolution}",
        f"AMEM_DISABLE_ANALYSIS={args.amem_disable_analysis}",
        f"MEMGAS_EMBEDDER={os.environ.get('MEMGAS_EMBEDDER', 'minilm')}",
        f"MEMGAS_DEVICE={os.environ.get('MEMGAS_DEVICE', 'cpu')}",
        f"MEMGAS_LLM_MAX_TOKENS={os.environ.get('MEMGAS_LLM_MAX_TOKENS', '256')}",
        f"MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH={os.environ.get('MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH', '1')}",
        f"MEMGAS_FORCE_HEURISTIC_GRANULARITY={os.environ.get('LONGDIALQA_MEMGAS_FORCE_HEURISTIC_GRANULARITY', '1')}",
        f"HIGMEM_SCENE_UNIT={args.higmem_scene_unit}",
        f"HIGMEM_SCENE_MAX_CHARS={args.higmem_scene_max_chars}",
        f"HIGMEM_LINK_WINDOW={args.higmem_link_window}",
        f"HIGMEM_K_EVENT_AFFILIATION={args.higmem_k_event_affiliation}",
        f"HIGMEM_DISABLE_QUERY_REWRITE={args.disable_higmem_query_rewrite}",
        f"HIGMEM_MAX_TOKENS={os.environ.get('HIGMEM_MAX_TOKENS', '2048')}",
        f"EMBEDDING_MODEL={args.embedding_model}",
        "COMMAND=" + " ".join(sys.argv),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    normalized_manifest = args.normalized_dir / "manifest.json"
    sessions_path = args.normalized_dir / "sessions.jsonl"
    args.run_root.mkdir(parents=True, exist_ok=True)
    subset, subset_path, subset_manifest = load_subset_from_manifest(args)
    allowed_shows = set(args.shows or [])
    if allowed_shows:
        subset = [row for row in subset if row.get("show") in allowed_shows]
    if not subset:
        raise ValueError(f"No rows remain after --shows filter: {args.shows}")
    sessions = load_sessions(sessions_path)

    out_dir = args.run_root / (args.run_name or args.baseline)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "raw_predictions.jsonl"
    retrieval_path = out_dir / "retrieved_context.jsonl"
    failures_path = out_dir / "failure_retry_logs.jsonl"
    metrics_path = out_dir / "metrics.json"
    stats_path = out_dir / "stats.json"
    summary_path = out_dir / "summary.md"
    run_log_path = out_dir / "run.log"
    save_command_env(out_dir / "command.env", args, subset_path, normalized_manifest)
    if args.resume:
        predictions = read_jsonl(pred_path) if pred_path.exists() else []
        retrieval_rows = read_jsonl(retrieval_path) if retrieval_path.exists() else []
        failures = read_jsonl(failures_path) if failures_path.exists() else []
        completed_ids = {str(row.get("qa_id")) for row in predictions if row.get("qa_id")}
    else:
        for path in (pred_path, retrieval_path, failures_path):
            if path.exists():
                path.unlink()
        predictions = []
        retrieval_rows = []
        failures = []
        completed_ids: set[str] = set()
    run_log_lines = [
        f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"baseline={args.baseline}",
        f"subset_manifest={args.subset_manifest}",
        f"subset_path={subset_path}",
        f"model={args.model}",
        f"shows={','.join(args.shows)}",
        f"resume={args.resume}",
        f"completed_before_start={len(completed_ids)}",
    ]

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url, max_retries=0)
    stats: dict[str, Any] = {
        "baseline": args.baseline,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "token_usage": Counter(),
        "shows": {},
        "shows_filter": list(args.shows),
        "subset_rows": len(subset),
        "subset_sha256": sha256_file(subset_path),
        "subset_manifest": str(args.subset_manifest),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
    }
    started = time.time()

    subset_by_show: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        subset_by_show[row["show"]].append(row)
    for show_rows in subset_by_show.values():
        show_rows.sort(key=lambda row: (int(row["session_ordinal"]), int(row["ask_turn_index"])))

    for show, qa_rows in sorted(subset_by_show.items()):
        show_started = time.time()
        run_log_lines.append(f"show_start={show} qa_rows={len(qa_rows)}")
        adapter = make_adapter(args.baseline, args, out_dir, show)
        show_sessions = sessions[show]
        next_session_idx = 0
        next_turn_idx_in_session = 0

        def feed_until(target_session_ordinal: int, target_turn_index: int) -> None:
            nonlocal next_session_idx, next_turn_idx_in_session
            while next_session_idx < len(show_sessions):
                session = show_sessions[next_session_idx]
                session_ordinal = int(session["session_ordinal"])
                turns = turn_records_for_session(session)
                if session_ordinal < target_session_ordinal:
                    while next_turn_idx_in_session < len(turns):
                        adapter.add_turn(turns[next_turn_idx_in_session])
                        next_turn_idx_in_session += 1
                    next_session_idx += 1
                    next_turn_idx_in_session = 0
                    continue
                if session_ordinal == target_session_ordinal:
                    limit = min(target_turn_index, len(turns))
                    while next_turn_idx_in_session < limit:
                        adapter.add_turn(turns[next_turn_idx_in_session])
                        next_turn_idx_in_session += 1
                    return
                return

        for qa_index, row in enumerate(qa_rows):
            qa_started = time.time()
            try:
                feed_until(int(row["session_ordinal"]), int(row["ask_turn_index"]))
                if str(row["id"]) in completed_ids:
                    continue
                retrieval_started = time.time()
                context, retrieved_records, retrieval_stats = adapter.retrieve(row, args.top_k)
                retrieval_seconds = time.time() - retrieval_started
                if args.baseline != "full_context" and args.retrieved_context_max_chars > 0 and len(context or "") > args.retrieved_context_max_chars:
                    retrieval_stats = dict(retrieval_stats)
                    retrieval_stats["answer_context_truncated"] = True
                    retrieval_stats["answer_context_original_chars"] = len(context or "")
                    context = (context or "")[: args.retrieved_context_max_chars]
                prompt = build_prompt(row, context)
                answer_started = time.time()
                answer, usage, error = answer_with_openai(client, args, prompt)
                answer_seconds = time.time() - answer_started
                stats["token_usage"].update(usage)
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
                    "baseline": args.baseline,
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
                }
                predictions.append(common)
                append_jsonl(pred_path, common)
                retrieval_rows.append(
                    {
                        **{key: common[key] for key in ["baseline", "qa_id", "show", "show_name", "question", "question_type"]},
                        **context_artifact(context, args.max_saved_context_chars),
                        **retrieved_records_artifact(retrieved_records, args.max_saved_records),
                        "retrieval_stats": retrieval_stats,
                        "retrieved_scene_ids": retrieved_scene_ids,
                        "evidence_scene_ids": evidence_scene_ids,
                        "evidence_recall_any": evidence_recall_any,
                        "evidence_recall_all": evidence_recall_all,
                    }
                )
                append_jsonl(retrieval_path, retrieval_rows[-1])
                completed_ids.add(str(row["id"]))
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "qa_id": row.get("id"),
                    "stage": "qa",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                append_jsonl(failures_path, failure)
                failed_prediction = {
                    "baseline": args.baseline,
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
                append_jsonl(pred_path, failed_prediction)
                completed_ids.add(str(row.get("id")))
        show_stats = adapter.finalize()
        stats["shows"][show] = {
            **show_stats,
            "qa_rows": len(qa_rows),
            "seconds": time.time() - show_started,
        }
        run_log_lines.append(f"show_done={show} seconds={stats['shows'][show]['seconds']:.3f} stats={json.dumps(show_stats, ensure_ascii=False)}")

    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stats["runtime_seconds"] = time.time() - started
    stats["token_usage"] = dict(stats["token_usage"])
    write_jsonl(pred_path, predictions)
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(failures_path, failures)
    metrics = build_metrics(predictions)
    metrics["artifact_paths"] = {
        "raw_predictions": str(pred_path),
        "retrieved_context": str(retrieval_path),
        "metrics": str(metrics_path),
        "stats": str(stats_path),
        "failures": str(failures_path),
        "command_env": str(out_dir / "command.env"),
        "run_log": str(run_log_path),
        "summary_md": str(summary_path),
        "subset": str(subset_path),
    }
    fraction = float(subset_manifest.get("fraction", 0.0))
    if subset_manifest.get("split_label") == "full" or fraction >= 0.999:
        subset_label = "full benchmark reproduction"
    elif fraction < 0.5:
        subset_label = "5% smoke subset result"
    else:
        subset_label = "50% subset result"
    metrics["dataset"] = {
        "normalized_manifest": str(normalized_manifest),
        "normalized_manifest_sha256": sha256_file(normalized_manifest),
        "subset_sha256": sha256_file(subset_path),
        "subset_manifest": str(args.subset_manifest),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "subset_rows": len(subset),
        "subset_label": subset_label,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    run_log_lines.append(f"finished_at={stats['finished_at']}")
    run_log_lines.append(f"runtime_seconds={stats['runtime_seconds']:.3f}")
    run_log_lines.append(f"overall={json.dumps(metrics['overall'], ensure_ascii=False, sort_keys=True)}")
    run_log_path.write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")
    summary_lines = [
        f"# LongDialQA/DialSim {args.baseline} Baseline",
        "",
        f"Result label: {metrics['dataset']['subset_label']}",
        f"Subset manifest: `{args.subset_manifest}`",
        f"Subset hash: `{metrics['dataset']['subset_sha256']}`",
        f"Model: `{args.model}`",
        "",
        "## Overall",
        "",
        f"- Count: {metrics['overall'].get('count')}",
        f"- Accuracy: {metrics['overall'].get('accuracy')}",
        f"- Strict accuracy: {metrics['overall'].get('strict_accuracy')}",
        f"- Mean retrieved K: {metrics['overall'].get('mean_retrieved_k')}",
        f"- Mean retrieved tokens approx: {metrics['overall'].get('mean_context_tokens_approx')}",
        f"- Runtime seconds: {stats['runtime_seconds']}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in metrics["artifact_paths"].items():
        summary_lines.append(f"- {key}: `{value}`")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"metrics": metrics, "stats": stats}


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps({"baseline": args.baseline, "overall": result["metrics"]["overall"], "run_root": str(args.run_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
