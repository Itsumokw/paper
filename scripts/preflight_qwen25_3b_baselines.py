#!/usr/bin/env python3
"""Preflight checks for the local Qwen2.5-3B baseline runs."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import socket
import sys
import urllib.request
from pathlib import Path


PAPER_ROOT = Path("/home/stu0032/paper")
PYTHON = PAPER_ROOT / ".venv" / "bin" / "python"
DATASET = PAPER_ROOT / "baseline" / "MAGMA" / "data" / "locomo10.json"
EMBEDDING_MODEL = os.environ.get(
    "LIGHTMEM_EMBEDDING_MODEL_PATH",
    os.environ.get("XMEMORY_EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def add_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def import_module(name: str) -> None:
    importlib.import_module(name)


def import_file(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def check_path(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"missing {description}: {path}")


def check_vllm() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    url = f"{base_url}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read(256)
    except Exception as exc:  # noqa: BLE001
        fail(f"vLLM/OpenAI endpoint not reachable at {url}: {exc}")

    from openai import OpenAI

    model = os.environ.get(
        "OPENAI_MODEL",
        os.environ.get(
            "LIGHTMEM_LLM_MODEL",
            os.environ.get("XMEMORY_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        ),
    )
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=4,
            temperature=0,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"vLLM/OpenAI chat completion failed at {base_url} with model {model}: {exc}")
    if not response.choices:
        fail(f"vLLM/OpenAI chat completion returned no choices for model {model}")


def check_embedding_cache() -> None:
    import_module("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)


def check_nltk_data() -> None:
    import nltk

    nltk.data.path.insert(0, os.environ.get("NLTK_DATA", "/home/stu0032/nltk_data"))
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError as exc:
            fail(f"missing NLTK data {resource}: {exc}")
    from nltk import word_tokenize
    from nltk.corpus import stopwords

    stopwords.words("english")
    word_tokenize("MemMachine BM25 tokenizer preflight.", language="english")


def check_lightmem() -> None:
    lightmem_root = PAPER_ROOT / "baseline" / "LightMem"
    locomo_dir = lightmem_root / "experiments" / "locomo"
    add_path(lightmem_root / "src")
    add_path(locomo_dir)

    check_path(DATASET, "LoCoMo dataset")

    import_module("openai")
    import_module("numpy")
    import_module("qdrant_client")
    import_module("tqdm")
    import_module("lightmem")
    import_file("lightmem_locomo_retrievers_preflight", locomo_dir / "retrievers.py")
    import_file("lightmem_locomo_search_preflight", locomo_dir / "search_locomo.py")
    check_embedding_cache()


def check_xmemory() -> None:
    root = PAPER_ROOT / "baseline" / "xMemory"
    locomo_dir = root / "evaluation" / "locomo"
    add_path(root / "src")
    add_path(root)
    add_path(locomo_dir)

    check_path(DATASET, "LoCoMo dataset")
    check_path(locomo_dir / "config.local_qwen25_3b.json", "xMemory local config template")

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    for module in ("dotenv", "tqdm", "numpy", "jinja2", "chromadb", "rank_bm25", "nltk"):
        import_module(module)
    import_file("xmemory_locomo_add_preflight", locomo_dir / "add.py")
    import_file("xmemory_locomo_search_preflight", locomo_dir / "xMemory_search_framework.py")
    import_file("xmemory_locomo_evals_preflight", locomo_dir / "evals.py")
    check_embedding_cache()
    check_nltk_data()


def check_neo4j() -> None:
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j_password")

    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", 7687))
    except OSError as exc:
        fail(f"Neo4j is not reachable at {uri}: {exc}")
    finally:
        sock.close()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            value = session.run("RETURN 1 AS ok").single()
            if value is None or value["ok"] != 1:
                fail(f"Neo4j auth check failed at {uri}: RETURN 1 did not return 1")
    except Exception as exc:  # noqa: BLE001
        fail(f"Neo4j auth check failed at {uri} for user {user}: {exc}")
    finally:
        driver.close()


def check_memmachine() -> None:
    root = PAPER_ROOT / "baseline" / "MemMachine"
    eval_dir = root / "evaluation" / "retrieval_agent"
    add_path(root)
    add_path(root / "packages" / "common" / "src")
    add_path(root / "packages" / "server" / "src")
    add_path(root / "packages" / "client" / "src")

    check_path(eval_dir / "configuration.yml", "MemMachine configuration.yml")
    check_path(eval_dir / ".." / "data" / "locomo10.json", "MemMachine LoCoMo dataset")

    for module in ("dotenv", "pandas", "yaml", "neo4j", "openai", "numpy", "nltk"):
        import_module(module)
    import_file("memmachine_locomo_ingest_preflight", eval_dir / "locomo_ingest.py")
    import_file("memmachine_locomo_search_preflight", eval_dir / "locomo_search.py")
    import_file("memmachine_evaluate_preflight", eval_dir / "evaluate.py")
    import_module("memmachine_server.common.episode_store")
    import_module("memmachine_server.common.utils")
    check_embedding_cache()
    check_nltk_data()
    check_neo4j()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="+",
        choices=("lightmem", "xmemory", "memmachine"),
        help="Baseline groups to preflight",
    )
    parser.add_argument("--skip-vllm", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")

    check_path(PYTHON, "project Python")
    if not args.skip_vllm:
        check_vllm()

    checks = {
        "lightmem": check_lightmem,
        "xmemory": check_xmemory,
        "memmachine": check_memmachine,
    }

    errors: list[str] = []
    for target in args.targets:
        try:
            checks[target]()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{target}: {exc}")
        else:
            print(f"[preflight] {target}: ok")

    if errors:
        print("[preflight] failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
