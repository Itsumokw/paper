#!/usr/bin/env bash
set -euo pipefail

source /home/stu0032/paper/scripts/memmachine_env.sh

mkdir -p "${MEMMACHINE_RUN_ROOT}" "${MEMMACHINE_EVAL_DIR}/result/final_score"

if [ ! -x "${MEMMACHINE_PYTHON}" ]; then
    echo "Missing Python executable: ${MEMMACHINE_PYTHON}" >&2
    exit 2
fi

if [ ! -f "${MEMMACHINE_CONFIG_SRC}" ]; then
    echo "Missing MemMachine local config template: ${MEMMACHINE_CONFIG_SRC}" >&2
    exit 2
fi

if ! cmp -s "${MEMMACHINE_CONFIG_SRC}" "${MEMMACHINE_CONFIG_PATH}"; then
    cp "${MEMMACHINE_CONFIG_SRC}" "${MEMMACHINE_CONFIG_PATH}"
fi

"${MEMMACHINE_PYTHON}" - <<'PY'
import os
import socket
import sys
import urllib.request
from pathlib import Path

import nltk
from memmachine_server.common.configuration import Configuration
from sentence_transformers import SentenceTransformer

config_path = Path(os.environ["MEMMACHINE_CONFIG_PATH"])
run_root = Path(os.environ["MEMMACHINE_RUN_ROOT"])
conf = Configuration.load_yml_file(str(config_path))

print(f"MemMachine config: {config_path}")
print(f"Run root: {run_root}")
print(f"LLM: {conf.retrieval_agent.llm_model}")
print(f"Vector graph store: {conf.episodic_memory.long_term_memory.vector_graph_store}")

java_home = Path(os.environ["JAVA_HOME"])
if (java_home / "bin" / "java").exists():
    print(f"Java: {java_home / 'bin' / 'java'}")
else:
    print(f"Java: missing at {java_home / 'bin' / 'java'}")

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
print("Embedding cache: sentence-transformers/all-MiniLM-L6-v2 OK")

nltk_data = os.environ.get("NLTK_DATA", "/home/stu0032/nltk_data")
nltk.data.path.insert(0, nltk_data)
for package, resource in (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, download_dir=nltk_data, quiet=True)
        nltk.data.find(resource)
from nltk import word_tokenize
from nltk.corpus import stopwords

stopwords.words("english")
word_tokenize("MemMachine BM25 tokenizer setup check.", language="english")
print("NLTK data: punkt/punkt_tab/stopwords OK")

try:
    with urllib.request.urlopen(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/models", timeout=2) as response:
        response.read(256)
except Exception as exc:
    print(f"vLLM/OpenAI endpoint: NOT reachable at {os.environ['OPENAI_BASE_URL']} ({exc})")
    if os.environ.get("MEMMACHINE_STRICT_VLLM", "0") == "1":
        raise SystemExit(11) from exc
else:
    print(f"vLLM/OpenAI endpoint: reachable at {os.environ['OPENAI_BASE_URL']}")

sock = socket.socket()
sock.settimeout(1.5)
try:
    sock.connect(("127.0.0.1", 7687))
except OSError as exc:
    print("Neo4j: NOT reachable at bolt://127.0.0.1:7687")
    print("MemMachine official LoCoMo path needs Neo4j before a full run.")
    if os.environ.get("MEMMACHINE_STRICT_NEO4J", "0") == "1":
        raise SystemExit(10) from exc
else:
    print("Neo4j: reachable at bolt://127.0.0.1:7687")
finally:
    sock.close()
PY

echo
echo "MemMachine env is prepared. To reuse it in the current shell:"
echo "source /home/stu0032/paper/scripts/memmachine_env.sh"
