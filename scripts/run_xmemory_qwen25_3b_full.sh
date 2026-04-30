#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper/baseline/xMemory/evaluation/locomo

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export XMEMORY_LLM_BACKEND="${XMEMORY_LLM_BACKEND:-openai}"
export XMEMORY_LLM_MODEL="${XMEMORY_LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export XMEMORY_USE_STREAMING="${XMEMORY_USE_STREAMING:-1}"
export XMEMORY_EMBEDDING_BACKEND="${XMEMORY_EMBEDDING_BACKEND:-sentence-transformer}"
export XMEMORY_EMBEDDING_DEVICE="${XMEMORY_EMBEDDING_DEVICE:-cpu}"
export XMEMORY_SENTENCE_TRANSFORMER_DEVICE="${XMEMORY_SENTENCE_TRANSFORMER_DEVICE:-cpu}"
export XMEMORY_BERTSCORE_DEVICE="${XMEMORY_BERTSCORE_DEVICE:-cpu}"
export XMEMORY_BUILD_WORKERS="${XMEMORY_BUILD_WORKERS:-1}"
export XMEMORY_SEARCH_WORKERS="${XMEMORY_SEARCH_WORKERS:-32}"
export XMEMORY_EVAL_WORKERS="${XMEMORY_EVAL_WORKERS:-32}"
export XMEMORY_JUDGE_MODEL="${XMEMORY_JUDGE_MODEL:-$XMEMORY_LLM_MODEL}"
export XMEMORY_SEARCH_STRATEGY="${XMEMORY_SEARCH_STRATEGY:-baseline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="/home/stu0032/paper/baseline/xMemory/src:/home/stu0032/paper/baseline/xMemory${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${XMEMORY_RUN_ROOT:-/home/stu0032/paper/runs/xmemory/qwen25_3b_full}"
CONFIG_TEMPLATE="/home/stu0032/paper/baseline/xMemory/evaluation/locomo/config.local_qwen25_3b.json"
CONFIG="$RUN_ROOT/config.runtime.json"
DATA="${XMEMORY_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}"
mkdir -p "$RUN_ROOT"

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/preflight_qwen25_3b_baselines.py xmemory

/home/stu0032/paper/.venv/bin/python - "$CONFIG_TEMPLATE" "$CONFIG" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

template_path, output_path, run_root = sys.argv[1:4]
data = json.loads(Path(template_path).read_text(encoding="utf-8"))
data["storage_path"] = f"{run_root}/evaluation_memories"
data["chroma_persist_directory"] = f"{run_root}/evaluation_memories/chroma_db"
Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY

if [ "${RESET_XMEMORY_RUN:-1}" = "1" ]; then
  rm -rf "$RUN_ROOT/evaluation_memories" "$RUN_ROOT/results.json" "$RUN_ROOT/metrics.json" "$RUN_ROOT/results.token_stats.json"
fi

/home/stu0032/paper/.venv/bin/python -u add.py \
  --data "$DATA" \
  --config "$CONFIG" \
  --batch-size 1 \
  --wait-timeout 1800 \
  --max-workers "$XMEMORY_BUILD_WORKERS" \
  --max-retries 15 \
  --llm-model "$XMEMORY_LLM_MODEL" \
  --verbose

/home/stu0032/paper/.venv/bin/python -u xMemory_search_framework.py \
  --data "$DATA" \
  --config "$CONFIG" \
  --output "$RUN_ROOT/results.json" \
  --top-k-episodes 10 \
  --top-k-semantic 20 \
  --search-method vector \
  --include-original-messages-top-k 10 \
  --max-workers "$XMEMORY_SEARCH_WORKERS" \
  --llm-model "$XMEMORY_LLM_MODEL" \
  --search-strategy "$XMEMORY_SEARCH_STRATEGY"

/home/stu0032/paper/.venv/bin/python -u evals.py \
  --input_file "$RUN_ROOT/results.json" \
  --output_file "$RUN_ROOT/metrics.json" \
  --max_workers "$XMEMORY_EVAL_WORKERS"
