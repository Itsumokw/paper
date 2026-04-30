#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper/baseline/LightMem/experiments/locomo

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export LIGHTMEM_LLM_MODEL="${LIGHTMEM_LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export LIGHTMEM_DATA_PATH="${LIGHTMEM_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}"
export LIGHTMEM_EMBEDDING_MODEL_PATH="${LIGHTMEM_EMBEDDING_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"
export LIGHTMEM_EMBEDDING_DEVICE="${LIGHTMEM_EMBEDDING_DEVICE:-cpu}"
export LIGHTMEM_SEARCH_WORKERS="${LIGHTMEM_SEARCH_WORKERS:-1}"
export LIGHTMEM_ALLOW_CATEGORIES="${LIGHTMEM_ALLOW_CATEGORIES:-1 2 3 4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="/home/stu0032/paper/baseline/LightMem/src:/home/stu0032/paper/baseline/LightMem/experiments/locomo${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${LIGHTMEM_RUN_ROOT:-/home/stu0032/paper/runs/lightmem/qwen25_3b_structmem_full}"
export LIGHTMEM_QDRANT_PRE_UPDATE_DIR="$RUN_ROOT/qdrant_pre_update"
export LIGHTMEM_RESULTS_DIR="${LIGHTMEM_RESULTS_DIR:-$RUN_ROOT/results}"
mkdir -p "$LIGHTMEM_RESULTS_DIR"

if [ ! -f "$LIGHTMEM_QDRANT_PRE_UPDATE_DIR/meta.json" ]; then
  echo "Missing completed LightMem build output: $LIGHTMEM_QDRANT_PRE_UPDATE_DIR/meta.json" >&2
  exit 2
fi

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/check_lightmem_qdrant_ready.py \
  --dataset "$LIGHTMEM_DATA_PATH" \
  --qdrant-dir "$LIGHTMEM_QDRANT_PRE_UPDATE_DIR" \
  --require-summaries

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/preflight_qwen25_3b_baselines.py lightmem

/home/stu0032/paper/.venv/bin/python -u search_locomo.py \
  --dataset "$LIGHTMEM_DATA_PATH" \
  --qdrant-dir "$LIGHTMEM_QDRANT_PRE_UPDATE_DIR" \
  --output-dir "$LIGHTMEM_RESULTS_DIR" \
  --retrieval-mode combined \
  --total-limit 60 \
  --allow-categories $LIGHTMEM_ALLOW_CATEGORIES \
  --embedder huggingface \
  --embedding-model-path "$LIGHTMEM_EMBEDDING_MODEL_PATH" \
  --enable-summary \
  --summary-limit 5 \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-base-url "$OPENAI_BASE_URL" \
  --llm-model "$LIGHTMEM_LLM_MODEL" \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-base-url "$OPENAI_BASE_URL" \
  --judge-model "$LIGHTMEM_LLM_MODEL" \
  --workers "$LIGHTMEM_SEARCH_WORKERS"
