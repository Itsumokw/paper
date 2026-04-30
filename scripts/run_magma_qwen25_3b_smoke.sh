#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper/baseline/MAGMA

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export MAGMA_OPENAI_BASE_URL="${MAGMA_OPENAI_BASE_URL:-$OPENAI_BASE_URL}"
export MAGMA_USE_STREAMING="${MAGMA_USE_STREAMING:-1}"
export MAGMA_BERTSCORE_DEVICE="${MAGMA_BERTSCORE_DEVICE:-cpu}"
export MAGMA_SENTENCE_TRANSFORMER_DEVICE="${MAGMA_SENTENCE_TRANSFORMER_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

MODEL="${MAGMA_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

/home/stu0032/paper/.venv/bin/python -u test_fixed_memory.py \
  --dataset data/locomo10.json \
  --sample 0 \
  --max-questions 1 \
  --category-to-test 1 \
  --model "$MODEL" \
  --embedding-model minilm \
  --n-workers 1 \
  --best-of-n 1
