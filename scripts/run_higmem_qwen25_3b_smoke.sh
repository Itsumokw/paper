#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper/baseline/HiGMem

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export HIGMEM_USE_STREAMING="${HIGMEM_USE_STREAMING:-1}"

# Keep the evaluation process on CPU so it does not compete with vLLM for GPU memory.
export CUDA_VISIBLE_DEVICES=""

# BERTScore can be recomputed later from predictions/references; skipping it keeps smoke runs fast.
export SKIP_BERTSCORE="${SKIP_BERTSCORE:-1}"

/home/stu0032/paper/.venv/bin/python -u run_fphm_evaluation.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --backend openai \
  --api_base "$OPENAI_API_BASE" \
  --api_key "$OPENAI_API_KEY" \
  --ablation-no-profile \
  --ablation-event-metadata-only \
  --ablation-no-link \
  --k_event 10 \
  --sample_index 0 \
  --max_turns 80 \
  --max_questions 5
