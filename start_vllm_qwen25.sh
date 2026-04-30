#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
mkdir -p runs

MODEL_DIR="${MODEL_DIR:-/home/stu0032/paper/models/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="*"
export no_proxy="*"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

EXTRA_ARGS=()
if [ -n "$VLLM_MAX_NUM_SEQS" ]; then
  EXTRA_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi

exec /home/stu0032/paper/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  "${EXTRA_ARGS[@]}" \
  --default-chat-template-kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"
