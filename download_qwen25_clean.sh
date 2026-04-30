#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
mkdir -p models

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-/home/stu0032/paper/models/Qwen3-8B}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="*"
export no_proxy="*"

if [ "${RESET_LOCAL_DIR:-0}" = "1" ] && [ -d "$MODEL_DIR" ]; then
  backup_dir="${MODEL_DIR}.bad.$(date -u +%Y%m%d-%H%M%S)"
  mv "$MODEL_DIR" "$backup_dir"
  echo "Moved existing model dir to $backup_dir"
fi

mkdir -p "$MODEL_DIR"

exec /home/stu0032/paper/.venv/bin/hf download \
  "$MODEL_ID" \
  --local-dir "$MODEL_DIR" \
  --force-download \
  --max-workers "$HF_MAX_WORKERS"
