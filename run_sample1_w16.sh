#!/usr/bin/env bash
set -u

: "${RUN_DIR:?RUN_DIR is required}"

mkdir -p "$RUN_DIR"

cd /home/stu0032/paper/baseline/SimpleMem || exit 1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT='https://hf-mirror.com'
export NLTK_DATA='/home/stu0032/nltk_data'

LOG="$RUN_DIR/run.log"
RESULT="$RUN_DIR/result.json"
EXIT_CODE="$RUN_DIR/exit.code"

{
  echo "started_at=$(date '+%F %T %Z')"
  echo "workers=16"
  echo "num_samples=1"
  echo "model=Qwen/Qwen3-8B"
} | tee "$LOG"

/usr/bin/time -p \
  env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/stu0032/paper/.venv/bin/python -u test_locomo10.py \
    --dataset /home/stu0032/paper/datasets/locomo/data/locomo10.json \
    --num-samples 1 \
    --parallel-questions \
    --test-workers 16 \
    --result-file "$RESULT" \
  >> "$LOG" 2>&1

status=$?
{
  echo "finished_at=$(date '+%F %T %Z')"
  echo "exit_status=$status"
} >> "$LOG"
echo "$status" > "$EXIT_CODE"

tail -120 "$LOG"
exit "$status"
