#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
mkdir -p runs

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LOG="runs/bertscore_roberta_large_download.log"
EXIT_FILE="runs/bertscore_roberta_large_download.exit"

echo "started_at=$(date '+%F %T %Z')" | tee "$LOG"

/home/stu0032/paper/.venv/bin/python -u - <<'PY' >> "$LOG" 2>&1
from transformers import AutoModel, AutoTokenizer

model_id = "roberta-large"
print(f"downloading {model_id}", flush=True)
AutoTokenizer.from_pretrained(model_id)
AutoModel.from_pretrained(model_id)
print(f"cached {model_id}", flush=True)
PY

status=$?
{
  echo "finished_at=$(date '+%F %T %Z')"
  echo "exit_status=$status"
} >> "$LOG"
echo "$status" > "$EXIT_FILE"

tail -120 "$LOG"
exit "$status"
