#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BASE_RUN_ROOT="/home/stu0032/paper/runs/2026_sota_memory_qwen25_3b_all"
if [ -z "${ALL_RUN_ROOT:-}" ] && [ -z "${RUN_TS:-}" ] && [ "${RESUME:-0}" = "1" ]; then
  LATEST_RUN="$(find "$BASE_RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
  if [ -n "$LATEST_RUN" ]; then
    ALL_RUN_ROOT="$LATEST_RUN"
  fi
fi
TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
ALL_RUN_ROOT="${ALL_RUN_ROOT:-$BASE_RUN_ROOT/$TS}"
export MEMGAS_RUN_ROOT="${MEMGAS_RUN_ROOT:-$ALL_RUN_ROOT/memgas}"
export OMNI_RUN_ROOT="${OMNI_RUN_ROOT:-$ALL_RUN_ROOT/omnisimplemem}"
export REME_RUN_ROOT="${REME_RUN_ROOT:-$ALL_RUN_ROOT/reme}"
mkdir -p "$ALL_RUN_ROOT"
LOG="$ALL_RUN_ROOT/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "[all] started_at=$(date '+%F %T %Z')"
echo "[all] all_run_root=$ALL_RUN_ROOT"
echo "[all] model=$OPENAI_MODEL base_url=$OPENAI_BASE_URL"

wait_for_vllm() {
  /home/stu0032/paper/.venv/bin/python - <<PY
from openai import OpenAI
client = OpenAI(api_key="${OPENAI_API_KEY}", base_url="${OPENAI_BASE_URL}")
client.models.list()
client.chat.completions.create(
    model="${OPENAI_MODEL}",
    messages=[{"role": "user", "content": "Reply OK."}],
    max_tokens=4,
    temperature=0,
)
PY
}

if ! wait_for_vllm; then
  echo "[all] vLLM not reachable; starting local Qwen2.5-3B server"
  /home/stu0032/paper/start_vllm_qwen25_3b.sh > "$ALL_RUN_ROOT/vllm_qwen25_3b.log" 2>&1 &
  for _ in $(seq 1 120); do
    if wait_for_vllm; then
      break
    fi
    sleep 5
  done
  wait_for_vllm
fi

if ! /home/stu0032/paper/.venv/bin/python - <<'PY'
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("roberta-large", local_files_only=True)
AutoModel.from_pretrained("roberta-large", local_files_only=True)
PY
then
  echo "[all] roberta-large for BERTScore is not cached; downloading"
  bash /home/stu0032/paper/download_bertscore_roberta.sh
fi

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py preflight \
  --targets memgas omni reme \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

bash /home/stu0032/paper/scripts/run_memgas_qwen25_3b_full.sh
bash /home/stu0032/paper/scripts/run_omnisimplemem_qwen25_3b_full.sh
bash /home/stu0032/paper/scripts/run_reme_qwen25_3b_full.sh

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py summarize \
  --run-root "$ALL_RUN_ROOT" \
  --memgas-dir "$MEMGAS_RUN_ROOT" \
  --omni-dir "$OMNI_RUN_ROOT" \
  --reme-dir "$REME_RUN_ROOT"

echo "[all] finished_at=$(date '+%F %T %Z')"
