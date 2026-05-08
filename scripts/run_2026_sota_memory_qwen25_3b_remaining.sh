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
if [ -z "${ALL_RUN_ROOT:-}" ]; then
  LATEST_RUN="$(find "$BASE_RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$LATEST_RUN" ]; then
    echo "[remaining] no previous all-run directory found under $BASE_RUN_ROOT" >&2
    exit 1
  fi
  ALL_RUN_ROOT="$LATEST_RUN"
fi

export MEMGAS_RUN_ROOT="${MEMGAS_RUN_ROOT:-$ALL_RUN_ROOT/memgas}"
export OMNI_RUN_ROOT="${OMNI_RUN_ROOT:-$ALL_RUN_ROOT/omnisimplemem}"
export REME_RUN_ROOT="${REME_RUN_ROOT:-$ALL_RUN_ROOT/reme}"

mkdir -p "$ALL_RUN_ROOT"
LOG="$ALL_RUN_ROOT/run_remaining.log"
exec > >(tee -a "$LOG") 2>&1

echo "[remaining] started_at=$(date '+%F %T %Z')"
echo "[remaining] all_run_root=$ALL_RUN_ROOT"
echo "[remaining] memgas_dir=$MEMGAS_RUN_ROOT"
echo "[remaining] omni_dir=$OMNI_RUN_ROOT"
echo "[remaining] reme_dir=$REME_RUN_ROOT"

if [ ! -f "$MEMGAS_RUN_ROOT/normalized_predictions.json" ]; then
  echo "[remaining] MemGAS normalized output not found; refusing to summarize a mixed/incomplete run" >&2
  exit 1
fi

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
  echo "[remaining] vLLM not reachable; starting local Qwen2.5-3B server"
  /home/stu0032/paper/start_vllm_qwen25_3b.sh > "$ALL_RUN_ROOT/vllm_qwen25_3b_remaining.log" 2>&1 &
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
  echo "[remaining] roberta-large for BERTScore is not cached; downloading"
  bash /home/stu0032/paper/download_bertscore_roberta.sh
fi

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py preflight \
  --targets omni reme \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

if [ "${FRESH_OMNI:-1}" = "1" ]; then
  echo "[remaining] removing previous Omni-SimpleMem partial output"
  rm -rf "$OMNI_RUN_ROOT"
fi

export OMNI_INGEST_WORKERS="${OMNI_INGEST_WORKERS:-1}"
export OMNI_WORKERS="${OMNI_WORKERS:-2}"
export REME_MAX_CONCURRENCY="${REME_MAX_CONCURRENCY:-1}"
export RESUME="${RESUME:-1}"

bash /home/stu0032/paper/scripts/run_omnisimplemem_qwen25_3b_full.sh
bash /home/stu0032/paper/scripts/run_reme_qwen25_3b_full.sh

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py summarize \
  --run-root "$ALL_RUN_ROOT" \
  --memgas-dir "$MEMGAS_RUN_ROOT" \
  --omni-dir "$OMNI_RUN_ROOT" \
  --reme-dir "$REME_RUN_ROOT"

echo "[remaining] finished_at=$(date '+%F %T %Z')"
