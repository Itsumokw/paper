#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
source /home/stu0032/paper/scripts/common_runtime_limits.sh
source /home/stu0032/paper/scripts/high_utilization_runtime.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${MEMGAS_CUDA_VISIBLE_DEVICES:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LOCOMO_OPENAI_TIMEOUT="${LOCOMO_OPENAI_TIMEOUT:-600}"
export LOCOMO_QA_MAX_RETRIES="${LOCOMO_QA_MAX_RETRIES:-2}"
export MEMGAS_OPENAI_TIMEOUT="${MEMGAS_OPENAI_TIMEOUT:-600}"
export MEMGAS_LLM_MAX_RETRIES="${MEMGAS_LLM_MAX_RETRIES:-3}"
export MEMGAS_LLM_RETRY_WAIT_SEC="${MEMGAS_LLM_RETRY_WAIT_SEC:-5}"
export MEMGAS_MIN_GENERATION_TOKENS="${MEMGAS_MIN_GENERATION_TOKENS:-512}"
export MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH="${MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH:-0}"
export MEMGAS_SUMMARY_WORD_LIMIT="${MEMGAS_SUMMARY_WORD_LIMIT:-120}"
export MEMGAS_KEYWORD_LIMIT="${MEMGAS_KEYWORD_LIMIT:-30}"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

SOURCE_DATA_PATH="${LOCOMO_SOURCE_DATA_PATH:-${LOCOMO_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}}"
MEMGAS_CATEGORIES="${MEMGAS_CATEGORIES:-1,2,3,4}"
DATA_PATH="$SOURCE_DATA_PATH"
RUN_ROOT="${MEMGAS_RUN_ROOT:-/home/stu0032/paper/runs/memgas/qwen25_3b_full/$(date +%Y%m%d_%H%M%S)}"
if [ "${RESUME:-0}" != "1" ] && [ "${MEMGAS_FRESH:-1}" = "1" ] && [ -d "$RUN_ROOT" ]; then
  rm -rf "$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"

if [[ "$MEMGAS_CATEGORIES" != "all" ]]; then
  SAFE_CATEGORIES="${MEMGAS_CATEGORIES//,/}"
  DATA_PATH="${MEMGAS_FILTERED_DATASET:-$RUN_ROOT/locomo10_cat${SAFE_CATEGORIES}.json}"
  /home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/filter_locomo_categories.py \
    --input "$SOURCE_DATA_PATH" \
    --output "$DATA_PATH" \
    --categories "$MEMGAS_CATEGORIES" \
    > "$RUN_ROOT/dataset_filter_summary.json"
fi

read -r MEMGAS_DATASET_SAMPLE_COUNT MEMGAS_DATASET_QA_COUNT < <(/home/stu0032/paper/.venv/bin/python - "$DATA_PATH" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(data), sum(len(sample.get("qa", [])) for sample in data))
PY
)
export MEMGAS_EXPECTED_SAMPLES="${MEMGAS_EXPECTED_SAMPLES:-$MEMGAS_DATASET_SAMPLE_COUNT}"
export MEMGAS_EXPECTED_QA="${MEMGAS_EXPECTED_QA:-$MEMGAS_DATASET_QA_COUNT}"
if [[ "$MEMGAS_CATEGORIES" == "1,2,3,4" ]]; then
  export MEMGAS_EXPECTED_CAT14_QA="${MEMGAS_EXPECTED_CAT14_QA:-$MEMGAS_DATASET_QA_COUNT}"
fi

LOG="$RUN_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

if locomo_high_util_enabled; then
  export MEMGAS_HIGH_UTIL_MIN_QA_WORKERS="${MEMGAS_HIGH_UTIL_MIN_QA_WORKERS:-12}"
  export LOCOMO_HIGH_UTIL_VLLM_GPU_MEMORY_UTILIZATION="${LOCOMO_HIGH_UTIL_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
  export LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_SEQS="${LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_SEQS:-32}"
  locomo_raise_env_int MEMGAS_QA_WORKERS "$MEMGAS_HIGH_UTIL_MIN_QA_WORKERS"
  locomo_enable_high_util_vllm_defaults
fi
export LOCOMO_MAX_MODEL_TOKENS="${LOCOMO_MAX_MODEL_TOKENS:-${VLLM_MAX_MODEL_LEN:-32000}}"
export LOCOMO_MAX_ANSWER_TOKENS="${LOCOMO_MAX_ANSWER_TOKENS:-${MEMGAS_ANSWER_MAX_TOKENS:-8192}}"
export LOCOMO_MAX_PROMPT_TOKENS="${LOCOMO_MAX_PROMPT_TOKENS:-18000}"
export LOCOMO_TOKEN_GUARD_BUFFER="${LOCOMO_TOKEN_GUARD_BUFFER:-1536}"

echo "[memgas] started_at=$(date '+%F %T %Z')"
echo "[memgas] run_root=$RUN_ROOT"
cat > "$RUN_ROOT/command.env" <<EOF
OPENAI_MODEL=$OPENAI_MODEL
OPENAI_BASE_URL=$OPENAI_BASE_URL
LOCOMO_SOURCE_DATA_PATH=$SOURCE_DATA_PATH
MEMGAS_CATEGORIES=$MEMGAS_CATEGORIES
LOCOMO_DATA_PATH=$DATA_PATH
MEMGAS_RUN_ROOT=$RUN_ROOT
MEMGAS_QA_WORKERS=${MEMGAS_QA_WORKERS:-12}
MEMGAS_HIGH_UTIL_MIN_QA_WORKERS=${MEMGAS_HIGH_UTIL_MIN_QA_WORKERS:-12}
MEMGAS_TOPK=${MEMGAS_TOPK:-20}
MEMGAS_ANSWER_MAX_TOKENS=${MEMGAS_ANSWER_MAX_TOKENS:-8192}
MEMGAS_SUMMARY_MAX_TOKENS=${MEMGAS_SUMMARY_MAX_TOKENS:-8192}
MEMGAS_MAX_MODEL_TOKENS=${MEMGAS_MAX_MODEL_TOKENS:-$LOCOMO_MAX_MODEL_TOKENS}
MEMGAS_MAX_PROMPT_TOKENS=${MEMGAS_MAX_PROMPT_TOKENS:-$LOCOMO_MAX_PROMPT_TOKENS}
LOCOMO_MAX_MODEL_TOKENS=$LOCOMO_MAX_MODEL_TOKENS
LOCOMO_MAX_PROMPT_TOKENS=$LOCOMO_MAX_PROMPT_TOKENS
LOCOMO_MAX_ANSWER_TOKENS=$LOCOMO_MAX_ANSWER_TOKENS
LOCOMO_TOKEN_GUARD_BUFFER=$LOCOMO_TOKEN_GUARD_BUFFER
MEMGAS_OPENAI_TIMEOUT=$MEMGAS_OPENAI_TIMEOUT
MEMGAS_LLM_MAX_RETRIES=$MEMGAS_LLM_MAX_RETRIES
MEMGAS_LLM_RETRY_WAIT_SEC=$MEMGAS_LLM_RETRY_WAIT_SEC
LOCOMO_QA_MAX_RETRIES=$LOCOMO_QA_MAX_RETRIES
MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH=$MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH
MEMGAS_SUMMARY_WORD_LIMIT=$MEMGAS_SUMMARY_WORD_LIMIT
MEMGAS_KEYWORD_LIMIT=$MEMGAS_KEYWORD_LIMIT
MEMGAS_EXPECTED_SAMPLES=$MEMGAS_EXPECTED_SAMPLES
MEMGAS_EXPECTED_QA=$MEMGAS_EXPECTED_QA
MEMGAS_EXPECTED_CAT14_QA=${MEMGAS_EXPECTED_CAT14_QA:-}
LOCOMO_HIGH_UTILIZATION_AFTER_AMEM=${LOCOMO_HIGH_UTILIZATION_AFTER_AMEM:-1}
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$(locomo_default_vllm_model_path)}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_GENERATION_CONFIG=${VLLM_GENERATION_CONFIG:-}
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-}
BERTSCORE_MODEL=$BERTSCORE_MODEL
BERTSCORE_NUM_LAYERS=$BERTSCORE_NUM_LAYERS
BERTSCORE_BATCH_SIZE=$BERTSCORE_BATCH_SIZE
BERTSCORE_DEVICE=$BERTSCORE_DEVICE
EOF

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
locomo_set_vllm_model_path_for_openai_model
export VLLM_SERVED_MODEL="$OPENAI_MODEL"
export VLLM_ALT_SERVED_MODEL="${OPENAI_MODEL##*/}"
if ! locomo_high_util_enabled; then
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
fi
locomo_start_vllm_with_safe_fallback
cat >> "$RUN_ROOT/command.env" <<EOF
VLLM_EFFECTIVE_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-}
VLLM_EFFECTIVE_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_EFFECTIVE_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VLLM_EFFECTIVE_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
VLLM_EFFECTIVE_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
EOF

if [ "${RESUME:-0}" = "1" ] && /home/stu0032/paper/.venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("$RUN_ROOT") / "normalized_predictions.json"
ok = False
if p.exists():
    data = json.loads(p.read_text())
    ok = len(data.get("records", [])) == int("${MEMGAS_EXPECTED_QA:-1986}")
raise SystemExit(0 if ok else 1)
PY
then
  echo "[memgas] complete normalized output exists; skipping because RESUME=1"
  exit 0
fi

export LOCOMO_PREFLIGHT_EXPECTED_SAMPLES="${LOCOMO_PREFLIGHT_EXPECTED_SAMPLES:-$MEMGAS_EXPECTED_SAMPLES}"
export LOCOMO_PREFLIGHT_EXPECTED_QA="${LOCOMO_PREFLIGHT_EXPECTED_QA:-$MEMGAS_EXPECTED_QA}"
/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py preflight \
  --data-path "$DATA_PATH" \
  --targets memgas \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

MEMGAS_RUN_ARGS=(
  /home/stu0032/paper/scripts/locomo_2026_sota.py run-memgas
  --data-path "$DATA_PATH"
  --output-dir "$RUN_ROOT"
  --model "$OPENAI_MODEL"
  --base-url "$OPENAI_BASE_URL"
  --api-key "$OPENAI_API_KEY"
  --embedder "${MEMGAS_EMBEDDER:-minilm}"
  --device "${MEMGAS_DEVICE:-cpu}"
  --method "${MEMGAS_METHOD:-memgas}"
  --topk "${MEMGAS_TOPK:-20}"
  --qa-workers "${MEMGAS_QA_WORKERS:-12}"
  --answer-max-tokens "${MEMGAS_ANSWER_MAX_TOKENS:-8192}"
  --summary-max-tokens "${MEMGAS_SUMMARY_MAX_TOKENS:-8192}"
)
if [[ -n "${MEMGAS_MAX_CONVERSATIONS:-}" ]]; then
  MEMGAS_RUN_ARGS+=(--max-conversations "$MEMGAS_MAX_CONVERSATIONS")
fi

/home/stu0032/paper/.venv/bin/python "${MEMGAS_RUN_ARGS[@]}"

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py validate \
  --input "$RUN_ROOT/normalized_predictions.json" \
  --expected-count "${MEMGAS_EXPECTED_QA:-1986}"

/home/stu0032/paper/.venv/bin/python - "$RUN_ROOT/normalized_predictions.json" "$RUN_ROOT/normalized_predictions_cat1_4.json" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text())
records = data.get("records", [])
cat14 = [row for row in records if str(row.get("category")) in {"1", "2", "3", "4"}]
expected = int(os.environ.get("MEMGAS_EXPECTED_CAT14_QA", "1540"))
if len(cat14) != expected:
    raise SystemExit(f"Expected {expected} MemGAS cat1-4 rows, got {len(cat14)}")
dst.write_text(json.dumps({"records": cat14, "summary": data.get("summary", {})}, ensure_ascii=False, indent=2))
print("[memgas] normalized cat1-4 predictions:", dst)
PY

METRICS_ARGS=(
  --input "$RUN_ROOT/normalized_predictions.json"
  --output "$RUN_ROOT/normalized_metrics_all.json"
  --prediction-key prediction
  --reference-key reference
)
if [ "${MEMGAS_SKIP_BERTSCORE:-1}" = "1" ]; then
  METRICS_ARGS+=(--skip-bertscore)
else
  METRICS_ARGS+=(
    --bertscore-model "${BERTSCORE_MODEL:-roberta-large}"
    --bertscore-batch-size "${BERTSCORE_BATCH_SIZE:-4}"
    --bertscore-num-layers "${BERTSCORE_NUM_LAYERS:-17}"
    --bertscore-device "${BERTSCORE_DEVICE:-cpu}"
    --fail-on-bertscore-error
  )
fi
/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/compute_locomo_text_metrics.py "${METRICS_ARGS[@]}"

CAT14_METRICS_ARGS=("${METRICS_ARGS[@]}")
for i in "${!CAT14_METRICS_ARGS[@]}"; do
  if [[ "${CAT14_METRICS_ARGS[$i]}" == "$RUN_ROOT/normalized_predictions.json" ]]; then
    CAT14_METRICS_ARGS[$i]="$RUN_ROOT/normalized_predictions_cat1_4.json"
  elif [[ "${CAT14_METRICS_ARGS[$i]}" == "$RUN_ROOT/normalized_metrics_all.json" ]]; then
    CAT14_METRICS_ARGS[$i]="$RUN_ROOT/normalized_metrics_cat1_4.json"
  fi
done
/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/compute_locomo_text_metrics.py "${CAT14_METRICS_ARGS[@]}"

echo "[memgas] finished_at=$(date '+%F %T %Z')"
