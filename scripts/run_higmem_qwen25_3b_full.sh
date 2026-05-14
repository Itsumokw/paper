#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper/baseline/HiGMem
source /home/stu0032/paper/scripts/common_runtime_limits.sh
source /home/stu0032/paper/scripts/high_utilization_runtime.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HIGMEM_USE_STREAMING="${HIGMEM_USE_STREAMING:-1}"
export HIGMEM_MAX_TOKENS="${HIGMEM_MAX_TOKENS:-8192}"
export HIGMEM_OPENAI_TIMEOUT="${HIGMEM_OPENAI_TIMEOUT:-300}"
export HIGMEM_RUN_ROOT="${HIGMEM_RUN_ROOT:-${QWEN3_8B_RUN_ROOT:+$QWEN3_8B_RUN_ROOT/higmem}}"
export HIGMEM_RUN_ROOT="${HIGMEM_RUN_ROOT:-/home/stu0032/paper/runs/higmem/qwen25_3b_full/$(date +%Y%m%d_%H%M%S)}"
export HIGMEM_SOURCE_DATASET="${HIGMEM_SOURCE_DATASET:-${HIGMEM_DATASET:-data/locomo10.json}}"
export HIGMEM_CATEGORIES="${HIGMEM_CATEGORIES:-1,2,3,4}"
export HIGMEM_EFFECTIVE_DATASET="$HIGMEM_SOURCE_DATASET"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

# Keep the evaluation process on CPU so it does not compete with vLLM for GPU memory.
export CUDA_VISIBLE_DEVICES=""

# HiGMem still reports F1/ROUGE/BLEU/METEOR/SBERT. Recompute BERTScore later if needed.
export SKIP_BERTSCORE="${SKIP_BERTSCORE:-1}"
mkdir -p "$HIGMEM_RUN_ROOT"

if [[ "$HIGMEM_CATEGORIES" != "all" ]]; then
  SAFE_CATEGORIES="${HIGMEM_CATEGORIES//,/}"
  export HIGMEM_EFFECTIVE_DATASET="${HIGMEM_FILTERED_DATASET:-$HIGMEM_RUN_ROOT/locomo10_cat${SAFE_CATEGORIES}.json}"
  /home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/filter_locomo_categories.py \
    --input "$HIGMEM_SOURCE_DATASET" \
    --output "$HIGMEM_EFFECTIVE_DATASET" \
    --categories "$HIGMEM_CATEGORIES" \
    > "$HIGMEM_RUN_ROOT/dataset_filter_summary.json"
fi

read -r HIGMEM_DATASET_SAMPLE_COUNT HIGMEM_DATASET_QA_COUNT < <(/home/stu0032/paper/.venv/bin/python - "$HIGMEM_EFFECTIVE_DATASET" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(data), sum(len(sample.get("qa", [])) for sample in data))
PY
)
export HIGMEM_EXPECTED_SAMPLES="${HIGMEM_EXPECTED_SAMPLES:-$HIGMEM_DATASET_SAMPLE_COUNT}"
export HIGMEM_EXPECTED_QA="${HIGMEM_EXPECTED_QA:-$HIGMEM_DATASET_QA_COUNT}"
if [[ "$HIGMEM_CATEGORIES" == "1,2,3,4" ]]; then
  export HIGMEM_EXPECTED_CAT14_QA="${HIGMEM_EXPECTED_CAT14_QA:-$HIGMEM_DATASET_QA_COUNT}"
fi

if locomo_high_util_enabled; then
  locomo_raise_env_int HIGMEM_WORKERS 16
  locomo_enable_high_util_vllm_defaults
fi

locomo_set_vllm_model_path_for_openai_model
export VLLM_SERVED_MODEL="$OPENAI_MODEL"
export VLLM_ALT_SERVED_MODEL="${OPENAI_MODEL##*/}"
locomo_start_vllm_with_safe_fallback

cat > "$HIGMEM_RUN_ROOT/command.env" <<EOF
OPENAI_MODEL=$OPENAI_MODEL
OPENAI_API_BASE=$OPENAI_API_BASE
HIGMEM_SOURCE_DATASET=$HIGMEM_SOURCE_DATASET
HIGMEM_CATEGORIES=$HIGMEM_CATEGORIES
HIGMEM_DATASET=$HIGMEM_EFFECTIVE_DATASET
HIGMEM_RUN_ROOT=$HIGMEM_RUN_ROOT
HIGMEM_USE_STREAMING=$HIGMEM_USE_STREAMING
HIGMEM_MAX_TOKENS=$HIGMEM_MAX_TOKENS
HIGMEM_OPENAI_TIMEOUT=$HIGMEM_OPENAI_TIMEOUT
HIGMEM_WORKERS=${HIGMEM_WORKERS:-5}
HIGMEM_EXPECTED_SAMPLES=$HIGMEM_EXPECTED_SAMPLES
HIGMEM_EXPECTED_QA=$HIGMEM_EXPECTED_QA
HIGMEM_EXPECTED_CAT14_QA=${HIGMEM_EXPECTED_CAT14_QA:-}
LOCOMO_HIGH_UTILIZATION_AFTER_AMEM=${LOCOMO_HIGH_UTILIZATION_AFTER_AMEM:-1}
VLLM_MODEL_PATH=$VLLM_MODEL_PATH
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_GENERATION_CONFIG=${VLLM_GENERATION_CONFIG:-}
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
BERTSCORE_MODEL=$BERTSCORE_MODEL
BERTSCORE_NUM_LAYERS=$BERTSCORE_NUM_LAYERS
BERTSCORE_BATCH_SIZE=$BERTSCORE_BATCH_SIZE
BERTSCORE_DEVICE=$BERTSCORE_DEVICE
EOF
exec > >(tee -a "$HIGMEM_RUN_ROOT/run.log") 2>&1

args=(
  --dataset "$HIGMEM_EFFECTIVE_DATASET"
  --model "${HIGMEM_MODEL:-$OPENAI_MODEL}"
  --backend openai
  --api_base "$OPENAI_API_BASE"
  --api_key "$OPENAI_API_KEY"
  --ablation-no-profile
  --ablation-event-metadata-only
  --ablation-no-link
  --k_event 10
  --num-workers "${HIGMEM_WORKERS:-5}"
  --output-dir "$HIGMEM_RUN_ROOT"
)
if [[ -n "${HIGMEM_SAMPLE_INDEX:-}" ]]; then
  args+=(--sample_index "$HIGMEM_SAMPLE_INDEX")
fi
if [[ -n "${HIGMEM_MAX_TURNS:-}" ]]; then
  args+=(--max_turns "$HIGMEM_MAX_TURNS")
fi
if [[ -n "${HIGMEM_MAX_QUESTIONS:-}" ]]; then
  args+=(--max_questions "$HIGMEM_MAX_QUESTIONS")
fi

/home/stu0032/paper/.venv/bin/python -u run_fphm_evaluation.py \
  "${args[@]}"

/home/stu0032/paper/.venv/bin/python - "$HIGMEM_RUN_ROOT/normalized_predictions.json" "$HIGMEM_RUN_ROOT/normalized_predictions_cat1_4.json" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text())
records = data.get("records", [])
expected_all = int(os.environ.get("HIGMEM_EXPECTED_QA", "1986"))
if len(records) != expected_all:
    raise SystemExit(f"Expected {expected_all} HiGMem QA rows, got {len(records)}")
empty = [i for i, row in enumerate(records) if not str(row.get("prediction") or "").strip()]
if empty:
    raise SystemExit(f"HiGMem has empty predictions at rows: {empty[:10]}")
cat14 = [row for row in records if str(row.get("category")) in {"1", "2", "3", "4"}]
expected_cat14 = int(os.environ.get("HIGMEM_EXPECTED_CAT14_QA", "1540"))
if len(cat14) != expected_cat14:
    raise SystemExit(f"Expected {expected_cat14} HiGMem cat1-4 rows, got {len(cat14)}")
dst.write_text(json.dumps({"records": cat14, "summary": data.get("summary", {})}, indent=2, ensure_ascii=False))
print(f"Validated HiGMem rows: all={len(records)} cat1-4={len(cat14)}")
PY

metrics_args=(
  --input "$HIGMEM_RUN_ROOT/normalized_predictions_cat1_4.json"
  --output "$HIGMEM_RUN_ROOT/higmem_metrics_cat1_4.json"
  --prediction-key prediction
  --reference-key reference
  --question-key question
  --category-key category
)
if [ "${HIGMEM_SKIP_BERTSCORE:-0}" = "1" ]; then
  metrics_args+=(--skip-bertscore)
else
  metrics_args+=(
    --bertscore-model "$BERTSCORE_MODEL"
    --bertscore-num-layers "$BERTSCORE_NUM_LAYERS"
    --bertscore-batch-size "$BERTSCORE_BATCH_SIZE"
    --bertscore-device "$BERTSCORE_DEVICE"
    --fail-on-bertscore-error
  )
fi
/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/compute_locomo_text_metrics.py "${metrics_args[@]}"
