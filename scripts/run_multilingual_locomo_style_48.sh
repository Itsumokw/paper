#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

export EXPERIMENT_CPU_THREADS="${MULTI_STYLE_CPU_THREADS:-${EXPERIMENT_CPU_THREADS:-8}}"
export EXPERIMENT_CPU_INTEROP_THREADS="${MULTI_STYLE_CPU_INTEROP_THREADS:-${EXPERIMENT_CPU_INTEROP_THREADS:-2}}"
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LOCOMO_HIGH_UTILIZATION_AFTER_AMEM="${LOCOMO_HIGH_UTILIZATION_AFTER_AMEM:-1}"
export LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART="${LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART:-0}"
export VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-900}"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-${MULTI_STYLE_BERTSCORE_BATCH_SIZE:-8}}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

PYTHON="/home/stu0032/paper/.venv/bin/python"
TS="${MULTI_STYLE_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MULTI_STYLE_RUN_ROOT:-/home/stu0032/paper/runs/multilingual_locomo_style_48/$TS}"
STATUS_DIR="$RUN_ROOT/status"
VLLM_PROFILE_FILE="$RUN_ROOT/status/current_vllm_profile.txt"
FAILURE_LOG="$RUN_ROOT/failure_repair_log.md"
DATASET_ROOT="${MULTI_STYLE_DATASET_ROOT:-/home/stu0032/paper/datasets/locomo_style_eval_repaired_20260513/primary}"
mkdir -p "$RUN_ROOT" "$STATUS_DIR" "$RUN_ROOT/datasets" "$RUN_ROOT/vllm_logs"
export VLLM_LOG_DIR="$RUN_ROOT/vllm_logs"
export MULTI_STYLE_RUN_JUDGE="${MULTI_STYLE_RUN_JUDGE:-1}"
export MULTI_STYLE_JUDGE_WORKERS="${MULTI_STYLE_JUDGE_WORKERS:-8}"
export MULTI_STYLE_JUDGE_PROTOCOL="${MULTI_STYLE_JUDGE_PROTOCOL:-locomo_binary}"

ALL_MODELS=(qwen25_3b qwen3_8b)
ALL_DATASETS=(perltqa opela jlongchat del1l2im)
ALL_METHODS=(full_context amem mem0 simplemem higmem memgas)

MODELS=(${MULTI_STYLE_MODELS:-${ALL_MODELS[*]}})
DATASETS=(${MULTI_STYLE_DATASETS:-${ALL_DATASETS[*]}})
METHODS=(${MULTI_STYLE_METHODS:-${ALL_METHODS[*]}})

declare -A DATASET_SOURCE=(
  [perltqa]="$DATASET_ROOT/PerLTQA-LoCoMo-style-eval.json"
  [opela]="$DATASET_ROOT/OPELA-LoCoMo-style-eval.json"
  [jlongchat]="$DATASET_ROOT/JLongChat-LoCoMo-style-eval.json"
  [del1l2im]="$DATASET_ROOT/deL1L2IM-LoCoMo-style-eval.json"
)
declare -A DATASET_LABEL=(
  [perltqa]="PerLTQA"
  [opela]="OPELA"
  [jlongchat]="JLongChat"
  [del1l2im]="deL1L2IM"
)
declare -A DATASET_SAMPLES=(
  [perltqa]=10
  [opela]=10
  [jlongchat]=10
  [del1l2im]=9
)
declare -A DATASET_CAT14_QA=(
  [perltqa]=320
  [opela]=200
  [jlongchat]=200
  [del1l2im]=180
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

usage() {
  cat <<'EOF'
Run multilingual LoCoMo-style 48-result baseline grid.

Environment filters:
  MULTI_STYLE_MODELS="qwen25_3b qwen3_8b"
  MULTI_STYLE_DATASETS="perltqa opela jlongchat del1l2im"
  MULTI_STYLE_METHODS="full_context amem mem0 simplemem higmem memgas"
  MULTI_STYLE_RUN_ROOT=/path/to/run
  MULTI_STYLE_DATASET_ROOT=/path/to/repaired/primary
  MULTI_STYLE_RUN_JUDGE=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

append_failure() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  local exit_code="$4"
  local log_file="$5"
  {
    printf '\n## %s %s %s failed at %s\n\n' "$model" "$dataset" "$method" "$(date '+%F %T %Z')"
    printf -- '- exit_code: `%s`\n' "$exit_code"
    printf -- '- run_log: `%s`\n' "$log_file"
    printf -- '- reason: command failed; inspect the run log tail and rerun the same task after repair.\n'
    printf -- '- repair: pending\n'
  } >> "$FAILURE_LOG"
}

mark_repair() {
  local note="$1"
  {
    printf '\n## repair note at %s\n\n' "$(date '+%F %T %Z')"
    printf -- '- %s\n' "$note"
  } >> "$FAILURE_LOG"
}

served_model_ready() {
  local model="$1"
  OPENAI_BASE_URL="$OPENAI_BASE_URL" OPENAI_MODEL="$model" "$PYTHON" - <<'PY' >/dev/null 2>&1
import os
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url=os.environ["OPENAI_BASE_URL"], timeout=5, max_retries=0)
models = [m.id for m in client.models.list().data]
raise SystemExit(0 if os.environ["OPENAI_MODEL"] in models else 1)
PY
}

configure_model_env() {
  local model_key="$1"
  local profile="$2"
  case "$model_key" in
    qwen25_3b)
      export OPENAI_MODEL="Qwen/Qwen2.5-3B-Instruct"
      export QWEN25_3B_MODEL_PATH="/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean"
      export VLLM_MODEL_PATH="$QWEN25_3B_MODEL_PATH"
      export VLLM_SERVED_MODEL="$OPENAI_MODEL"
      export VLLM_ALT_SERVED_MODEL="Qwen2.5-3B-Instruct"
      export VLLM_GPU_MEMORY_UTILIZATION="${MULTI_STYLE_3B_VLLM_GPU_MEMORY_UTILIZATION:-0.96}"
      export VLLM_MAX_MODEL_LEN="${MULTI_STYLE_3B_VLLM_MAX_MODEL_LEN:-32000}"
      export VLLM_MAX_NUM_SEQS="${MULTI_STYLE_3B_VLLM_MAX_NUM_SEQS:-32}"
      export VLLM_MAX_NUM_BATCHED_TOKENS="${MULTI_STYLE_3B_VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"
      export VLLM_GENERATION_CONFIG="${MULTI_STYLE_VLLM_GENERATION_CONFIG:-vllm}"
      export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${MULTI_STYLE_VLLM_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
      export VLLM_EXTRA_ARGS="${MULTI_STYLE_3B_VLLM_EXTRA_ARGS:---enforce-eager}"
      ;;
    qwen3_8b)
      export OPENAI_MODEL="Qwen/Qwen3-8B"
      export QWEN3_8B_MODEL_PATH="/home/stu0032/paper/models/Qwen3-8B"
      export QWEN25_3B_MODEL_PATH="$QWEN3_8B_MODEL_PATH"
      export VLLM_MODEL_PATH="$QWEN3_8B_MODEL_PATH"
      export VLLM_SERVED_MODEL="$OPENAI_MODEL"
      export VLLM_ALT_SERVED_MODEL="Qwen3-8B"
      if [[ "$profile" == "memgas8b" ]]; then
        export VLLM_GPU_MEMORY_UTILIZATION="${MULTI_STYLE_8B_MEMGAS_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
        export VLLM_MAX_MODEL_LEN="${MULTI_STYLE_8B_MEMGAS_VLLM_MAX_MODEL_LEN:-30000}"
        export VLLM_MAX_NUM_SEQS="${MULTI_STYLE_8B_MEMGAS_VLLM_MAX_NUM_SEQS:-32}"
        export VLLM_MAX_NUM_BATCHED_TOKENS="${MULTI_STYLE_8B_MEMGAS_VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"
        export VLLM_EXTRA_ARGS="${MULTI_STYLE_8B_MEMGAS_VLLM_EXTRA_ARGS:---enforce-eager}"
      else
        export VLLM_GPU_MEMORY_UTILIZATION="${MULTI_STYLE_8B_VLLM_GPU_MEMORY_UTILIZATION:-0.98}"
        export VLLM_MAX_MODEL_LEN="${MULTI_STYLE_8B_VLLM_MAX_MODEL_LEN:-32000}"
        export VLLM_MAX_NUM_SEQS="${MULTI_STYLE_8B_VLLM_MAX_NUM_SEQS:-32}"
        export VLLM_MAX_NUM_BATCHED_TOKENS="${MULTI_STYLE_8B_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
        export VLLM_EXTRA_ARGS="${MULTI_STYLE_8B_VLLM_EXTRA_ARGS:-}"
      fi
      export VLLM_GENERATION_CONFIG="${MULTI_STYLE_VLLM_GENERATION_CONFIG:-vllm}"
      export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${MULTI_STYLE_VLLM_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
      ;;
    *)
      echo "Unknown model key: $model_key" >&2
      exit 2
      ;;
  esac
}

ensure_vllm_profile() {
  local model_key="$1"
  local profile="$2"
  local profile_key="${model_key}:${profile}"
  configure_model_env "$model_key" "$profile"

  if [[ -f "$VLLM_PROFILE_FILE" ]] && [[ "$(cat "$VLLM_PROFILE_FILE")" == "$profile_key" ]] && served_model_ready "$OPENAI_MODEL"; then
    log "vLLM profile already active: $profile_key"
    return 0
  fi

  local force_restart=1
  if served_model_ready "$OPENAI_MODEL" && [[ ! -f "$VLLM_PROFILE_FILE" ]]; then
    force_restart=0
  fi
  if served_model_ready "$OPENAI_MODEL" && [[ -f "$VLLM_PROFILE_FILE" ]] && [[ "$(cat "$VLLM_PROFILE_FILE")" == "$profile_key" ]]; then
    force_restart=0
  fi

  log "Ensuring vLLM profile $profile_key force_restart=$force_restart"
  CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}" VLLM_FORCE_RESTART="$force_restart" bash scripts/start_vllm_qwen25_3b.sh
  printf '%s\n' "$profile_key" > "$VLLM_PROFILE_FILE"
}

dataset_cat14_path() {
  local dataset="$1"
  printf '%s/datasets/%s_cat1234.json' "$RUN_ROOT" "$dataset"
}

prepare_dataset() {
  local dataset="$1"
  local source="${DATASET_SOURCE[$dataset]}"
  local output
  output="$(dataset_cat14_path "$dataset")"
  local filter_summary="$RUN_ROOT/datasets/${dataset}_filter_summary.json"
  local need_filter=1
  if [[ -s "$output" && -s "$filter_summary" ]]; then
    local summary_input
    summary_input="$("$PYTHON" - "$filter_summary" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("input", ""))
except Exception:
    print("")
PY
)"
    if [[ "$summary_input" == "$source" ]]; then
      need_filter=0
    fi
  fi
  if [[ "$need_filter" == "1" ]]; then
    log "Filtering ${DATASET_LABEL[$dataset]} to categories 1-4 -> $output"
    "$PYTHON" scripts/filter_locomo_categories.py \
      --input "$source" \
      --output "$output" \
      --categories 1,2,3,4 \
      > "$filter_summary"
  fi
  "$PYTHON" scripts/normalize_locomo_style_runner_dates.py \
    --input "$output" \
    --output "$output" \
    --summary "$RUN_ROOT/datasets/${dataset}_date_normalization_summary.json" \
    > "$RUN_ROOT/datasets/${dataset}_date_normalization_stdout.json"
  "$PYTHON" - "$dataset" "$output" "${DATASET_SAMPLES[$dataset]}" "${DATASET_CAT14_QA[$dataset]}" <<'PY'
import json
import sys
from pathlib import Path

name, path, expected_samples, expected_qa = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
data = json.loads(path.read_text(encoding="utf-8"))
samples = len(data)
qa = sum(len(sample.get("qa", [])) for sample in data)
if samples != expected_samples or qa != expected_qa:
    raise SystemExit(f"{name}: expected samples/qa {expected_samples}/{expected_qa}, got {samples}/{qa}")
PY
}

task_complete() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  local judge_args=()
  if [[ "$MULTI_STYLE_RUN_JUDGE" != "1" ]]; then
    judge_args+=(--skip-judge)
  fi
  "$PYTHON" scripts/check_multilingual_locomo_style_48.py \
    --run-root "$RUN_ROOT" \
    --model "$model" \
    --dataset "$dataset" \
    --method "$method" \
    --quiet \
    --no-write \
    "${judge_args[@]}" \
    --judge-protocol "$MULTI_STYLE_JUDGE_PROTOCOL" \
    --fail-on-incomplete
}

task_text_complete() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  "$PYTHON" scripts/check_multilingual_locomo_style_48.py \
    --run-root "$RUN_ROOT" \
    --model "$model" \
    --dataset "$dataset" \
    --method "$method" \
    --quiet \
    --no-write \
    --skip-judge \
    --fail-on-incomplete
}

run_task_command() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  local log_file="$4"
  shift 4
  local start_ts
  start_ts="$(date +%s)"
  log "START $model/$dataset/$method"
  set +e
  "$@" 2>&1 | tee "$log_file"
  local exit_code=${PIPESTATUS[0]}
  set -e
  local elapsed=$(( $(date +%s) - start_ts ))
  if [[ "$exit_code" -ne 0 ]]; then
    log "FAILED $model/$dataset/$method exit=$exit_code elapsed=${elapsed}s"
    append_failure "$model" "$dataset" "$method" "$exit_code" "$log_file"
    return "$exit_code"
  fi
  log "DONE $model/$dataset/$method elapsed=${elapsed}s"
  return 0
}

method_prediction_path() {
  local task_root="$1"
  local method="$2"
  case "$method" in
    full_context) printf '%s/full_context_fixed/full_context/full_context_predictions_flat.json' "$task_root" ;;
    amem) printf '%s/amem_official/official_predictions_cat1_4.json' "$task_root" ;;
    mem0) printf '%s/mem0_fixed/mem0/mem0_predictions_flat.json' "$task_root" ;;
    simplemem) printf '%s/simplemem/normalized_predictions_cat1_4.json' "$task_root" ;;
    higmem) printf '%s/higmem/normalized_predictions_cat1_4.json' "$task_root" ;;
    memgas) printf '%s/memgas/normalized_predictions_cat1_4.json' "$task_root" ;;
    *) echo "Unknown method for prediction path: $method" >&2; return 2 ;;
  esac
}

method_judge_path() {
  local task_root="$1"
  local method="$2"
  case "$method" in
    full_context) printf '%s/full_context_fixed/full_context/full_context_judge_metrics_cat1_4.json' "$task_root" ;;
    amem) printf '%s/amem_official/official_judge_metrics_cat1_4.json' "$task_root" ;;
    mem0) printf '%s/mem0_fixed/mem0/mem0_judge_metrics_cat1_4.json' "$task_root" ;;
    simplemem) printf '%s/simplemem/simplemem_judge_metrics_cat1_4.json' "$task_root" ;;
    higmem) printf '%s/higmem/higmem_judge_metrics_cat1_4.json' "$task_root" ;;
    memgas) printf '%s/memgas/normalized_judge_metrics_cat1_4.json' "$task_root" ;;
    *) echo "Unknown method for judge path: $method" >&2; return 2 ;;
  esac
}

run_judge_for_method() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  local task_root="$RUN_ROOT/$model/$dataset"
  local pred_path judge_path pred_key ref_key
  if [[ "$MULTI_STYLE_RUN_JUDGE" != "1" ]]; then
    return 0
  fi
  pred_path="$(method_prediction_path "$task_root" "$method")"
  judge_path="$(method_judge_path "$task_root" "$method")"
  if [[ "$method" == "full_context" || "$method" == "mem0" ]]; then
    pred_key="model_answer"
    ref_key="golden_answer"
  else
    pred_key="prediction"
    ref_key="reference"
  fi
  if [[ ! -s "$pred_path" ]]; then
    echo "Prediction file missing for judge: $pred_path" >&2
    return 1
  fi
  log "JUDGE $model/$dataset/$method -> $judge_path"
  "$PYTHON" scripts/compute_locomo_llm_judge_metrics.py \
    --input "$pred_path" \
    --output "$judge_path" \
    --prediction-key "$pred_key" \
    --reference-key "$ref_key" \
    --question-key question \
    --category-key category \
    --model "${MULTI_STYLE_JUDGE_MODEL:-$OPENAI_MODEL}" \
    --base-url "${MULTI_STYLE_JUDGE_BASE_URL:-$OPENAI_BASE_URL}" \
    --api-key "${MULTI_STYLE_JUDGE_API_KEY:-$OPENAI_API_KEY}" \
    --protocol "$MULTI_STYLE_JUDGE_PROTOCOL" \
    --max-workers "$MULTI_STYLE_JUDGE_WORKERS" \
    --max-retries "${MULTI_STYLE_JUDGE_MAX_RETRIES:-3}" \
    --resume \
    --fail-on-error \
    2>&1 | tee "$task_root/${method}_judge.log"
}

run_full_context() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  local data_path
  data_path="$(dataset_cat14_path "$dataset")"
  mkdir -p "$task_root/full_context_fixed"
  run_task_command "$model" "$dataset" "full_context" "$task_root/full_context_fixed/run.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      OPENAI_API_BASE="$OPENAI_BASE_URL" \
      QWEN25_3B_MODEL_PATH="$VLLM_MODEL_PATH" \
      LOCOMO_DATASET="$data_path" \
      CORE_BASELINE_RUN_ROOT="$task_root/full_context_fixed" \
      CORE_BASELINE_START_IDX=0 \
      CORE_BASELINE_END_IDX="${DATASET_SAMPLES[$dataset]}" \
      CORE_BASELINE_EXPECTED_TRAJECTORIES="${DATASET_SAMPLES[$dataset]}" \
      CORE_BASELINE_EXPECTED_QA_COUNT="${DATASET_CAT14_QA[$dataset]}" \
      CORE_BASELINE_BUILD_WORKERS="${MULTI_STYLE_FULL_CONTEXT_BUILD_WORKERS:-4}" \
      CORE_BASELINE_SEARCH_WORKERS="${MULTI_STYLE_FULL_CONTEXT_SEARCH_WORKERS:-4}" \
      CORE_BASELINE_API_PARALLEL="${MULTI_STYLE_FULL_CONTEXT_API_PARALLEL:-16}" \
      CORE_BASELINE_QA_BATCH="${MULTI_STYLE_FULL_CONTEXT_QA_BATCH:-16}" \
      CORE_BASELINE_QA_MAX_TOKENS="${MULTI_STYLE_QA_MAX_TOKENS:-8192}" \
      CORE_BASELINE_MAX_MODEL_TOKENS="$VLLM_MAX_MODEL_LEN" \
      CORE_BASELINE_CONTEXT_TOKEN_BUFFER="${MULTI_STYLE_CONTEXT_TOKEN_BUFFER:-1024}" \
      CORE_BASELINE_FORCE_METRICS=1 \
      CORE_BASELINE_RUN_JUDGE="$MULTI_STYLE_RUN_JUDGE" \
      CORE_BASELINE_JUDGE_MODEL="${MULTI_STYLE_JUDGE_MODEL:-$OPENAI_MODEL}" \
      CORE_BASELINE_JUDGE_BASE_URL="${MULTI_STYLE_JUDGE_BASE_URL:-$OPENAI_BASE_URL}" \
      CORE_BASELINE_JUDGE_API_KEY="${MULTI_STYLE_JUDGE_API_KEY:-$OPENAI_API_KEY}" \
      CORE_BASELINE_JUDGE_WORKERS="$MULTI_STYLE_JUDGE_WORKERS" \
      CORE_BASELINE_JUDGE_PROTOCOL="$MULTI_STYLE_JUDGE_PROTOCOL" \
      CORE_BASELINE_SKIP_BERTSCORE=0 \
      CORE_BASELINE_BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      CORE_BASELINE_BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      CORE_BASELINE_BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      CORE_BASELINE_BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      bash scripts/run_locomo_fixed_missing_baselines_qwen25_3b.sh FullContext
}

run_mem0() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  local data_path
  local mem0_isolate_parallel="${MULTI_STYLE_MEM0_ISOLATE_BUILD_PARALLEL:-}"
  if [[ -z "$mem0_isolate_parallel" ]]; then
    if [[ "$model" == "qwen25_3b" ]]; then
      mem0_isolate_parallel=10
    else
      mem0_isolate_parallel=4
    fi
  fi
  data_path="$(dataset_cat14_path "$dataset")"
  mkdir -p "$task_root/mem0_fixed"
  run_task_command "$model" "$dataset" "mem0" "$task_root/mem0_fixed/run.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      OPENAI_API_BASE="$OPENAI_BASE_URL" \
      QWEN25_3B_MODEL_PATH="$VLLM_MODEL_PATH" \
      LOCOMO_DATASET="$data_path" \
      CORE_BASELINE_RUN_ROOT="$task_root/mem0_fixed" \
      CORE_BASELINE_START_IDX=0 \
      CORE_BASELINE_END_IDX="${DATASET_SAMPLES[$dataset]}" \
      CORE_BASELINE_EXPECTED_TRAJECTORIES="${DATASET_SAMPLES[$dataset]}" \
      CORE_BASELINE_EXPECTED_QA_COUNT="${DATASET_CAT14_QA[$dataset]}" \
      CORE_BASELINE_BUILD_WORKERS="${MULTI_STYLE_MEM0_BUILD_WORKERS:-1}" \
      CORE_BASELINE_SEARCH_WORKERS="${MULTI_STYLE_MEM0_SEARCH_WORKERS:-2}" \
      CORE_BASELINE_ISOLATE_BUILD=1 \
      CORE_BASELINE_ISOLATE_BUILD_PARALLEL="$mem0_isolate_parallel" \
      CORE_BASELINE_API_PARALLEL="${MULTI_STYLE_MEM0_API_PARALLEL:-12}" \
      CORE_BASELINE_QA_BATCH="${MULTI_STYLE_MEM0_QA_BATCH:-12}" \
      CORE_BASELINE_TOPK_MEMORY="${MULTI_STYLE_MEM0_TOPK:-40}" \
      CORE_BASELINE_QA_MAX_TOKENS="${MULTI_STYLE_QA_MAX_TOKENS:-8192}" \
      CORE_BASELINE_MAX_MODEL_TOKENS="$VLLM_MAX_MODEL_LEN" \
      CORE_BASELINE_CONTEXT_TOKEN_BUFFER="${MULTI_STYLE_CONTEXT_TOKEN_BUFFER:-1024}" \
      CORE_BASELINE_FORCE_METRICS=1 \
      CORE_BASELINE_RUN_JUDGE="$MULTI_STYLE_RUN_JUDGE" \
      CORE_BASELINE_JUDGE_MODEL="${MULTI_STYLE_JUDGE_MODEL:-$OPENAI_MODEL}" \
      CORE_BASELINE_JUDGE_BASE_URL="${MULTI_STYLE_JUDGE_BASE_URL:-$OPENAI_BASE_URL}" \
      CORE_BASELINE_JUDGE_API_KEY="${MULTI_STYLE_JUDGE_API_KEY:-$OPENAI_API_KEY}" \
      CORE_BASELINE_JUDGE_WORKERS="$MULTI_STYLE_JUDGE_WORKERS" \
      CORE_BASELINE_JUDGE_PROTOCOL="$MULTI_STYLE_JUDGE_PROTOCOL" \
      CORE_BASELINE_SKIP_BERTSCORE=0 \
      CORE_BASELINE_BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      CORE_BASELINE_BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      CORE_BASELINE_BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      CORE_BASELINE_BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      MEM0_LLM_MAX_TOKENS="${MULTI_STYLE_MEM0_LLM_MAX_TOKENS:-24576}" \
      LIGHTMEM_MEM0_STRICT_SEARCH=1 \
      LIGHTMEM_MEM0_REQUIRE_NONEMPTY=1 \
      MEM0_DIR="$task_root/mem0_fixed/.mem0" \
      bash scripts/run_locomo_fixed_missing_baselines_qwen25_3b.sh MemZero
}

run_amem() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  local data_path
  local amem_parallel="${MULTI_STYLE_AMEM_SHARD_PARALLEL:-}"
  if [[ -z "$amem_parallel" ]]; then
    if [[ "$model" == "qwen25_3b" ]]; then
      amem_parallel=10
    else
      amem_parallel=4
    fi
  fi
  data_path="$(dataset_cat14_path "$dataset")"
  mkdir -p "$task_root/amem_official"
  run_task_command "$model" "$dataset" "amem" "$task_root/amem_official/run.wrapper.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      QWEN25_3B_MODEL_PATH="$VLLM_MODEL_PATH" \
      VLLM_FORCE_RESTART=0 \
      LOCOMO_DATASET="$data_path" \
      AMEM_OFFICIAL_RUN_ROOT="$task_root/amem_official" \
      AMEM_OFFICIAL_FRESH=1 \
      AMEM_EXPECTED_QA_COUNT="${DATASET_CAT14_QA[$dataset]}" \
      AMEM_EXPECTED_CAT14_COUNT="${DATASET_CAT14_QA[$dataset]}" \
      AMEM_RATIO=1.0 \
      AMEM_RETRIEVE_K="${MULTI_STYLE_AMEM_RETRIEVE_K:-10}" \
      AMEM_SHARD_PARALLEL="$amem_parallel" \
      AMEM_SHARD_COPY_EXISTING_CACHE=0 \
      AMEM_SKIP_BERTSCORE=0 \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      bash scripts/run_amem_official_sharded_qwen25_3b_full.sh
}

run_simplemem() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  mkdir -p "$task_root/simplemem"
  run_task_command "$model" "$dataset" "simplemem" "$task_root/simplemem/run.wrapper.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      VLLM_FORCE_RESTART=0 \
      VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
      SIMPLEMEM_SOURCE_DATASET="${DATASET_SOURCE[$dataset]}" \
      SIMPLEMEM_CATEGORIES=1,2,3,4 \
      SIMPLEMEM_RUN_ROOT="$task_root/simplemem" \
      SIMPLEMEM_EXPECTED_SAMPLES="${DATASET_SAMPLES[$dataset]}" \
      SIMPLEMEM_EXPECTED_QA="${DATASET_CAT14_QA[$dataset]}" \
      SIMPLEMEM_EXPECTED_CAT14_QA="${DATASET_CAT14_QA[$dataset]}" \
      SIMPLEMEM_BUILD_WORKERS="${MULTI_STYLE_SIMPLEMEM_BUILD_WORKERS:-10}" \
      SIMPLEMEM_RETRIEVAL_WORKERS="${MULTI_STYLE_SIMPLEMEM_RETRIEVAL_WORKERS:-24}" \
      SIMPLEMEM_TEST_WORKERS="${MULTI_STYLE_SIMPLEMEM_TEST_WORKERS:-24}" \
      SIMPLEMEM_MAX_OUTPUT_TOKENS="${MULTI_STYLE_SIMPLEMEM_MAX_OUTPUT_TOKENS:-16384}" \
      SIMPLEMEM_MAX_MODEL_TOKENS="$VLLM_MAX_MODEL_LEN" \
      SIMPLEMEM_OPENAI_TIMEOUT="${MULTI_STYLE_OPENAI_TIMEOUT:-600}" \
      SIMPLEMEM_SKIP_BERTSCORE=0 \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART=0 \
      bash scripts/run_simplemem_qwen25_3b_full.sh
}

run_higmem() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  mkdir -p "$task_root/higmem"
  run_task_command "$model" "$dataset" "higmem" "$task_root/higmem/run.wrapper.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_API_BASE="$OPENAI_BASE_URL" \
      VLLM_FORCE_RESTART=0 \
      VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
      HIGMEM_SOURCE_DATASET="${DATASET_SOURCE[$dataset]}" \
      HIGMEM_CATEGORIES=1,2,3,4 \
      HIGMEM_RUN_ROOT="$task_root/higmem" \
      HIGMEM_EXPECTED_SAMPLES="${DATASET_SAMPLES[$dataset]}" \
      HIGMEM_EXPECTED_QA="${DATASET_CAT14_QA[$dataset]}" \
      HIGMEM_EXPECTED_CAT14_QA="${DATASET_CAT14_QA[$dataset]}" \
      HIGMEM_WORKERS="${MULTI_STYLE_HIGMEM_WORKERS:-16}" \
      HIGMEM_MAX_TOKENS="${MULTI_STYLE_QA_MAX_TOKENS:-8192}" \
      HIGMEM_OPENAI_TIMEOUT="${MULTI_STYLE_OPENAI_TIMEOUT:-600}" \
      HIGMEM_SKIP_BERTSCORE=0 \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART=0 \
      bash scripts/run_higmem_qwen25_3b_full.sh
}

run_memgas() {
  local model="$1"
  local dataset="$2"
  local task_root="$RUN_ROOT/$model/$dataset"
  mkdir -p "$task_root/memgas"
  run_task_command "$model" "$dataset" "memgas" "$task_root/memgas/run.wrapper.log" \
    env \
      OPENAI_MODEL="$OPENAI_MODEL" \
      OPENAI_BASE_URL="$OPENAI_BASE_URL" \
      VLLM_FORCE_RESTART=0 \
      VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
      LOCOMO_SOURCE_DATA_PATH="${DATASET_SOURCE[$dataset]}" \
      MEMGAS_CATEGORIES=1,2,3,4 \
      MEMGAS_RUN_ROOT="$task_root/memgas" \
      MEMGAS_EXPECTED_SAMPLES="${DATASET_SAMPLES[$dataset]}" \
      MEMGAS_EXPECTED_QA="${DATASET_CAT14_QA[$dataset]}" \
      MEMGAS_EXPECTED_CAT14_QA="${DATASET_CAT14_QA[$dataset]}" \
      MEMGAS_QA_WORKERS="${MULTI_STYLE_MEMGAS_QA_WORKERS:-12}" \
      MEMGAS_TOPK="${MULTI_STYLE_MEMGAS_TOPK:-20}" \
      MEMGAS_ANSWER_MAX_TOKENS="${MULTI_STYLE_MEMGAS_ANSWER_MAX_TOKENS:-768}" \
      MEMGAS_SUMMARY_MAX_TOKENS="${MULTI_STYLE_MEMGAS_SUMMARY_MAX_TOKENS:-1024}" \
      MEMGAS_MAX_MODEL_TOKENS="$VLLM_MAX_MODEL_LEN" \
      LOCOMO_MAX_MODEL_TOKENS="$VLLM_MAX_MODEL_LEN" \
      LOCOMO_MAX_PROMPT_TOKENS="${MULTI_STYLE_MEMGAS_MAX_PROMPT_TOKENS:-18000}" \
      LOCOMO_TOKEN_GUARD_BUFFER="${MULTI_STYLE_MEMGAS_TOKEN_GUARD_BUFFER:-1536}" \
      MEMGAS_ACCEPT_TRUNCATED_ON_LENGTH=1 \
      MEMGAS_SUMMARY_WORD_LIMIT="${MULTI_STYLE_MEMGAS_SUMMARY_WORD_LIMIT:-120}" \
      MEMGAS_KEYWORD_LIMIT="${MULTI_STYLE_MEMGAS_KEYWORD_LIMIT:-30}" \
      MEMGAS_OPENAI_TIMEOUT="${MULTI_STYLE_OPENAI_TIMEOUT:-600}" \
      MEMGAS_LLM_MAX_RETRIES="${MULTI_STYLE_MEMGAS_LLM_MAX_RETRIES:-3}" \
      LOCOMO_QA_MAX_RETRIES="${MULTI_STYLE_LOCOMO_QA_MAX_RETRIES:-2}" \
      MEMGAS_SKIP_BERTSCORE=0 \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART=0 \
      bash scripts/run_memgas_qwen25_3b_full.sh
}

run_method() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  if task_complete "$model" "$dataset" "$method"; then
    log "SKIP complete $model/$dataset/$method"
    return 0
  fi
  if task_text_complete "$model" "$dataset" "$method"; then
    log "Text metrics complete; adding missing judge $model/$dataset/$method"
  else
    case "$method" in
      full_context) run_full_context "$model" "$dataset" ;;
      amem) run_amem "$model" "$dataset" ;;
      mem0) run_mem0 "$model" "$dataset" ;;
      simplemem) run_simplemem "$model" "$dataset" ;;
      higmem) run_higmem "$model" "$dataset" ;;
      memgas) run_memgas "$model" "$dataset" ;;
      *)
        echo "Unknown method: $method" >&2
        exit 2
        ;;
    esac
  fi
  run_judge_for_method "$model" "$dataset" "$method"
  local judge_args=()
  if [[ "$MULTI_STYLE_RUN_JUDGE" != "1" ]]; then
    judge_args+=(--skip-judge)
  fi
  "$PYTHON" scripts/check_multilingual_locomo_style_48.py \
    --run-root "$RUN_ROOT" \
    --model "$model" \
    --dataset "$dataset" \
    --method "$method" \
    "${judge_args[@]}" \
    --judge-protocol "$MULTI_STYLE_JUDGE_PROTOCOL" \
    --fail-on-incomplete
}

if [[ -f "$FAILURE_LOG" ]]; then
  {
    printf '\n## resume at %s\n\n' "$(date '+%F %T %Z')"
    printf -- '- scope_models: `%s`\n' "${MODELS[*]}"
    printf -- '- scope_datasets: `%s`\n' "${DATASETS[*]}"
    printf -- '- scope_methods: `%s`\n' "${METHODS[*]}"
  } >> "$FAILURE_LOG"
else
  {
    printf '# multilingual LoCoMo-style 48 failure/repair log\n\n'
    printf -- '- run_root: `%s`\n' "$RUN_ROOT"
    printf -- '- started_at: `%s`\n' "$(date '+%F %T %Z')"
    printf -- '- scope_models: `%s`\n' "${MODELS[*]}"
    printf -- '- scope_datasets: `%s`\n' "${DATASETS[*]}"
    printf -- '- scope_methods: `%s`\n' "${METHODS[*]}"
  } > "$FAILURE_LOG"
fi

log "Run root: $RUN_ROOT"
for dataset in "${DATASETS[@]}"; do
  prepare_dataset "$dataset"
done

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    profile="default"
    if [[ "$model" == "qwen3_8b" && "$method" == "memgas" ]]; then
      profile="memgas8b"
    fi
    ensure_vllm_profile "$model" "$profile"
    for dataset in "${DATASETS[@]}"; do
      if ! run_method "$model" "$dataset" "$method"; then
        if [[ "$model" == "qwen3_8b" && "$method" == "memgas" && "$profile" != "memgas8b" ]]; then
          mark_repair "Retrying $model/$dataset/$method with memgas8b vLLM profile."
          ensure_vllm_profile "$model" "memgas8b"
          run_method "$model" "$dataset" "$method"
        else
          exit 1
        fi
      fi
    done
  done
done

final_selection_args=()
for model in "${MODELS[@]}"; do
  final_selection_args+=(--model "$model")
done
for dataset in "${DATASETS[@]}"; do
  final_selection_args+=(--dataset "$dataset")
done
for method in "${METHODS[@]}"; do
  final_selection_args+=(--method "$method")
done
final_judge_args=()
if [[ "$MULTI_STYLE_RUN_JUDGE" != "1" ]]; then
  final_judge_args+=(--skip-judge)
fi
"$PYTHON" scripts/check_multilingual_locomo_style_48.py \
  --run-root "$RUN_ROOT" \
  "${final_selection_args[@]}" \
  "${final_judge_args[@]}" \
  --judge-protocol "$MULTI_STYLE_JUDGE_PROTOCOL" \
  --fail-on-incomplete
log "All selected multilingual LoCoMo-style experiments completed."
log "Summary: $RUN_ROOT/summary/results_f1_desc.md"
