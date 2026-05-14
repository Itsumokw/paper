#!/usr/bin/env bash
set -uo pipefail

cd /home/stu0032/paper
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TS="${FOUR_BASELINES_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="/home/stu0032/paper/runs/four_memory_baselines_qwen25_3b_full/$TS"
mkdir -p "$RUN_ROOT"
MASTER_LOG="$RUN_ROOT/master.log"
STATUS_JSON="$RUN_ROOT/status.json"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_MODEL="${BERTSCORE_MODEL:-roberta-large}"
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$MASTER_LOG"
}

write_status() {
  /home/stu0032/paper/.venv/bin/python - "$STATUS_JSON" "$RUN_ROOT" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
records = []
for item in sys.argv[3:]:
    name, status, code, log_path = item.split("::", 3)
    records.append({
        "name": name,
        "status": status,
        "exit_code": None if code == "" else int(code),
        "log": log_path,
    })
path.write_text(json.dumps({
    "run_root": str(run_root),
    "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    "steps": records,
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

declare -a STEP_STATUS=()

set_step_status() {
  local name="$1"
  local status="$2"
  local code="${3:-}"
  local log_path="${4:-}"
  local found=0
  for i in "${!STEP_STATUS[@]}"; do
    if [[ "${STEP_STATUS[$i]}" == "$name::"* ]]; then
      STEP_STATUS[$i]="$name::$status::$code::$log_path"
      found=1
    fi
  done
  if [[ "$found" == "0" ]]; then
    STEP_STATUS+=("$name::$status::$code::$log_path")
  fi
  write_status "${STEP_STATUS[@]}"
}

run_step_once() {
  local name="$1"
  local log_path="$2"
  shift 2
  set_step_status "$name" "running" "" "$log_path"
  log "START $name"
  log "LOG $log_path"
  "$@" > >(tee -a "$log_path") 2> >(tee -a "$log_path" >&2) &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1200
    log "HEARTBEAT $name pid=$pid still running"
    tail -40 "$log_path" | sed "s/^/[${name} tail] /" | tee -a "$MASTER_LOG" || true
  done
  wait "$pid"
  local code=$?
  if [[ "$code" == "0" ]]; then
    set_step_status "$name" "completed" "$code" "$log_path"
    log "DONE $name"
  else
    set_step_status "$name" "failed" "$code" "$log_path"
    log "FAILED $name exit=$code"
  fi
  return "$code"
}

run_step_with_retry() {
  local name="$1"
  local fallback_env="$2"
  shift 2
  local log_path="$RUN_ROOT/${name}.log"
  run_step_once "$name" "$log_path" "$@"
  local code=$?
  if [[ "$code" == "0" ]]; then
    return 0
  fi
  log "RETRY $name with fallback env: $fallback_env"
  local retry_log="$RUN_ROOT/${name}.retry.log"
  # shellcheck disable=SC2086
  env $fallback_env "$@" > >(tee -a "$retry_log") 2> >(tee -a "$retry_log" >&2) &
  local pid=$!
  set_step_status "$name" "retrying" "" "$retry_log"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1200
    log "HEARTBEAT $name retry pid=$pid still running"
    tail -40 "$retry_log" | sed "s/^/[${name} retry tail] /" | tee -a "$MASTER_LOG" || true
  done
  wait "$pid"
  code=$?
  if [[ "$code" == "0" ]]; then
    set_step_status "$name" "completed_after_retry" "$code" "$retry_log"
    log "DONE $name after retry"
  else
    set_step_status "$name" "failed_after_retry" "$code" "$retry_log"
    log "FAILED $name retry exit=$code"
  fi
  return "$code"
}

log "four-baselines run_root=$RUN_ROOT"
bash scripts/start_vllm_qwen25_3b.sh 2>&1 | tee -a "$MASTER_LOG"

run_step_with_retry \
  simplemem \
  "SIMPLEMEM_TEST_WORKERS=16 SIMPLEMEM_RETRIEVAL_WORKERS=16 SIMPLEMEM_BUILD_WORKERS=8" \
  env SIMPLEMEM_TEST_WORKERS="${SIMPLEMEM_TEST_WORKERS:-16}" \
      SIMPLEMEM_BUILD_WORKERS="${SIMPLEMEM_BUILD_WORKERS:-8}" \
      SIMPLEMEM_RETRIEVAL_WORKERS="${SIMPLEMEM_RETRIEVAL_WORKERS:-16}" \
      SIMPLEMEM_SKIP_BERTSCORE="${SIMPLEMEM_SKIP_BERTSCORE:-0}" \
      SIMPLEMEM_FAIL_ON_FAILED_ANSWERS="${SIMPLEMEM_FAIL_ON_FAILED_ANSWERS:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_simplemem_qwen25_3b_full.sh

run_step_with_retry \
  memgas \
  "MEMGAS_QA_WORKERS=8 MEMGAS_TOPK=20 MEMGAS_SUMMARY_MAX_TOKENS=8192 MEMGAS_ANSWER_MAX_TOKENS=8192" \
  env MEMGAS_QA_WORKERS="${MEMGAS_QA_WORKERS:-16}" \
      MEMGAS_TOPK="${MEMGAS_TOPK:-20}" \
      MEMGAS_SUMMARY_MAX_TOKENS="${MEMGAS_SUMMARY_MAX_TOKENS:-8192}" \
      MEMGAS_ANSWER_MAX_TOKENS="${MEMGAS_ANSWER_MAX_TOKENS:-8192}" \
      MEMGAS_SKIP_BERTSCORE="${MEMGAS_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_memgas_qwen25_3b_full.sh

run_step_with_retry \
  mem0 \
  "MEM0_BUILD_WORKERS=1 MEM0_SEARCH_WORKERS=1 MEM0_QA_BATCH=4 MEM0_API_PARALLEL=4 MEM0_QA_MAX_TOKENS=8192 MEM0_LLM_MAX_TOKENS=24576" \
  env MEM0_WAIT_FOR_ACTIVE_BASELINES=0 \
      MEM0_BUILD_WORKERS="${MEM0_BUILD_WORKERS:-2}" \
      MEM0_SEARCH_WORKERS="${MEM0_SEARCH_WORKERS:-2}" \
      MEM0_QA_BATCH="${MEM0_QA_BATCH:-8}" \
      MEM0_API_PARALLEL="${MEM0_API_PARALLEL:-8}" \
      MEM0_QA_MAX_TOKENS="${MEM0_QA_MAX_TOKENS:-8192}" \
      MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-24576}" \
      MEM0_SKIP_BERTSCORE="${MEM0_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_mem0_qwen25_3b_full.sh

run_step_with_retry \
  amem_official \
  "AMEM_RETRIEVE_K=10 AMEM_LLM_MAX_TOKENS=8192" \
  env AMEM_OFFICIAL_FRESH="${AMEM_OFFICIAL_FRESH:-1}" \
      AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}" \
      AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
      AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_amem_official_qwen25_3b_full.sh

log "four-baselines completed. status=$STATUS_JSON"
