#!/usr/bin/env bash
set -uo pipefail

cd /home/stu0032/paper || exit 1
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen3-8B}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_PATH="${QWEN3_8B_MODEL_PATH:-/home/stu0032/paper/models/Qwen3-8B}"
TS="${QWEN3_8B_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${QWEN3_8B_RUN_ROOT:-/home/stu0032/paper/runs/core_baselines_qwen3_8b_full/$TS}"
mkdir -p "$RUN_ROOT"
MASTER_LOG="$RUN_ROOT/master.log"
STATUS_JSON="$RUN_ROOT/status.json"
QWEN3_CONTEXT_TOKENS="${QWEN3_CONTEXT_TOKENS:-32000}"
QWEN3_QA_MAX_TOKENS="${QWEN3_QA_MAX_TOKENS:-8192}"
QWEN3_CONTEXT_BUFFER_TOKENS="${QWEN3_CONTEXT_BUFFER_TOKENS:-1024}"
QWEN3_PROMPT_TOKENS="$((QWEN3_CONTEXT_TOKENS - QWEN3_QA_MAX_TOKENS - QWEN3_CONTEXT_BUFFER_TOKENS))"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$QWEN3_CONTEXT_TOKENS}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$QWEN3_CONTEXT_TOKENS}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
export VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-vllm}"
if (( QWEN3_PROMPT_TOKENS < 4096 )); then
  echo "ERROR: Qwen3 prompt budget too small: $QWEN3_PROMPT_TOKENS" >&2
  exit 2
fi

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_MODEL="${BERTSCORE_MODEL:-roberta-large}"
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

write_status() {
  /home/stu0032/paper/.venv/bin/python - "$STATUS_JSON" "$RUN_ROOT" "$@" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
run_root = sys.argv[2]
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
    "run_root": run_root,
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "steps": records,
}, indent=2), encoding="utf-8")
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

stop_vllm_on_port() {
  local port="${VLLM_PORT:-8000}"
  local pids
  pids="$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" || true)"
  if [[ -n "$pids" ]]; then
    log "Stopping existing vLLM on port $port: $pids"
    kill $pids 2>/dev/null || true
    sleep 8
    pids="$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

run_step() {
  local name="$1"
  shift
  local step_log="$RUN_ROOT/$name.log"
  set_step_status "$name" "running" "" "$step_log"
  log "START $name"
  "$@" 2>&1 | tee "$step_log"
  local code=${PIPESTATUS[0]}
  if [[ "$code" == "0" ]]; then
    set_step_status "$name" "completed" "$code" "$step_log"
    log "END $name exit=0"
  else
    set_step_status "$name" "failed" "$code" "$step_log"
    log "END $name exit=$code; continuing"
  fi
  return 0
}

preflight_served_model() {
  /home/stu0032/paper/.venv/bin/python - <<PY
from openai import OpenAI
client = OpenAI(api_key="${OPENAI_API_KEY}", base_url="${OPENAI_BASE_URL}", timeout=60, max_retries=0)
models = [m.id for m in client.models.list().data]
print("served models:", models)
if "${OPENAI_MODEL}" not in models:
    raise SystemExit("Expected ${OPENAI_MODEL}, got " + repr(models))
resp = client.chat.completions.create(
    model="${OPENAI_MODEL}",
    messages=[{"role": "user", "content": "Reply OK."}],
    max_tokens=4,
    temperature=0,
    timeout=60,
)
print("Qwen3-8B preflight OK:", resp.choices[0].message.content)
PY
}

log "qwen3-8b core baseline run_root=$RUN_ROOT"
log "model_path=$MODEL_PATH model=$OPENAI_MODEL context=$QWEN3_CONTEXT_TOKENS qa_max=$QWEN3_QA_MAX_TOKENS prompt_budget=$QWEN3_PROMPT_TOKENS"

stop_vllm_on_port
if ! VLLM_FORCE_RESTART=1 \
  VLLM_MODEL_PATH="$MODEL_PATH" \
  VLLM_SERVED_MODEL="$OPENAI_MODEL" \
  VLLM_ALT_SERVED_MODEL="Qwen3-8B" \
  bash scripts/start_vllm_qwen25_3b.sh 2>&1 | tee -a "$MASTER_LOG"; then
  log "ERROR failed to start Qwen3-8B vLLM; aborting 8B run before any baseline"
  exit 1
fi
if ! preflight_served_model 2>&1 | tee -a "$MASTER_LOG"; then
  log "ERROR Qwen3-8B preflight failed; aborting 8B run before any baseline"
  exit 1
fi

if [[ "${QWEN3_SKIP_FULL_CONTEXT:-0}" == "1" ]]; then
  set_step_status "full_context" "skipped" "0" ""
  log "SKIP full_context because QWEN3_SKIP_FULL_CONTEXT=1"
else
  run_step full_context \
    env OPENAI_MODEL="$OPENAI_MODEL" \
        QWEN25_3B_MODEL_PATH="$MODEL_PATH" \
        CORE_BASELINE_RUN_ROOT="$RUN_ROOT/lightmem_fixed" \
        CORE_BASELINE_START_IDX=0 \
        CORE_BASELINE_END_IDX=10 \
        CORE_BASELINE_EXPECTED_TRAJECTORIES=10 \
        CORE_BASELINE_EXPECTED_QA_COUNT=1540 \
        CORE_BASELINE_BUILD_WORKERS="${FULL_CONTEXT_BUILD_WORKERS:-2}" \
        CORE_BASELINE_SEARCH_WORKERS="${FULL_CONTEXT_SEARCH_WORKERS:-2}" \
        CORE_BASELINE_API_PARALLEL="${FULL_CONTEXT_API_PARALLEL:-1}" \
        CORE_BASELINE_QA_BATCH="${FULL_CONTEXT_QA_BATCH:-1}" \
        CORE_BASELINE_FULL_CONTEXT_WINDOW="${FULL_CONTEXT_WINDOW:-$QWEN3_PROMPT_TOKENS}" \
        CORE_BASELINE_MAX_MODEL_TOKENS="${CORE_BASELINE_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
        CORE_BASELINE_QA_MAX_TOKENS="${CORE_BASELINE_QA_MAX_TOKENS:-$QWEN3_QA_MAX_TOKENS}" \
        CORE_BASELINE_CONTEXT_TOKEN_BUFFER="${CORE_BASELINE_CONTEXT_TOKEN_BUFFER:-$QWEN3_CONTEXT_BUFFER_TOKENS}" \
        CORE_BASELINE_SKIP_BERTSCORE="${CORE_BASELINE_SKIP_BERTSCORE:-0}" \
        BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
        BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
        BERTSCORE_MODEL="$BERTSCORE_MODEL" \
        BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
        bash scripts/run_locomo_fixed_missing_baselines_qwen25_3b.sh FullContext
fi

run_step amem_official \
  env OPENAI_MODEL="$OPENAI_MODEL" \
      QWEN25_3B_MODEL_PATH="$MODEL_PATH" \
      AMEM_OFFICIAL_RUN_ROOT="$RUN_ROOT/amem_official" \
      AMEM_OFFICIAL_FRESH="${AMEM_OFFICIAL_FRESH:-1}" \
      AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}" \
      AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
      AMEM_QA_MAX_TOKENS="${AMEM_QA_MAX_TOKENS:-$QWEN3_QA_MAX_TOKENS}" \
      AMEM_RETRIEVAL_MAX_TOKENS="${AMEM_RETRIEVAL_MAX_TOKENS:-8192}" \
      AMEM_MEMORY_MAX_TOKENS="${AMEM_MEMORY_MAX_TOKENS:-8192}" \
      AMEM_KEYWORD_MAX_TOKENS="${AMEM_KEYWORD_MAX_TOKENS:-8192}" \
      AMEM_ANALYZE_MAX_TOKENS="${AMEM_ANALYZE_MAX_TOKENS:-8192}" \
      AMEM_EVOLUTION_MAX_TOKENS="${AMEM_EVOLUTION_MAX_TOKENS:-8192}" \
      AMEM_STRENGTHEN_MAX_TOKENS="${AMEM_STRENGTHEN_MAX_TOKENS:-8192}" \
      AMEM_UPDATE_NEIGHBORS_MAX_TOKENS="${AMEM_UPDATE_NEIGHBORS_MAX_TOKENS:-8192}" \
      AMEM_MAX_MODEL_TOKENS="${AMEM_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      AMEM_MAX_PROMPT_TOKENS="${AMEM_MAX_PROMPT_TOKENS:-$QWEN3_PROMPT_TOKENS}" \
      AMEM_OPENAI_TIMEOUT="${AMEM_OPENAI_TIMEOUT:-300}" \
      AMEM_LLM_MAX_RETRIES="${AMEM_LLM_MAX_RETRIES:-1}" \
      AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_amem_official_qwen25_3b_full.sh

run_step mem0 \
  env OPENAI_MODEL="$OPENAI_MODEL" \
      QWEN25_3B_MODEL_PATH="$MODEL_PATH" \
      CORE_BASELINE_RUN_ROOT="$RUN_ROOT/mem0_fixed" \
      MEM0_WAIT_FOR_ACTIVE_BASELINES=0 \
      MEM0_BUILD_WORKERS="${MEM0_BUILD_WORKERS:-1}" \
      MEM0_SEARCH_WORKERS="${MEM0_SEARCH_WORKERS:-2}" \
      MEM0_QA_BATCH="${MEM0_QA_BATCH:-1}" \
      MEM0_API_PARALLEL="${MEM0_API_PARALLEL:-1}" \
      MEM0_QA_MAX_TOKENS="${MEM0_QA_MAX_TOKENS:-$QWEN3_QA_MAX_TOKENS}" \
      MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-24576}" \
      CORE_BASELINE_ISOLATE_BUILD="${CORE_BASELINE_ISOLATE_BUILD:-1}" \
      CORE_BASELINE_MAX_MODEL_TOKENS="${CORE_BASELINE_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      CORE_BASELINE_CONTEXT_TOKEN_BUFFER="${CORE_BASELINE_CONTEXT_TOKEN_BUFFER:-$QWEN3_CONTEXT_BUFFER_TOKENS}" \
      LIGHTMEM_MAX_MODEL_TOKENS="${LIGHTMEM_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      LIGHTMEM_MAX_PROMPT_TOKENS="${LIGHTMEM_MAX_PROMPT_TOKENS:-$QWEN3_PROMPT_TOKENS}" \
      LIGHTMEM_OPENAI_TIMEOUT="${LIGHTMEM_OPENAI_TIMEOUT:-300}" \
      LIGHTMEM_MEM0_MAX_RETRIES="${LIGHTMEM_MEM0_MAX_RETRIES:-1}" \
      MEM0_SKIP_BERTSCORE="${MEM0_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_mem0_qwen25_3b_full.sh

run_step simplemem \
  env OPENAI_MODEL="$OPENAI_MODEL" \
      QWEN25_3B_MODEL_PATH="$MODEL_PATH" \
      SIMPLEMEM_RUN_ROOT="$RUN_ROOT/simplemem" \
      SIMPLEMEM_TEST_WORKERS="${SIMPLEMEM_TEST_WORKERS:-2}" \
      SIMPLEMEM_BUILD_WORKERS="${SIMPLEMEM_BUILD_WORKERS:-2}" \
      SIMPLEMEM_RETRIEVAL_WORKERS="${SIMPLEMEM_RETRIEVAL_WORKERS:-2}" \
      SIMPLEMEM_MAX_OUTPUT_TOKENS="${SIMPLEMEM_MAX_OUTPUT_TOKENS:-24576}" \
      SIMPLEMEM_MAX_MODEL_TOKENS="${SIMPLEMEM_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      SIMPLEMEM_TOKEN_GUARD_BUFFER="${SIMPLEMEM_TOKEN_GUARD_BUFFER:-$QWEN3_CONTEXT_BUFFER_TOKENS}" \
      SIMPLEMEM_OPENAI_TIMEOUT="${SIMPLEMEM_OPENAI_TIMEOUT:-300}" \
      SIMPLEMEM_SKIP_BERTSCORE="${SIMPLEMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_simplemem_qwen25_3b_full.sh

run_step higmem \
  env OPENAI_MODEL="$OPENAI_MODEL" \
      HIGMEM_MODEL="$OPENAI_MODEL" \
      HIGMEM_WORKERS="${HIGMEM_WORKERS:-1}" \
      HIGMEM_MAX_TOKENS="${HIGMEM_MAX_TOKENS:-$QWEN3_QA_MAX_TOKENS}" \
      HIGMEM_OPENAI_TIMEOUT="${HIGMEM_OPENAI_TIMEOUT:-300}" \
      bash scripts/run_higmem_qwen25_3b_full.sh

run_step memgas \
  env OPENAI_MODEL="$OPENAI_MODEL" \
      QWEN25_3B_MODEL_PATH="$MODEL_PATH" \
      MEMGAS_RUN_ROOT="$RUN_ROOT/memgas" \
      MEMGAS_QA_WORKERS="${MEMGAS_QA_WORKERS:-2}" \
      MEMGAS_TOPK="${MEMGAS_TOPK:-20}" \
      MEMGAS_SUMMARY_MAX_TOKENS="${MEMGAS_SUMMARY_MAX_TOKENS:-24576}" \
      MEMGAS_ANSWER_MAX_TOKENS="${MEMGAS_ANSWER_MAX_TOKENS:-$QWEN3_QA_MAX_TOKENS}" \
      MEMGAS_MAX_MODEL_TOKENS="${MEMGAS_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      MEMGAS_MAX_PROMPT_TOKENS="${MEMGAS_MAX_PROMPT_TOKENS:-$QWEN3_PROMPT_TOKENS}" \
      LOCOMO_MAX_MODEL_TOKENS="${LOCOMO_MAX_MODEL_TOKENS:-$QWEN3_CONTEXT_TOKENS}" \
      LOCOMO_MAX_PROMPT_TOKENS="${LOCOMO_MAX_PROMPT_TOKENS:-$QWEN3_PROMPT_TOKENS}" \
      MEMGAS_SKIP_BERTSCORE="${MEMGAS_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_memgas_qwen25_3b_full.sh

log "DONE qwen3-8b core baseline run. status=$STATUS_JSON"
