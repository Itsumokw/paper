#!/usr/bin/env bash
set -uo pipefail

cd /home/stu0032/paper || exit 1
source /home/stu0032/paper/scripts/common_runtime_limits.sh

TS="${MEM0_AMEM_RERUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MEM0_AMEM_RERUN_ROOT:-/home/stu0032/paper/runs/mem0_amem_rerun_qwen25_3b_full/$TS}"
mkdir -p "$RUN_ROOT"
MASTER_LOG="$RUN_ROOT/master.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  local step_log="$RUN_ROOT/$name.log"
  log "START $name"
  "$@" 2>&1 | tee "$step_log"
  local code=${PIPESTATUS[0]}
  log "END $name exit=$code log=$step_log"
  return 0
}

log "Mem0 + A-MEM rerun root=$RUN_ROOT"

run_step mem0 \
  env MEM0_WAIT_FOR_ACTIVE_BASELINES=0 \
      MEM0_BUILD_WORKERS="${MEM0_BUILD_WORKERS:-1}" \
      MEM0_SEARCH_WORKERS="${MEM0_SEARCH_WORKERS:-2}" \
      MEM0_QA_BATCH="${MEM0_QA_BATCH:-8}" \
      MEM0_API_PARALLEL="${MEM0_API_PARALLEL:-8}" \
      MEM0_QA_MAX_TOKENS="${MEM0_QA_MAX_TOKENS:-8192}" \
      MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-24576}" \
      CORE_BASELINE_ISOLATE_BUILD="${CORE_BASELINE_ISOLATE_BUILD:-1}" \
      MEM0_SKIP_BERTSCORE="${MEM0_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}" \
      BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}" \
      bash scripts/run_mem0_qwen25_3b_full.sh

run_step amem_official \
  env AMEM_OFFICIAL_FRESH="${AMEM_OFFICIAL_FRESH:-1}" \
      AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}" \
      AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
      AMEM_OPENAI_TIMEOUT="${AMEM_OPENAI_TIMEOUT:-180}" \
      AMEM_LLM_MAX_RETRIES="${AMEM_LLM_MAX_RETRIES:-1}" \
      AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}" \
      BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}" \
      bash scripts/run_amem_official_qwen25_3b_full.sh

log "DONE all requested reruns"
