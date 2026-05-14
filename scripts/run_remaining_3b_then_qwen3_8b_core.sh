#!/usr/bin/env bash
set -uo pipefail

cd /home/stu0032/paper || exit 1
source /home/stu0032/paper/scripts/common_runtime_limits.sh

TS="${REMAINING_PLUS_8B_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${REMAINING_PLUS_8B_ROOT:-/home/stu0032/paper/runs/remaining_3b_then_qwen3_8b_core/$TS}"
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

log "remaining 3B then Qwen3-8B run_root=$RUN_ROOT"

run_step qwen25_3b_mem0_amem \
  env MEM0_AMEM_RERUN_ROOT="$RUN_ROOT/qwen25_3b_mem0_amem" \
      MEM0_BUILD_WORKERS="${MEM0_BUILD_WORKERS:-1}" \
      MEM0_SEARCH_WORKERS="${MEM0_SEARCH_WORKERS:-2}" \
      MEM0_QA_BATCH="${MEM0_QA_BATCH:-8}" \
      MEM0_API_PARALLEL="${MEM0_API_PARALLEL:-8}" \
      MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-24576}" \
      CORE_BASELINE_ISOLATE_BUILD="${CORE_BASELINE_ISOLATE_BUILD:-1}" \
      MEM0_SKIP_BERTSCORE="${MEM0_SKIP_BERTSCORE:-0}" \
      AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
      AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}" \
      BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}" \
      bash scripts/run_mem0_then_amem_qwen25_3b_full.sh

run_step qwen3_8b_core_baselines \
  env QWEN3_8B_RUN_ROOT="$RUN_ROOT/qwen3_8b_core_baselines" \
      BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}" \
      BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}" \
      bash scripts/run_core_baselines_qwen3_8b_full.sh

log "DONE remaining 3B then Qwen3-8B. Logs in $RUN_ROOT"
