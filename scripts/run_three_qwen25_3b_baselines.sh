#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
source /home/stu0032/paper/scripts/common_runtime_limits.sh
mkdir -p runs

TS="${THREE_BASELINES_TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="${THREE_BASELINES_LOG:-/home/stu0032/paper/runs/three_baselines_qwen25_3b_${TS}.log}"

export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"

echo "[three-baselines] started_at=$(date '+%F %T %Z')" | tee "$MASTER_LOG"
echo "[three-baselines] master_log=$MASTER_LOG" | tee -a "$MASTER_LOG"

run_step() {
  local name="$1"
  shift
  echo "[three-baselines] >>> START $name at $(date '+%F %T %Z')" | tee -a "$MASTER_LOG"
  set +e
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    echo "[three-baselines] !!! FAILED $name status=$status at $(date '+%F %T %Z')" | tee -a "$MASTER_LOG"
    exit "$status"
  fi
  echo "[three-baselines] <<< DONE $name at $(date '+%F %T %Z')" | tee -a "$MASTER_LOG"
}

run_step "SimpleMem" env \
  SIMPLEMEM_TEST_WORKERS="${SIMPLEMEM_TEST_WORKERS:-16}" \
  SIMPLEMEM_BUILD_WORKERS="${SIMPLEMEM_BUILD_WORKERS:-8}" \
  SIMPLEMEM_RETRIEVAL_WORKERS="${SIMPLEMEM_RETRIEVAL_WORKERS:-16}" \
  SIMPLEMEM_SKIP_BERTSCORE="${SIMPLEMEM_SKIP_BERTSCORE:-0}" \
  BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
  BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
  BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
  bash scripts/run_simplemem_qwen25_3b_full.sh

run_step "MemGAS" env \
  MEMGAS_QA_WORKERS="${MEMGAS_QA_WORKERS:-16}" \
  MEMGAS_TOPK="${MEMGAS_TOPK:-20}" \
  MEMGAS_SKIP_BERTSCORE="${MEMGAS_SKIP_BERTSCORE:-0}" \
  BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
  BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
  BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
  bash scripts/run_memgas_qwen25_3b_full.sh

run_step "A-MEM official" env \
  AMEM_OFFICIAL_FRESH="${AMEM_OFFICIAL_FRESH:-1}" \
  AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}" \
  AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
  AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
  BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
  BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
  BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
  bash scripts/run_amem_official_qwen25_3b_full.sh

echo "[three-baselines] finished_at=$(date '+%F %T %Z')" | tee -a "$MASTER_LOG"
