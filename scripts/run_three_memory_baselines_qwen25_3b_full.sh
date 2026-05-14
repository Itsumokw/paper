#!/usr/bin/env bash
set -u

cd /home/stu0032/paper || exit 1
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

TS="${THREE_BASELINES_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${THREE_BASELINES_RUN_ROOT:-/home/stu0032/paper/runs/three_memory_baselines_qwen25_3b_full/$TS}"
mkdir -p "$RUN_ROOT"
MASTER_LOG="$RUN_ROOT/master.log"
STATUS_JSON="$RUN_ROOT/status.json"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

write_status() {
  /home/stu0032/paper/.venv/bin/python - "$STATUS_JSON" "$RUN_ROOT" "$@" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

status_path = Path(sys.argv[1])
run_root = sys.argv[2]
name, status, code, log_path = sys.argv[3:7]
data = {"run_root": run_root, "updated_at": datetime.now().isoformat(timespec="seconds"), "steps": []}
if status_path.exists():
    data = json.loads(status_path.read_text())
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
seen = False
for step in data["steps"]:
    if step["name"] == name:
        step.update({"status": status, "exit_code": None if code == "null" else int(code), "log": log_path})
        seen = True
if not seen:
    data["steps"].append({"name": name, "status": status, "exit_code": None if code == "null" else int(code), "log": log_path})
status_path.write_text(json.dumps(data, indent=2))
PY
}

run_step() {
  local name="$1"
  shift
  local log_path="$RUN_ROOT/$name.log"
  write_status "$name" "running" "null" "$log_path"
  log "START $name"
  "$@" 2>&1 | tee "$log_path"
  local code=${PIPESTATUS[0]}
  if [[ "$code" == "0" ]]; then
    write_status "$name" "completed" "$code" "$log_path"
    log "DONE $name"
  else
    write_status "$name" "failed" "$code" "$log_path"
    log "FAILED $name exit=$code"
  fi
  return "$code"
}

log "three-baselines run_root=$RUN_ROOT"
bash scripts/start_vllm_qwen25_3b.sh 2>&1 | tee -a "$MASTER_LOG"

run_step \
  memgas \
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

run_step \
  mem0 \
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

run_step \
  amem_official \
  env AMEM_OFFICIAL_FRESH="${AMEM_OFFICIAL_FRESH:-1}" \
      AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}" \
      AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}" \
      AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}" \
      BERTSCORE_BATCH_SIZE="$BERTSCORE_BATCH_SIZE" \
      BERTSCORE_DEVICE="$BERTSCORE_DEVICE" \
      BERTSCORE_MODEL="$BERTSCORE_MODEL" \
      BERTSCORE_NUM_LAYERS="$BERTSCORE_NUM_LAYERS" \
      bash scripts/run_amem_official_qwen25_3b_full.sh

log "three-baselines finished. status=$STATUS_JSON"
