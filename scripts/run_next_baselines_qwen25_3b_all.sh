#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export LIGHTMEM_DATA_PATH="${LIGHTMEM_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}"
export LIGHTMEM_RUN_ROOT="${LIGHTMEM_RUN_ROOT:-/home/stu0032/paper/runs/lightmem/qwen25_3b_structmem_full}"
export LIGHTMEM_SEARCH_WORKERS="${LIGHTMEM_SEARCH_WORKERS:-1}"
export XMEMORY_BUILD_WORKERS="${XMEMORY_BUILD_WORKERS:-1}"
export XMEMORY_SEARCH_WORKERS="${XMEMORY_SEARCH_WORKERS:-32}"
export XMEMORY_EVAL_WORKERS="${XMEMORY_EVAL_WORKERS:-32}"
export MEMMACHINE_INGEST_CONCURRENCY="${MEMMACHINE_INGEST_CONCURRENCY:-8}"
export MEMMACHINE_SEARCH_CONCURRENCY="${MEMMACHINE_SEARCH_CONCURRENCY:-32}"
export MEMMACHINE_JUDGE_CONCURRENCY="${MEMMACHINE_JUDGE_CONCURRENCY:-32}"

if pgrep -af 'test_fixed_memory.py|run_magma_qwen25_3b_full.sh' >/dev/null && [ "${ALLOW_CONCURRENT_EXPERIMENTS:-0}" != "1" ]; then
  echo "MAGMA is still running. Set ALLOW_CONCURRENT_EXPERIMENTS=1 to run anyway, or wait for it to finish."
  exit 2
fi

wait_for_vllm() {
  /home/stu0032/paper/.venv/bin/python - <<'PY'
from openai import OpenAI
try:
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
    client.models.list()
except Exception:
    raise SystemExit(1)
PY
}

if ! wait_for_vllm; then
  mkdir -p /home/stu0032/paper/runs/next_baselines
  /home/stu0032/paper/start_vllm_qwen25_3b.sh > /home/stu0032/paper/runs/next_baselines/vllm_qwen25_3b.log 2>&1 &
  for _ in $(seq 1 120); do
    if wait_for_vllm; then
      break
    fi
    sleep 5
  done
  wait_for_vllm
fi

/home/stu0032/paper/scripts/download_next_baseline_models.sh
if /home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/check_lightmem_qdrant_ready.py \
  --dataset "$LIGHTMEM_DATA_PATH" \
  --qdrant-dir "$LIGHTMEM_RUN_ROOT/qdrant_pre_update" \
  --require-summaries; then
  echo "LightMem build output is non-empty; running resume search only."
  /home/stu0032/paper/scripts/run_lightmem_structmem_qwen25_3b_resume_search.sh
else
  echo "LightMem build output is missing or empty; running full LightMem build and search."
  /home/stu0032/paper/scripts/run_lightmem_structmem_qwen25_3b_full.sh
fi
/home/stu0032/paper/scripts/run_xmemory_qwen25_3b_full.sh
/home/stu0032/paper/scripts/setup_memmachine_local_env.sh

if /home/stu0032/paper/.venv/bin/python - <<'PY'
import socket
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect(("127.0.0.1", 7687))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
then
  /home/stu0032/paper/scripts/run_memmachine_qwen25_3b_full.sh
else
  echo "MemMachine requires Neo4j on 127.0.0.1:7687; not skipping because this run must complete all three experiments."
  exit 10
fi
