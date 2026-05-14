#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MODEL_PATH="${VLLM_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}"
SERVED_MODEL="${VLLM_SERVED_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ALT_SERVED_MODEL="${VLLM_ALT_SERVED_MODEL:-Qwen2.5-3B-Instruct}"
LOG_DIR="${VLLM_LOG_DIR:-/home/stu0032/paper/runs/vllm}"
mkdir -p "$LOG_DIR"

served_model_ready() {
  /home/stu0032/paper/.venv/bin/python - <<PY >/dev/null 2>&1
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://${HOST}:${PORT}/v1", timeout=5, max_retries=0)
models = [m.id for m in client.models.list().data]
if "${SERVED_MODEL}" not in models:
    raise SystemExit(2)
PY
}

endpoint_ready() {
  /home/stu0032/paper/.venv/bin/python - <<PY >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://${HOST}:${PORT}/v1/models", timeout=2).read(1)
PY
}

if pgrep -af "vllm.entrypoints.openai.api_server.*--port ${PORT}" >/dev/null || endpoint_ready; then
  if served_model_ready && [[ "${VLLM_FORCE_RESTART:-0}" != "1" ]]; then
    echo "vLLM already serves ${SERVED_MODEL} at http://${HOST}:${PORT}/v1."
    exit 0
  fi
  if [[ "${VLLM_FORCE_RESTART:-0}" != "1" ]]; then
    echo "ERROR: port ${PORT} is occupied but does not serve ${SERVED_MODEL}. Set VLLM_FORCE_RESTART=1 to replace it." >&2
    /home/stu0032/paper/.venv/bin/python - <<PY >&2 || true
from openai import OpenAI
try:
    client = OpenAI(api_key="EMPTY", base_url="http://${HOST}:${PORT}/v1", timeout=5, max_retries=0)
    print("served models:", [m.id for m in client.models.list().data])
except Exception as exc:
    print("served model check failed:", exc)
PY
    exit 1
  fi
  pids="$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping existing vLLM on port ${PORT}: $pids"
    kill $pids 2>/dev/null || true
    sleep 8
    pids="$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
fi

if endpoint_ready
then
  echo "ERROR: OpenAI-compatible endpoint is still reachable at http://${HOST}:${PORT}/v1 but does not serve ${SERVED_MODEL}." >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
safe_served="${SERVED_MODEL//\//_}"
safe_served="${safe_served//:/_}"
LOG_FILE="$LOG_DIR/${safe_served}_${TS}.log"

extra_args=()
if [[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  extra_args+=(--default-chat-template-kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS")
fi
if [[ -n "${VLLM_GENERATION_CONFIG:-}" ]]; then
  extra_args+=(--generation-config "$VLLM_GENERATION_CONFIG")
fi
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args+=(${VLLM_EXTRA_ARGS})
fi

setsid /home/stu0032/paper/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" "$ALT_SERVED_MODEL" \
  --dtype half \
  --trust-remote-code \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-2}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}" \
  --enable-prefix-caching \
  --disable-log-stats \
  "${extra_args[@]}" \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
disown "$PID" 2>/dev/null || true
echo "$PID" > "$LOG_DIR/${safe_served}.pid"
echo "Started vLLM PID=$PID"
echo "Log: $LOG_FILE"

ready_timeout_seconds="${VLLM_READY_TIMEOUT_SECONDS:-600}"
ready_checks=$(( (ready_timeout_seconds + 1) / 2 ))
for _ in $(seq 1 "$ready_checks"); do
  if served_model_ready
  then
    echo "vLLM endpoint is ready: http://${HOST}:${PORT}/v1 model=${SERVED_MODEL}"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null
  then
    echo "vLLM process exited before endpoint became ready. Last log lines:" >&2
    tail -80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "vLLM did not become ready within ${ready_timeout_seconds} seconds. Last log lines:" >&2
tail -80 "$LOG_FILE" >&2 || true
exit 1
