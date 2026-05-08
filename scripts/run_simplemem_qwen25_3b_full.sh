#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run SimpleMem on LoCoMo10 with local Qwen/Qwen2.5-3B-Instruct.

Usage:
  SIMPLEMEM_TEST_WORKERS=20 bash scripts/run_simplemem_qwen25_3b_full.sh
EOF
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments: $*" >&2
  echo "Use --help for usage." >&2
  exit 2
fi

# Never route local vLLM calls through a shell/screen proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost,*"
export no_proxy="$NO_PROXY"

PAPER_ROOT="/home/stu0032/paper"
SIMPLEMEM_ROOT="$PAPER_ROOT/baseline/SimpleMem"
DATASET="$PAPER_ROOT/datasets/locomo/data/locomo10.json"
CONF="$SIMPLEMEM_ROOT/config.py"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PAPER_ROOT/runs/simplemem/qwen25_3b_full_$TS"
PYTHON="$PAPER_ROOT/.venv/bin/python"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

mkdir -p "$OUT"
cp "$CONF" "$OUT/config.before.py"

restore_config() {
  cp "$OUT/config.before.py" "$CONF"
}
trap restore_config EXIT INT TERM

check_vllm() {
  OPENAI_BASE_URL="$OPENAI_BASE_URL" OPENAI_MODEL="$OPENAI_MODEL" "$PYTHON" - <<'PY'
import os
import sys
from openai import OpenAI

try:
    client = OpenAI(
        api_key="EMPTY",
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=20,
        max_retries=0,
    )
    models = [m.id for m in client.models.list().data]
    if os.environ["OPENAI_MODEL"] not in models:
        print(f"vLLM preflight failed: {os.environ['OPENAI_MODEL']} not in served models: {models}", file=sys.stderr)
        raise SystemExit(2)
    response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=4,
        timeout=20,
    )
    print("vLLM preflight OK:", response.choices[0].message.content)
except Exception as exc:
    print(f"vLLM preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

vllm_port_listening() {
  ss -ltn | grep -q '127\.0\.0\.1:8000'
}

start_vllm() {
  echo "[simplemem-qwen25] vLLM is not reachable; starting Qwen2.5-3B server..."
  mkdir -p "$PAPER_ROOT/runs"
  if screen -ls | grep -q '[.]paper'; then
    screen -S paper -X screen -t vllm25 bash -lc \
      'cd /home/stu0032/paper && unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy && export NO_PROXY="*" no_proxy="*" HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && exec ./start_vllm_qwen25_3b.sh >> /home/stu0032/paper/runs/vllm_qwen25_3b_current.log 2>&1'
  else
    nohup bash -lc \
      'cd /home/stu0032/paper && unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy && export NO_PROXY="*" no_proxy="*" HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && exec ./start_vllm_qwen25_3b.sh' \
      >> "$PAPER_ROOT/runs/vllm_qwen25_3b_current.log" 2>&1 &
  fi
}

wait_for_vllm() {
  if check_vllm; then
    return 0
  fi
  if vllm_port_listening; then
    echo "[simplemem-qwen25] port 8000 is already listening; waiting for existing vLLM instead of starting another one."
  else
    start_vllm
  fi
  for _ in $(seq 1 180); do
    sleep 2
    if check_vllm; then
      return 0
    fi
    if ! vllm_port_listening; then
      start_vllm
    fi
  done
  echo "[simplemem-qwen25] ERROR: vLLM did not become ready at $OPENAI_BASE_URL" >&2
  echo "[simplemem-qwen25] See: $PAPER_ROOT/runs/vllm_qwen25_3b_current.log" >&2
  return 1
}

"$PYTHON" - <<'PY'
from pathlib import Path
import re

p = Path("/home/stu0032/paper/baseline/SimpleMem/config.py")
s = p.read_text()
s = re.sub(r'^OPENAI_API_KEY = .*$', 'OPENAI_API_KEY = "EMPTY"', s, flags=re.M)
s = re.sub(r'^OPENAI_BASE_URL = .*$', 'OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"', s, flags=re.M)
s = re.sub(r'^LLM_MODEL = .*$', 'LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"', s, flags=re.M)
p.write_text(s)
PY

cp "$CONF" "$OUT/config_qwen25_3b.py"
cat > "$OUT/command.txt" <<EOF
bash scripts/run_simplemem_qwen25_3b_full.sh
EOF

wait_for_vllm

if [[ "${SIMPLEMEM_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[simplemem-qwen25] preflight only; not starting LoCoMo evaluation."
  echo "[simplemem-qwen25] output: $OUT"
  exit 0
fi

echo "[simplemem-qwen25] output: $OUT"
cd "$SIMPLEMEM_ROOT"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u test_locomo10.py \
  --dataset "$DATASET" \
  --result-file "$OUT/result.json" \
  --parallel-questions \
  --test-workers "${SIMPLEMEM_TEST_WORKERS:-20}" \
  2>&1 | tee "$OUT/run.log"
