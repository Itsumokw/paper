#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export no_proxy="${no_proxy:-*}"

PAPER_ROOT="/home/stu0032/paper"
SIMPLEMEM_ROOT="$PAPER_ROOT/baseline/SimpleMem"
DATASET="$PAPER_ROOT/datasets/locomo/data/locomo10.json"
CONF="$SIMPLEMEM_ROOT/config.py"
PYTHON="$PAPER_ROOT/.venv/bin/python"
PATCH="$PAPER_ROOT/scripts/patch_simplemem_config_qwen25_3b.py"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PAPER_ROOT/runs/simplemem/qwen25_3b_full_$TS"

mkdir -p "$OUT"
cp "$CONF" "$OUT/config.before.py"

restore_config() {
  cp "$OUT/config.before.py" "$CONF"
  echo "[restore] config.py restored"
}
trap restore_config EXIT INT TERM

# Patch config
"$PYTHON" "$PATCH"

# Save patched config
cp "$CONF" "$OUT/config_qwen25_3b.py"

# Preflight check
echo "[preflight] checking vLLM at http://127.0.0.1:8000/v1 ..."
"$PYTHON" "$PAPER_ROOT/scripts/vllm_preflight.py"

echo "[simplemem-qwen25] output: $OUT"
echo "[simplemem-qwen25] workers: ${SIMPLEMEM_TEST_WORKERS:-20}"

cd "$SIMPLEMEM_ROOT"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u test_locomo10.py \
  --dataset "$DATASET" \
  --result-file "$OUT/result.json" \
  --parallel-questions \
  --test-workers "${SIMPLEMEM_TEST_WORKERS:-20}" \
  2>&1 | tee "$OUT/run.log"
