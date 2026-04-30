#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${MEMGAS_CUDA_VISIBLE_DEVICES:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

DATA_PATH="${LOCOMO_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}"
RUN_ROOT="${MEMGAS_RUN_ROOT:-/home/stu0032/paper/runs/memgas/qwen25_3b_full/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "[memgas] started_at=$(date '+%F %T %Z')"
echo "[memgas] run_root=$RUN_ROOT"

if [ "${RESUME:-0}" = "1" ] && /home/stu0032/paper/.venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("$RUN_ROOT") / "normalized_predictions.json"
ok = False
if p.exists():
    data = json.loads(p.read_text())
    ok = len(data.get("records", [])) == 1986
raise SystemExit(0 if ok else 1)
PY
then
  echo "[memgas] complete normalized output exists; skipping because RESUME=1"
  exit 0
fi

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py preflight \
  --data-path "$DATA_PATH" \
  --targets memgas \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py run-memgas \
  --data-path "$DATA_PATH" \
  --output-dir "$RUN_ROOT" \
  --model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --embedder "${MEMGAS_EMBEDDER:-minilm}" \
  --device "${MEMGAS_DEVICE:-cpu}" \
  --method "${MEMGAS_METHOD:-memgas}" \
  --topk "${MEMGAS_TOPK:-20}" \
  --qa-workers "${MEMGAS_QA_WORKERS:-4}" \
  --fresh

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py validate \
  --input "$RUN_ROOT/normalized_predictions.json"

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/compute_locomo_text_metrics.py \
  --input "$RUN_ROOT/normalized_predictions.json" \
  --output "$RUN_ROOT/normalized_metrics.json" \
  --prediction-key prediction \
  --reference-key reference \
  --bertscore-model "${BERTSCORE_MODEL:-roberta-large}" \
  --bertscore-batch-size "${BERTSCORE_BATCH_SIZE:-4}" \
  --bertscore-device "${BERTSCORE_DEVICE:-cpu}"

echo "[memgas] finished_at=$(date '+%F %T %Z')"
