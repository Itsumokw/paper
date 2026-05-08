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
export CUDA_VISIBLE_DEVICES="${OMNI_CUDA_VISIBLE_DEVICES:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMNI_EXTRACTION_MAX_TOKENS="${OMNI_EXTRACTION_MAX_TOKENS:-8192}"
export OMNI_ENTITY_MAX_TOKENS="${OMNI_ENTITY_MAX_TOKENS:-4096}"
export OMNI_ANSWER_MAX_TOKENS="${OMNI_ANSWER_MAX_TOKENS:-1024}"

DATA_PATH="${LOCOMO_DATA_PATH:-/home/stu0032/paper/baseline/MAGMA/data/locomo10.json}"
OMNI_ROOT="/home/stu0032/paper/baseline/SimpleMem/OmniSimpleMem"
RUN_ROOT="${OMNI_RUN_ROOT:-/home/stu0032/paper/runs/omnisimplemem/qwen25_3b_full/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "[omni] started_at=$(date '+%F %T %Z')"
echo "[omni] run_root=$RUN_ROOT"

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
  echo "[omni] complete normalized output exists; skipping because RESUME=1"
  exit 0
fi

/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py preflight \
  --data-path "$DATA_PATH" \
  --targets omni \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

cd "$OMNI_ROOT"
PYTHONPATH="$OMNI_ROOT" /home/stu0032/paper/.venv/bin/python benchmarks/locomo/run_locomo.py \
  --data-path "$DATA_PATH" \
  --model "$OPENAI_MODEL" \
  --ingest-concurrency "${OMNI_INGEST_WORKERS:-1}" \
  --concurrency "${OMNI_WORKERS:-4}" \
  --fresh \
  -o "$RUN_ROOT"

cd /home/stu0032/paper
/home/stu0032/paper/.venv/bin/python /home/stu0032/paper/scripts/locomo_2026_sota.py normalize-omni \
  --output-dir "$RUN_ROOT"

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

echo "[omni] finished_at=$(date '+%F %T %Z')"
