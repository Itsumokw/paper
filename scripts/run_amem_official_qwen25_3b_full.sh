#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
source /home/stu0032/paper/scripts/common_runtime_limits.sh

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export AMEM_INLINE_BERTSCORE="${AMEM_INLINE_BERTSCORE:-0}"
export AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-8192}"
export AMEM_QA_MAX_TOKENS="${AMEM_QA_MAX_TOKENS:-8192}"
export AMEM_RETRIEVAL_MAX_TOKENS="${AMEM_RETRIEVAL_MAX_TOKENS:-8192}"
export AMEM_MEMORY_MAX_TOKENS="${AMEM_MEMORY_MAX_TOKENS:-8192}"
export AMEM_KEYWORD_MAX_TOKENS="${AMEM_KEYWORD_MAX_TOKENS:-8192}"
export AMEM_ANALYZE_MAX_TOKENS="${AMEM_ANALYZE_MAX_TOKENS:-8192}"
export AMEM_EVOLUTION_MAX_TOKENS="${AMEM_EVOLUTION_MAX_TOKENS:-8192}"
export AMEM_STRENGTHEN_MAX_TOKENS="${AMEM_STRENGTHEN_MAX_TOKENS:-8192}"
export AMEM_UPDATE_NEIGHBORS_MAX_TOKENS="${AMEM_UPDATE_NEIGHBORS_MAX_TOKENS:-8192}"
export AMEM_OPENAI_TIMEOUT="${AMEM_OPENAI_TIMEOUT:-180}"
export AMEM_LLM_MAX_RETRIES="${AMEM_LLM_MAX_RETRIES:-1}"
export AMEM_MIN_OUTPUT_TOKENS="${AMEM_MIN_OUTPUT_TOKENS:-64}"
export AMEM_EMBEDDING_DEVICE="${AMEM_EMBEDDING_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

PYTHON="/home/stu0032/paper/.venv/bin/python"
DATASET="${LOCOMO_DATASET:-/home/stu0032/paper/datasets/locomo/data/locomo10.json}"
MODEL_PATH="${QWEN25_3B_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}"
BERTSCORE_MODEL="${BERTSCORE_MODEL:-/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59}"
BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"
AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}"
AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}"
AMEM_RATIO="${AMEM_RATIO:-1.0}"

TS="${AMEM_OFFICIAL_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${AMEM_OFFICIAL_RUN_ROOT:-/home/stu0032/paper/runs/amem_official/qwen25_3b_full/$TS}"
mkdir -p "$RUN_ROOT"

echo "Run root:       $RUN_ROOT"
echo "Dataset:        $DATASET"
echo "Model:          $OPENAI_MODEL"
echo "Retrieve k:     $AMEM_RETRIEVE_K"
echo "Ratio:          $AMEM_RATIO"
echo "BERTScore:      skip=$AMEM_SKIP_BERTSCORE model=$BERTSCORE_MODEL"
echo "Max new tokens: $AMEM_LLM_MAX_TOKENS"
echo "A-MEM caps:     qa=$AMEM_QA_MAX_TOKENS retrieval=$AMEM_RETRIEVAL_MAX_TOKENS memory=$AMEM_MEMORY_MAX_TOKENS keyword=$AMEM_KEYWORD_MAX_TOKENS analyze=$AMEM_ANALYZE_MAX_TOKENS evolution=$AMEM_EVOLUTION_MAX_TOKENS update_neighbors=$AMEM_UPDATE_NEIGHBORS_MAX_TOKENS"

CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}" \
VLLM_MODEL_PATH="${QWEN25_3B_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}" \
VLLM_SERVED_MODEL="$OPENAI_MODEL" \
VLLM_ALT_SERVED_MODEL="${OPENAI_MODEL##*/}" \
bash scripts/start_vllm_qwen25_3b.sh
export CUDA_VISIBLE_DEVICES=""

"$PYTHON" - <<PY
from openai import OpenAI
client = OpenAI(api_key="${OPENAI_API_KEY}", base_url="${OPENAI_BASE_URL}")
resp = client.chat.completions.create(
    model="${OPENAI_MODEL}",
    messages=[{"role": "user", "content": "Reply OK."}],
    max_tokens=4,
    temperature=0,
    timeout=120,
)
print("vLLM preflight OK:", resp.choices[0].message.content)
PY

if [ "${AMEM_OFFICIAL_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "Preflight-only mode completed."
  exit 0
fi

cache_dir="/home/stu0032/paper/baseline/A-MEM/cached_memories_robust_vllm_${OPENAI_MODEL}"
if [ "${AMEM_OFFICIAL_FRESH:-1}" = "1" ]; then
  echo "Removing generated A-MEM cache: $cache_dir"
  rm -rf "$cache_dir"
fi

cat > "$RUN_ROOT/command.env" <<EOF
OPENAI_MODEL=$OPENAI_MODEL
OPENAI_BASE_URL=$OPENAI_BASE_URL
LOCOMO_DATASET=$DATASET
AMEM_RETRIEVE_K=$AMEM_RETRIEVE_K
AMEM_RATIO=$AMEM_RATIO
AMEM_LLM_MAX_TOKENS=$AMEM_LLM_MAX_TOKENS
AMEM_QA_MAX_TOKENS=$AMEM_QA_MAX_TOKENS
AMEM_RETRIEVAL_MAX_TOKENS=$AMEM_RETRIEVAL_MAX_TOKENS
AMEM_MEMORY_MAX_TOKENS=$AMEM_MEMORY_MAX_TOKENS
AMEM_KEYWORD_MAX_TOKENS=$AMEM_KEYWORD_MAX_TOKENS
AMEM_ANALYZE_MAX_TOKENS=$AMEM_ANALYZE_MAX_TOKENS
AMEM_EVOLUTION_MAX_TOKENS=$AMEM_EVOLUTION_MAX_TOKENS
AMEM_STRENGTHEN_MAX_TOKENS=$AMEM_STRENGTHEN_MAX_TOKENS
AMEM_UPDATE_NEIGHBORS_MAX_TOKENS=$AMEM_UPDATE_NEIGHBORS_MAX_TOKENS
AMEM_MIN_OUTPUT_TOKENS=$AMEM_MIN_OUTPUT_TOKENS
AMEM_OFFICIAL_FRESH=${AMEM_OFFICIAL_FRESH:-1}
AMEM_SKIP_BERTSCORE=$AMEM_SKIP_BERTSCORE
BERTSCORE_MODEL=$BERTSCORE_MODEL
BERTSCORE_NUM_LAYERS=$BERTSCORE_NUM_LAYERS
BERTSCORE_BATCH_SIZE=$BERTSCORE_BATCH_SIZE
BERTSCORE_DEVICE=$BERTSCORE_DEVICE
EOF

pushd /home/stu0032/paper/baseline/A-MEM >/dev/null
"$PYTHON" -u test_advanced_robust.py \
  --backend vllm \
  --model "$OPENAI_MODEL" \
  --dataset "$DATASET" \
  --output "$RUN_ROOT/result.json" \
  --ratio "$AMEM_RATIO" \
  --retrieve_k "$AMEM_RETRIEVE_K" \
  --sglang_host http://127.0.0.1 \
  --sglang_port 8000 \
  2>&1 | tee "$RUN_ROOT/run.log"
popd >/dev/null

"$PYTHON" - "$RUN_ROOT/result.json" "$RUN_ROOT/official_predictions_all.json" "$RUN_ROOT/official_predictions_cat1_4.json" <<'PY'
import json
import os
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
all_path = Path(sys.argv[2])
cat14_path = Path(sys.argv[3])
data = json.loads(result_path.read_text())
records = data.get("individual_results", [])
expected = int(os.environ.get("AMEM_EXPECTED_QA_COUNT", "1986")) if float(data.get("ratio", 1.0) or 1.0) == 1.0 else None
if not records:
    raise SystemExit("A-MEM result has no individual_results")
if expected is not None and len(records) != expected:
    raise SystemExit(f"Expected {expected} QA rows, got {len(records)}")
empty = [i for i, r in enumerate(records) if not str(r.get("prediction") or "").strip()]
if empty:
    raise SystemExit(f"Found empty predictions at indices {empty[:10]}")

normalized = {
    "records": [
        {
            "sample_id": r.get("sample_id"),
            "question": r.get("question"),
            "prediction": r.get("prediction", ""),
            "reference": r.get("reference", ""),
            "category": r.get("category"),
            "metrics": r.get("metrics", {}),
            "latency_seconds": r.get("latency_seconds"),
            "token_usage": r.get("token_usage"),
        }
        for r in records
    ]
}
all_path.write_text(json.dumps(normalized, indent=2))
cat14 = {
    "records": [
        r for r in normalized["records"]
        if str(r.get("category")) in {"1", "2", "3", "4"}
    ]
}
expected_cat14 = int(os.environ.get("AMEM_EXPECTED_CAT14_COUNT", "1540")) if expected is not None else None
if expected_cat14 is not None and len(cat14["records"]) != expected_cat14:
    raise SystemExit(f"Expected {expected_cat14} cat1-4 rows, got {len(cat14['records'])}")
cat14_path.write_text(json.dumps(cat14, indent=2))
print(f"Validated {len(records)} rows; cat1-4 rows={len(cat14['records'])}")
PY

bertscore_args=()
if [ "$AMEM_SKIP_BERTSCORE" = "1" ]; then
  bertscore_args+=(--skip-bertscore)
else
  bertscore_args+=(
    --bertscore-model "$BERTSCORE_MODEL"
    --bertscore-num-layers "$BERTSCORE_NUM_LAYERS"
    --bertscore-batch-size "$BERTSCORE_BATCH_SIZE"
    --bertscore-device "$BERTSCORE_DEVICE"
    --fail-on-bertscore-error
  )
fi

"$PYTHON" scripts/compute_locomo_text_metrics.py \
  --input "$RUN_ROOT/official_predictions_cat1_4.json" \
  --output "$RUN_ROOT/official_metrics_cat1_4.json" \
  --prediction-key prediction \
  --reference-key reference \
  --question-key question \
  --category-key category \
  "${bertscore_args[@]}"

"$PYTHON" scripts/compute_locomo_text_metrics.py \
  --input "$RUN_ROOT/official_predictions_all.json" \
  --output "$RUN_ROOT/official_metrics_all.json" \
  --prediction-key prediction \
  --reference-key reference \
  --question-key question \
  --category-key category \
  "${bertscore_args[@]}"

echo "A-MEM official run completed."
echo "Run root: $RUN_ROOT"
