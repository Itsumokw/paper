#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper
export EXPERIMENT_CPU_THREADS="${AMEM_SHARD_CPU_THREADS:-${EXPERIMENT_CPU_THREADS:-4}}"
export EXPERIMENT_CPU_INTEROP_THREADS="${AMEM_SHARD_CPU_INTEROP_THREADS:-${EXPERIMENT_CPU_INTEROP_THREADS:-1}}"
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
export AMEM_LLM_MAX_TOKENS="${AMEM_LLM_MAX_TOKENS:-1024}"
export AMEM_QA_MAX_TOKENS="${AMEM_QA_MAX_TOKENS:-1024}"
export AMEM_RETRIEVAL_MAX_TOKENS="${AMEM_RETRIEVAL_MAX_TOKENS:-512}"
export AMEM_MEMORY_MAX_TOKENS="${AMEM_MEMORY_MAX_TOKENS:-512}"
export AMEM_KEYWORD_MAX_TOKENS="${AMEM_KEYWORD_MAX_TOKENS:-64}"
export AMEM_ANALYZE_MAX_TOKENS="${AMEM_ANALYZE_MAX_TOKENS:-512}"
export AMEM_EVOLUTION_MAX_TOKENS="${AMEM_EVOLUTION_MAX_TOKENS:-128}"
export AMEM_STRENGTHEN_MAX_TOKENS="${AMEM_STRENGTHEN_MAX_TOKENS:-256}"
export AMEM_UPDATE_NEIGHBORS_MAX_TOKENS="${AMEM_UPDATE_NEIGHBORS_MAX_TOKENS:-512}"
export AMEM_OPENAI_TIMEOUT="${AMEM_OPENAI_TIMEOUT:-180}"
export AMEM_LLM_MAX_RETRIES="${AMEM_LLM_MAX_RETRIES:-1}"
export AMEM_MIN_OUTPUT_TOKENS="${AMEM_MIN_OUTPUT_TOKENS:-64}"
export AMEM_EMBEDDING_DEVICE="${AMEM_EMBEDDING_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

PYTHON="/home/stu0032/paper/.venv/bin/python"
PAPER_ROOT="/home/stu0032/paper"
AMEM_ROOT="$PAPER_ROOT/baseline/A-MEM"
DATASET="${LOCOMO_DATASET:-$PAPER_ROOT/datasets/locomo/data/locomo10.json}"
MODEL_PATH="${QWEN25_3B_MODEL_PATH:-$PAPER_ROOT/models/Qwen2.5-3B-Instruct-clean}"
BERTSCORE_MODEL="${BERTSCORE_MODEL:-/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59}"
BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-8}"
BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"
AMEM_RETRIEVE_K="${AMEM_RETRIEVE_K:-10}"
AMEM_SKIP_BERTSCORE="${AMEM_SKIP_BERTSCORE:-0}"
AMEM_RATIO="${AMEM_RATIO:-1.0}"
AMEM_SHARD_PARALLEL="${AMEM_SHARD_PARALLEL:-4}"
AMEM_SHARD_COPY_EXISTING_CACHE="${AMEM_SHARD_COPY_EXISTING_CACHE:-0}"

TS="${AMEM_OFFICIAL_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${AMEM_OFFICIAL_RUN_ROOT:-$PAPER_ROOT/runs/amem_official/qwen25_3b_sharded/$TS}"
WORK_ROOT="$RUN_ROOT/shards"
mkdir -p "$RUN_ROOT" "$WORK_ROOT"
LOG_FILE="$RUN_ROOT/run.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[amem-sharded] started_at=$(date '+%F %T %Z')"
echo "[amem-sharded] run_root=$RUN_ROOT"
echo "[amem-sharded] dataset=$DATASET"
echo "[amem-sharded] model=$OPENAI_MODEL"
echo "[amem-sharded] retrieve_k=$AMEM_RETRIEVE_K parallel=$AMEM_SHARD_PARALLEL"

read -r AMEM_DATASET_SAMPLE_COUNT AMEM_DATASET_QA_COUNT AMEM_DATASET_CAT14_QA_COUNT < <("$PYTHON" - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cat14 = 0
qa = 0
for sample in data:
    rows = sample.get("qa", [])
    qa += len(rows)
    cat14 += sum(1 for row in rows if str(row.get("category")) in {"1", "2", "3", "4"})
print(len(data), qa, cat14)
PY
)
AMEM_EXPECTED_QA_COUNT="${AMEM_EXPECTED_QA_COUNT:-$AMEM_DATASET_QA_COUNT}"
AMEM_EXPECTED_CAT14_COUNT="${AMEM_EXPECTED_CAT14_COUNT:-$AMEM_DATASET_CAT14_QA_COUNT}"
echo "[amem-sharded] samples=$AMEM_DATASET_SAMPLE_COUNT expected_qa=$AMEM_EXPECTED_QA_COUNT expected_cat14=$AMEM_EXPECTED_CAT14_COUNT"

CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}" \
VLLM_MODEL_PATH="$MODEL_PATH" \
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

cat > "$RUN_ROOT/command.env" <<EOF
OPENAI_MODEL=$OPENAI_MODEL
OPENAI_BASE_URL=$OPENAI_BASE_URL
LOCOMO_DATASET=$DATASET
AMEM_SHARDED=1
AMEM_SHARD_PARALLEL=$AMEM_SHARD_PARALLEL
AMEM_DATASET_SAMPLE_COUNT=$AMEM_DATASET_SAMPLE_COUNT
AMEM_EXPECTED_QA_COUNT=$AMEM_EXPECTED_QA_COUNT
AMEM_EXPECTED_CAT14_COUNT=$AMEM_EXPECTED_CAT14_COUNT
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
AMEM_SKIP_BERTSCORE=$AMEM_SKIP_BERTSCORE
BERTSCORE_MODEL=$BERTSCORE_MODEL
BERTSCORE_NUM_LAYERS=$BERTSCORE_NUM_LAYERS
BERTSCORE_BATCH_SIZE=$BERTSCORE_BATCH_SIZE
BERTSCORE_DEVICE=$BERTSCORE_DEVICE
VLLM_MODEL_PATH=$MODEL_PATH
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
VLLM_GENERATION_CONFIG=${VLLM_GENERATION_CONFIG:-}
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
EOF

"$PYTHON" - "$DATASET" "$WORK_ROOT" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
work_root = Path(sys.argv[2])
data = json.loads(dataset.read_text(encoding="utf-8"))
manifest = []
for idx, sample in enumerate(data):
    shard_dir = work_root / f"shard_{idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_dataset = shard_dir / f"locomo_sample_{idx}.json"
    shard_dataset.write_text(json.dumps([sample], ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.append({"sample_index": idx, "dataset": str(shard_dataset), "qa_count": len(sample.get("qa", []))})
(work_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[amem-sharded] wrote {len(manifest)} shard datasets")
PY

for ((i=0; i<AMEM_DATASET_SAMPLE_COUNT; i++)); do
  shard_dir="$WORK_ROOT/shard_$i"
  find "$AMEM_ROOT" -maxdepth 1 -type f -name '*.py' -exec ln -sf {} "$shard_dir"/ \;
  mkdir -p "$shard_dir/logs"
  if [ "$AMEM_SHARD_COPY_EXISTING_CACHE" = "1" ]; then
    src_cache="$AMEM_ROOT/cached_memories_robust_vllm_Qwen/Qwen2.5-3B-Instruct"
    dst_cache="$shard_dir/cached_memories_robust_vllm_Qwen/Qwen2.5-3B-Instruct"
    if [ -f "$src_cache/memory_cache_sample_${i}.pkl" ]; then
      mkdir -p "$dst_cache"
      cp "$src_cache/memory_cache_sample_${i}.pkl" "$dst_cache/memory_cache_sample_0.pkl"
      if [ -f "$src_cache/retriever_cache_sample_${i}.pkl" ]; then
        cp "$src_cache/retriever_cache_sample_${i}.pkl" "$dst_cache/retriever_cache_sample_0.pkl"
      fi
      if [ -f "$src_cache/retriever_cache_embeddings_sample_${i}.npy" ]; then
        cp "$src_cache/retriever_cache_embeddings_sample_${i}.npy" "$dst_cache/retriever_cache_embeddings_sample_0.npy"
      fi
      echo "[amem-sharded] copied existing cache for original sample $i"
    fi
  fi
done

run_shard() {
  local idx="$1"
  local shard_dir="$WORK_ROOT/shard_$idx"
  local shard_dataset="$shard_dir/locomo_sample_${idx}.json"
  echo "[amem-sharded] shard $idx started_at=$(date '+%F %T %Z')"
  (
    cd "$shard_dir"
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_MODEL="$OPENAI_MODEL" \
    AMEM_LLM_MAX_TOKENS="$AMEM_LLM_MAX_TOKENS" \
    AMEM_QA_MAX_TOKENS="$AMEM_QA_MAX_TOKENS" \
    AMEM_RETRIEVAL_MAX_TOKENS="$AMEM_RETRIEVAL_MAX_TOKENS" \
    AMEM_MEMORY_MAX_TOKENS="$AMEM_MEMORY_MAX_TOKENS" \
    AMEM_KEYWORD_MAX_TOKENS="$AMEM_KEYWORD_MAX_TOKENS" \
    AMEM_ANALYZE_MAX_TOKENS="$AMEM_ANALYZE_MAX_TOKENS" \
    AMEM_EVOLUTION_MAX_TOKENS="$AMEM_EVOLUTION_MAX_TOKENS" \
    AMEM_STRENGTHEN_MAX_TOKENS="$AMEM_STRENGTHEN_MAX_TOKENS" \
    AMEM_UPDATE_NEIGHBORS_MAX_TOKENS="$AMEM_UPDATE_NEIGHBORS_MAX_TOKENS" \
    AMEM_OPENAI_TIMEOUT="$AMEM_OPENAI_TIMEOUT" \
    AMEM_LLM_MAX_RETRIES="$AMEM_LLM_MAX_RETRIES" \
    AMEM_MIN_OUTPUT_TOKENS="$AMEM_MIN_OUTPUT_TOKENS" \
    AMEM_EMBEDDING_DEVICE="$AMEM_EMBEDDING_DEVICE" \
    "$PYTHON" -u test_advanced_robust.py \
      --backend vllm \
      --model "$OPENAI_MODEL" \
      --dataset "$shard_dataset" \
      --output "$shard_dir/result.json" \
      --ratio "$AMEM_RATIO" \
      --retrieve_k "$AMEM_RETRIEVE_K" \
      --sglang_host http://127.0.0.1 \
      --sglang_port 8000 \
      > "$shard_dir/run.log" 2>&1
  )
  echo "[amem-sharded] shard $idx finished_at=$(date '+%F %T %Z')"
}

active=0
failed=0
for ((i=0; i<AMEM_DATASET_SAMPLE_COUNT; i++)); do
  run_shard "$i" &
  active=$((active + 1))
  if (( active >= AMEM_SHARD_PARALLEL )); then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done
if (( failed != 0 )); then
  echo "[amem-sharded] ERROR: at least one shard failed" >&2
  exit 1
fi

echo "[amem-sharded] shard logs follow"
for ((i=0; i<AMEM_DATASET_SAMPLE_COUNT; i++)); do
  echo "===== A-MEM shard $i log ====="
  cat "$WORK_ROOT/shard_$i/run.log"
done

"$PYTHON" - "$WORK_ROOT" "$RUN_ROOT" "$OPENAI_MODEL" "$DATASET" "$AMEM_DATASET_SAMPLE_COUNT" "$AMEM_EXPECTED_QA_COUNT" "$AMEM_EXPECTED_CAT14_COUNT" "$AMEM_RETRIEVE_K" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

work_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
model = sys.argv[3]
dataset = sys.argv[4]
sample_count = int(sys.argv[5])
expected_qa = int(sys.argv[6])
expected_cat14 = int(sys.argv[7])
retrieve_k = int(sys.argv[8])
records = []
all_metrics = []
all_categories = []
category_counts = defaultdict(int)

for sample_idx in range(sample_count):
    path = work_root / f"shard_{sample_idx}" / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    shard_records = data.get("individual_results", [])
    if not shard_records:
        raise SystemExit(f"Shard {sample_idx} has no individual_results")
    for row in shard_records:
        row = dict(row)
        row["sample_id"] = sample_idx
        records.append(row)
        metrics = row.get("metrics") or {}
        category = row.get("category")
        all_metrics.append(metrics)
        all_categories.append(category)
        category_counts[category] += 1

if len(records) != expected_qa:
    raise SystemExit(f"Expected {expected_qa} merged A-MEM rows, got {len(records)}")
empty = [i for i, row in enumerate(records) if not str(row.get("prediction") or "").strip()]
if empty:
    raise SystemExit(f"Empty A-MEM predictions after merge: {empty[:10]}")

sys.path.insert(0, "/home/stu0032/paper/baseline/A-MEM")
from utils import aggregate_metrics  # noqa: E402

merged = {
    "model": model,
    "dataset": dataset,
    "memory_layer": "robust",
    "ratio": 1.0,
    "retrieve_k": retrieve_k,
    "sharded": True,
    "total_questions": len(records),
    "category_distribution": {str(k): v for k, v in sorted(category_counts.items(), key=lambda item: str(item[0]))},
    "aggregate_metrics": aggregate_metrics(all_metrics, all_categories),
    "individual_results": records,
}
(run_root / "result.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

normalized = {
    "records": [
        {
            "sample_id": row.get("sample_id"),
            "question": row.get("question"),
            "prediction": row.get("prediction", ""),
            "reference": row.get("reference", ""),
            "category": row.get("category"),
            "metrics": row.get("metrics", {}),
            "latency_seconds": row.get("latency_seconds"),
            "token_usage": row.get("token_usage"),
        }
        for row in records
    ]
}
(run_root / "official_predictions_all.json").write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
cat14 = {"records": [row for row in normalized["records"] if str(row.get("category")) in {"1", "2", "3", "4"}]}
if len(cat14["records"]) != expected_cat14:
    raise SystemExit(f"Expected {expected_cat14} cat1-4 rows, got {len(cat14['records'])}")
(run_root / "official_predictions_cat1_4.json").write_text(json.dumps(cat14, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[amem-sharded] merged rows={len(records)} cat1-4={len(cat14['records'])}")
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

echo "[amem-sharded] finished_at=$(date '+%F %T %Z')"
echo "[amem-sharded] run_root=$RUN_ROOT"
