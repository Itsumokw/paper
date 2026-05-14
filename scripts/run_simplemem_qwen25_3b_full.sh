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
source /home/stu0032/paper/scripts/common_runtime_limits.sh
source /home/stu0032/paper/scripts/high_utilization_runtime.sh
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost,*"
export no_proxy="$NO_PROXY"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SIMPLEMEM_MAX_QUESTION_WORKER_CAP="${SIMPLEMEM_MAX_QUESTION_WORKER_CAP:-32}"
export SIMPLEMEM_INLINE_BERTSCORE="${SIMPLEMEM_INLINE_BERTSCORE:-0}"

LOCAL_BERTSCORE_MODEL="/home/stu0032/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
if [[ -z "${BERTSCORE_MODEL:-}" && -d "$LOCAL_BERTSCORE_MODEL" ]]; then
  export BERTSCORE_MODEL="$LOCAL_BERTSCORE_MODEL"
fi
export BERTSCORE_NUM_LAYERS="${BERTSCORE_NUM_LAYERS:-17}"
export BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-2}"
export BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-cpu}"

PAPER_ROOT="/home/stu0032/paper"
SIMPLEMEM_ROOT="$PAPER_ROOT/baseline/SimpleMem"
SOURCE_DATASET="${SIMPLEMEM_SOURCE_DATASET:-${SIMPLEMEM_DATASET:-$PAPER_ROOT/datasets/locomo/data/locomo10.json}}"
SIMPLEMEM_CATEGORIES="${SIMPLEMEM_CATEGORIES:-1,2,3,4}"
DATASET="$SOURCE_DATASET"
CONF="$SIMPLEMEM_ROOT/config.py"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${SIMPLEMEM_RUN_ROOT:-$PAPER_ROOT/runs/simplemem/qwen25_3b_full_$TS}"
PYTHON="$PAPER_ROOT/.venv/bin/python"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LOCAL_QWEN3_EMBEDDING="/home/stu0032/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
if [[ -z "${SIMPLEMEM_EMBEDDING_MODEL:-}" && -d "$LOCAL_QWEN3_EMBEDDING" ]]; then
  export SIMPLEMEM_EMBEDDING_MODEL="$LOCAL_QWEN3_EMBEDDING"
fi

mkdir -p "$OUT"

if [[ "$SIMPLEMEM_CATEGORIES" != "all" ]]; then
  SAFE_CATEGORIES="${SIMPLEMEM_CATEGORIES//,/}"
  DATASET="${SIMPLEMEM_FILTERED_DATASET:-$OUT/locomo10_cat${SAFE_CATEGORIES}.json}"
  "$PYTHON" "$PAPER_ROOT/scripts/filter_locomo_categories.py" \
    --input "$SOURCE_DATASET" \
    --output "$DATASET" \
    --categories "$SIMPLEMEM_CATEGORIES" \
    > "$OUT/dataset_filter_summary.json"
fi

read -r SIMPLEMEM_DATASET_SAMPLE_COUNT SIMPLEMEM_DATASET_QA_COUNT < <("$PYTHON" - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(data), sum(len(sample.get("qa", [])) for sample in data))
PY
)
export SIMPLEMEM_EXPECTED_SAMPLES="${SIMPLEMEM_EXPECTED_SAMPLES:-$SIMPLEMEM_DATASET_SAMPLE_COUNT}"
export SIMPLEMEM_EXPECTED_QA="${SIMPLEMEM_EXPECTED_QA:-$SIMPLEMEM_DATASET_QA_COUNT}"
if [[ "$SIMPLEMEM_CATEGORIES" == "1,2,3,4" ]]; then
  export SIMPLEMEM_EXPECTED_CAT14_QA="${SIMPLEMEM_EXPECTED_CAT14_QA:-$SIMPLEMEM_DATASET_QA_COUNT}"
fi

cp "$CONF" "$OUT/config.before.py"

if locomo_high_util_enabled; then
  locomo_raise_env_int SIMPLEMEM_BUILD_WORKERS 10
  locomo_raise_env_int SIMPLEMEM_RETRIEVAL_WORKERS 24
  locomo_raise_env_int SIMPLEMEM_TEST_WORKERS 24
  locomo_enable_high_util_vllm_defaults
fi

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
  echo "[simplemem-qwen25] vLLM is not reachable; starting server for $OPENAI_MODEL..."
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
  locomo_set_vllm_model_path_for_openai_model
  export VLLM_SERVED_MODEL="$OPENAI_MODEL"
  export VLLM_ALT_SERVED_MODEL="${OPENAI_MODEL##*/}"
  if locomo_high_util_enabled; then
    locomo_enable_high_util_vllm_defaults
  else
    export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
    export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
  fi
  locomo_start_vllm_with_safe_fallback
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
s = re.sub(r'^OPENAI_API_KEY = .*$', 'OPENAI_API_KEY = __import__("os").environ.get("OPENAI_API_KEY", "EMPTY")', s, flags=re.M)
s = re.sub(r'^OPENAI_BASE_URL = .*$', 'OPENAI_BASE_URL = __import__("os").environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")', s, flags=re.M)
s = re.sub(r'^LLM_MODEL = .*$', 'LLM_MODEL = __import__("os").environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-3B-Instruct")', s, flags=re.M)
s = re.sub(r'^MAX_OUTPUT_TOKENS = .*$', 'MAX_OUTPUT_TOKENS = int(__import__("os").environ.get("SIMPLEMEM_MAX_OUTPUT_TOKENS", "8192"))', s, flags=re.M)
s = re.sub(r'^WINDOW_SIZE = .*$', 'WINDOW_SIZE = int(__import__("os").environ.get("SIMPLEMEM_WINDOW_SIZE", "40"))', s, flags=re.M)
s = re.sub(r'^OVERLAP_SIZE = .*$', 'OVERLAP_SIZE = int(__import__("os").environ.get("SIMPLEMEM_OVERLAP_SIZE", "2"))', s, flags=re.M)
s = re.sub(r'^EMBEDDING_MODEL = .*$', 'EMBEDDING_MODEL = __import__("os").environ.get("SIMPLEMEM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")', s, flags=re.M)
s = re.sub(r'^ENABLE_THINKING = .*$', 'ENABLE_THINKING = False', s, flags=re.M)
s = re.sub(r'^USE_STREAMING = .*$', 'USE_STREAMING = True', s, flags=re.M)
s = re.sub(r'^USE_JSON_FORMAT = .*$', 'USE_JSON_FORMAT = False', s, flags=re.M)
s = re.sub(r'^ENABLE_PARALLEL_PROCESSING = .*$', 'ENABLE_PARALLEL_PROCESSING = True', s, flags=re.M)
s = re.sub(r'^MAX_PARALLEL_WORKERS = .*$', 'MAX_PARALLEL_WORKERS = int(__import__("os").environ.get("SIMPLEMEM_BUILD_WORKERS", "16"))', s, flags=re.M)
s = re.sub(r'^ENABLE_PARALLEL_RETRIEVAL = .*$', 'ENABLE_PARALLEL_RETRIEVAL = True', s, flags=re.M)
s = re.sub(r'^MAX_RETRIEVAL_WORKERS = .*$', 'MAX_RETRIEVAL_WORKERS = int(__import__("os").environ.get("SIMPLEMEM_RETRIEVAL_WORKERS", "32"))', s, flags=re.M)
s = re.sub(r'^MAX_TEST_QUESTION_WORKERS = .*$', 'MAX_TEST_QUESTION_WORKERS = int(__import__("os").environ.get("SIMPLEMEM_TEST_WORKERS", "32"))', s, flags=re.M)
p.write_text(s)
PY

cp "$CONF" "$OUT/config_qwen25_3b.py"
cat > "$OUT/command.txt" <<EOF
bash scripts/run_simplemem_qwen25_3b_full.sh
EOF
cat > "$OUT/command.env" <<EOF
OPENAI_MODEL=$OPENAI_MODEL
OPENAI_BASE_URL=$OPENAI_BASE_URL
SIMPLEMEM_SOURCE_DATASET=$SOURCE_DATASET
SIMPLEMEM_CATEGORIES=$SIMPLEMEM_CATEGORIES
SIMPLEMEM_DATASET=$DATASET
SIMPLEMEM_RUN_ROOT=$OUT
SIMPLEMEM_EXPECTED_SAMPLES=$SIMPLEMEM_EXPECTED_SAMPLES
SIMPLEMEM_EXPECTED_QA=$SIMPLEMEM_EXPECTED_QA
SIMPLEMEM_EXPECTED_CAT14_QA=${SIMPLEMEM_EXPECTED_CAT14_QA:-}
SIMPLEMEM_TEST_WORKERS=${SIMPLEMEM_TEST_WORKERS:-32}
SIMPLEMEM_BUILD_WORKERS=${SIMPLEMEM_BUILD_WORKERS:-16}
SIMPLEMEM_RETRIEVAL_WORKERS=${SIMPLEMEM_RETRIEVAL_WORKERS:-32}
SIMPLEMEM_MAX_OUTPUT_TOKENS=${SIMPLEMEM_MAX_OUTPUT_TOKENS:-8192}
SIMPLEMEM_MAX_MODEL_TOKENS=${SIMPLEMEM_MAX_MODEL_TOKENS:-32768}
SIMPLEMEM_TOKEN_GUARD_BUFFER=${SIMPLEMEM_TOKEN_GUARD_BUFFER:-1024}
SIMPLEMEM_OPENAI_TIMEOUT=${SIMPLEMEM_OPENAI_TIMEOUT:-300}
LOCOMO_HIGH_UTILIZATION_AFTER_AMEM=${LOCOMO_HIGH_UTILIZATION_AFTER_AMEM:-1}
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$(locomo_default_vllm_model_path)}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_GENERATION_CONFIG=${VLLM_GENERATION_CONFIG:-}
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
BERTSCORE_MODEL=$BERTSCORE_MODEL
BERTSCORE_NUM_LAYERS=$BERTSCORE_NUM_LAYERS
BERTSCORE_BATCH_SIZE=$BERTSCORE_BATCH_SIZE
BERTSCORE_DEVICE=$BERTSCORE_DEVICE
EOF

if [[ "${SIMPLEMEM_POSTPROCESS_ONLY:-0}" != "1" ]]; then
  wait_for_vllm

  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SIMPLEMEM_ROOT" "$PYTHON" - <<'PY'
from utils.embedding import EmbeddingModel

model = EmbeddingModel()
vector = model.encode_query("embedding preflight")
if len(vector) == 0:
    raise SystemExit("SimpleMem embedding preflight returned an empty vector")
print("[simplemem-qwen25] embedding preflight OK:", getattr(model, "model_name", "unknown"), getattr(vector, "shape", None))
PY

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
    --test-workers "${SIMPLEMEM_TEST_WORKERS:-32}" \
    2>&1 | tee "$OUT/run.log"

  if grep -q "All 15 attempts failed" "$OUT/run.log"; then
    echo "[simplemem-qwen25] ERROR: memory build had windows that failed all extraction attempts." >&2
    echo "[simplemem-qwen25] Refusing to score a degraded SimpleMem run. See $OUT/run.log" >&2
    exit 1
  fi
else
  echo "[simplemem-qwen25] postprocess only: using existing $OUT/result.json"
  if [[ ! -s "$OUT/result.json" ]]; then
    echo "[simplemem-qwen25] ERROR: SIMPLEMEM_POSTPROCESS_ONLY=1 but $OUT/result.json is missing or empty" >&2
    exit 1
  fi
fi

"$PYTHON" - "$OUT/result.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
details = data.get("detailed_results") or []
summary = data.get("summary") or {}
overall = (data.get("aggregated_metrics") or {}).get("overall") or {}
expected_samples = int(os.environ.get("SIMPLEMEM_EXPECTED_SAMPLES", "10"))
expected_questions = int(os.environ.get("SIMPLEMEM_EXPECTED_QA", "1986"))
if summary.get("num_samples") != expected_samples or summary.get("num_questions") != expected_questions:
    raise SystemExit(f"Unexpected SimpleMem summary: {summary}")
if len(details) != expected_questions:
    raise SystemExit(f"Expected {expected_questions} detailed results, found {len(details)}")
bad = [i for i, row in enumerate(details) if not str(row.get("answer") or "").strip()]
if bad:
    raise SystemExit(f"SimpleMem has empty answers at rows: {bad[:10]}")
failure_markers = {"Error during processing", "Failed to generate answer"}
failed = [
    {"row": i, "category": row.get("category"), "question": row.get("question")}
    for i, row in enumerate(details)
    if str(row.get("answer") or "").strip() in failure_markers
]
if failed:
    diagnostics = path.with_name("failed_question_outputs.json")
    diagnostics.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.environ.get("SIMPLEMEM_FAIL_ON_FAILED_ANSWERS", "0") == "1":
        raise SystemExit(f"SimpleMem has failed question outputs: {failed[:10]}")
    print(f"[simplemem-qwen25] warning: {len(failed)} failed question outputs written to {diagnostics}")
inline_f1_count = (overall.get("f1") or {}).get("count")
if inline_f1_count != expected_questions:
    print(
        f"[simplemem-qwen25] warning: inline overall F1 count {inline_f1_count} "
        f"!= expected {expected_questions}; external metrics will score all detailed results."
    )
print("[simplemem-qwen25] validation OK:", path)
PY

"$PYTHON" - "$OUT/result.json" "$OUT/normalized_predictions.json" <<'PY'
import json
import sys
from pathlib import Path

def estimate_tokens(text):
    if text is None:
        return 0
    value = str(text)
    return max(1, len(value) // 4) if value else 0

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text())
records = []
for idx, row in enumerate(data.get("detailed_results") or []):
    prediction = row.get("answer", "")
    records.append(
        {
            "source": "simplemem",
            "qa_idx": idx,
            "category": row.get("category"),
            "question": row.get("question", ""),
            "prediction": prediction,
            "reference": row.get("reference", ""),
            "latency_seconds": row.get("total_time"),
            "retrieval_latency_seconds": row.get("retrieval_time"),
            "answer_latency_seconds": row.get("answer_time"),
            "token_usage": {
                "prompt_tokens": estimate_tokens(row.get("question", "")),
                "completion_tokens": estimate_tokens(prediction),
                "total_tokens": estimate_tokens(row.get("question", "")) + estimate_tokens(prediction),
                "note": "SimpleMem per-question prompt context is not saved by the upstream evaluator; prompt_tokens here count the visible question only. Full LLM-call aggregate is in result.summary.llm_usage.",
            },
        }
    )
dst.write_text(json.dumps({
    "records": records,
    "summary": {
        "method": "SimpleMem",
        "source_result": str(src),
        "num_records": len(records),
        "latency": data.get("summary", {}),
        "llm_usage": (data.get("summary") or {}).get("llm_usage"),
    },
}, ensure_ascii=False, indent=2))
print("[simplemem-qwen25] normalized predictions:", dst)
PY

"$PYTHON" - "$OUT/normalized_predictions.json" "$OUT/normalized_predictions_cat1_4.json" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text())
records = data.get("records", [])
cat14 = [row for row in records if str(row.get("category")) in {"1", "2", "3", "4"}]
expected = int(os.environ.get("SIMPLEMEM_EXPECTED_CAT14_QA", "1540"))
if len(cat14) != expected:
    raise SystemExit(f"Expected {expected} SimpleMem cat1-4 rows, got {len(cat14)}")
dst.write_text(json.dumps({"records": cat14, "summary": data.get("summary", {})}, ensure_ascii=False, indent=2))
print("[simplemem-qwen25] normalized cat1-4 predictions:", dst)
PY

METRICS_ARGS=(
  --input "$OUT/normalized_predictions.json"
  --output "$OUT/simplemem_metrics_all.json"
  --prediction-key prediction
  --reference-key reference
)
if [ "${SIMPLEMEM_SKIP_BERTSCORE:-1}" = "1" ]; then
  METRICS_ARGS+=(--skip-bertscore)
else
  METRICS_ARGS+=(
    --bertscore-model "${BERTSCORE_MODEL:-roberta-large}"
    --bertscore-batch-size "${BERTSCORE_BATCH_SIZE:-4}"
    --bertscore-num-layers "${BERTSCORE_NUM_LAYERS:-17}"
    --bertscore-device "${BERTSCORE_DEVICE:-cpu}"
    --fail-on-bertscore-error
  )
fi
"$PYTHON" "$PAPER_ROOT/scripts/compute_locomo_text_metrics.py" "${METRICS_ARGS[@]}"

CAT14_METRICS_ARGS=("${METRICS_ARGS[@]}")
for i in "${!CAT14_METRICS_ARGS[@]}"; do
  if [[ "${CAT14_METRICS_ARGS[$i]}" == "$OUT/normalized_predictions.json" ]]; then
    CAT14_METRICS_ARGS[$i]="$OUT/normalized_predictions_cat1_4.json"
  elif [[ "${CAT14_METRICS_ARGS[$i]}" == "$OUT/simplemem_metrics_all.json" ]]; then
    CAT14_METRICS_ARGS[$i]="$OUT/simplemem_metrics_cat1_4.json"
  fi
done
"$PYTHON" "$PAPER_ROOT/scripts/compute_locomo_text_metrics.py" "${CAT14_METRICS_ARGS[@]}"
