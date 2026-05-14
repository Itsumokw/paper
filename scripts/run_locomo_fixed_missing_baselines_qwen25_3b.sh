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
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

PYTHON="/home/stu0032/paper/.venv/bin/python"
TOOLKIT="/home/stu0032/paper/baseline/LightMem/src/lightmem/memory_toolkits"
LIGHTMEM_SRC="/home/stu0032/paper/baseline/LightMem/src"
MEM0_VENDOR="$TOOLKIT/memories/layers/baselines"
DATASET="${LOCOMO_DATASET:-/home/stu0032/paper/datasets/locomo/data/locomo10.json}"
EMBEDDING_MODEL="${CORE_BASELINE_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
MODEL_PATH="${QWEN25_3B_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}"

export PYTHONPATH="$MEM0_VENDOR:$TOOLKIT:$LIGHTMEM_SRC${PYTHONPATH:+:$PYTHONPATH}"

TS="${CORE_BASELINE_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${CORE_BASELINE_RUN_ROOT:-/home/stu0032/paper/runs/locomo_fixed_baselines/qwen25_3b_${TS}}"
CONFIG_DIR="$RUN_ROOT/configs"
mkdir -p "$CONFIG_DIR"

export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"
export MEM0_DIR="${MEM0_DIR:-$RUN_ROOT/.mem0}"
mkdir -p "$MEM0_DIR"

BUILD_WORKERS="${CORE_BASELINE_BUILD_WORKERS:-2}"
SEARCH_WORKERS="${CORE_BASELINE_SEARCH_WORKERS:-2}"
API_PARALLEL="${CORE_BASELINE_API_PARALLEL:-8}"
QA_BATCH="${CORE_BASELINE_QA_BATCH:-8}"
ISOLATE_BUILD="${CORE_BASELINE_ISOLATE_BUILD:-0}"
ISOLATE_BUILD_PARALLEL="${CORE_BASELINE_ISOLATE_BUILD_PARALLEL:-1}"
TOPK_MEMORY="${CORE_BASELINE_TOPK_MEMORY:-40}"
MAX_MODEL_TOKENS="${CORE_BASELINE_MAX_MODEL_TOKENS:-32768}"
QA_MAX_TOKENS="${CORE_BASELINE_QA_MAX_TOKENS:-8192}"
CONTEXT_TOKEN_BUFFER="${CORE_BASELINE_CONTEXT_TOKEN_BUFFER:-1024}"
MAX_INPUT_TOKENS="${CORE_BASELINE_MAX_INPUT_TOKENS:-$((MAX_MODEL_TOKENS - QA_MAX_TOKENS - CONTEXT_TOKEN_BUFFER))}"
FULL_CONTEXT_WINDOW="${CORE_BASELINE_FULL_CONTEXT_WINDOW:-$MAX_INPUT_TOKENS}"
START_IDX="${CORE_BASELINE_START_IDX:-0}"
END_IDX="${CORE_BASELINE_END_IDX:-10}"
EXPECTED_TRAJECTORIES="${CORE_BASELINE_EXPECTED_TRAJECTORIES:-$((END_IDX - START_IDX))}"
EXPECTED_QA_COUNT="${CORE_BASELINE_EXPECTED_QA_COUNT:-1540}"
FORCE_BUILD="${CORE_BASELINE_FORCE_BUILD:-${CORE_BASELINE_RERUN:-0}}"
FORCE_SEARCH="${CORE_BASELINE_FORCE_SEARCH:-0}"
FORCE_ANSWER="${CORE_BASELINE_FORCE_ANSWER:-0}"
FORCE_METRICS="${CORE_BASELINE_FORCE_METRICS:-1}"

export LIGHTMEM_OPENAI_TIMEOUT="${LIGHTMEM_OPENAI_TIMEOUT:-180}"
export LIGHTMEM_OPENAI_CLIENT_MAX_RETRIES="${LIGHTMEM_OPENAI_CLIENT_MAX_RETRIES:-0}"
export LIGHTMEM_AMEM_MAX_TOKENS="${LIGHTMEM_AMEM_MAX_TOKENS:-8192}"
export LIGHTMEM_AMEM_MAX_RETRIES="${LIGHTMEM_AMEM_MAX_RETRIES:-1}"
export LIGHTMEM_MEM0_MAX_RETRIES="${LIGHTMEM_MEM0_MAX_RETRIES:-1}"
export LIGHTMEM_MIN_GENERATION_TOKENS="${LIGHTMEM_MIN_GENERATION_TOKENS:-512}"
export LIGHTMEM_MEM0_STRICT_ADD="${LIGHTMEM_MEM0_STRICT_ADD:-0}"
export LIGHTMEM_MEM0_STRICT_SEARCH="${LIGHTMEM_MEM0_STRICT_SEARCH:-0}"
export LIGHTMEM_MEM0_REQUIRE_NONEMPTY="${LIGHTMEM_MEM0_REQUIRE_NONEMPTY:-0}"

BASELINES=("$@")
if [ "${#BASELINES[@]}" -eq 0 ]; then
  BASELINES=(FullContext A-MEM MemZero)
fi

MEM0_LLM_MAX_TOKENS="${MEM0_LLM_MAX_TOKENS:-24576}"
BERTSCORE_MODEL="${CORE_BASELINE_BERTSCORE_MODEL:-${BERTSCORE_MODEL:-roberta-large}}"
BERTSCORE_BATCH_SIZE="${CORE_BASELINE_BERTSCORE_BATCH_SIZE:-2}"
BERTSCORE_DEVICE="${CORE_BASELINE_BERTSCORE_DEVICE:-cpu}"
BERTSCORE_NUM_LAYERS="${CORE_BASELINE_BERTSCORE_NUM_LAYERS:-${BERTSCORE_NUM_LAYERS:-}}"
SKIP_BERTSCORE="${CORE_BASELINE_SKIP_BERTSCORE:-1}"
FAIL_ON_BERTSCORE_ERROR="${CORE_BASELINE_FAIL_ON_BERTSCORE_ERROR:-1}"
RUN_JUDGE="${CORE_BASELINE_RUN_JUDGE:-0}"
JUDGE_MODEL="${CORE_BASELINE_JUDGE_MODEL:-${LLM_JUDGE_MODEL:-$OPENAI_MODEL}}"
JUDGE_BASE_URL="${CORE_BASELINE_JUDGE_BASE_URL:-${LLM_JUDGE_BASE_URL:-$OPENAI_BASE_URL}}"
JUDGE_API_KEY="${CORE_BASELINE_JUDGE_API_KEY:-${LLM_JUDGE_API_KEY:-$OPENAI_API_KEY}}"
JUDGE_WORKERS="${CORE_BASELINE_JUDGE_WORKERS:-${LLM_JUDGE_WORKERS:-$API_PARALLEL}}"
JUDGE_MAX_RETRIES="${CORE_BASELINE_JUDGE_MAX_RETRIES:-${LLM_JUDGE_MAX_RETRIES:-3}}"
JUDGE_PROTOCOL="${CORE_BASELINE_JUDGE_PROTOCOL:-${LOCOMO_JUDGE_PROTOCOL:-${LLM_JUDGE_PROTOCOL:-locomo_binary}}}"

safe_model="${OPENAI_MODEL//\//_}"
safe_model="${safe_model//:/_}"

echo "Run root: $RUN_ROOT"
echo "Model:    $OPENAI_MODEL"
echo "Dataset:  $DATASET"
echo "Workers:  build=$BUILD_WORKERS search=$SEARCH_WORKERS api_parallel=$API_PARALLEL qa_batch=$QA_BATCH"
echo "Token guard: max_model=$MAX_MODEL_TOKENS max_input=$MAX_INPUT_TOKENS qa_max_output=$QA_MAX_TOKENS buffer=$CONTEXT_TOKEN_BUFFER"
echo "FullContext window: $FULL_CONTEXT_WINDOW memory tokens"
echo "Sample range: start=$START_IDX end=$END_IDX"
echo "Resume:   expected_trajectories=$EXPECTED_TRAJECTORIES expected_qa=$EXPECTED_QA_COUNT force_build=$FORCE_BUILD force_search=$FORCE_SEARCH force_answer=$FORCE_ANSWER force_metrics=$FORCE_METRICS isolate_build=$ISOLATE_BUILD"
echo "Isolated build parallelism: $ISOLATE_BUILD_PARALLEL"
echo "Baselines: ${BASELINES[*]}"
echo "BERTScore: skip=$SKIP_BERTSCORE model=$BERTSCORE_MODEL device=$BERTSCORE_DEVICE batch=$BERTSCORE_BATCH_SIZE"
echo "LLM judge: enabled=$RUN_JUDGE model=$JUDGE_MODEL workers=$JUDGE_WORKERS protocol=$JUDGE_PROTOCOL"

"$PYTHON" - <<PY
from openai import OpenAI
base_url = "${OPENAI_BASE_URL}".rstrip("/")
client = OpenAI(api_key="${OPENAI_API_KEY}", base_url=base_url)
resp = client.chat.completions.create(
    model="${OPENAI_MODEL}",
    messages=[{"role": "user", "content": "Reply OK."}],
    max_tokens=4,
    temperature=0,
    timeout=120,
)
print("vLLM preflight OK:", resp.choices[0].message.content)
PY

"$PYTHON" - <<PY
import json
from pathlib import Path

config_dir = Path("${CONFIG_DIR}")
config_dir.mkdir(parents=True, exist_ok=True)
api_parallel = int("${API_PARALLEL}")
api_config = {
    "api_keys": ["${OPENAI_API_KEY}"] * api_parallel,
    "base_urls": ["${OPENAI_BASE_URL}"] * api_parallel,
}
(config_dir / "api_config.json").write_text(json.dumps(api_config, indent=2))

model = "${OPENAI_MODEL}"
embedding_model = "${EMBEDDING_MODEL}"
base_url = "${OPENAI_BASE_URL}"

configs = {
    "FullContext": {
        "user_id": "dummy",
        "llm_backend": "openai",
        "llm_model": model,
        "tokenizer_name_or_path": "${MODEL_PATH}",
        "context_window": int("${FULL_CONTEXT_WINDOW}"),
    },
    "A-MEM": {
        "user_id": "dummy",
        "embedder_provider": "sentence-transformers",
        "retriever_name_or_path": embedding_model,
        "base_url": base_url,
        "llm_backend": "openai",
        "llm_model": model,
        "evo_threshold": 100,
        "api_key": "${OPENAI_API_KEY}",
    },
    "MemZero": {
        "user_id": "dummy",
        "llm_backend": "openai",
        "llm_model": model,
        "llm_max_tokens": int("${MEM0_LLM_MAX_TOKENS}"),
        "embedder_provider": "huggingface",
        "retriever_name_or_path": embedding_model,
        "embedding_model_dims": 384,
        "use_gpu": "cpu",
        "vector_store_provider": "qdrant",
        "qdrant_on_disk": True,
    },
}
for name, cfg in configs.items():
    (config_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2))
PY

json_list_len() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(-1)
    raise SystemExit
try:
    data = json.loads(path.read_text())
except Exception:
    print(-1)
    raise SystemExit
print(len(data) if isinstance(data, list) else -1)
PY
}

eval_json_is_complete() {
  local path="$1"
  local expected="$2"
  "$PYTHON" - "$path" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
if not isinstance(data, list) or len(data) != expected:
    raise SystemExit(1)
for item in data:
    if item.get("qa_error"):
        raise SystemExit(1)
    if not str(item.get("prediction") or "").strip():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

validate_memory_snapshots() {
  local out_dir="$1"
  local expected="$2"
  local require_nonempty="${3:-0}"
  local inventory="$out_dir/memory_inventory.json"
  "$PYTHON" - "$out_dir" "$expected" "$require_nonempty" "$inventory" <<'PY'
import json
import pickle
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
require_nonempty = sys.argv[3] == "1"
inventory_path = Path(sys.argv[4])

rows = []
bad = []
for pkl_path in sorted(out_dir.rglob("*.pkl")):
    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            memory_count = len(data)
        elif isinstance(data, dict):
            if isinstance(data.get("memories"), dict):
                memory_count = len(data["memories"])
            elif isinstance(data.get("memories"), list):
                memory_count = len(data["memories"])
            elif isinstance(data.get("ordered_ids"), list):
                memory_count = len(data["ordered_ids"])
            elif isinstance(data.get("memory_units"), list):
                memory_count = len(data["memory_units"])
            else:
                memory_count = len(data)
        else:
            memory_count = -1
        error = None
    except Exception as exc:  # noqa: BLE001
        memory_count = -1
        error = str(exc)
    row = {
        "path": str(pkl_path.relative_to(out_dir)),
        "memory_count": memory_count,
        "error": error,
    }
    rows.append(row)
    if error or memory_count < 0 or (require_nonempty and memory_count == 0):
        bad.append(row)

summary = {
    "expected_snapshots": expected,
    "snapshot_count": len(rows),
    "total_memories": sum(row["memory_count"] for row in rows if row["memory_count"] > 0),
    "require_nonempty": require_nonempty,
    "bad_snapshots": bad,
    "snapshots": rows,
}
inventory_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

if len(rows) < expected:
    print(f"Expected {expected} memory snapshots, found {len(rows)}", file=sys.stderr)
    raise SystemExit(1)
if bad:
    print(f"Invalid memory snapshots written to {inventory_path}", file=sys.stderr)
    raise SystemExit(1)
print(f"Memory snapshot inventory OK: {len(rows)} snapshots, {summary['total_memories']} memories")
PY
}

run_one() {
  local method="$1"
  local slug topk
  case "$method" in
    FullContext)
      slug="full_context"
      topk="-1"
      context_truncation_strategy="keep_tail"
      ;;
    A-MEM)
      slug="amem"
      topk="$TOPK_MEMORY"
      context_truncation_strategy="keep_head"
      ;;
    MemZero)
      slug="mem0"
      topk="$TOPK_MEMORY"
      context_truncation_strategy="keep_head"
      ;;
    *)
      echo "Unknown baseline: $method" >&2
      return 2
      ;;
  esac

  local out_dir="$RUN_ROOT/$slug"
  local config_path="$CONFIG_DIR/$method.json"
  mkdir -p "$out_dir"
  pushd "$out_dir" >/dev/null

  echo "======================================================================"
  echo "Running $method -> $out_dir"
  echo "======================================================================"

  local memory_count
  memory_count="$(find "$out_dir" -name '*.pkl' | wc -l | tr -d ' ')"
  if [ "$FORCE_BUILD" != "1" ] && [ "$memory_count" -ge "$EXPECTED_TRAJECTORIES" ]; then
    echo "Skipping $method construction: found $memory_count existing memory snapshots."
  else
    local build_args=(
      --memory-type "$method"
      --dataset-type LoCoMo
      --dataset-path "$DATASET"
      --config-path "$config_path"
      --num-workers "$BUILD_WORKERS"
      --start-idx "$START_IDX"
      --end-idx "$END_IDX"
      --token-cost-save-filename "$out_dir/token_cost_build"
      --tokenizer-path "$MODEL_PATH"
    )
    build_args+=(--message-preprocessor memory_construction:locomo_speaker_message_preprocessor)
    if [ "$FORCE_BUILD" = "1" ]; then
      build_args+=(--rerun)
    fi
    if [ "$method" = "MemZero" ] && [ "$ISOLATE_BUILD" = "1" ]; then
      if [ "$FORCE_BUILD" = "1" ]; then
        : > "$out_dir/01_build.log"
      else
        touch "$out_dir/01_build.log"
      fi
      run_isolated_memzero_build() {
        local idx="$1"
        local next_idx=$((idx + 1))
        local isolated_log="$out_dir/01_build_${idx}.log"
        local isolated_mem0_dir="$RUN_ROOT/.mem0_isolated_${idx}"
        mkdir -p "$isolated_mem0_dir"
        echo "---- isolated MemZero construction idx=$idx end=$next_idx ----" > "$isolated_log"
        local isolated_args=(
          --memory-type "$method"
          --dataset-type LoCoMo
          --dataset-path "$DATASET"
          --config-path "$config_path"
          --num-workers 1
          --start-idx "$idx"
          --end-idx "$next_idx"
          --token-cost-save-filename "$out_dir/token_cost_build_$idx"
          --tokenizer-path "$MODEL_PATH"
          --message-preprocessor memory_construction:locomo_speaker_message_preprocessor
        )
        if [ "$FORCE_BUILD" = "1" ]; then
          isolated_args+=(--rerun)
        fi
        MEM0_DIR="$isolated_mem0_dir" LIGHTMEM_FORCE_PROCESS_EXIT_AFTER_CONSTRUCTION=1 \
          "$PYTHON" -u "$TOOLKIT/memory_construction.py" "${isolated_args[@]}" \
          >> "$isolated_log" 2>&1
      }

      local idx
      if ! [[ "$ISOLATE_BUILD_PARALLEL" =~ ^[0-9]+$ ]] || (( ISOLATE_BUILD_PARALLEL < 1 )); then
        ISOLATE_BUILD_PARALLEL=1
      fi
      if (( ISOLATE_BUILD_PARALLEL == 1 )); then
        for ((idx=START_IDX; idx<END_IDX; idx++)); do
          if [ "$FORCE_BUILD" != "1" ]; then
            local existing_snapshot
            existing_snapshot="$(find "$out_dir" -path "*user_LoCoMo_locomo_${idx}/user_LoCoMo_locomo_${idx}.pkl" -print -quit)"
            if [ -n "$existing_snapshot" ]; then
              echo "Skipping isolated MemZero construction idx=$idx: found existing snapshot $existing_snapshot" | tee -a "$out_dir/01_build.log"
              continue
            fi
          fi
          echo "---- isolated MemZero construction idx=$idx end=$((idx + 1)) ----" | tee -a "$out_dir/01_build.log"
          run_isolated_memzero_build "$idx"
          cat "$out_dir/01_build_${idx}.log" >> "$out_dir/01_build.log"
        done
      else
        local active=0
        local failed=0
        for ((idx=START_IDX; idx<END_IDX; idx++)); do
          if [ "$FORCE_BUILD" != "1" ]; then
            local existing_snapshot
            existing_snapshot="$(find "$out_dir" -path "*user_LoCoMo_locomo_${idx}/user_LoCoMo_locomo_${idx}.pkl" -print -quit)"
            if [ -n "$existing_snapshot" ]; then
              echo "Skipping isolated MemZero construction idx=$idx: found existing snapshot $existing_snapshot" | tee -a "$out_dir/01_build.log"
              continue
            fi
          fi
          echo "Launching isolated MemZero construction idx=$idx parallel=$ISOLATE_BUILD_PARALLEL" | tee -a "$out_dir/01_build.log"
          run_isolated_memzero_build "$idx" &
          active=$((active + 1))
          if (( active >= ISOLATE_BUILD_PARALLEL )); then
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
        for ((idx=START_IDX; idx<END_IDX; idx++)); do
          if [ -f "$out_dir/01_build_${idx}.log" ]; then
            cat "$out_dir/01_build_${idx}.log" >> "$out_dir/01_build.log"
          fi
        done
        if (( failed != 0 )); then
          echo "At least one isolated MemZero construction failed" >&2
          exit 1
        fi
      fi
    else
      "$PYTHON" -u "$TOOLKIT/memory_construction.py" "${build_args[@]}" \
        2>&1 | tee "$out_dir/01_build.log"
    fi
  fi

  memory_count="$(find "$out_dir" -name '*.pkl' | wc -l | tr -d ' ')"
  if [ "$memory_count" -lt "$EXPECTED_TRAJECTORIES" ]; then
    echo "Expected memory snapshots for $EXPECTED_TRAJECTORIES LoCoMo trajectories, found $memory_count" >&2
    exit 1
  fi
  validate_memory_snapshots "$out_dir" "$EXPECTED_TRAJECTORIES" "$LIGHTMEM_MEM0_REQUIRE_NONEMPTY"

  local search_json="$out_dir/${method}_${safe_model}_LoCoMo_${topk}_${START_IDX}_${END_IDX}.json"
  local search_count=-1
  if [ -f "$search_json" ]; then
    search_count="$(json_list_len "$search_json")"
  fi
  if [ "$FORCE_SEARCH" != "1" ] && [ "$search_count" -eq "$EXPECTED_QA_COUNT" ]; then
    echo "Skipping $method search: found complete $search_json ($search_count rows)."
  else
    "$PYTHON" -u "$TOOLKIT/memory_search.py" \
      --memory-type "$method" \
      --dataset-type LoCoMo \
      --dataset-path "$DATASET" \
      --config-path "$config_path" \
      --num-workers "$SEARCH_WORKERS" \
      --top-k "$topk" \
      --start-idx "$START_IDX" \
      --end-idx "$END_IDX" \
      --strict \
      2>&1 | tee "$out_dir/02_search.log"
  fi

  if [ ! -f "$search_json" ]; then
    echo "Search output not found: $search_json" >&2
    find "$out_dir" -maxdepth 1 -type f -name '*.json' -print >&2
    exit 1
  fi
  search_count="$(json_list_len "$search_json")"
  if [ "$search_count" -ne "$EXPECTED_QA_COUNT" ]; then
    echo "Expected $EXPECTED_QA_COUNT search rows, found $search_count in $search_json" >&2
    exit 1
  fi

  local eval_json="${search_json%.json}_evaluation.json"
  if [ "$FORCE_ANSWER" != "1" ] && eval_json_is_complete "$eval_json" "$EXPECTED_QA_COUNT"; then
    echo "Skipping $method QA: found complete $eval_json."
  else
    "$PYTHON" -u "$TOOLKIT/memory_evaluation.py" \
      --search-results-path "$search_json" \
      --qa-model "$OPENAI_MODEL" \
      --judge-model "$OPENAI_MODEL" \
      --qa-batch-size "$QA_BATCH" \
      --judge-batch-size "$QA_BATCH" \
      --api-config-path "$CONFIG_DIR/api_config.json" \
      --dataset-type LoCoMo \
      --tokenizer-path "$MODEL_PATH" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --max-output-tokens "$QA_MAX_TOKENS" \
      --context-truncation-strategy "$context_truncation_strategy" \
      --skip-judge \
      2>&1 | tee "$out_dir/03_answer.log"
  fi
  if ! eval_json_is_complete "$eval_json" "$EXPECTED_QA_COUNT"; then
    echo "QA output is incomplete or contains empty/error predictions: $eval_json" >&2
    exit 1
  fi

  local flat_json="$out_dir/${slug}_predictions_flat.json"
  local metrics_json="$out_dir/${slug}_metrics_cat1_4.json"
  local judge_json="$out_dir/${slug}_judge_metrics_cat1_4.json"

  if [ "$FORCE_METRICS" != "1" ] && [ -s "$metrics_json" ]; then
    echo "Skipping $method metrics: found $metrics_json."
  else
    "$PYTHON" /home/stu0032/paper/scripts/normalize_lightmem_toolkit_locomo_results.py \
      --input "$eval_json" \
      --output "$flat_json" \
      --method "$method" \
      --model "$OPENAI_MODEL"

    metrics_args=(
      --input "$flat_json"
      --output "$metrics_json"
      --prediction-key model_answer
      --reference-key golden_answer
      --question-key question
      --category-key category
    )
    if [ "$SKIP_BERTSCORE" = "1" ]; then
      metrics_args+=(--skip-bertscore)
    else
      metrics_args+=(
        --bertscore-model "$BERTSCORE_MODEL"
        --bertscore-batch-size "$BERTSCORE_BATCH_SIZE"
        --bertscore-device "$BERTSCORE_DEVICE"
      )
      if [ -n "$BERTSCORE_NUM_LAYERS" ]; then
        metrics_args+=(--bertscore-num-layers "$BERTSCORE_NUM_LAYERS")
      fi
      if [ "$FAIL_ON_BERTSCORE_ERROR" = "1" ]; then
        metrics_args+=(--fail-on-bertscore-error)
      fi
    fi
    "$PYTHON" /home/stu0032/paper/scripts/compute_locomo_text_metrics.py "${metrics_args[@]}" \
      2>&1 | tee "$out_dir/04_metrics.log"
  fi

  if [ "$RUN_JUDGE" = "1" ]; then
    "$PYTHON" /home/stu0032/paper/scripts/compute_locomo_llm_judge_metrics.py \
      --input "$flat_json" \
      --output "$judge_json" \
      --prediction-key model_answer \
      --reference-key golden_answer \
      --question-key question \
      --category-key category \
      --model "$JUDGE_MODEL" \
      --base-url "$JUDGE_BASE_URL" \
      --api-key "$JUDGE_API_KEY" \
      --protocol "$JUDGE_PROTOCOL" \
      --max-workers "$JUDGE_WORKERS" \
      --max-retries "$JUDGE_MAX_RETRIES" \
      --resume \
      2>&1 | tee "$out_dir/05_judge.log"
  fi

  popd >/dev/null
}

for method in "${BASELINES[@]}"; do
  run_one "$method"
done

echo "All requested baselines completed."
echo "Run root: $RUN_ROOT"
