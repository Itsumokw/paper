#!/usr/bin/env bash
# Shared concurrency knobs for post-A-MEM local runs.
#
# The defaults intentionally raise utilization conservatively for a 24GB RTX 4090
# serving Qwen3-8B at 32k context. They improve batching without changing prompts,
# retrieval, memory construction, or evaluation semantics.

locomo_high_util_enabled() {
  [[ "${LOCOMO_HIGH_UTILIZATION_AFTER_AMEM:-1}" != "0" ]]
}

locomo_raise_env_int() {
  local name="$1"
  local minimum="$2"
  local current="${!name:-}"
  if ! [[ "$current" =~ ^[0-9]+$ ]] || (( current < minimum )); then
    export "$name=$minimum"
  fi
}

locomo_enable_high_util_vllm_defaults() {
  if ! locomo_high_util_enabled; then
    return 0
  fi
  export VLLM_FORCE_RESTART="${LOCOMO_HIGH_UTIL_FORCE_VLLM_RESTART:-1}"
  export VLLM_GPU_MEMORY_UTILIZATION="${LOCOMO_HIGH_UTIL_VLLM_GPU_MEMORY_UTILIZATION:-0.98}"
  export VLLM_MAX_MODEL_LEN="${LOCOMO_HIGH_UTIL_VLLM_MAX_MODEL_LEN:-32000}"
  export VLLM_GENERATION_CONFIG="${LOCOMO_HIGH_UTIL_VLLM_GENERATION_CONFIG:-vllm}"
  if [[ -n "${LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="$LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS"
  else
    export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  fi
  export VLLM_MAX_NUM_SEQS="${LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_SEQS:-32}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${LOCOMO_HIGH_UTIL_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
}

locomo_default_vllm_model_path() {
  local model="${OPENAI_MODEL:-}"
  case "$model" in
    *Qwen3-8B*|*qwen3-8b*)
      echo "${QWEN3_8B_MODEL_PATH:-/home/stu0032/paper/models/Qwen3-8B}"
      ;;
    *Qwen2.5-3B*|*Qwen2_5-3B*|*qwen2.5-3b*|*qwen2_5-3b*)
      echo "${QWEN25_3B_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}"
      ;;
    *)
      echo "${QWEN25_3B_MODEL_PATH:-/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean}"
      ;;
  esac
}

locomo_set_vllm_model_path_for_openai_model() {
  export VLLM_MODEL_PATH="$(locomo_default_vllm_model_path)"
}

locomo_start_vllm_with_safe_fallback() {
  local requested_seqs="${VLLM_MAX_NUM_SEQS:-}"
  local requested_batched="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
  if bash /home/stu0032/paper/scripts/start_vllm_qwen25_3b.sh; then
    return 0
  fi
  if locomo_high_util_enabled && [[ "$requested_seqs" != "1" ]]; then
    echo "[high-util] vLLM failed with max_num_seqs=${requested_seqs}, max_num_batched_tokens=${requested_batched}; retrying safe seq=1." >&2
    export VLLM_FORCE_RESTART=1
    export VLLM_GPU_MEMORY_UTILIZATION="${LOCOMO_HIGH_UTIL_FALLBACK_GPU_MEMORY_UTILIZATION:-0.97}"
    export VLLM_MAX_MODEL_LEN="${LOCOMO_HIGH_UTIL_FALLBACK_MAX_MODEL_LEN:-32000}"
    export VLLM_GENERATION_CONFIG="${LOCOMO_HIGH_UTIL_VLLM_GENERATION_CONFIG:-vllm}"
    if [[ -n "${LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS:-}" ]]; then
      export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="$LOCOMO_HIGH_UTIL_VLLM_CHAT_TEMPLATE_KWARGS"
    else
      export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    fi
    export VLLM_MAX_NUM_SEQS=1
    export VLLM_MAX_NUM_BATCHED_TOKENS="${LOCOMO_HIGH_UTIL_FALLBACK_BATCHED_TOKENS:-32000}"
    bash /home/stu0032/paper/scripts/start_vllm_qwen25_3b.sh
    return $?
  fi
  return 1
}
