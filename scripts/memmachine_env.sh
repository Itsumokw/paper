#!/usr/bin/env bash
# Source this file before running MemMachine reproduction commands.

export PAPER_ROOT="${PAPER_ROOT:-/home/stu0032/paper}"
export MEMMACHINE_ROOT="${MEMMACHINE_ROOT:-${PAPER_ROOT}/baseline/MemMachine}"
export MEMMACHINE_EVAL_DIR="${MEMMACHINE_EVAL_DIR:-${MEMMACHINE_ROOT}/evaluation/retrieval_agent}"
export MEMMACHINE_RUN_ROOT="${MEMMACHINE_RUN_ROOT:-${PAPER_ROOT}/runs/memmachine/qwen25_3b_full}"
export MEMMACHINE_CONFIG_SRC="${MEMMACHINE_CONFIG_SRC:-${MEMMACHINE_EVAL_DIR}/configuration.local_qwen25_vllm.yml}"
export MEMMACHINE_CONFIG_PATH="${MEMMACHINE_CONFIG_PATH:-${MEMMACHINE_EVAL_DIR}/configuration.yml}"
export MEMMACHINE_PYTHON="${MEMMACHINE_PYTHON:-${PAPER_ROOT}/.venv/bin/python}"
export MEMMACHINE_JAVA_HOME="${MEMMACHINE_JAVA_HOME:-${PAPER_ROOT}/tools/java/jdk-21.0.11+10-jre}"
export MEMMACHINE_NEO4J_HOME="${MEMMACHINE_NEO4J_HOME:-${PAPER_ROOT}/tools/neo4j/neo4j-community-5.26.6}"
export JAVA_HOME="${JAVA_HOME:-${MEMMACHINE_JAVA_HOME}}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export MEMMACHINE_USE_STREAMING="${MEMMACHINE_USE_STREAMING:-1}"
export MEMMACHINE_CLOSE_ON_FIRST_JSON="${MEMMACHINE_CLOSE_ON_FIRST_JSON:-1}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j_password}"

if [ -x "${JAVA_HOME}/bin/java" ]; then
    export PATH="${JAVA_HOME}/bin:${PAPER_ROOT}/.venv/bin:${PATH}"
else
    export PATH="${PAPER_ROOT}/.venv/bin:${PATH}"
fi
export PYTHONPATH="${MEMMACHINE_ROOT}:${MEMMACHINE_ROOT}/packages/common/src:${MEMMACHINE_ROOT}/packages/server/src:${MEMMACHINE_ROOT}/packages/client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
