#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

MODEL_DIR=/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean \
SERVED_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}" \
./start_vllm_qwen25.sh
