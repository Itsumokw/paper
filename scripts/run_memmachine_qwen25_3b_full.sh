#!/usr/bin/env bash
set -euo pipefail

source /home/stu0032/paper/scripts/memmachine_env.sh

RUNNING_BASELINE_PATTERN='[r]un_xmemory_qwen25_3b_full|[a]dd.py|[x]Memory_search_framework.py|[e]vals.py|[r]un_lightmem|[s]earch_locomo.py|[a]dd_locomo.py'
if pgrep -af "${RUNNING_BASELINE_PATTERN}" >/dev/null && [ "${ALLOW_CONCURRENT_EXPERIMENTS:-0}" != "1" ]; then
    echo "Another baseline experiment is still running. Set ALLOW_CONCURRENT_EXPERIMENTS=1 to run MemMachine anyway." >&2
    exit 2
fi

cd "${MEMMACHINE_EVAL_DIR}"

mkdir -p "${MEMMACHINE_RUN_ROOT}" result/final_score

if [ ! -f "${MEMMACHINE_CONFIG_SRC}" ]; then
    echo "Missing MemMachine local config template: ${MEMMACHINE_CONFIG_SRC}" >&2
    exit 2
fi

if ! cmp -s "${MEMMACHINE_CONFIG_SRC}" "${MEMMACHINE_CONFIG_PATH}"; then
    cp "${MEMMACHINE_CONFIG_SRC}" "${MEMMACHINE_CONFIG_PATH}"
fi

"${MEMMACHINE_PYTHON}" - <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect(("127.0.0.1", 7687))
except OSError:
    print("MemMachine requires a running Neo4j at bolt://127.0.0.1:7687.")
    print("Start Neo4j first and set NEO4J_PASSWORD, then rerun this script.")
    sys.exit(10)
finally:
    sock.close()
PY

POSTFIX="${MEMMACHINE_POSTFIX:-qwen25_3b_full}"
INGEST_CONCURRENCY="${MEMMACHINE_INGEST_CONCURRENCY:-8}"
SEARCH_CONCURRENCY="${MEMMACHINE_SEARCH_CONCURRENCY:-32}"
JUDGE_CONCURRENCY="${MEMMACHINE_JUDGE_CONCURRENCY:-32}"

"${MEMMACHINE_PYTHON}" /home/stu0032/paper/scripts/preflight_qwen25_3b_baselines.py memmachine

./run_test.sh locomo "$POSTFIX" delete memmachine
./run_test.sh locomo "$POSTFIX" ingest memmachine --ingest-concurrency "$INGEST_CONCURRENCY"
./run_test.sh locomo "$POSTFIX" search memmachine --search-concurrency "$SEARCH_CONCURRENCY" --judge-concurrency "$JUDGE_CONCURRENCY"
