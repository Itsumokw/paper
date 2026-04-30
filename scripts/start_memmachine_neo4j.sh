#!/usr/bin/env bash
set -euo pipefail

source /home/stu0032/paper/scripts/memmachine_env.sh

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files neo4j.service >/dev/null 2>&1; then
    printf '%s\n' "${SUDO_PASSWORD:?Set SUDO_PASSWORD to start system Neo4j}" | sudo -S -p '' systemctl start neo4j
    "${MEMMACHINE_PYTHON}" - <<'PY'
import socket
import time

deadline = time.time() + 60
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 7687))
    except OSError:
        time.sleep(1)
    else:
        print("Neo4j reachable at bolt://127.0.0.1:7687")
        raise SystemExit(0)
    finally:
        sock.close()

raise SystemExit("Neo4j did not become reachable within 60 seconds")
PY
    exit 0
fi

if [ ! -x "${JAVA_HOME}/bin/java" ]; then
    echo "Local Java is missing. Run:" >&2
    echo "  /home/stu0032/paper/scripts/install_memmachine_neo4j_local.sh" >&2
    exit 2
fi

if [ ! -x "${MEMMACHINE_NEO4J_HOME}/bin/neo4j" ]; then
    echo "Local Neo4j is missing at ${MEMMACHINE_NEO4J_HOME}." >&2
    echo "Run:" >&2
    echo "  /home/stu0032/paper/scripts/install_memmachine_neo4j_local.sh" >&2
    exit 2
fi

mkdir -p "${PAPER_ROOT}/runs/memmachine/neo4j"

CONF="${MEMMACHINE_NEO4J_HOME}/conf/neo4j.conf"
append_conf() {
    local key="$1"
    local value="$2"
    if grep -qE "^#?${key}=" "${CONF}"; then
        sed -i "s|^#\\?${key}=.*|${key}=${value}|" "${CONF}"
    else
        printf '%s=%s\n' "${key}" "${value}" >> "${CONF}"
    fi
}

append_conf "server.default_listen_address" "127.0.0.1"
append_conf "server.bolt.enabled" "true"
append_conf "server.bolt.listen_address" ":7687"
append_conf "server.http.enabled" "true"
append_conf "server.http.listen_address" ":7474"
append_conf "server.memory.heap.initial_size" "${NEO4J_HEAP_INITIAL:-1G}"
append_conf "server.memory.heap.max_size" "${NEO4J_HEAP_MAX:-2G}"

if [ ! -f "${MEMMACHINE_NEO4J_HOME}/data/dbms/auth.ini" ] && [ ! -f "${MEMMACHINE_NEO4J_HOME}/data/dbms/auth" ]; then
    "${MEMMACHINE_NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}"
fi

"${MEMMACHINE_NEO4J_HOME}/bin/neo4j" start

"${MEMMACHINE_PYTHON}" - <<'PY'
import socket
import time

deadline = time.time() + 60
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 7687))
    except OSError:
        time.sleep(1)
    else:
        print("Neo4j reachable at bolt://127.0.0.1:7687")
        raise SystemExit(0)
    finally:
        sock.close()

raise SystemExit("Neo4j did not become reachable within 60 seconds")
PY
