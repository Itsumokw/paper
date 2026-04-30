#!/usr/bin/env bash
set -euo pipefail

source /home/stu0032/paper/scripts/memmachine_env.sh

cd "${PAPER_ROOT}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

mkdir -p tools/downloads tools/java tools/neo4j

JRE_VERSION="21.0.11_10"
JRE_DIR="${PAPER_ROOT}/tools/java/jdk-21.0.11+10-jre"
JRE_ARCHIVE="${PAPER_ROOT}/tools/downloads/OpenJDK21U-jre_x64_linux_hotspot_${JRE_VERSION}.tar.gz"
JRE_URL="${JRE_URL:-https://mirrors.tuna.tsinghua.edu.cn/Adoptium/21/jre/x64/linux/OpenJDK21U-jre_x64_linux_hotspot_${JRE_VERSION}.tar.gz}"

if [ ! -x "${JRE_DIR}/bin/java" ]; then
    echo "Installing local Java runtime from ${JRE_URL}"
    if [ ! -s "${JRE_ARCHIVE}" ]; then
        curl -L --retry 3 --connect-timeout 20 --max-time 600 -o "${JRE_ARCHIVE}" "${JRE_URL}"
    fi
    rm -rf "${JRE_DIR}"
    tar -xzf "${JRE_ARCHIVE}" -C "${PAPER_ROOT}/tools/java"
fi

"${JRE_DIR}/bin/java" -version

NEO4J_VERSION="${NEO4J_VERSION:-5.26.6}"
NEO4J_DIR="${PAPER_ROOT}/tools/neo4j/neo4j-community-${NEO4J_VERSION}"
NEO4J_ARCHIVE="${NEO4J_ARCHIVE:-${PAPER_ROOT}/tools/downloads/neo4j-community-${NEO4J_VERSION}-unix.tar.gz}"
NEO4J_URL="${NEO4J_URL:-https://neo4j.com/artifact.php?name=neo4j-community-${NEO4J_VERSION}-unix.tar.gz}"

if [ ! -x "${NEO4J_DIR}/bin/neo4j" ]; then
    if [ ! -s "${NEO4J_ARCHIVE}" ]; then
        echo "Downloading Neo4j from ${NEO4J_URL}"
        echo "If this fails with HTTP 403, place the archive at:"
        echo "  ${NEO4J_ARCHIVE}"
        curl -L --fail --retry 2 --connect-timeout 20 --max-time 900 -A 'Mozilla/5.0' -o "${NEO4J_ARCHIVE}.tmp" "${NEO4J_URL}"
        tar -tzf "${NEO4J_ARCHIVE}.tmp" >/dev/null
        mv "${NEO4J_ARCHIVE}.tmp" "${NEO4J_ARCHIVE}"
    fi
    rm -rf "${NEO4J_DIR}"
    tar -xzf "${NEO4J_ARCHIVE}" -C "${PAPER_ROOT}/tools/neo4j"
fi

echo "Neo4j installed at ${NEO4J_DIR}"
echo "Start it with:"
echo "  /home/stu0032/paper/scripts/start_memmachine_neo4j.sh"
