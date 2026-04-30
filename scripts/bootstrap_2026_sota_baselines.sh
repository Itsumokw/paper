#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p baseline

clone_checkout() {
  local dir="$1"
  local url="$2"
  local commit="$3"

  if [ -d "$dir/.git" ]; then
    echo "[bootstrap] updating $dir"
    git -C "$dir" remote set-url origin "$url"
    git -C "$dir" fetch origin
  else
    if [ -e "$dir" ]; then
      echo "[bootstrap] $dir exists but is not a git checkout; leaving it untouched"
      return 0
    fi
    echo "[bootstrap] cloning $url -> $dir"
    git clone "$url" "$dir"
  fi

  git -C "$dir" checkout "$commit"
}

clone_checkout \
  baseline/MemGAS \
  https://github.com/Applied-Machine-Learning-Lab/ICLR2026_MemGAS.git \
  c2d4e9fdc331074802a711baf4371197f9194399

clone_checkout \
  baseline/ReMe \
  https://github.com/agentscope-ai/ReMe.git \
  e0d0e3e568e6d2163c068ad05af2cf4536c42ad2

if [ -f baseline/MAGMA/data/locomo10.json ]; then
  echo "[bootstrap] LoCoMo10 dataset already present at baseline/MAGMA/data/locomo10.json"
else
  clone_checkout \
    baseline/MAGMA \
    https://github.com/FredJiang0324/MAGMA.git \
    6ba49dd64b1ba674bb8c39addd5fb2a60068703b
fi

echo "[bootstrap] done"
