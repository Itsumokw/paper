#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT="/home/stu0032/paper"
PYTHON="$PAPER_ROOT/.venv/bin/python"
SIMPLEMEM="$PAPER_ROOT/baseline/SimpleMem"
DATASET="$PAPER_ROOT/datasets/locomo/data/locomo10.json"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PAPER_ROOT/runs/simplemem/run_$TS"

mkdir -p "$OUT"

# 1) backup config
cp "$SIMPLEMEM/config.py" "$OUT/config_backup.py"

# 2) restore on exit
trap 'cp "$OUT/config_backup.py" "$SIMPLEMEM/config.py"' EXIT

# 3) patch config inline
$PYTHON -c "
from pathlib import Path
import re
p = Path('$SIMPLEMEM/config.py')
s = p.read_text()
s = re.sub(r'^OPENAI_API_KEY = .*', 'OPENAI_API_KEY = \"EMPTY\"', s, flags=re.M)
s = re.sub(r'^OPENAI_BASE_URL = .*', 'OPENAI_BASE_URL = \"http://127.0.0.1:8000/v1\"', s, flags=re.M)
s = re.sub(r'^LLM_MODEL = .*', 'LLM_MODEL = \"Qwen/Qwen2.5-3B-Instruct\"', s, flags=re.M)
p.write_text(s)
print('Config patched OK')
"

# 4) preflight
$PYTHON -c "
from openai import OpenAI
c = OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8000/v1')
r = c.chat.completions.create(model='Qwen/Qwen2.5-3B-Instruct', messages=[{'role':'user','content':'ok'}], max_tokens=4)
print('vLLM OK:', r.choices[0].message.content)
"

# 5) run test
echo "Output dir: $OUT"
cd "$SIMPLEMEM"
CUDA_VISIBLE_DEVICES="" $PYTHON -u test_locomo10.py \
  --dataset "$DATASET" \
  --result-file "$OUT/result.json" \
  --parallel-questions \
  --test-workers 20 \
  2>&1 | tee "$OUT/run.log"
