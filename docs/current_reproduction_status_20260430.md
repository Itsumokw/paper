# Current Reproduction Status

Snapshot time: 2026-04-30 18:52:27 CST

## Active run

- Command family: `scripts/run_2026_sota_memory_qwen25_3b_all.sh`
- Run root: `runs/2026_sota_memory_qwen25_3b_all/20260430_183900`
- Model endpoint: `http://127.0.0.1:8000/v1`
- Served model: `Qwen/Qwen2.5-3B-Instruct`
- Dataset: `baseline/MAGMA/data/locomo10.json`
- Dataset size: 10 conversations, 1986 QA
- Category counts: cat1 282, cat2 321, cat3 96, cat4 841, cat5 446

The run is intentionally not tracked in git because `runs/` is live output and can grow while experiments are running.

## Live progress at snapshot

The active all-in-one run was still in the first model, MemGAS.

- Active process: `locomo_2026_sota.py run-memgas`
- MemGAS device setting: `--device cpu`
- MemGAS top-k: 20
- MemGAS QA workers: 4
- Predictions completed: 335 / 1986
- Built memory states: `conv-26`, `conv-30`, `conv-41`
- Completed QA by sample:
  - `conv-26`: 199 / 199
  - `conv-30`: 105 / 105
  - `conv-41`: 31 / 193
- Prediction record errors: 0
- Ingest/prediction error files: not present at snapshot
- Last record: `conv-41`, `qa_idx=30`, `category=1`, `error=None`

GPU at snapshot:

- GPU memory: 22752 / 24564 MiB
- GPU utilization: 100%
- GPU temperature: 90C
- Power draw: about 445 W

## What is committed

This commit tracks the small, portable pieces needed to understand and restart the work:

- Reproduction scripts under `scripts/`
- Current vLLM launch helpers
- Shared LoCoMo text metrics script with F1, BLEU1, ROUGE-L, and optional BERTScore-F1
- Local Omni-SimpleMem compatibility config shim
- SimpleMem local fixes already used in earlier reproductions
- Reproduction notes and previous compact result summaries under `docs/` and `reproductions/`
- Research survey markdown/manifest/README
- A bootstrap script that restores the external checkouts and the LoCoMo10 dataset path

## What is intentionally not committed

The following local artifacts are ignored and should not be pushed to GitHub:

- `.venv/`
- `models/`
- `runs/`
- External baseline checkouts:
  - `baseline/MemGAS/`
  - `baseline/ReMe/`
- `baseline/MAGMA/`
  - `baseline/LightMem/`
  - `baseline/xMemory/`
  - `baseline/MemMachine/`
  - `baseline/HiGMem/`
- Downloaded paper PDFs and web snapshots under `research/**/papers/` and `research/**/web/`
- Local Java/Neo4j downloads under `tools/`

The full local research pack remains available on the server as:

- `research/agent_memory_survey_20260430.tar.gz`
- `research/agent_memory_survey_20260430/papers/`
- `research/agent_memory_survey_20260430/web/`

## External baselines used

Use `scripts/bootstrap_2026_sota_baselines.sh` after pull to restore the external checkouts used by the current scripts.

- MemGAS: `https://github.com/Applied-Machine-Learning-Lab/ICLR2026_MemGAS.git`
  - Commit: `c2d4e9fdc331074802a711baf4371197f9194399`
- ReMe: `https://github.com/agentscope-ai/ReMe.git`
  - Commit: `e0d0e3e568e6d2163c068ad05af2cf4536c42ad2`
- MAGMA dataset source: `https://github.com/FredJiang0324/MAGMA.git`
  - Commit observed locally: `6ba49dd64b1ba674bb8c39addd5fb2a60068703b`
  - Used for `baseline/MAGMA/data/locomo10.json`

## Reproduction caveat

The current three-model pipeline is a local adapted comparable benchmark, not a strict untouched paper reproduction.

- MemGAS uses the upstream quickstart API plus the local LoCoMo QA adapter.
- Omni-SimpleMem uses the repository code plus a small local `omni_memory/core/config.py` shim because the checkout referenced that module but did not include it.
- ReMe uses the PyPI runtime package plus the GitHub benchmark wrapper and local CPU sentence-transformer embeddings.
- Embedding and BERTScore are forced to CPU by default to avoid OOM while vLLM occupies the 4090.

This preserves the dataset, local Qwen2.5-3B endpoint, retrieval/answering output shape, and unified F1/BLEU1/ROUGE-L/BERTScore evaluation path, but it should be described as local adapted reproduction unless the official end-to-end paper scripts are run unchanged.

## Pull-side quick start

```bash
git pull origin main
cd paper
bash scripts/bootstrap_2026_sota_baselines.sh
```

Then restore the Python environment and model cache as needed on the target machine. The large local model directory is intentionally not committed.
