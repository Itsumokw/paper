# HiGMemPlus Audit

Date: 2026-05-17

## Scope

This audit follows `docs/higmem_plus_goal_20260517.md`. The active experimental scope is:

- LongDialQA/DialSim baseline reproduction first: FullContext, A-Mem, MemGAS, original HiGMem.
- HiGMemPlus methods later: Evidence-Component, Repairable-Episode, Adaptive-Routing.
- Only deterministic 5% smoke and 50% subset runs are allowed. Full-dataset runs are out of scope.

## Required Documents Read

- `CODEX_FIRST_READ.md`
- `docs/reproduction_results_20260508.md`
- `locomo_baseline_reproduction_acceptance.md`
- Current `scripts/` inventory
- `reproductions/successful/HiGMem/2026-04-28_qwen2.5-3b_locomo10-full_w10/`

## Local Code Availability

| Component | Local Path | Status | Commit / Evidence |
|---|---:|---|---|
| Original HiGMem | `baseline/HiGMem` | Present | `f275072f25323a01a8bff3680edbb34ed97d33be` |
| HiGMem reproduced artifact | `reproductions/successful/HiGMem/2026-04-28_qwen2.5-3b_locomo10-full_w10` | Present | README records Qwen2.5-3B LoCoMo10 run, 1986 QA |
| DialSim official code | `baseline/DialSim` | Present | `0dd4db4db90740dbcf047f18a8e8adc83e7ba0f0` |
| LightMem toolkit adapters | `baseline/LightMem` | Present | `b11eccd23c7cf6f0fde390b44c25efef818a5c5e` |
| A-MEM local checkout | `baseline/A-MEM` and `baseline/A-MEM-SYS` | Present | `baseline/A-MEM` commit `0c8039f28fdcc08189a23c07a3437d9d2482f9c2` |
| MemGAS local checkout | `baseline/MemGAS` | Present | `c2d4e9fdc331074802a711baf4371197f9194399` |
| HiGMemPlus extension package | `baseline/HiGMemPlus` | Started | Non-invasive package, original HiGMem untouched |

No official code needed to be cloned during this audit because all required baseline source trees are already local.

## Dataset Availability

| Dataset | Local Path | Status | Hash / Count |
|---|---:|---|---|
| LoCoMo10 | `datasets/locomo/data/locomo10.json` | Present | sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, 10 samples, 1986 QA |
| DialSim v1.1 zip | `datasets/DialSim/dialsim_v1.1.zip` | Present | sha256 `e7772260089ff1ecc7f46da29da33aa50c894526848d01fec8bf55b6cad6c10c` |
| DialSim v1.0 zip | `datasets/DialSim/dialsim_v1.0.zip` | Present | sha256 `b32517d8e08221362dfe1e0c7349fa91ec8930691c53f87ef680acfc5f74596f` |
| DialSim extracted v1.1 pickles | `datasets/DialSim/v1.1/` | Present | per-file hashes recorded in normalized manifest |
| LongDialQA normalized adapter output | `datasets/DialSim/longdialqa_normalized_v1.1_seed0/` | Present | 3940 sessions, 3277 seeded selected QA |

The LongDialQA normalized adapter is `scripts/normalize_longdialqa_dialsim.py`. It treats DialSim as protocol and LongDialQA as dataset, using official v1.1 pickle files and deterministic seeded replay of the official question-selection logic. It records session/date-level evidence where oracle target dates exist; turn-level evidence is not present in the official pickle schema.

## Deterministic Subsets

Subset builder: `scripts/build_higmem_plus_subsets.py`

Seed: `20260517`

| Dataset | Fraction | Dataset Path | Manifest Path | Count |
|---|---:|---:|---:|---:|
| LoCoMo10 | 50% | `datasets/subsets/locomo10_50pct_seed20260517.json` | `datasets/subsets/locomo10_50pct_seed20260517_manifest.json` | 992 / 1986 QA |
| LoCoMo10 | 5% smoke | `datasets/subsets/locomo10_5pct_seed20260517.json` | `datasets/subsets/locomo10_5pct_seed20260517_manifest.json` | 99 / 1986 QA |
| LongDialQA/DialSim | 50% | `datasets/subsets/longdialqa_50pct_seed20260517.json` | `datasets/subsets/longdialqa_50pct_seed20260517_manifest.json` | 1637 / 3277 questions |
| LongDialQA/DialSim | 5% smoke | `datasets/subsets/longdialqa_5pct_seed20260517.json` | `datasets/subsets/longdialqa_5pct_seed20260517_manifest.json` | 175 / 3277 questions |

LoCoMo subsets are stratified by official category. LongDialQA subsets are stratified by show, question source/type, and answerability.

## Runner Scripts

Existing relevant scripts:

- `scripts/run_higmem_qwen25_3b_full.sh`: original HiGMem LoCoMo wrapper. It defaults to full LoCoMo and is not used for HiGMemPlus subset experiments.
- `scripts/run_memgas_qwen25_3b_full.sh`: MemGAS LoCoMo wrapper. It defaults to full LoCoMo and is not used directly for this subset-only goal.
- `scripts/run_locomo_fixed_missing_baselines_qwen25_3b.sh`: LightMem toolkit runner for FullContext, A-MEM, MemZero on LoCoMo. It can default to full ranges; not used directly for subset-only HiGMemPlus runs.
- `scripts/normalize_longdialqa_dialsim.py`: LongDialQA/DialSim adapter.
- `scripts/build_higmem_plus_subsets.py`: deterministic subset builder.
- `scripts/run_longdialqa_baseline.py`: LongDialQA baseline runner. It has been patched to require `--subset-manifest` and refuse manifest-less full-data defaults.

Planned required runner:

- `scripts/run_higmem_plus.py`: unified subset-only runner for `locomo|longdialqa`, original HiGMem baseline, and three HiGMemPlus methods. This runner must require `--subset-manifest`.

## Environment

Model endpoint:

- OpenAI-compatible vLLM endpoint: `http://127.0.0.1:8000/v1`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Status during audit: restarted successfully after an earlier 502/unhealthy endpoint state.

Local GPU:

- NVIDIA RTX 4090, 24GB VRAM.
- vLLM startup uses controlled memory parameters to avoid OOM.

Important environment assumptions:

- `OPENAI_API_KEY=EMPTY` is acceptable for local vLLM.
- No proxy should be used for localhost calls.
- Embedding models and baseline dependencies are local or already installed in `.venv`; missing model downloads can still block MemGAS/A-MEM smoke tests.

## Risks

- Original HiGMem and several existing shell wrappers default to full LoCoMo. The new HiGMemPlus runner must require `--subset-manifest` and should not call these wrappers directly.
- FullContext on LongDialQA can produce very long prompts; context budgets must be fixed and identical across comparable runs.
- A-MEM and MemGAS construction call the same local LLM and can be slow even on 5% smoke subsets.
- MemGAS summary/keyword extraction may fail if the local endpoint returns truncated JSON; fallback behavior must be logged.
- LongDialQA official evidence is mostly session/date-level, not turn-level. Evidence recall should be labeled accordingly.
- Prior local results are LoCoMo reproductions, not LongDialQA/DialSim reproductions. LongDialQA baseline results must not be claimed until raw predictions, retrieved evidence, metrics, stats, logs, command.env, subset manifest, and dataset hashes exist.
