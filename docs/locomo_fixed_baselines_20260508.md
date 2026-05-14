# LoCoMo Fixed Baseline Set

Date: 2026-05-08

This document records the fixed baseline combination for the main LoCoMo-style
long-term memory evaluation.

## Fixed Baselines

| Method | Role in the comparison |
|---|---|
| Full Context | Long-context control; measures what the same LLM can do when given the full conversation history directly. |
| A-Mem | High-frequency agentic memory baseline with dynamic note/link organization. |
| Mem0 | Production-oriented long-term memory baseline and common external comparator. |
| SimpleMem | Lightweight compression and retrieval baseline; current backbone candidate for method improvement. |
| HiGMem | Hierarchical and LLM-guided memory baseline; already reproduced locally on LoCoMo10. |
| MemGAS | Recent accepted multi-granularity graph/routing memory baseline; already reproduced locally on LoCoMo10. |

## LoCoMo Evaluation Scope

Use the same LoCoMo10 data and the same local LLM endpoint for the comparable
table whenever possible.

- Dataset: `datasets/locomo/data/locomo10.json` or an exact copy of the same
  10-sample LoCoMo subset.
- Total QA: 1,986.
- Main comparable score: categories 1-4, 1,540 QA, token F1 and BLEU-1.
- Category 5: report separately because adversarial refusal scoring is not
  always implemented consistently across baselines.
- Preferred local model for a fair baseline table: `Qwen/Qwen2.5-3B-Instruct`
  served through the local OpenAI-compatible vLLM endpoint.

## Current LoCoMo Status

| Method | Local status | Notes |
|---|---|---|
| Full Context | Done | Qwen2.5-3B cat1-4 run exists under `runs/locomo_fixed_baselines/qwen25_3b_20260508_220722/full_context`. |
| A-Mem | Pending rerun | Use the official A-MEM runner path, not the earlier LightMem adapter result. The adapter run is invalid for final reporting because it used a non-official retrieval/QA path and old memory cache lost true speaker names. |
| Mem0 | Missing | Need setup/run; can use the LightMem toolkit implementation or official Mem0, but must keep the same metric script. |
| SimpleMem | Done | Qwen2.5-3B and Qwen3-8B runs exist; use Qwen2.5-3B for the same-model baseline table. |
| HiGMem | Done | Qwen2.5-3B LoCoMo10 full run exists. |
| MemGAS | Done | Qwen2.5-3B normalized metrics exist. |

## Required Remaining LoCoMo Runs

1. Full Context with `Qwen/Qwen2.5-3B-Instruct`.
2. A-Mem official runner with `Qwen/Qwen2.5-3B-Instruct`.
3. Mem0 with `Qwen/Qwen2.5-3B-Instruct`.
4. Optional verification reruns for SimpleMem, HiGMem, and MemGAS only if the
   final table requires one completely fresh synchronized timestamp.

After each run, save raw predictions and compute the same normalized metrics:
cat1-4 F1/BLEU-1, all-category F1/BLEU-1, and category-wise breakdown.

## 2026-05-08 Preparation Notes

Prepared scripts:

- `scripts/start_vllm_qwen25_3b.sh`
- `scripts/run_locomo_fixed_missing_baselines_qwen25_3b.sh`
- `scripts/run_amem_official_qwen25_3b_full.sh`
- `scripts/normalize_lightmem_toolkit_locomo_results.py`

Downloaded source references:

- `baseline/A-MEM` at `0c8039f`
- `baseline/A-MEM-SYS` at `f303dfc`
- `baseline/mem0` at `a623cfa`

Execution uses the LightMem `memory_toolkits` adapters for FullContext and
MemZero so those methods share one LoCoMo construction/retrieval/QA pipeline and
one metric script. A-MEM must be run through `baseline/A-MEM/test_advanced_robust.py`
because the official A-MEM LoCoMo path generates LLM retrieval keywords and uses
category-specific answer prompts; the LightMem adapter does not match that
evaluation path. The current run script enables a token guard for
QA generation: with the local `Qwen2.5-3B-Instruct` 32,768-token context, it
defaults to 256 output tokens, a 1,024-token safety buffer, and 31,488 maximum
input tokens. Retrieved memories are dropped from the low-priority tail before
the OpenAI-compatible request is sent if a prompt would exceed that budget.
FullContext also caps stored retrieved memory text at 30,000 memory tokens by
default, with the QA-time guard acting as the final protection against vLLM
context-length failures.

## A-MEM Rerun Correction

The `runs/locomo_fixed_baselines/qwen25_3b_20260508_220722/amem` result should
not be used in the final table. It is an adapter-only run: old memories contained
`Speaker user says: ...` instead of true LoCoMo speaker names, and the retrieval
and QA path did not follow the official A-MEM `test_advanced_robust.py` flow.
The corrected official script stores raw output in
`runs/amem_official/qwen25_3b_full/<timestamp>/result.json` and normalized
metrics in `official_metrics_cat1_4.json` and `official_metrics_all.json`.
