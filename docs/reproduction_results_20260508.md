# LoCoMo Reproduction Results Snapshot

Date: 2026-05-08

This document summarizes the completed local LoCoMo reproductions in this
workspace. The ranking table uses the most comparable text-metric setting:
LoCoMo categories 1-4, token F1, and BLEU-1. Category 5 is reported separately
or in the all-category columns because several systems score adversarial
refusal with a different rule.

## Ranked Results: Cat1-4

| Rank | Method | Run | N | F1 | BLEU-1 | All-cat F1 | All-cat BLEU-1 | Result artifact |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | SimpleMem | Qwen3-8B full | 1540 | 0.3721 | 0.3078 | 0.4851 | 0.4350 | `runs/simplemem/full_20260427_183242/result.json`; tracked copy in `reproductions/successful/SimpleMem/2026-04-27_qwen3-8b_locomo10-full_16mem-30qa/result.json` |
| 2 | HiGMem | Qwen2.5-3B full | 1540 | 0.3402 | 0.2792 | 0.4189 | 0.3678 | `baseline/HiGMem/fphm_runs/.../aggregated_results.json`; tracked copy in `reproductions/successful/HiGMem/2026-04-28_qwen2.5-3b_locomo10-full_w10/aggregated_results.json` |
| 3 | xMemory | Qwen2.5-3B full | 1540 | 0.3390 | 0.2816 | N/A | N/A | `reproductions/results_snapshot_20260508/xmemory_qwen25_3b_metrics.json` |
| 4 | MAGMA | Qwen2.5-3B full | 1540 | 0.2698 | 0.2278 | 0.4250 | 0.3926 | `reproductions/results_snapshot_20260508/magma_qwen25_3b_aggregate_samples_0_9.json` |
| 5 | MemMachine | Qwen2.5-3B full | 1540 | 0.2090 | 0.1397 | 0.1798 | 0.1183 | `reproductions/results_snapshot_20260508/memmachine_qwen25_3b_text_metrics_all_categories.json` |
| 6 | SimpleMem | Qwen2.5-3B full | 1540 | 0.1813 | 0.1469 | 0.3519 | 0.3248 | `reproductions/successful/SimpleMem/2026-05-01_qwen2.5-3b_locomo10-full_20qa/result.json` |
| 7 | MemGAS | Qwen2.5-3B full | 1540 | 0.1794 | 0.1226 | 0.1392 | 0.0951 | `reproductions/results_snapshot_20260508/memgas_qwen25_3b_normalized_metrics.json` |
| 8 | Omni-SimpleMem | Qwen2.5-3B full | 1540 | 0.1778 | 0.1449 | 0.1379 | 0.1123 | `reproductions/results_snapshot_20260508/omnisimplemem_qwen25_3b_normalized_metrics.json` |

## Judge-Only Results

| Method | Run | Scope | N | LLM Judge / Accuracy | Text F1 | Artifact |
|---|---|---|---:|---:|---:|---|
| LightMem | Qwen2.5-3B structmem | cat1-4 | 1540 | 0.2058 | N/A | `reproductions/results_snapshot_20260508/lightmem_qwen25_3b_cat1_4_summary.json` |
| LightMem | Qwen2.5-3B structmem | cat5 | 446 | 0.1256 | N/A | `reproductions/results_snapshot_20260508/lightmem_qwen25_3b_cat5_summary.json` |
| MemMachine | Qwen2.5-3B full | cat1-4 | 1540 | 0.7286 | 0.2090 | `reproductions/results_snapshot_20260508/memmachine_qwen25_3b_cat1_4_final_score.result` |
| MemMachine | Qwen2.5-3B full | cat5 | 446 | 0.1951 | 0.0789 | `reproductions/results_snapshot_20260508/memmachine_qwen25_3b_cat5_final_score.result` |

## Notes

- Raw `runs/`, model weights, vector databases, and long logs are intentionally
  not tracked in Git because they include large artifacts and are ignored by
  `.gitignore`.
- This snapshot tracks small metric/result files needed to audit the final
  scores. Large prediction logs remain on the server under `runs/`.
- MAGMA all-category F1 is high because category 5 is very strong. For method
  comparison across systems, prefer the cat1-4 ranking above.
- Omni-SimpleMem has two relevant F1 conventions: its internal evaluator gives
  category 5 credit for refusal; the normalized text metric treats category 5
  as token overlap. The table uses cat1-4 normalized token F1 for comparability.
