# Results Snapshot 2026-05-08

This directory contains small metric artifacts copied out of ignored local
`runs/` and external baseline directories so the reproduction state can be
pulled from Git without committing large raw logs, vector databases, or model
weights.

Included artifacts:

| Artifact | Source run |
|---|---|
| `xmemory_qwen25_3b_metrics.json` | `runs/xmemory/qwen25_3b_full/metrics.json` |
| `memmachine_qwen25_3b_text_metrics_all_categories.json` | `runs/memmachine/qwen25_3b_full/locomo_memmachine_text_metrics_qwen25_3b_full_all_categories.json` |
| `memgas_qwen25_3b_normalized_metrics.json` | `runs/2026_sota_memory_qwen25_3b_all/20260430_183900/memgas/normalized_metrics.json` |
| `omnisimplemem_qwen25_3b_normalized_metrics.json` | `runs/2026_sota_memory_qwen25_3b_all/20260430_183900/omnisimplemem/normalized_metrics.json` |
| `magma_qwen25_3b_aggregate_samples_0_9.json` | `baseline/MAGMA/results_Qwen/Qwen2_5_3B_Instruct/fixed_results_aggregate_samples_0_1_2_3_4_5_6_7_8_9.json` |
| `lightmem_qwen25_3b_cat1_4_summary.json` | `runs/lightmem/qwen25_3b_structmem_full/results_qa_eval_20260429_120459/summary.json` |
| `lightmem_qwen25_3b_cat5_summary.json` | `runs/lightmem/qwen25_3b_structmem_full/results_qa_eval_cat5_20260429_132929/summary.json` |
| `memmachine_qwen25_3b_cat1_4_final_score.result` | `baseline/MemMachine/evaluation/retrieval_agent/result/final_score/locomo_memmachine_qwen25_3b_full.result` |
| `memmachine_qwen25_3b_cat5_final_score.result` | `baseline/MemMachine/evaluation/retrieval_agent/result/final_score/locomo_memmachine_qwen25_3b_full_cat5.result` |

See `docs/reproduction_results_20260508.md` for the ranked table.
