# HiGMemPlus Final Results Snapshot - 2026-05-17

This snapshot includes only completed final results. In-progress MemGAS and
HiGMem full DialSim runs are intentionally excluded until their full artifact
directories finish.

## LongDialQA/DialSim FullContext Full Reproduction

- Result label: full benchmark reproduction
- Completed rows: 3277
- Artifact directory: `reproductions/higmem_plus/baselines_longdialqa_full/full_context`
- Subset/full manifest: `datasets/subsets/longdialqa_full_seed20260517_manifest.json`
- Dataset SHA256: `98aa8e3989a647231776d44b52bc40d647361506006a45571b0dbc806b9b37e2`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Failures: 0

| Split | Count | Accuracy | Strict Acc | Mean F1 | Avg K | Approx Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 3277 | 0.227647 | 0.227647 | 0.233252 | 9445.77 | 14892.82 |
| Friends | 783 | 0.227331 | 0.227331 | 0.228967 | 9350.87 | 14893.60 |
| The Big Bang Theory | 788 | 0.220812 | 0.220812 | 0.228263 | 10109.39 | 14919.88 |
| The Office | 1706 | 0.230950 | 0.230950 | 0.237523 | 9182.79 | 14879.97 |

## Uploaded Compact Artifacts

- `command.env`
- `failure_retry_logs.jsonl`
- `metrics.json`
- `metrics_compact.json`
- `run.log`
- `stats.json`
- `summary.md`

Large local artifacts are not included in this compact result commit:

- `raw_predictions.jsonl`: 83,168,222 bytes, SHA256 `ebc98e33847b55c2c191a083714975c9a37f12e50f393042dc29f81c3d0d60d1`
- `retrieved_context.jsonl`: 494,012,562 bytes, SHA256 `764dbc11ac6aba15cb1baa10f819aea37483c4a482563499097cfd5e8349d02d`

## LongDialQA/DialSim A-Mem Full Reproduction

- Result label: full benchmark reproduction
- Completed rows: 3277
- Artifact directory: `reproductions/higmem_plus/baselines_longdialqa_full/a_mem`
- Subset/full manifest: `datasets/subsets/longdialqa_full_seed20260517_manifest.json`
- Dataset SHA256: `98aa8e3989a647231776d44b52bc40d647361506006a45571b0dbc806b9b37e2`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Failures: 0

| Split | Count | Accuracy | Strict Acc | Mean F1 | Avg K | Approx Tokens | Evidence Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 3277 | 0.239854 | 0.239854 | 0.245333 | 10.00 | 835.81 | 0.527038 |
| Friends | 783 | 0.237548 | 0.237548 | 0.239184 | 10.00 | 804.19 | 0.478333 |
| The Big Bang Theory | 788 | 0.228426 | 0.228426 | 0.235370 | 9.99 | 823.42 | 0.516234 |
| The Office | 1706 | 0.246190 | 0.246190 | 0.252758 | 10.00 | 856.04 | 0.553892 |

Uploaded compact A-Mem artifacts:

- `command.env`
- `failure_retry_logs.jsonl`
- `metrics.json`
- `metrics_compact.json`
- `run.log`
- `stats.json`
- `summary.md`

Large local A-Mem artifacts are not included in this compact result commit:

- `raw_predictions.jsonl`: 7,221,523 bytes, SHA256 `3096a824ec962b946599a668b0f1bc7f584b32bdb4d0c9e139bb2b9e9847e867`
- `retrieved_context.jsonl`: 43,523,986 bytes, SHA256 `cda3099e523c4bdf463d0e6fecaebe228fda7f4b903d0b6c833e9667f72f2043`

## Completed HiGMemPlus Engineering Smoke

These are smoke-test checks only, not benchmark results.

| Method | Count | Accuracy | Avg K | Approx Tokens | Artifact |
|---|---:|---:|---:|---:|---|
| Evidence-Component HiGMem | 3 | 0.3333 | 10.0 | 1154.0 | `reproductions/higmem_plus/smoke_higmem_plus_min/evidence_component/metrics.json` |
| Repairable-Episode HiGMem | 3 | 0.3333 | 18.0 | 1383.7 | `reproductions/higmem_plus/smoke_higmem_plus_min/repairable_episode/metrics.json` |
| Adaptive-Routing HiGMem | 3 | 0.3333 | 21.7 | 2004.3 | `reproductions/higmem_plus/smoke_higmem_plus_min/adaptive_routing/metrics.json` |
