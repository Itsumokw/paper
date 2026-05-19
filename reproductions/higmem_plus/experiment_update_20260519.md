# HiGMemPlus Experiment Update 2026-05-19

## LoCoMo Main Full

| Dataset | Model | Method | Count | F1 | BLEU-1 | Judge | Support | Drill |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | Qwen3-3B | HiGMem baseline | 1986 | 0.438958 | 0.397167 | 0.433535 | 0.687311 | 0.000000 |
| LoCoMo | Qwen3-3B | Evidence Frame Routing | 1986 | 0.432405 | 0.387381 | 0.419436 | 0.720544 | 0.348439 |
| LoCoMo | Qwen3-8B | HiGMem baseline | 1986 | 0.482777 | 0.440543 | 0.480363 | 0.723565 | 0.000000 |
| LoCoMo | Qwen3-8B | Evidence Frame Routing | 1986 | 0.466727 | 0.423000 | 0.449144 | 0.747231 | 0.348439 |

## Multilingual Full, Improved Method

The multilingual-48 memory checkpoints are the baseline checkpoints; the table below evaluates Evidence Frame Routing over those checkpoints.

| Dataset | Model | Count | F1 | BLEU-1 | Judge | Support | Drill |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PerLTQA | Qwen3-8B | 320 | 0.506495 | 0.258355 | 0.515625 | 0.509375 | 0.937500 |
| OPELA | Qwen3-8B | 200 | 0.512388 | 0.096334 | 0.535000 | 0.725000 | 0.900000 |
| JLongChat | Qwen3-8B | 200 | 0.431664 | 0.131641 | 0.480000 | 0.750000 | 0.900000 |
| deL1L2IM | Qwen3-8B | 180 | 0.538025 | 0.494989 | 0.494444 | 0.661111 | 0.900000 |

## LongDialQA 10%, Improved Method

| Dataset | Model | Method | Count | Accuracy | Strict Acc | Parse | Mean Token F1 | Evidence Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LongDialQA 10% | Qwen3-8B | Evidence Frame Routing, cached 3B base retrieval | 335 | 0.391045 | 0.391045 | 1.000000 | 0.394795 | 0.609375 |

Note: `longdialqa_10pct_8b_evidence_frame_cached3bbase_fast` is excluded because the 8-worker run overloaded the 8B vLLM server, producing API connection failures and invalid empty predictions. The stable result above uses the same cached base retrieval with `answer_workers=3`.
