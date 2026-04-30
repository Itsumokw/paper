# SimpleMem LoCoMo10 Full Reproduction

Status: successful full run

Run identity:

- Paper/system: SimpleMem
- Dataset: LoCoMo10
- Model served by vLLM: `Qwen/Qwen3-8B`
- vLLM model path: `/home/stu0032/paper/models/Qwen3-8B`
- Run directory copied from: `/home/stu0032/paper/runs/simplemem/full_20260427_183242`
- Started at: `2026-04-27 18:32:42`
- Result saved at: `2026-04-27 23:10:09`
- Runtime: `4:37:27`
- Memory workers: `16`
- QA workers: `30`

Archived files:

- `result.json`: final structured evaluation output
- `run.log`: full console log
- `command.txt`: exact evaluation command
- `config_redacted.py`: copied config snapshot
- `meta.txt`: run metadata

Main metrics from `result.json`:

| Scope | n | EM | F1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | SBERT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 1986 | 27.69 | 48.51 | 48.50 | 43.50 | 29.06 | 41.75 | 64.92 |
| Cat 1 | 282 | 2.13 | 30.88 | 28.53 | 23.95 | 7.00 | 18.08 | 53.69 |
| Cat 2 | 321 | 4.05 | 39.87 | 39.05 | 28.44 | 8.05 | 21.26 | 67.65 |
| Cat 3 | 96 | 4.17 | 14.93 | 14.96 | 13.04 | 3.50 | 10.37 | 40.97 |
| Cat 4 | 841 | 16.53 | 40.87 | 41.89 | 35.98 | 16.62 | 37.11 | 58.02 |
| Cat 5 | 446 | 87.00 | 87.53 | 87.61 | 87.43 | 87.11 | 86.98 | 88.22 |

Non-adversarial Cat 1-4:

| Scope | n | EM | F1 | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | SBERT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cat 1-4 | 1540 | 10.52 | 37.21 | 37.18 | 30.78 | 12.25 | 28.65 | 58.17 |

Timing:

- Average retrieval time: `35.73s`
- Average answer time: `3.17s`
- Average total time: `38.90s`
- Average retrieved memory entries per question: `47.89`

Known caveat:

- BERTScore in this run is not valid. `bert_f1` was zero for `1981 / 1986`
  questions due to `Cannot copy out of meta tensor` during concurrent scoring.
  Recompute BERTScore from `answer` and `reference` in `result.json` if needed.

