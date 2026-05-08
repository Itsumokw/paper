# SimpleMem Qwen2.5-3B LoCoMo10 Full Reproduction

Status: successful full run

Run identity:

- Paper/system: SimpleMem
- Dataset: LoCoMo10
- Model served by vLLM: `Qwen/Qwen2.5-3B-Instruct`
- Run directory copied from: `/home/stu0032/paper/runs/simplemem/qwen25_3b_full_20260501_180413`
- Result saved at: `2026-05-01 23:34:59`
- QA workers: `20`

Main metrics from `result.json`:

| Scope | N | F1 | BLEU-1 | ROUGE-L |
|---|---:|---:|---:|---:|
| Overall cat1-5 | 1986 | 0.3519 | 0.3248 | 0.3603 |
| Cat1-4 | 1540 | 0.1813 | 0.1469 | N/A |
| Cat1 | 282 | 0.1500 | 0.1106 | N/A |
| Cat2 | 321 | 0.1714 | 0.1208 | N/A |
| Cat3 | 96 | 0.1124 | 0.0975 | N/A |
| Cat4 | 841 | 0.2034 | 0.1746 | N/A |
| Cat5 | 446 | 0.9410 | 0.9393 | N/A |

Tracked files:

- `result.json`: final structured evaluation output
- `config_qwen25_3b.py`: local config snapshot with redacted/empty API key
