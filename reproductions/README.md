# Paper Reproduction Registry

This directory is the single entry point for paper reproduction records. Keep raw source checkouts, model weights, datasets, and active run outputs in their existing top-level locations unless a user explicitly approves moving or copying them.

## Repository Map

- `baseline/`: paper codebases and reference papers. Current main codebase: `baseline/SimpleMem`.
- `baseline/papers/`: downloaded paper PDFs, including `SimpleMem_2601.02553v3.pdf`.
- `datasets/`: benchmark or evaluation data. Current LoCoMo subset: `datasets/locomo/data/locomo10.json`.
- `models/`: local model weights, including `Qwen3-8B` and `Qwen2.5-3B-Instruct-clean`.
- `runs/`: execution outputs. Current SimpleMem runs are under `runs/simplemem/`.
- `scripts/`: reusable orchestration or merge scripts, including SimpleMem batch and result merge helpers.
- `research/`: literature tracking and follow-up notes.

## Standard Layout

Use one directory per paper and status:

```text
reproductions/
  successful/
    <PaperName>/
      <date>_<model>_<dataset>_<run-tag>/
        README.md
        result.json
        run.log
        command.txt
        config_redacted.py
        meta.txt
  planned/
    <PaperName>/
      README.md
      commands.md
      configs/
      logs/
      results/
      notes/
```

For large logs or outputs, prefer storing relative pointers to `runs/` unless a copy is explicitly needed for preservation. Do not edit archived `result.json` files in place. If metrics need to be recomputed, write a derived file next to it, for example `result_with_recomputed_bertscore.json`.

Suggested run naming:

```text
YYYY-MM-DD_<model-or-method>_<dataset>_<short-purpose>
```

Each run record should include:

- source code path and commit hash
- dataset path and version
- model path or model identifier
- exact command
- environment notes, including Python path and key package versions when relevant
- output path for `result.json`, logs, config snapshots, and exit status
- metric summary and known caveats

## Current Successful Runs

- `successful/SimpleMem/2026-04-27_qwen3-8b_locomo10-full_16mem-30qa`
  - source run: `runs/simplemem/full_20260427_183242`
  - source repo: `baseline/SimpleMem`
  - source commit: `94ef7d76786af96878dea6e87ea2c7f5eaeae168`
  - dataset: `datasets/locomo/data/locomo10.json`
  - summary: 10 samples, 1986 questions, average total time 38.897s
  - main metrics: F1 0.4851, ROUGE-L F 0.4850, SBERT similarity 0.6492
  - caveat: BERTScore emitted runtime errors in the log, so BERT metrics should not be treated as reliable for this run.

## Current Prepared Runs

- `baseline/HiGMem`
  - planned paper: HiGMem
  - planned model: `Qwen/Qwen2.5-3B-Instruct`
  - guide: `docs/higmem_reproduction_plan.md`
  - smoke script: `scripts/run_higmem_qwen25_3b_smoke.sh`
  - full script: `scripts/run_higmem_qwen25_3b_full.sh`

## Minimal Workflow For New Papers

1. Create `reproductions/planned/<PaperName>/README.md` before running.
2. Register source paths, dataset paths, model paths, and paper PDF path.
3. Put exact setup and run commands in `commands.md`.
4. Save redacted configs under `configs/`.
5. Save compact result summaries under `results/`; copy full `result.json` only when approved.
6. Link large raw logs from `runs/` or copy selected logs under `logs/` when approved.
7. Promote a completed run to `successful/<PaperName>/<run-id>/` only after confirming the result is final.
