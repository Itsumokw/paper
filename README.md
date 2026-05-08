# Paper Workspace

This repository is the working tree for our undergraduate thesis project on long-term memory evaluation for LLM agents.

Read [CODEX_FIRST_READ.md](/C:/Users/itsumo/Desktop/paper/CODEX_FIRST_READ.md) first before making research, dataset, or evaluation changes.

## What is here

- `baseline/`: local baselines and their supporting code.
- `datasets/`: downloaded benchmark and candidate multilingual datasets.
- `docs/`: short research notes, dataset summaries, prompts, and status docs.
- `locomo_f1_open_reproducible_2026/`: screened LoCoMo/F1/open-source paper collection and notes.
- `reproductions/`: reproduction outputs and result snapshots.
- `scripts/`: experiment runners and helper scripts.

## Current focus

We are not only reproducing LoCoMo results. The main research direction is to build a multilingual, heterogeneous long-term memory evaluation setting and then compare memory baselines such as `SimpleMem`, `A-Mem`, `Mem0`, and related methods under a unified LoCoMo-style QA/F1 protocol.
