# CODEX FIRST READ

This file compresses the main research discussion so future Codex sessions can resume quickly without rereading the whole chat history.

## Project goal

Primary thesis direction:

- Build a multilingual, heterogeneous long-term memory evaluation setting around LoCoMo-style QA/F1.
- Reuse and compare memory baselines such as `SimpleMem`, `A-Mem`, `Mem0`, and related methods.
- Study cross-dataset robustness, not just single-benchmark leaderboard gains.

The intended thesis story is:

- `LoCoMo` remains the English anchor benchmark.
- `PerLTQA` is the Chinese main extension set.
- `OPELA`, `Japanese Long-term Chat`, and `deL1L2IM` are multilingual expansion candidates after conversion to a LoCoMo-style format.

## Important conclusions already made

1. `PerLTQA` is a better Chinese extension target than `DuLeMon`.
   It is much closer to personal long-term memory QA and requires far less manual reconstruction.

2. We should not claim that cross-dataset score differences are caused only by language.
   The datasets differ in language, dialogue structure, session granularity, length, and QA style.
   Safer claim: we are evaluating cross-lingual and cross-structure robustness of memory methods.

3. `OPELA` should not be treated as a fully long-context benchmark without filtering.
   Most samples are short or medium length, so use a long subset or report turn-length buckets.

4. LLM-generated turn expansion is allowed only as controlled augmentation or stress testing.
   It must not replace the main evaluation set.

5. If we compare methods across datasets, compare relative behavior inside each dataset first.
   Example:
   `SimpleMem` vs `Mem0` vs `A-Mem` vs `Full Context` on the same dataset.
   Use cross-dataset comparisons only for robustness trends, not for direct language-only claims.

## Datasets currently kept in this repo

### 1. `datasets/locomo`

Role:
English anchor benchmark.

Why it matters:
It already contains long conversations, QA, and evidence annotations.

How it should be used:
Run baselines directly first. This is the calibration set for the whole pipeline.

### 2. `datasets/PerLTQA`

Role:
Chinese main extension benchmark.

Why it matters:
It already provides profile memory, social relationships, events, and QA.

How it should be used:
Convert to a LoCoMo-style unified format.
Do not describe it as a Chinese translation of LoCoMo. It is a different memory structure.

### 3. `datasets/OPELA`

Role:
Korean open-domain persona and empathy dialogue expansion set.

Why it matters:
Useful for Korean memory evaluation, but only after conversion.

Important caution:
Most samples are not long enough to serve as a strong long-context benchmark.
Prefer a filtered subset such as `turns >= 50` or length buckets.

### 4. `datasets/japanese-long-term-chat`

Role:
Japanese multi-session candidate set.

Why it matters:
It has cleaner session structure than OPELA.

Important caution:
`JMSC` is multi-session but not very long in turns per pair.
`LAC` has a few truly longer real chat rooms and may be more useful for long-memory stress.

### 5. `datasets/deL1L2IM`

Role:
German low-resource long-chat supplement.

Why it matters:
The per-file chat logs are actually long in turns, often hundreds of messages.

Important caution:
It is small and narrow in domain, so it should be presented as a supplement, not a main benchmark.

## Best research framing

Do not frame the thesis as:

- proving pure language effects, or
- showing that one language is harder than another.

Preferred framing:

- multilingual and heterogeneous long-term memory evaluation,
- cross-dataset robustness of memory methods,
- conversion of different long-memory resources into a unified LoCoMo-style QA/F1 setting,
- analysis of when existing methods stop generalizing.

## Fixed baseline set

The baseline set for the main LoCoMo-style evaluation is now fixed as:

- `Full Context`
- `A-Mem`
- `Mem0`
- `SimpleMem`
- `HiGMem`
- `MemGAS`

Treat this as the default comparison set before adding any extra method. The
purpose is to cover a long-context control, two widely used memory baselines,
one lightweight compression baseline, one hierarchical memory system, and one
recent accepted multi-granularity memory method.

Related methods also repeatedly discussed:

- `MAGMA`
- `xMemory`
- `MemoryOS`
- `ReadAgent`
- `LightMem`
- `AriadneMem`

## Paper survey status

The repo includes a screened collection of LoCoMo/F1/open-source/reproducible memory papers under:

- [locomo_f1_open_reproducible_2026/qualified_papers](/C:/Users/itsumo/Desktop/paper/locomo_f1_open_reproducible_2026/qualified_papers)

This line of work is already partially summarized into CSV files and reading-note tables.

Important survey finding:

- There are many long-term memory benchmarks.
- There are some bilingual or non-English memory resources.
- We did not identify a mature, widely adopted benchmark line that already does the exact same thing as our intended multilingual LoCoMo-style evaluation.

## Recommended next implementation steps

1. Define one unified JSON schema for all converted datasets.
2. Convert `LoCoMo` first and keep it as the reference implementation.
3. Convert `PerLTQA` next because it already has QA.
4. Convert `Japanese Long-term Chat` and `OPELA` after that, with manual or semi-automatic QA/evidence construction.
5. Use `deL1L2IM` as a smaller German supplement.
6. Run baseline comparisons inside each dataset before writing cross-dataset conclusions.

## Ground rules for future Codex sessions

- Read this file before changing dataset strategy or experiment framing.
- Prefer preserving datasets, baseline code, and reproduction outputs.
- Avoid deleting useful experiment scripts even if they look temporary.
- Be careful with nested `.git` directories inside downloaded datasets.
- Keep the repo organized around reproducibility and handoff, not just ad hoc local exploration.
