# multilingual LoCoMo-style 48 failure/repair log

- run_root: `/home/stu0032/paper/runs/multilingual_locomo_style_repaired_48/20260513_123300`
- started_at: `2026-05-13 12:34:45 CST`
- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## qwen25_3b perltqa full_context failed at 2026-05-13 12:39:51 CST

- exit_code: `143`
- run_log: `/home/stu0032/paper/runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen25_3b/perltqa/full_context_fixed/run.log`
- reason: command failed; inspect the run log tail and rerun the same task after repair.
- repair: pending

## resume at 2026-05-13 12:41:38 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## judge protocol repair at 2026-05-13 13:06:20 CST

- finding: official LoCoMo QA code reports token F1; local LightMem LoCoMo evaluation uses `locomo-judge` with binary `CORRECT`/`WRONG` accuracy for LLM-as-a-judge.
- repair: changed `scripts/compute_locomo_llm_judge_metrics.py` default protocol to `locomo_binary`, matching LightMem's `locomo-judge` prompt and binary score.
- cache guard: judge cache keys now include protocol/version; `scripts/check_multilingual_locomo_style_48.py` requires `judge.protocol == locomo_binary`.
- rerun: qwen25_3b Full Context judge files for PerLTQA, OPELA, JLongChat, and deL1L2IM were recomputed with `locomo_binary`; all four passed audit with zero judge errors.

## A-MEM token-cap repair at 2026-05-13 13:17:33 CST

- finding: A-MEM memory-construction prompts defaulted to 8192 output tokens, causing local Qwen2.5-3B to run into length-limited completions before any shard finished.
- repair: capped non-QA A-MEM structured outputs to short-format budgets (`keyword=64`, `evolution=128`, `strengthen=256`, `analyze/memory/update=512`, `retrieval=512`) while keeping QA at `1024`.
- rerun: interrupted the slow PerLTQA A-MEM attempt before merge, then resumed the same run root; Full Context tasks were skipped as complete and A-MEM restarted with the new caps recorded in `amem_official/command.env`.

## resume at 2026-05-13 13:17:33 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## resume at 2026-05-13 16:14:14 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## SimpleMem JSON-shape repair at 2026-05-13 17:18:00 CST

- finding: PerLTQA SimpleMem first hit length-limited memory extraction with `MULTI_STYLE_SIMPLEMEM_MAX_OUTPUT_TOKENS=1024`; after increasing the cap, Qwen sometimes returned a top-level JSON array where SimpleMem expected a JSON object.
- symptom: repeated question failures with `'list' object has no attribute 'get'` in `baseline/SimpleMem/core/hybrid_retriever.py`.
- repair: normalized LLM JSON outputs in SimpleMem retrieval planning/reflection so object-shaped prompts tolerate top-level arrays and malformed list fields without crashing.
- validation: `.venv/bin/python -m py_compile baseline/SimpleMem/core/hybrid_retriever.py` passed.
- rerun: resume same run root with `MULTI_STYLE_SIMPLEMEM_MAX_OUTPUT_TOKENS=16384` and `SIMPLEMEM_FAIL_ON_FAILED_ANSWERS=1` so failed-answer runs cannot be counted as valid.

## SimpleMem query-analysis array repair at 2026-05-13 17:32:00 CST

- finding: after the crash fix, Sample 0 progressed into QA but `_analyze_query` still retried three times when Qwen returned a top-level JSON array of keywords.
- impact: non-fatal, but it wasted judge/model calls and slowed every SimpleMem dataset.
- repair: treat top-level JSON arrays from query analysis as keyword lists instead of retrying/falling back.
- validation: `.venv/bin/python -m py_compile baseline/SimpleMem/core/hybrid_retriever.py` passed.
- rerun: interrupted the partial PerLTQA SimpleMem attempt before metrics and restarted from scratch in the same output directory.

## resume at 2026-05-13 16:30:08 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## resume at 2026-05-13 16:38:12 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## qwen25_3b opela simplemem failed at 2026-05-13 18:13:35 CST

- exit_code: `143`
- run_log: `/home/stu0032/paper/runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen25_3b/opela/simplemem/run.wrapper.log`
- reason: intentionally interrupted after a sample-1 memory extraction window hit the 16384-token generation limit and began an expensive retry loop; the partial run had no normalized predictions, metrics, or judge file, so it was not countable.
- repair: added conservative truncated-JSON salvage in `baseline/SimpleMem/core/memory_builder.py` so complete `entries` objects from a length-limited response are retained and only the final incomplete object is discarded.
- validation: `.venv/bin/python -m py_compile baseline/SimpleMem/core/memory_builder.py` passed.
- rerun: resume the same run root; already complete tasks should be skipped and `qwen25_3b/opela/simplemem` restarted with the repair loaded.

## resume at 2026-05-13 18:14:03 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## qwen25_3b jlongchat simplemem failed at 2026-05-13 20:02:24 CST

- exit_code: `143`
- run_log: `/home/stu0032/paper/runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen25_3b/jlongchat/simplemem/run.wrapper.log`
- reason: intentionally interrupted after a query-analysis JSON response inherited the 16384-token SimpleMem build cap and spent minutes in length-limited retries; the partial run had no normalized predictions, metrics, or judge file, so it was not countable.
- repair: added an optional per-call `max_tokens` parameter to `baseline/SimpleMem/utils/llm_client.py`; capped SimpleMem retrieval/planning/reflection JSON calls in `baseline/SimpleMem/core/hybrid_retriever.py` to `RETRIEVAL_OUTPUT_TOKENS` defaulting to 512 while keeping memory build output at 16384.
- validation: `.venv/bin/python -m py_compile baseline/SimpleMem/utils/llm_client.py baseline/SimpleMem/core/hybrid_retriever.py` passed.
- rerun: resume the same run root; already complete tasks should be skipped and `qwen25_3b/jlongchat/simplemem` restarted with the repair loaded.

## resume at 2026-05-13 20:03:56 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## HiGMem JSON-shape repair at 2026-05-13 22:42:10 CST

- finding: qwen25_3b/perltqa/higmem sample 0 crashed during memory build with `'list' object has no attribute 'get'`; log inspection showed `_get_llm_json_response` salvaged a truncated multi-field object response by extracting a JSON array and returning it as the top-level response.
- impact: the multi-sample HiGMem runner would otherwise continue and later produce incomplete predictions, so the partial run was intentionally terminated before metrics.
- repair: `baseline/HiGMem/fphm_core.py` now only returns dicts for object schemas; extracted arrays are accepted only for single-array-field schemas and otherwise fall back to schema-shaped empty values.
- guard: `baseline/HiGMem/run_fphm_evaluation.py` now prints traceback for sample failures and fails the full task if any sample fails, preventing partial HiGMem outputs from being counted.
- validation: `.venv/bin/python -m py_compile baseline/HiGMem/fphm_core.py baseline/HiGMem/run_fphm_evaluation.py` passed; a small JSON-shape regression check confirmed multi-field object schemas no longer return lists.
- rerun: resume the same run root; already complete tasks should be skipped and `qwen25_3b/perltqa/higmem` restarted with the repair loaded.

## resume at 2026-05-13 22:42:34 CST

- scope_models: `qwen25_3b qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`

## qwen3_8b del1l2im full_context failed at 2026-05-14 05:59:33 CST

- exit_code: `1`
- run_log: `/home/stu0032/paper/runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen3_8b/del1l2im/full_context_fixed/run.log`
- reason: command failed; inspect the run log tail and rerun the same task after repair.
- repair: pending

## resume at 2026-05-14 06:20:29 CST

- scope_models: `qwen3_8b`
- scope_datasets: `perltqa opela jlongchat del1l2im`
- scope_methods: `full_context amem mem0 simplemem higmem memgas`
