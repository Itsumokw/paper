# Goal: Build a Source-faithful LoCoMo-style Multilingual Eval Set

## Objective

Construct exactly one final LoCoMo-style inference evaluation version for each source dataset:

- `PerLTQA-LoCoMo-style-eval`
- `OPELA-LoCoMo-style-eval`
- `JLongChat-LoCoMo-style-eval`
- `deL1L2IM-LoCoMo-style-eval`

Optionally merge the four source-specific eval sets into:

- `multilingual_locomo_style_eval.json`

The final benchmark must be LoCoMo-loader compatible, source-faithful, and suitable for long-term memory evaluation. It must not claim to be LoCoMo-equivalent in data generation process or naturalness.

## Non-negotiable Rules

1. Produce only one final eval version per source dataset.
2. Keep only one final QA set: `locomo_style_main`.
3. Do not create train/dev/test splits. Mark all samples as `split: "eval"`.
4. Do not tune prompts, retrieval top-k, chunking, memory compression, context truncation, or cat5 refusal rules on this benchmark.
5. All baseline prompt, retrieval, chunking, compression, and refusal settings must be fixed from LoCoMo or a pre-declared configuration.
6. Primary eval JSON must contain only fields required by LoCoMo-style loaders.
7. Put provenance, fact ledger, hashes, answer-fact tracing, and audit details into sidecar files.
8. Main model input should use conversation turns only. `observation`, `session_summary`, and `event_summary` must not enter the default context unless a separate summary-memory setting is explicitly reported.
9. Preserve original source text exactly except minimal normalization.
10. Synthetic content must never overwrite, rewrite, translate, reorder, split, merge, or be inserted inside original turns.
11. Every answer-critical fact in final QA must trace back to an original turn, memory, event, persona, or fact-ledger entry.
12. If any core entity, event, preference, relationship, or temporal change in an answer cannot trace back to the original fact ledger, delete that QA.
13. Dataset construction scripts must not silently call local or remote model services to generate dialogue, QA, summaries, or evidence.
14. A tiny chat request is allowed only as an optional service-readiness preflight for later baseline experiments. It must not contribute text to the dataset.

## Output Files

Use a two-layer output design.

### Primary Eval JSON

The primary eval JSON is the file used by LoCoMo-style baseline loaders. It should contain only loader-facing fields and should not include extra fields that may change loader behavior.

Required top-level sample fields:

- `sample_id`
- `source_dataset`
- `language`
- `split`
- `conversation`
- `observation`
- `session_summary`
- `event_summary`
- `qa`

Required conversation fields:

- `speaker_a`
- `speaker_b`
- `session_i_date_time`
- `session_i`
- turn-level `speaker`
- turn-level `dia_id`
- turn-level `text`

Required answerable QA fields:

- `question`
- `answer`
- `category`
- `evidence`

Required cat5 adversarial QA fields:

- `question`
- `category`
- `evidence`
- `adversarial_answer`

Evidence IDs must use LoCoMo-style turn IDs:

```text
D{session_id}:{turn_id}
```

Example:

```text
D2:7
```

### Sidecar Files

Use sidecar files for all construction and audit metadata:

- `*_fact_ledger.jsonl`
- `*_provenance.jsonl`
- `*_qa_audit.jsonl`
- `*_hash_check.jsonl`
- `*_construction_report.md`

Sidecar metadata should include:

- `source_dataset`
- `source_file`
- `source_record_id`
- `source_turn_id`
- `raw_text_hash`
- `source_origin`
- `source_fact_id`
- `answer_facts`
- `evidence_detail`
- `negative_evidence`
- `adversarial_reason`
- `whether_cross_session`
- `difficulty`
- verifier decisions
- human audit status

## Source Fidelity Contract

For every original turn:

- preserve text exactly;
- preserve source order;
- preserve session order;
- store `raw_text_hash`;
- store `source_turn_id`;
- mark provenance as `original_turn`.

Allowed text normalization:

- remove illegal control characters;
- normalize line breaks;
- apply necessary Unicode normalization.

Forbidden operations:

- translation;
- polishing;
- paraphrasing;
- inserting synthetic turns inside an original session;
- merging multiple raw turns;
- splitting one raw turn;
- treating summaries as original dialogue;
- reordering source sessions to make the story look more LoCoMo-like.

Original sessions are immutable blocks. Synthetic turns may only appear in separate synthetic sessions or bridge sessions, never inside an original session block.

## Provenance Labels

Use only explicit provenance labels:

- `original_turn`
- `original_memory`
- `memory_anchor_turn`
- `synthetic_bridge_turn`
- `synthetic_continuation_turn`
- `llm_summary`

Do not use ambiguous labels such as `source-derived`.

## Model Input Policy

Default evaluation input:

- include chronological `conversation` turns;
- exclude `observation`;
- exclude `session_summary`;
- exclude `event_summary`;
- exclude sidecar metadata.

If a method intentionally uses summaries or observations as memory, report it as a separate summary-memory setting. Do not mix summary-visible and conversation-only methods in the same main table.

## Generation and Preflight Policy

The benchmark construction path is no-model by default:

- conversion, hashing, schema validation, and loader smoke tests must run without vLLM, OpenAI-compatible endpoints, or external APIs;
- generated or synthetic dataset content must be explicit, reviewed, and provenance-marked;
- scripts must not hide model calls behind build, validate, or loader-smoke commands;
- if assistant-authored expansion or QA rewriting is used, it must be treated as a declared generation step and then validated against the source fact ledger.

Allowed model/service check:

- a minimal chat completion may be sent only to verify that a baseline experiment service is reachable;
- this check should be recorded as `service_preflight_only`;
- its response must not be copied into primary eval JSON or sidecar facts;
- failing or skipping this check must not block dataset construction validation.

## PlanModes

### PlanMode A: Raw-preserving Native Conversion

Use for:

- `deL1L2IM`
- Japanese LAC
- already multi-session JMSC samples

Rules:

- no LLM dialogue expansion;
- synthetic turns = 0;
- original turns only;
- QA evidence only from `original_turn`;
- preserve original message order;
- do not generate new learning events or personal facts.

### PlanMode B: Light Session Completion

Use for:

- short JMSC samples with around 3 sessions

Rules:

- lightly complete to 5-8 sessions;
- synthetic turns <= 40%;
- each synthetic session must be anchored to at least one original persona, event, or turn;
- do not insert synthetic turns into original sessions;
- answer facts must trace back to original facts.

### PlanMode C: Persona-grounded Continuation

Use for:

- `OPELA`

Rules:

- preserve Korean original turns;
- use pause metadata and persona/user summaries as anchors;
- synthetic turns <= 60%;
- do not invent major identity, family, occupation, trauma, or life-event facts;
- empathy and self-disclosure labels may be auxiliary metadata, but not sole answer evidence;
- every QA must include at least one original evidence turn or original fact anchor.

### PlanMode D: Memory-anchored Dialogization

Use for:

- `PerLTQA`

Rules:

- build a fact ledger from profile, relationships, events, dialogues, and original QA;
- convert memory facts into `memory_anchor_turn`;
- use synthetic bridge turns only for natural dialogue flow;
- do not mark `memory_anchor_turn` as `original_turn`;
- describe PerLTQA as a memory-anchored dialogue-style eval, not a naturally occurring multi-session dialogue corpus;
- answer facts must 100% trace back to original fact-ledger entries.

PerLTQA-specific reporting must include:

- original-turn evidence ratio;
- memory-anchor evidence ratio;
- synthetic-bridge-turn ratio;
- QA percentage whose answer facts are fully backed by original facts.

## Per-source Construction Requirements

### PerLTQA-LoCoMo-style-eval

Pipeline:

1. Extract `profile`, `social_relationship`, `events`, `dialogues`, and original QA.
2. Build a fact ledger with `fact_id`, `source_type`, `source_text`, and `source_id`.
3. Convert selected memory facts into `memory_anchor_turn`.
4. Add conservative `synthetic_bridge_turn` turns only for dialogue flow.
5. Generate LoCoMo-style QA from frozen conversation and fact ledger.
6. Delete QA if any answer fact lacks a source `fact_id`.

Original PerLTQA QA should not enter final eval directly. It can be used to check whether original memory facts survive conversion, or rewritten into LoCoMo-style QA only after new evidence is assigned.

### OPELA-LoCoMo-style-eval

Pipeline:

1. Preserve `persona_text_all` and `user_text_all` as original turns.
2. Use `pause_count`, `pause_hour`, and `total_minutes` only as session-gap hints.
3. Use `persona_summary` and `user_summary` as anchors, not as sole answer evidence.
4. Add persona-grounded continuation sessions only when needed.
5. Generate QA about persona consistency, user preferences, emotional state changes, relationship development, topic recurrence, and memory of earlier mentions.

Do not exaggerate pause metadata into months of timeline unless the source supports it.

### JLongChat-LoCoMo-style-eval

Pipeline:

1. For LAC, map `room` to sample and `dayid` to session.
2. For JMSC, map `pair_id` to sample, `sid` to session, and `tid` to turn.
3. Use the persona file as persona or memory anchors.
4. Apply PlanMode A to already multi-session samples.
5. Apply PlanMode B only to short samples that need minimal session completion.

Rules:

- do not translate Japanese;
- do not polish Japanese;
- do not change `dayid` or `sid` order;
- do not treat persona summaries as original utterances.

### deL1L2IM-LoCoMo-style-eval

Pipeline:

1. Parse TEI XML messages into sessions and turns.
2. Map dyads to samples.
3. Map native speaker and L2 learner roles to `speaker_a` and `speaker_b`.
4. Generate QA only from original messages.

Rules:

- no expansion;
- no synthetic sessions;
- no splitting longest XML to inflate sample count;
- no new learning events;
- QA evidence only from `original_turn`.

## QA Generation Rules

Generate only one final QA set: `locomo_style_main`.

Recommended QA volume:

- 20-40 QA per sample.

QA generation order:

1. Select evidence turns.
2. Extract answer-critical facts.
3. Write the answer.
4. Write the question.
5. Assign LoCoMo category.
6. Run source-entailment verifier.
7. Delete QA if any answer-critical fact lacks original support.

Categories:

- cat1: single-hop factual
- cat2: multi-hop
- cat3: temporal
- cat4: commonsense / open-domain reasoning
- cat5: adversarial / unanswerable

For cat1-4:

- use `evidence`;
- every evidence ID must point to a valid `dia_id`;
- answer must be supported by evidence;
- cross-session QA should cite evidence from multiple sessions.

For cat5:

- set `evidence: []`;
- omit ordinary `answer`;
- include `adversarial_answer`;
- use `negative_evidence`;
- include `adversarial_reason`, such as `time_swap`, `entity_swap`, or `unsupported_fact`;
- report refusal accuracy and unsupported claim rate instead of F1, BLEU, or ROUGE.

## Verifier Requirements

The verifier must judge source entailment, not plausibility.

For every answer fact, check:

1. Does it have a `source_fact_id`?
2. Does the `source_fact_id` come from original source material?
3. Does the original source fact entail the answer fact?
4. Is the evidence turn ID valid?
5. Is the evidence sufficient for the category?

Delete the QA if any answer-critical fact fails these checks.

## Validation Requirements

Before release:

- run a no-model LoCoMo-style loader smoke test;
- optionally run a tiny service-readiness chat preflight before baseline experiments;
- verify original turn hashes;
- verify original session order;
- verify every evidence ID exists;
- verify answer facts against source fact ledger;
- verify no synthetic-only answer-critical QA enters the main set;
- verify cat5 has no gold answer and no ordinary evidence;
- audit cat2, cat4, and cat5 carefully.

Human audit minimum:

- fully review at least 2 complete samples per source dataset;
- review at least 30% of cat2, cat4, and cat5 QA;
- fully review all synthetic-adjacent QA;
- review at least 50% of PerLTQA memory-anchor QA;
- perform 100% hash checking for deL1L2IM, LAC, and JMSC original turns.

## Long-memory Diagnostic

Run recent-session diagnostic baselines:

- last session only;
- last 3 sessions only;
- full conversation.

If last-session or last-3-session baselines perform close to full conversation, revise the QA because the long-term memory demand is too weak.

## Final Evaluation

Run fixed baseline settings only:

- Full Context
- A-MEM
- Mem0
- SimpleMem
- HiGMem

MemGAS should enter only if clean metrics are available.

Final fixed-baseline acceptance is based on per-method prediction JSONL files generated against the audited multilingual eval file with the predeclared Qwen3-8B settings. Old LoCoMo10/Qwen2.5 runner defaults may only be used as implementation adapters with explicit overrides; they must not define the final dataset, model, counts, or accepted metrics.

Report results by:

- source dataset;
- language;
- category;
- cross-session status;
- evidence provenance.

Main score:

- cat1-4 answerable QA.

Separate score:

- cat5 refusal / adversarial behavior.

## Execution Priority

Follow this priority order when tradeoffs arise:

1. Source fidelity.
2. LoCoMo-loader compatibility.
3. Answer-fact traceability.
4. Multi-session long-memory measurability.
5. QA quality and category balance.
6. Baseline evaluation and reporting.

Do not increase dataset size by weakening source fidelity or evidence quality.
