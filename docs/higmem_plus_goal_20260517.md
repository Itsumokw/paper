Goal: Build and evaluate three macro-level HiGMem improvement directions on LoCoMo and LongDialQA/DialSim using fixed 50% subset experiments only.

Working directory:
- Prefer /home/stu0032/paper if it exists.
- Otherwise use the current repository root.

Research story:
HiGMem proves that event summaries can efficiently locate relevant memory, but event summaries are not always evidence-faithful. The next step is not more summarization, but evidence-aware and repairable hierarchical memory. We will test three macro directions:
1. Evidence-Component HiGMem
2. Repairable-Episode HiGMem
3. Adaptive-Routing HiGMem

These are not small tweaks. Each direction must have a clear motivation, implementation, ablation, and evaluation story.

Hard constraints:
- Never run full-dataset experiments.
- Only run deterministic 50% subset experiments.
- Run a 5% smoke subset first.
- If any script defaults to full data, stop and patch it to require --subset-manifest.
- Preserve original HiGMem as baseline.
- Put new code in a separate non-invasive package, e.g. baseline/HiGMemPlus.
- Do not destructively edit upstream baseline code.
- Do not use destructive git commands.
- Use identical model, decoding, answer prompt, and metric script across comparable methods.
- Do not tune prompts or hyperparameters separately per method.
- Every reported result must include artifact path, command.env, dataset hash, subset manifest, raw predictions, retrieved evidence, metrics.json, runtime/token stats, and logs.
- Label all results as “50% subset results”, never full benchmark results.

Phase 0: Baseline Reproduction on LongDialQA/DialSim First
Before implementing new variants, reproduce baseline performance on LongDialQA/DialSim 50% subset for:
1. FullContext / RawContext control
2. A-Mem
3. MemGAS
4. Original HiGMem
5. Optional BM25 Session Entire sanity control

Requirements:
- Treat DialSim as simulator/protocol and LongDialQA as dataset.
- Use official DialSim/LongDialQA data/code if not present locally.
- Build a normalized LongDialQA adapter.
- Report by show:
  Friends
  The Big Bang Theory
  The Office
- Metrics:
  multiple-choice accuracy if MC format is used;
  otherwise EM/F1.
  Also report Avg K, retrieved tokens, runtime, and evidence recall if evidence annotations exist.
- Preserve full allowed history up to each selected question. Do not delete memory history just because QA subset is 50%.
- Save baseline results under:
  reproductions/higmem_plus/baselines_longdialqa_50pct/

Phase 1: Repository and Data Audit
Read:
- CODEX_FIRST_READ.md
- docs/reproduction_results_20260508.md
- locomo_baseline_reproduction_acceptance.md
- existing scripts under scripts/
- reproductions/successful/HiGMem if present

Determine:
- Whether HiGMem source exists locally. If only artifacts exist, clone official HiGMem into a new clean directory without overwriting.
- Whether A-Mem and MemGAS source/runners exist locally. If not, fetch or adapt from existing local baselines.
- Whether LongDialQA/DialSim data exists locally. If not, fetch official data/code.
- What model endpoint is available, e.g. local vLLM OpenAI-compatible endpoint.

Write:
- docs/higmem_plus_audit.md

Phase 2: Fixed 50% Subset Construction
Create deterministic subset manifests before running any experiment.

Use seed:
- 20260517

LoCoMo subset:
- Select approximately 50% of QA from LoCoMo10.
- Preserve full conversation history for selected QA.
- Stratify by official categories:
  Cat1 = multi-hop
  Cat2 = temporal
  Cat3 = open-domain / commonsense
  Cat4 = single-hop
  Cat5 = adversarial / unanswerable
- Save:
  datasets/subsets/locomo10_50pct_seed20260517.json
  datasets/subsets/locomo10_50pct_seed20260517_manifest.json

LongDialQA/DialSim subset:
- Select approximately 50% of questions.
- Preserve all three shows.
- Stratify by show and question source/type if available:
  fan quiz
  TKG one-hop
  TKG two-hop
  answerable/unanswerable
- Preserve full allowed dialogue history up to each selected question.
- Do not leak future scenes/sessions.
- Save:
  datasets/subsets/longdialqa_50pct_seed20260517.json
  datasets/subsets/longdialqa_50pct_seed20260517_manifest.json

Also create 5% smoke manifests for both datasets.

Phase 3: Unified Evaluation Interface
Create a common runner:

python scripts/run_higmem_plus.py \
  --dataset locomo|longdialqa \
  --dataset-path ... \
  --subset-manifest ... \
  --method baseline_higmem|evidence_component|repairable_episode|adaptive_routing \
  --baseline fullcontext|amem|memgas|higmem \
  --model Qwen/Qwen2.5-3B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --output-dir runs/higmem_plus/...

Each run directory must contain:
- command.env
- dataset_manifest.json
- raw_predictions.jsonl
- retrieved_evidence.jsonl
- component_traces.jsonl if applicable
- graph_traces.jsonl if applicable
- repair_traces.jsonl if applicable
- episode_traces.jsonl if applicable
- route_traces.jsonl if applicable
- metrics.json
- summary.md
- run.log

Phase 4: Implement Three Macro Variants

Method 1: Evidence-Component HiGMem
This combines:
- Decoupled Evidence HiGMem
- Typed Evidence Graph HiGMem

Motivation:
HiGMem event summaries are efficient semantic anchors, but they can conflate multiple facts, weaken temporal anchors, and hide reasoning premises. Inspired by xMemory, MAGMA, GAM, and AriadneMem, add an evidence layer: atomic components with typed relations and source provenance.

Implementation:
- Add evidence components between Event and Turn.
- Component schema:
  component_id
  event_id
  text
  subject
  predicate/action
  object
  time_expr
  event_time if inferable
  mentioned_at
  source_turn_ids
  source_session_or_episode_id
  confidence
- Components must always point back to raw turns.
- Build typed edges between components:
  entity
  temporal
  causal
  preference/attribute
  social
  event-update
- Retrieval can enter through:
  event summary similarity
  component similarity
  typed relation/path retrieval
- Temporal questions should prefer temporal components/edges.
- Open-domain questions should retrieve premise components/paths.
- Save component and graph traces.

Ablations:
- baseline HiGMem
- + evidence components only
- + typed graph retrieval
- + source-turn verification

Method 2: Repairable-Episode HiGMem
This combines:
- Evidence-Gap Repair HiGMem
- Episodic Reconstruction HiGMem

Motivation:
A compressed memory system should know when retrieved evidence is insufficient. When summary/component evidence is missing time, subject, causal premise, or local context, the system should reconstruct the original local episode. This is generic: LoCoMo sessions and DialSim scenes/sessions are both local evidence episodes.

Implementation:
- Define generic Episode object:
  episode_id
  dataset
  session_or_scene_id
  chronological_order
  participants
  turns
  summary
  linked_events
  linked_components
- For LoCoMo:
  episode = session or coherent local session chunk.
- For LongDialQA/DialSim:
  episode = scene/session.
- After candidate retrieval, run an evidence sufficiency checker.
- Missing slots:
  missing_time
  missing_subject
  missing_object
  missing_relation
  missing_causal_premise
  missing_speaker
  missing_episode_context
- If insufficient, expand:
  component -> linked raw turns
  event -> linked turns
  episode -> full local episode under budget
- Save repair traces:
  repair_needed
  missing_slots
  expanded_sources
  final_context_before
  final_context_after

Ablations:
- baseline HiGMem
- + sufficiency checker only
- + linked raw-turn repair
- + full episode reconstruction under budget

Method 3: Adaptive-Routing HiGMem
This uses Method 1 and Method 2 as available tools.

Motivation:
Different questions need different memory granularity. Fixed k_event/k_turn retrieval is inefficient and brittle: single-hop can use compact components, temporal needs normalized temporal evidence, open-domain needs typed premise paths, and multi-hop/multi-party questions may need episode reconstruction. Inspired by BudgetMem and xMemory uncertainty expansion, route each query to the cheapest sufficient memory path.

Implementation:
- Build a router that predicts:
  question_type
  evidence_risk
  needed_layers
  retrieval_budget
- Routes:
  single-hop -> event/component + few turns
  temporal -> temporal component/edge + source verification
  open-domain -> typed premise path + sufficiency checker
  multi-hop -> diverse components + episode repair
  adversarial -> evidence absence check + conservative answer
  longdialogue/multi-party -> episode reconstruction + speaker/entity filter
  uncertain -> raw episode fallback
- Implement heuristic router first.
- Optional LLM router second if cheap.
- Oracle router using labels is allowed only as upper-bound ablation.
- Save route traces.

Ablations:
- fixed baseline HiGMem
- heuristic router
- heuristic router + repair
- oracle category router if labels exist

Phase 5: Smoke Tests
Run 5% smoke tests first:
- LoCoMo 5% subset including temporal and open-domain examples.
- LongDialQA/DialSim 5% subset across all three shows if feasible; otherwise at least one show.

Verify:
- no empty predictions
- baseline and methods use same final answer prompt
- subset manifests are honored
- evidence ids are saved
- component/graph/repair/episode/route traces are saved
- metrics script runs

Phase 6: 50% Subset Evaluation Only
Run on fixed 50% subsets.

First, LongDialQA/DialSim baseline reproduction:
- FullContext
- A-Mem
- MemGAS
- HiGMem
- optional BM25 Session Entire sanity control

Then run:
- baseline_higmem
- evidence_component
- repairable_episode
- adaptive_routing

Run both:
- LoCoMo 50%
- LongDialQA/DialSim 50%

Do not run complete datasets.

Phase 7: Reports
Write:
- docs/higmem_plus_experiment_report.md
- reproductions/higmem_plus/summary.json

Report must include:
1. Baseline LongDialQA/DialSim 50% results for FullContext, A-Mem, MemGAS, HiGMem.
2. LoCoMo 50% category results for baseline and three methods.
3. LongDialQA/DialSim 50% results by show.
4. Which method improves temporal most.
5. Which method improves open-domain most.
6. Which method improves LongDialQA/DialSim most.
7. Whether evidence components improve evidence precision/recall.
8. Whether typed graph improves open-domain premise/path coverage.
9. Whether episode repair actually helps when compressed memory is insufficient.
10. Whether adaptive routing improves accuracy-cost tradeoff.
11. Cost tradeoff: Avg K, tokens, runtime.
12. At least 5 real-data error analyses with:
    question
    gold answer
    baseline prediction
    improved prediction
    retrieved evidence
    source ids
    why the method helped or failed

Tables required:
- LongDialQA/DialSim baseline reproduction table.
- LoCoMo 50% category F1 table.
- LoCoMo retrieval/cost table.
- LongDialQA/DialSim 50% table by show.
- Method ablation table.
- Error analysis table.

Final response:
Give a concise summary with:
- implemented files
- commands run
- artifact paths
- main 50% subset results
- failed/skipped parts and why
- recommended next experiment
