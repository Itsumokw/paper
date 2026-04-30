# Next LoCoMo Baselines Reproduction Prep

Date: 2026-04-28

## Shared Runtime

All scripts use the existing environment:

- Python: `/home/stu0032/paper/.venv/bin/python`
- Chat model endpoint: `http://127.0.0.1:8000/v1`
- Served model: `Qwen/Qwen2.5-3B-Instruct`
- Dataset: `/home/stu0032/paper/baseline/MAGMA/data/locomo10.json`
- Proxy policy: all scripts unset `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`; HF downloads use `HF_ENDPOINT=https://hf-mirror.com`.

The helper script `scripts/download_next_baseline_models.sh` has already cached:

- `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- `sentence-transformers/all-MiniLM-L6-v2`
- NLTK `punkt`, `punkt_tab`, `wordnet`

## One Command

Run this inside `screen -r paper` after the current MAGMA run is finished:

```bash
cd /home/stu0032/paper
./scripts/run_next_baselines_qwen25_3b_all.sh
```

The all-in-one script:

1. Refuses to run if MAGMA is still active, unless `ALLOW_CONCURRENT_EXPERIMENTS=1`.
2. Starts Qwen2.5-3B vLLM if `127.0.0.1:8000` is not available.
3. Runs LightMem/StructMem.
4. Runs xMemory with local sentence-transformer embeddings and vLLM LLM calls.
5. Runs MemMachine only if Neo4j is already reachable at `127.0.0.1:7687`; otherwise it skips with a clear message.

## Baseline Notes

### LightMem / StructMem

Local changes:

- `add_locomo.py` now reads model, dataset, Qdrant paths, devices, and workers from environment variables.
- LLM JSON calls use streaming by default and close as soon as the first complete top-level JSON object/array is detected.
- No hard `max_tokens` is sent by default (`LIGHTMEM_MAX_TOKENS=0`).
- LLMLingua and embedding run on CPU by default, so the process does not compete with vLLM for GPU memory.
- QA/search has sample-level workers via `--workers`.

Max useful native parallelism:

- Memory build: `LIGHTMEM_BUILD_WORKERS=10` because LoCoMo10 has 10 samples.
- Internal extraction: up to 5 API calls per sample in the original manager.
- QA/search: `LIGHTMEM_SEARCH_WORKERS=10` sample-level workers.

Direct command:

```bash
cd /home/stu0032/paper
./scripts/run_lightmem_structmem_qwen25_3b_full.sh
```

Outputs:

- `/home/stu0032/paper/runs/lightmem/qwen25_3b_structmem_full/results/summary.json`
- Per-sample JSON files in the same result directory.

### xMemory

Local changes:

- `EmbeddingClient` supports local `sentence-transformers/all-MiniLM-L6-v2`, avoiding OpenAI embedding calls.
- `XMEMORY_LLM_BACKEND=openai` makes xMemory use the local vLLM OpenAI-compatible endpoint.
- JSON LLM calls use streaming and first-complete-JSON closure on the OpenAI-compatible backend.
- The original HF backend also has JSON early stopping via a Transformers stopping criterion for users who later run adaptive logprob-heavy search without vLLM.
- Metrics avoid online NLTK downloads and use CPU for local semantic models.

Max useful native parallelism:

- Build sample workers: `XMEMORY_BUILD_WORKERS=10`.
- Internal semantic generation workers: `semantic_generation_workers=20` in `config.local_qwen25_3b.json`.
- Search workers: `XMEMORY_SEARCH_WORKERS=10`.
- Eval workers: `XMEMORY_EVAL_WORKERS=20`.

Direct command:

```bash
cd /home/stu0032/paper
RESET_XMEMORY_RUN=1 ./scripts/run_xmemory_qwen25_3b_full.sh
```

Outputs:

- `/home/stu0032/paper/runs/xmemory/qwen25_3b_full/results.json`
- `/home/stu0032/paper/runs/xmemory/qwen25_3b_full/metrics.json`

Important: the default script uses xMemory `baseline` search to stay compatible with vLLM. The repo's `adaptive_hier` path relies on local HF logprob/cache methods; run it later with `XMEMORY_LLM_BACKEND=hf` after freeing the GPU.

### MemMachine

Local changes:

- Installed MemMachine local packages and dependencies into the shared venv.
- Added a local Qwen2.5/vLLM config:
  - `evaluation/retrieval_agent/configuration.local_qwen25_vllm.yml`
  - copied to `configuration.yml` for `run_test.sh`
- LLM uses `openai-chat-completions` against local vLLM.
- Embedding uses local `sentence-transformers/all-MiniLM-L6-v2`.
- Reranking uses local BM25.
- Chat-completions LLM client streams by default and closes at first complete JSON for structured responses.
- Added sourceable runtime environment:
  - `scripts/memmachine_env.sh`
  - `scripts/setup_memmachine_local_env.sh`
- Added local Neo4j helpers:
  - `scripts/install_memmachine_neo4j_local.sh`
  - `scripts/start_memmachine_neo4j.sh`

Prepare/check command:

```bash
cd /home/stu0032/paper
./scripts/setup_memmachine_local_env.sh
```

Remaining external service:

- Official MemMachine LoCoMo retrieval-agent path requires Neo4j as `vector_graph_store`.
- Neo4j is installed as a system service from the official Neo4j Debian repository.
- It is configured to listen only on localhost:
  - Bolt: `bolt://127.0.0.1:7687`
  - HTTP: `http://127.0.0.1:7474`
- MemMachine uses `NEO4J_PASSWORD=neo4j_password` by default.

Start/check command:

```bash
cd /home/stu0032/paper
SUDO_PASSWORD='<sudo-password>' ./scripts/start_memmachine_neo4j.sh
./scripts/setup_memmachine_local_env.sh
```

Max useful native parallelism:

- Ingest: `MEMMACHINE_INGEST_CONCURRENCY=10`.
- Search: `MEMMACHINE_SEARCH_CONCURRENCY=10`.
- Judge/eval: `MEMMACHINE_JUDGE_CONCURRENCY=30`.

Direct command after starting Neo4j:

```bash
cd /home/stu0032/paper
NEO4J_PASSWORD='<your-password>' ./scripts/run_memmachine_qwen25_3b_full.sh
```

Outputs:

- `/home/stu0032/paper/baseline/MemMachine/evaluation/retrieval_agent/result/`

## Reproduction Impact

The JSON streaming/first-closure changes do not alter the memory schema or retrieval algorithm. They only stop local models from continuing after a complete JSON result has already been emitted. This is the same stabilization strategy used for SimpleMem/MAGMA/HiGMem: preserve the first valid structured result, avoid runaway generation, and keep metrics computed from the normal result files.
