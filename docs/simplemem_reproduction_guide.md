# SimpleMem 复现指导

本文档记录当前 `/home/stu0032/paper/baseline/SimpleMem` 的 LoCoMo10 复现配置、关键改动和成功运行口径。基准成功 run 为：

`/home/stu0032/paper/runs/simplemem/full_20260427_183242`

成功 run 信息：

- 数据集：`/home/stu0032/paper/datasets/locomo/data/locomo10.json`
- SimpleMem repo：`/home/stu0032/paper/baseline/SimpleMem`
- commit：`94ef7d76786af96878dea6e87ea2c7f5eaeae168`
- Python：`/home/stu0032/paper/.venv/bin/python`
- 开始时间：`2026-04-27 18:32:42`
- 结果文件：`result.json`
- 日志文件：`run.log`

## 1. 模型与 vLLM 设置

当前 SimpleMem 侧配置在 `baseline/SimpleMem/config.py`：

```python
OPENAI_API_KEY = "dummy"
OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "Qwen/Qwen3-8B"
MAX_OUTPUT_TOKENS = 15000

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
EMBEDDING_CONTEXT_LENGTH = 32768

ENABLE_THINKING = False
USE_STREAMING = True
USE_JSON_FORMAT = False
```

vLLM 用本地模型目录启动，服务名必须和 `LLM_MODEL` 一致：

```bash
cd /home/stu0032/paper
./start_vllm_qwen25.sh
```

脚本实际展开为：

```bash
/home/stu0032/paper/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/stu0032/paper/models/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.92 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

关键点：

- `OPENAI_BASE_URL` 指向本机 vLLM：`http://127.0.0.1:8000/v1`。
- `OPENAI_API_KEY` 只是 OpenAI SDK 必填占位，当前为 `dummy`。
- `ENABLE_THINKING=False`，vLLM 启动也通过 chat template 关闭 thinking。
- `MAX_OUTPUT_TOKENS=15000` 是防止坏 JSON 一直生成到 40960 上限的硬限制。
- `USE_JSON_FORMAT=False`，不要在全流程打开 OpenAI `json_object` 模式；当前只在 memory extraction 局部传入 vLLM JSON schema。

## 2. 不走代理

vLLM 和模型下载脚本都显式关闭代理：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
```

原因：

- 本地 vLLM 服务在 `127.0.0.1:8000`，走代理会导致连接异常或延迟。
- HuggingFace 下载使用 `HF_ENDPOINT=https://hf-mirror.com`，不依赖系统代理。

如果手动启动 vLLM 或手动跑评测，也应先执行上述环境变量。确认方式：

```bash
curl http://127.0.0.1:8000/v1/models
```

能看到 `Qwen/Qwen3-8B` 即可。

## 3. 离线缓存

当前复现默认依赖本地缓存和离线加载：

- LLM 权重：`/home/stu0032/paper/models/Qwen3-8B`
- embedding：`Qwen/Qwen3-Embedding-0.6B`，由 SentenceTransformer 加载，本地缓存必须存在。
- SBERT similarity：`all-MiniLM-L6-v2`，本地缓存必须存在。
- BERTScore：默认会用 `roberta-large`，本地缓存不完整会出问题。
- NLTK：`test_locomo10.py` 只检查本地资源，不再自动下载；缺资源会打印 warning。

vLLM 启动脚本设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
```

模型下载脚本：

```bash
cd /home/stu0032/paper
./download_qwen25_clean.sh
```

BERTScore 的 roberta-large 可单独预缓存：

```bash
cd /home/stu0032/paper
./download_bertscore_roberta.sh
```

注意：成功 run 中 BERTScore 仍然不可用，见第 7 节。

## 4. 16/30 workers 设置

当前 memory 构建并行来自 `config.py`：

```python
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = 16

ENABLE_PARALLEL_RETRIEVAL = False
MAX_RETRIEVAL_WORKERS = 16
MAX_TEST_QUESTION_WORKERS = 16
```

成功 run 的评测命令传入了 `--parallel-questions --test-workers 30`：

```bash
/home/stu0032/paper/.venv/bin/python -u test_locomo10.py \
  --dataset /home/stu0032/paper/datasets/locomo/data/locomo10.json \
  --result-file /home/stu0032/paper/runs/simplemem/full_20260427_183242/result.json \
  --parallel-questions \
  --test-workers 30
```

需要区分两个层面：

- memory build：实际用 `MAX_PARALLEL_WORKERS=16`，日志显示 `Processing ... batches in parallel with 16 workers`。
- QA 测试：命令请求 `--test-workers 30`，但 `test_locomo10.py` 内部又执行 `min(max_workers, len(qa_list), 20)`，所以当前代码实际每个 sample 最多 20 个 QA worker。日志中会显示 `Using 20 parallel workers`。

汇报时建议写：`memory workers=16, QA workers requested=30, effective QA cap=20`。如果论文表格必须写复现实验参数，可写 `16/30 requested`，并在脚注说明当前测试脚本 cap 到 20。

## 5. 运行方式与保存 result

推荐通过 wrapper 跑，它会自动创建 run 目录、保存命令、日志、配置快照和结果：

```bash
cd /home/stu0032/paper
/home/stu0032/paper/.venv/bin/python run_simplemem.py smoke5 \
  --parallel-questions \
  --test-workers 30
```

完整 LoCoMo10：

```bash
cd /home/stu0032/paper
/home/stu0032/paper/.venv/bin/python run_simplemem.py full \
  --parallel-questions \
  --test-workers 30
```

固定输出目录：

```bash
cd /home/stu0032/paper
/home/stu0032/paper/.venv/bin/python run_simplemem.py full \
  --parallel-questions \
  --test-workers 30 \
  --output-dir /home/stu0032/paper/runs/simplemem/full_YYYYMMDD_HHMMSS
```

wrapper 会保存：

- `result.json`：最终结构化结果。
- `run.log`：完整 stdout/stderr。
- `command.txt`：实际传给 `test_locomo10.py` 的命令。
- `config_redacted.py`：脱敏后的 `config.py` 快照。
- `meta.txt`：preset、开始时间、repo、dataset、commit、python。

也可以直接运行 `test_locomo10.py`，但必须手动指定 `--result-file`，否则默认写到 SimpleMem repo 工作目录下的 `locomo10_test_results.json`：

```bash
cd /home/stu0032/paper/baseline/SimpleMem
/home/stu0032/paper/.venv/bin/python -u test_locomo10.py \
  --dataset /home/stu0032/paper/datasets/locomo/data/locomo10.json \
  --result-file /home/stu0032/paper/runs/simplemem/manual/result.json \
  --parallel-questions \
  --test-workers 30
```

`result.json` 结构：

- `summary`：样本数、问题数、平均 retrieval/answer/total 时间。
- `aggregated_metrics`：overall 和 category 级别的 metric 统计。
- `detailed_results`：每题 question、answer、reference、category、耗时、retrieved 数、metrics。

## 6. JSON schema、streaming 与早停

本次能稳定跑完的关键修改在 `core/memory_builder.py` 和 `utils/llm_client.py`。

memory extraction prompt 约束：

- 顶层 JSON 必须是对象。
- 仅有一个 key：`entries`。
- `entries` 是 memory entry 数组。
- 每个 window 最多 80 条 memory entries。
- 每条 `lossless_restatement` 必填。
- 15000 output token 上限内必须闭合 JSON。

`core/memory_builder.py` 的 `_memory_response_format()` 传给 vLLM：

- `type=json_schema`
- `strict=True`
- 顶层对象只允许 `entries`
- `entries.maxItems=80`
- 单条 entry 的 `lossless_restatement.maxLength=1500`
- list 字段如 `keywords/persons/entities` 限制 `maxItems=30`

`utils/llm_client.py` 的 streaming 早停逻辑：

- `USE_STREAMING=True` 时，OpenAI SDK 以 stream 模式请求 vLLM。
- 每收到增量文本就扫描第一个完整、平衡、可解析的顶层 JSON object/array。
- 一旦 JSON 完整闭合，立即 `stream.close()`，返回 JSON 前缀。
- 如果 finish_reason 是 `length`，抛出 `LLMTruncatedOutputError`，避免把截断 JSON 当成结果。

这套组合解决了两个问题：

- 不等待模型继续输出多余解释或重复内容。
- 坏输出不会无限生成到 40960 context 上限。

注意：`USE_JSON_FORMAT=False` 是全局策略。只对 memory extraction 使用 schema；retrieval、answer、judge 仍走 prompt 约束和 `extract_json()` 容错解析。

## 7. BERTScore 问题

成功 run 中 BERTScore 不应作为有效指标汇报。

日志反复出现：

```text
Error calculating BERTScore: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
when moving module from meta to a different device.
```

结果：

- `bert_f1` 为 0 的题数：`1981 / 1986`
- `overall bert_f1 mean=0.0022`，这个值没有可解释性。

原因判断：

- `test_locomo10.py` 在每题 `calculate_metrics()` 中直接调用 `bert_score()`。
- QA 并行时多个 worker 并发触发 BERTScore/transformers 模型加载或迁移。
- roberta-large 权重或 device/meta tensor 状态异常导致 scoring 失败，函数捕获异常后返回 0。

汇报建议：

- 主表不要汇报 BERTScore，或者标记为 invalid/not reported。
- 如果必须汇报 BERTScore，需要从 `result.json` 的 `detailed_results[*].answer/reference` 离线串行重算。
- 成功 run 的可用语义指标优先看 `sbert_similarity`，但也要注明它来自 `all-MiniLM-L6-v2`。

## 8. 指标汇报口径

成功 run 的 `result.json` 主要指标如下。百分数为 mean * 100。

| Scope | n | EM | F1 | ROUGE-L | METEOR | SBERT |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 1986 | 27.69 | 48.51 | 48.50 | 41.75 | 64.92 |
| Cat 1 | 282 | 2.13 | 30.88 | 28.53 | 18.08 | 53.69 |
| Cat 2 | 321 | 4.05 | 39.87 | 39.05 | 21.26 | 67.65 |
| Cat 3 | 96 | 4.17 | 14.93 | 14.96 | 10.37 | 40.97 |
| Cat 4 | 841 | 16.53 | 40.87 | 41.89 | 37.11 | 58.02 |
| Cat 5 | 446 | 87.00 | 87.53 | 87.61 | 86.98 | 88.22 |

Timing：

- 平均 retrieval time：`35.73s`
- 平均 answer time：`3.17s`
- 平均 total time：`38.90s`
- 总 retrieval time：`70953.45s`
- 总 answer time：`6296.01s`

注意：

- `llm_judge_score` 在成功 run 中全为 0，因为没有使用 `--llm-judge`。
- `category_5` 的 reference 固定为 `Not mentioned in the conversation`，测试脚本对该类关闭 reflection 并使用二选一 prompt。
- `result.json` 的 `count` 是参与该 metric 的题数，本次 overall 为 `1986`。

建议报告格式：

```text
SimpleMem, LoCoMo10 full, Qwen3-8B via local vLLM.
Memory workers=16; QA workers requested=30, effective cap=20 in current script.
Overall: EM 27.69, F1 48.51, ROUGE-L 48.50, METEOR 41.75, SBERT 64.92.
BERTScore invalid due to meta tensor failures in 1981/1986 questions; not reported.
```

## 9. 常见故障

### 9.1 连接 vLLM 失败

现象：

- `Connection refused`
- `Max retries exceeded`
- OpenAI client 指向了错误地址

检查：

```bash
ps -ef | rg 'vllm|Qwen3|8000'
curl http://127.0.0.1:8000/v1/models
```

处理：

- 重新启动 `./start_vllm_qwen25.sh`。
- 确认 `config.py` 中 `OPENAI_BASE_URL="http://127.0.0.1:8000/v1"`。
- 清掉代理变量并设置 `NO_PROXY="*"`。

### 9.2 vLLM 启动后模型名不匹配

现象：

- API 返回 model not found。

处理：

- `--served-model-name` 必须是 `Qwen/Qwen3-8B`。
- `config.py` 的 `LLM_MODEL` 也必须是 `Qwen/Qwen3-8B`。

### 9.3 离线模式找不到模型

现象：

- HuggingFace cache miss。
- `TRANSFORMERS_OFFLINE=1` 下加载失败。

处理：

- 先运行 `./download_qwen25_clean.sh` 下载 LLM。
- 确认 embedding 和 `all-MiniLM-L6-v2` 已在本机缓存。
- 需要 BERTScore 时先运行 `./download_bertscore_roberta.sh`，再串行重算。

### 9.4 NLTK warning

现象：

```text
Warning: NLTK resource 'wordnet' not found locally; related metrics may be zero.
No download attempted.
```

说明：

- 当前脚本不会在评测时联网下载 NLTK。
- 缺 `wordnet` 主要影响 METEOR。
- 若要严格汇报 METEOR，需要提前准备 NLTK 本地资源。

### 9.5 JSON 解析失败或 generation 截断

现象：

- `Failed to extract valid JSON`
- `LLMTruncatedOutputError`
- memory window 多次 retry，最终生成 0 entries

处理：

- 保持 `MAX_OUTPUT_TOKENS=15000`。
- 保持 memory schema 的 `maxItems=80` 和字段长度限制。
- 不要全局打开 `USE_JSON_FORMAT=True`。
- 先跑 `smoke5` 验证，再跑 full。

### 9.6 BERTScore 大量为 0

现象：

- `Cannot copy out of meta tensor; no data!`
- `bert_f1` 大面积为 0。

处理：

- 不把本次 BERTScore 纳入主结果。
- 用 `result.json` 离线串行重算，或临时关闭并行 QA 后单独跑 BERTScore。

### 9.7 workers 与日志不一致

现象：

- 命令写了 `--test-workers 30`，日志显示 `Using 20 parallel workers`。

原因：

- `test_locomo10.py` 内部把 QA worker cap 到 20。

处理：

- 复现成功 run 时保持现状。
- 如果要真正用 30 个 QA worker，需要修改 cap 并重新跑；这会改变复现条件。

### 9.8 LanceDB/FTS 问题

现象：

- 表已存在、FTS index 创建失败或旧数据干扰。

处理：

- `SimpleMemSystem(clear_db=True)` 会在每个 run 初始化时清库。
- `run_test()` 每个 sample 前也会 `vector_store.clear()`。
- 如仍异常，可删除 SimpleMem repo 下的 `lancedb_data` 后重跑 smoke。

## 10. 复现前检查清单

1. vLLM 正在监听 `127.0.0.1:8000`，`/v1/models` 返回 `Qwen/Qwen3-8B`。
2. 代理变量已 unset，`NO_PROXY="*"`。
3. `/home/stu0032/paper/models/Qwen3-8B` 存在。
4. `config.py` 使用本地 vLLM、`USE_STREAMING=True`、`USE_JSON_FORMAT=False`、`MAX_OUTPUT_TOKENS=15000`。
5. memory workers 为 16。
6. full run 使用 `--parallel-questions --test-workers 30`，并接受当前 effective QA cap=20。
7. 用 wrapper 保存 run 目录，不手动覆盖已有成功结果。
8. 汇报时排除或标注 BERTScore invalid。
