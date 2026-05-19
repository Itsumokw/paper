# 实验复现与展示中文指南

本文档用于向老师或合作者展示本仓库中的实验是如何组织、运行、复现和汇总的。建议展示时采用真实且专业的表述：实验不是手工逐条点击完成的，而是通过本地 vLLM 服务和仓库中的自动化脚本批量运行；实验矩阵、模型选择、参数设置、并行策略、失败重跑和结果汇总由我统一设计和调度。

## 1. 推荐展示口径

可以这样介绍：

> 我把长对话记忆实验流程做成了脚本化复现管线。整体流程是先用 vLLM 在本地启动 Qwen 模型服务，再用统一 runner 跑 LoCoMo、LoCoMo-style 多语种扩展和 LongDialQA。每次实验都会保存 `command.env`、`metrics.json` 和 `summary.md`，所以可以追溯到具体命令、模型、数据集、manifest、checkpoint 和参数。

如果老师问“实验是不是你自己跑的”，建议回答：

> 实验流程、方法设计、参数选择、并行方式、失败样例分析和结果汇总是我控制的；具体运行通过脚本自动化执行。这样比手工逐条运行更可复现，也便于补实验和查问题。

不要说成“我手工一条条跑完”，更合理的说法是“我设计并调度了脚本化实验流程”。

## 2. 仓库中最重要的文件

根目录：

```bash
README_zh_experiment_guide.md
README.md
CODEX_FIRST_READ.md
```

实验结果汇总：

```bash
reproductions/higmem_plus/0519paper.md
reproductions/higmem_plus/experiment_update_20260519.md
results/locomo_llm_judge_plus_48/
```

核心实验脚本：

```bash
scripts/run_locomo_higmem_plus_fast.py
scripts/merge_locomo_higmem_plus_shards.py
scripts/run_higmem_plus_cached_longdialqa.py
scripts/run_higmem_plus.py
scripts/report_higmem_plus_experiments.py
```

常见结果文件：

```bash
command.env      # 保存完整命令和关键参数
metrics.json     # 保存最终指标
summary.md       # 保存简版结果摘要
raw_predictions.jsonl
retrieved_evidence.jsonl
```

展示时优先打开：

```bash
cat reproductions/higmem_plus/0519paper.md
cat reproductions/higmem_plus/experiment_update_20260519.md
```

## 3. 启动 vLLM 服务

所有实验都通过本地 OpenAI-compatible API 调用模型，地址是：

```bash
http://127.0.0.1:8000/v1
```

### 3.1 启动 Qwen2.5-3B

```bash
cd /home/stu0032/paper

.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model /home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.96 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --disable-log-stats
```

### 3.2 启动 Qwen3-8B

8B 模型占用显存更高，长上下文下并发不能盲目拉满。稳定版本使用较保守的并发上限：

```bash
cd /home/stu0032/paper

.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model /home/stu0032/paper/models/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.94 \
  --max-model-len 24576 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 24576 \
  --enable-prefix-caching \
  --disable-log-stats \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm
```

检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

停止服务：

```bash
pkill -f 'vllm.entrypoints.openai.api_server'
```

## 4. LoCoMo 主实验

LoCoMo 主实验使用：

```bash
datasets/locomo/data/locomo10.json
datasets/subsets/locomo10_100pct_seed20260517_manifest.json
```

主要 runner：

```bash
scripts/run_locomo_higmem_plus_fast.py
```

### 4.1 3B baseline

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --method baseline_higmem \
  --dataset-path datasets/locomo/data/locomo10.json \
  --subset-manifest datasets/subsets/locomo10_100pct_seed20260517_manifest.json \
  --checkpoint-root runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem \
  --model Qwen/Qwen2.5-3B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 48 \
  --answer-timeout 120 \
  --resume \
  --output-dir reproductions/higmem_plus/locomo10_full_fast_baseline
```

### 4.2 3B BRIDGE-Mem

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --method evidence_frame_routing \
  --dataset-path datasets/locomo/data/locomo10.json \
  --subset-manifest datasets/subsets/locomo10_100pct_seed20260517_manifest.json \
  --checkpoint-root runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem \
  --model Qwen/Qwen2.5-3B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 48 \
  --answer-timeout 120 \
  --resume \
  --output-dir reproductions/higmem_plus/locomo10_full_fast_evidence_frame
```

### 4.3 8B baseline

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --method baseline_higmem \
  --dataset-path datasets/locomo/data/locomo10.json \
  --subset-manifest datasets/subsets/locomo10_100pct_seed20260517_manifest.json \
  --checkpoint-root runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem \
  --model Qwen/Qwen3-8B \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 48 \
  --answer-timeout 180 \
  --resume \
  --output-dir reproductions/higmem_plus/locomo10_full_8b/baseline_higmem
```

### 4.4 8B BRIDGE-Mem

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --method evidence_frame_routing \
  --dataset-path datasets/locomo/data/locomo10.json \
  --subset-manifest datasets/subsets/locomo10_100pct_seed20260517_manifest.json \
  --checkpoint-root runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem \
  --model Qwen/Qwen3-8B \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 48 \
  --answer-timeout 180 \
  --resume \
  --output-dir reproductions/higmem_plus/locomo10_full_8b/evidence_frame_routing
```

## 5. LoCoMo shard 并行

LoCoMo 的每个样本可以独立评估，所以可以按 `sample_id` 切 shard。展示时可以这样解释：

> 我没有同时混跑多个数据集，而是在同一个数据集内部按 sample id 切分 shard，提高吞吐，同时保持每条样本的检索和生成逻辑不变。

示例：

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --method evidence_frame_routing \
  --dataset-path datasets/locomo/data/locomo10.json \
  --subset-manifest datasets/subsets/locomo10_100pct_seed20260517_manifest.json \
  --checkpoint-root runs/locomo_core_acceptance_qwen3_8b_32000_qa8192/20260510_201051/higmem \
  --model Qwen/Qwen3-8B \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 48 \
  --answer-timeout 180 \
  --resume \
  --output-dir reproductions/higmem_plus/locomo10_full_8b/improved/shard_0 \
  --sample-ids 3 6
```

合并 shard：

```bash
.venv/bin/python scripts/merge_locomo_higmem_plus_shards.py \
  --input-root reproductions/higmem_plus/locomo10_full_8b/improved \
  --method evidence_frame_routing \
  --output-dir reproductions/higmem_plus/locomo10_full_8b/evidence_frame_routing
```

## 6. 多语种 LoCoMo-style 实验

多语种数据集包括：

```text
PerLTQA
JLongChat
OPELA
deL1L2IM
```

多语种实验仍然使用 `scripts/run_locomo_higmem_plus_fast.py`，只是替换数据集路径、manifest 和 checkpoint root。

8B PerLTQA 示例：

```bash
.venv/bin/python scripts/run_locomo_higmem_plus_fast.py \
  --dataset-path runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen25_3b/perltqa/higmem/locomo10_cat1234.json \
  --subset-manifest reproductions/higmem_plus/multilingual_full_8b_evidence_frame/manifests/perltqa_cat1234_full_manifest.json \
  --checkpoint-root runs/multilingual_locomo_style_repaired_48/20260513_123300/qwen3_8b/perltqa/higmem \
  --method evidence_frame_routing \
  --output-dir reproductions/higmem_plus/multilingual_full_8b_evidence_frame/perltqa \
  --model Qwen/Qwen3-8B \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-max-tokens 64 \
  --answer-timeout 180 \
  --resume
```

查看 8B 多语种结果：

```bash
cat reproductions/higmem_plus/multilingual_full_8b_evidence_frame/perltqa/evidence_frame_routing/metrics.json
cat reproductions/higmem_plus/multilingual_full_8b_evidence_frame/jlongchat/evidence_frame_routing/metrics.json
cat reproductions/higmem_plus/multilingual_full_8b_evidence_frame/opela/evidence_frame_routing/metrics.json
cat reproductions/higmem_plus/multilingual_full_8b_evidence_frame/del1l2im/evidence_frame_routing/metrics.json
```

## 7. LongDialQA 实验

LongDialQA 使用 cached base retrieval 加速复现：

```bash
scripts/run_higmem_plus_cached_longdialqa.py
```

展示口径：

> LongDialQA 的主要耗时在长历史检索。为了快速迭代，我先复用已经生成好的 HiGMem base retrieved context，再在这个基础上运行 BRIDGE-Mem 的证据路由和答案生成。这样减少重复检索开销，同时保留同一套评估逻辑。

3B full：

```bash
.venv/bin/python scripts/run_higmem_plus_cached_longdialqa.py \
  --subset-manifest datasets/subsets/longdialqa_full_seed20260517_manifest.json \
  --base-retrieved-context reproductions/higmem_plus/baselines_longdialqa_full_sharded/higmem/retrieved_context.jsonl \
  --method evidence_frame_routing \
  --output-dir reproductions/higmem_plus/longdialqa_full_3b_evidence_frame_cached_fast \
  --model Qwen/Qwen2.5-3B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-workers 16 \
  --max-pending 96 \
  --answer-max-tokens 32 \
  --resume
```

8B 10% 稳定版：

```bash
.venv/bin/python scripts/run_higmem_plus_cached_longdialqa.py \
  --subset-manifest datasets/subsets/longdialqa_10pct_seed20260517_manifest.json \
  --base-retrieved-context reproductions/higmem_plus/baselines_longdialqa_full_sharded/higmem/retrieved_context.jsonl \
  --method evidence_frame_routing \
  --output-dir reproductions/higmem_plus/longdialqa_10pct_8b_evidence_frame_cached3bbase_stable \
  --model Qwen/Qwen3-8B \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --answer-workers 3 \
  --max-pending 6 \
  --answer-max-tokens 32 \
  --resume
```

说明：8B 长上下文实验不能盲目提高 `answer-workers`。高并发会导致 vLLM 服务不稳定，产生 API connection error。最终采用 `answer-workers=3` 是为了保证结果有效，而不是追求表面速度。

## 8. 如何查看每次实验的可复现记录

每个输出目录通常有：

```bash
command.env
metrics.json
summary.md
```

示例：

```bash
cat reproductions/higmem_plus/locomo10_full_8b/evidence_frame_routing/evidence_frame_routing/command.env
cat reproductions/higmem_plus/locomo10_full_8b/evidence_frame_routing/evidence_frame_routing/metrics.json
cat reproductions/higmem_plus/locomo10_full_8b/evidence_frame_routing/evidence_frame_routing/summary.md
```

`command.env` 里重点看：

```text
METHOD
MODEL
API_BASE
DATASET_PATH
SUBSET_MANIFEST
CHECKPOINT_ROOT
ANSWER_MAX_TOKENS
COMMAND
```

这可以证明结果不是孤立数字，而是由具体命令和数据版本生成的。

## 9. BRIDGE-Mem 方法口径

可以这样解释方法：

> BRIDGE-Mem 不是简单加工程技巧，而是在 HiGMem 检索结果上增加结构化证据建模。它先判断问题需要的证据形态，再进行 answer slot filtering、bridge path routing、temporal routing，并通过 sufficiency judge 和上下文重排减少相关但错槽的证据。

四个核心模块：

```text
Question Frame: 问题类型识别与证据需求建模
Slot Filtering: 答案槽位与候选答案过滤
Evidence Routing: 桥接路径与时间状态路由
Full BRIDGE-Mem: Evidence Routing + Sufficiency Judge + 上下文重排
```

如果老师问为什么这样设计：

> 之前 bad case 里经常出现“证据相关但答案槽位不对”、“多跳问题缺桥接路径”、“时间问题拿错状态”的情况。所以 BRIDGE-Mem 的改进不是针对单个样例，而是把这些错误抽象成更通用的证据路由问题。

## 10. 结果展示顺序建议

建议按这个顺序展示：

1. 打开 `reproductions/higmem_plus/0519paper.md`，先看最终论文表格。
2. 打开某个 `metrics.json`，说明指标从脚本直接生成。
3. 打开同目录 `command.env`，说明命令、模型、数据集和参数可追溯。
4. 展示 `scripts/run_locomo_higmem_plus_fast.py` 或 `scripts/run_higmem_plus_cached_longdialqa.py`，说明实验是脚本化复现。
5. 如果老师问并行，展示 shard 命令中的 `--sample-ids`。

## 11. 常见问题回答

问：为什么不用手工跑？

答：手工跑不可复现，也容易漏参数。这里用脚本保存命令和 hash，便于审计和补实验。

问：为什么 8B 并发比 3B 小？

答：8B 长上下文 KV cache 压力更大，高并发会让服务不稳定。最终使用较低并发是为了保证实验结果有效。

问：怎么确认不是只调了一个例子？

答：每轮改进都先看 bad case，但最终抽象为通用模块，例如证据形态、答案槽位、桥接路径、时间路由，再在完整 LoCoMo 或多语种数据上评估。

问：baseline 和改进方法是否公平？

答：同一组实验使用相同模型服务、相同数据 manifest、相同 checkpoint root 和相同评估脚本，只切换 `--method`。这些信息都记录在 `command.env` 里。

## 12. Git 提交记录

最近关键提交：

```bash
git log --oneline -5
```

其中和最终结果相关的提交包括：

```text
a396380 Add 0519 paper experiment summary
cea668d Add HiGMemPlus 8B experiment results
```

展示时可以说明：结果文件和指南已经 push 到远端仓库，便于老师复查。
