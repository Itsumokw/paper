# HiGMem 复现准备说明

## 推荐结论

下一篇最建议优先复现：

**HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents**

原因：

- 论文够新：ACL Findings 2026。
- 开源：`https://github.com/ZeroLoss-Lab/HiGMem`。
- 任务对齐：仓库自带 LoCoMo10 reproduction。
- 指标对齐：代码计算 EM、F1、ROUGE、BLEU、METEOR、BERTScore、SBERT。
- 工程对齐：README 明确支持 OpenAI-compatible backend，可直接接本地 vLLM。
- 研究方向对齐：它不是单纯 rerank，而是做层级 turn/event memory 和 LLM-guided memory construction，适合作为 SimpleMem 后的“记忆构建性能改进”主 baseline。

本地代码位置：

```text
/home/stu0032/paper/baseline/HiGMem
```

当前 clone commit：

```text
f275072
```

## 与 SimpleMem 的关系

SimpleMem 把若干 dialogue window 压缩成 memory entries，再做混合检索。

HiGMem 采用更细的层级结构：

- Turn layer：保留原始对话 turn note。
- Event layer：把相关 turn 增量归并为事件，并维护事件 metadata / summary。
- Retrieval：先检索 event，再检索/过滤 turn evidence，最后生成答案。

因此它很适合作为后续改进方向的比较对象：

- SimpleMem：窗口级语义压缩。
- HiGMem：turn/event 层级记忆构建。
- 你的改进：可以围绕 evidence-aware construction、event/turn provenance、query-aware retrieval 或 raw fallback 展开。

## 本地模型

计划使用：

```text
/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean
```

served model name：

```text
Qwen/Qwen2.5-3B-Instruct
```

Qwen2.5-3B-Instruct 本地 config 的最大上下文为：

```text
32768
```

## 启动 vLLM

如果 `paper` screen 里还跑着 Qwen3-8B，需要先停掉旧 vLLM，再启动 Qwen2.5-3B。

启动脚本：

```bash
cd /home/stu0032/paper
./start_vllm_qwen25_3b.sh > runs/vllm_qwen25_3b_current.log 2>&1
```

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```

应看到：

```text
Qwen/Qwen2.5-3B-Instruct
```

## 先跑 smoke

先只跑 sample 0 的前 80 turns 和 5 个问题，确认 JSON、检索、指标都能跑通：

```bash
cd /home/stu0032/paper
./scripts/run_higmem_qwen25_3b_smoke.sh
```

输出位置由 HiGMem 原仓库控制：

```text
/home/stu0032/paper/baseline/HiGMem/results/
/home/stu0032/paper/baseline/HiGMem/fphm_logs/
/home/stu0032/paper/baseline/HiGMem/checkpoints/
```

## 跑 full

官方 README 的主设置是：

```text
--ablation-no-profile --ablation-event-metadata-only --ablation-no-link --k_event 10
```

本地 full 脚本：

```bash
cd /home/stu0032/paper
./scripts/run_higmem_qwen25_3b_full.sh
```

默认：

```text
HIGMEM_WORKERS=5
```

如果 Qwen2.5-3B 或 vLLM 排队明显，可以调低：

```bash
HIGMEM_WORKERS=3 ./scripts/run_higmem_qwen25_3b_full.sh
```

full 输出位置：

```text
/home/stu0032/paper/baseline/HiGMem/fphm_runs/
```

跑完后把成功 run 归档到：

```text
/home/stu0032/paper/reproductions/successful/HiGMem/<date>_qwen2.5-3b_locomo10-full/
```

## 指标口径

HiGMem 的 `utils.py` 会计算：

- `exact_match`
- `f1`
- `rouge1_f`
- `rouge2_f`
- `rougeL_f`
- `bleu1`
- `bleu2`
- `bleu3`
- `bleu4`
- `meteor`
- `bert_f1`
- `sbert_similarity`

本地脚本默认设置：

```bash
SKIP_BERTSCORE=1
```

原因是 BERTScore 在并行评测时很容易占用 GPU 或触发 meta tensor 问题。这个设置不影响 F1、ROUGE、BLEU、METEOR；如果论文必须汇报 BERTScore，建议 full 跑完后基于预测和 reference 单独补算。

## LLM 调用与 retry 防护

已把 SimpleMem 复现中的“流式读取 + JSON 闭合即停止”经验移植到 HiGMem：

- OpenAI-compatible client 设置 `max_retries=0`，避免 OpenAI SDK 自己在内部重复 retry。
- OpenAI-compatible client 默认使用 streaming。
- 不设置 `max_tokens` 输出上限，让 vLLM 使用当前模型/context 的默认上限。
- streaming 过程中检测第一个完整顶层 JSON object/array；一旦 JSON 闭合，立即关闭 stream 并返回这个 JSON。
- 不改 prompt、schema、temperature、检索逻辑、并发参数、外层 retry 或评测指标。

默认环境变量：

```bash
export HIGMEM_USE_STREAMING=1
```

如需严格回到非 streaming 调用，可临时关闭：

```bash
HIGMEM_USE_STREAMING=0 ./scripts/run_higmem_qwen25_3b_full.sh
```

## 已完成准备

- 已 clone HiGMem 到 `baseline/HiGMem`。
- 已确认本地 `Qwen2.5-3B-Instruct-clean` 权重存在。
- 已确认当前 `/home/stu0032/paper/.venv` 可 import HiGMem 主要依赖；缺少 `ollama`，但 OpenAI-compatible vLLM backend 不需要它。
- 已通过 `py_compile` 检查 `run_fphm_evaluation.py`、`fphm_core.py`、`memory_layer.py`、`utils.py`、`load_dataset.py`。
- 已确认 HiGMem 自带 LoCoMo10 数据可加载，共 10 samples / 1986 QAs。
