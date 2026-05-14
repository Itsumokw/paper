# LoCoMo 核心 Baseline 复现与统计验收要求

## 目标

在 `/home/stu0032/paper` 中稳定复现并统一评测以下核心 baseline：

- Full Context
- A-MEM
- Mem0
- SimpleMem
- HiGMem
- MemGAS

实际是 6 个 baseline，其中 Full Context 是参照组，其余 5 个是 memory baseline。目标是完整跑通、不中断、可复查、可写进论文表格，并保证对原代码的修改只服务于本地复现稳定性和统计完整性，不改变算法核心逻辑。

## 评测指标口径

依据 LoCoMo、SimpleMem、LightMem、A-MEM、MemGAS 等论文/报告常见口径，主表指标固定为：

- F1
- BLEU-1
- ROUGE-L
- BERTScore-F1
- Token Cost / Avg Tokens
- Latency

参考来源：

- LoCoMo 原始 benchmark: <https://arxiv.org/abs/2402.17753>
- SimpleMem: <https://arxiv.org/abs/2601.02553>
- MemGAS: <https://openreview.net/forum?id=i2yIvZARnG>
- LightMem: <https://openreview.net/pdf?id=BAqc2UU95N>
- A-MEM: <https://openreview.net/pdf?id=FiM0M8gcct>

统一规则：

- 主实验默认 LoCoMo10 cat1-4，1540 QA。
- 如果跑 all categories 或 cat5，必须单独命名，不能混入主表。
- F1、BLEU-1、ROUGE-L、BERTScore 必须使用同一份 `compute_locomo_text_metrics.py` 或统一封装后的等价实现。
- 所有 baseline 使用同一 normalization：
  - 小写化
  - 去标点
  - 去冠词
  - whitespace normalize
  - 空 prediction 记 0 分
- BERTScore 使用同一模型、同一 device、同一 batch size，并记录模型路径。
- LLM Judge 可作为补充分析，不作为主表唯一指标。
- 每个指标都要给 overall 和 per-category 结果。

## 必须统计的效率数据

每个 baseline 必须统计并落盘以下数据，供论文效率表、消融表和复现报告使用。

### Token 统计

- total prompt tokens
- total completion tokens
- total tokens
- avg prompt tokens / QA
- avg completion tokens / QA
- avg total tokens / QA
- p50 / p95 / max total tokens
- token reduction vs Full Context

### Latency 统计

- total wall-clock time
- memory build time
- retrieval time
- QA generation time
- metric computation time
- avg latency / QA
- p50 / p95 / p99 latency
- throughput: QA/min

### Memory / Retrieval 统计

- number of memories created
- avg memories per sample
- retrieved memory count per query
- avg retrieved context tokens
- p95 retrieved context tokens
- index/vector store size if available
- memory build token cost
- retrieval + answer token cost

### Reliability 统计

- retry count
- timeout count
- length-finish count
- fallback count
- empty prediction count
- JSON parse failure count
- memory action error count
- fatal error count

最终每个 baseline 的 `metrics.json` 必须包含 `overall`、`categories`、`runtime`、`tokens`、`reliability` 五类字段。

## 原始代码对比与修改约束

实现者必须为 A-MEM、Mem0、SimpleMem、HiGMem、MemGAS 获取或保留 clean upstream 原始代码，并与当前修改版对比。

每项修改必须分类。

### 低风险修改

- 路径适配
- vLLM/OpenAI-compatible API 适配
- timeout/retry
- max token 参数暴露
- token/latency 统计
- 日志落盘
- 评测格式归一化

### 中风险修改

- 对 length finish 使用部分输出
- 对 malformed JSON 做修复解析
- 对单条失败样本 fallback

### 高风险修改

- 改 prompt
- 改 retrieval top-k
- 改 memory build 逻辑
- 改 memory update/delete 行为
- 改排序、过滤、压缩、合并策略

高风险修改默认不允许保留。若必须保留，必须写清楚原因，并通过 smoke test + 小样本对比证明不是性能异常来源。

最终必须生成 `diff_audit.md`，说明：

- 修改了什么
- 为什么改
- 是否影响算法语义
- 对性能的潜在影响
- 是否需要在论文复现说明中披露

## 运行稳定性要求

所有 baseline 必须满足：

- 不允许因单条 QA 报错导致整个 baseline 中止。
- 不允许因一个 baseline 失败导致总控脚本中止。
- 不允许无限等待 worker。
- 不允许无限 retry。
- 不允许再次出现 `max_tokens=512` 这类内部默认值。
- 不允许 Connection refused 后直接整轮失败，必须先检查 vLLM health。
- 不允许 CUDA OOM 后继续污染结果。
- 所有错误必须记录到该 baseline 的 reliability 统计中。

统一 token 策略：

- 明确记录 vLLM `max_model_len`。
- 明确区分 input prompt budget 和 output max tokens。
- QA output 默认不低于 8192。
- memory/summary 类长输出可高于 QA output，但不得超过模型上下文可承受范围。
- 如果发生 length finish：
  - 可用内容可以保留，但必须计数。
  - 不可用内容必须 fallback，并计数。
  - 不得静默忽略。

统一并发策略：

- 先低并发保证完整跑通。
- 如果提高并发，必须确认不会引入 timeout、OOM、队列堆积或性能差异。
- 记录每次 full run 的 worker 数、batch size、vLLM `max_num_seqs`。

## Smoke Test 与 Full Run

完整实验前必须 smoke test：

- A-MEM
- Mem0
- SimpleMem
- HiGMem
- MemGAS

Smoke test 使用 LoCoMo sample 0 的少量 session 和至少 4 条 QA，必须覆盖：

- memory build
- retrieval
- QA generation
- metric computation
- token/latency 统计
- reliability 统计

Smoke 通过标准：

- exit code 为 0
- prediction 数量正确
- metrics 可解析
- runtime/token 字段存在
- 无 fatal Traceback
- 无 CUDA OOM
- 无 Connection refused
- 无 `max_tokens=512`
- 无无限重试
- status 为 completed

Full run 顺序：

1. Full Context
2. A-MEM
3. Mem0
4. SimpleMem
5. HiGMem
6. MemGAS

如果 Full Context 已经有同模型、同数据、同评测口径的完整结果，可以复用，但必须在 summary 里记录路径、配置和完成时间。

## 最终产物

每个 baseline 输出：

- `predictions.json`
- `metrics.json`
- `runtime.json` 或 metrics 内嵌 runtime
- `reliability.json` 或 metrics 内嵌 reliability
- `run.log`
- `status.json`
- `command.env`
- `diff_audit.md`

总控脚本输出：

- `overall_summary.json`
- `overall_summary.md`
- `leaderboard_f1.md`
- `leaderboard_efficiency.md`
- `failure_report.md`

论文主表建议字段：

| Method | F1 | BLEU-1 | ROUGE-L | BERTScore-F1 | Avg Tokens | Token Reduction | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|

论文补充表建议字段：

| Method | Build Time | QA Time | p95 Latency | Total Tokens | Memories | Fallbacks | Length Finish | Empty Pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 最终验收标准

任务只有在以下条件全部满足时才算完成：

- 6 个 baseline 都有完整 cat1-4 结果，或失败项有明确、可复现、可解释的原因。
- 每个 baseline 都有 F1、BLEU-1、ROUGE-L、BERTScore-F1。
- 每个 baseline 都有 token、latency、memory/retrieval、reliability 统计。
- 总 QA 数为 1540，除非明确标记为 smoke/cat5/all-category。
- 所有结果路径清楚，主结果和历史无关结果分开管理。
- 所有代码修改都和原始代码做过 diff audit。
- 没有未解释的性能异常下降。
- 另一个工程师只看 `overall_summary.md`、`command.env` 和 run 目录，就能复现实验、理解指标、判断结果是否可信。

默认优先级：

1. 正确复现口径
2. 完整不中断
3. 指标和效率统计齐全
4. 结果可审计
5. 再考虑速度和并发
