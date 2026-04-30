# 长期记忆 / SimpleMem 模块改进调研包

生成日期：2026-04-30  
工作目录：`/home/stu0032/paper`  
资料目录：`research/agent_memory_survey_20260430/`  
论文 PDF：`research/agent_memory_survey_20260430/papers/`  
网页资料：`research/agent_memory_survey_20260430/web/`

## 0. 最短结论

你现在最稳的科研主线不是“再提出一个新的 memory 框架”，而是：

> 在 SimpleMem / LoCoMo 复现基础上，系统研究语义压缩记忆的证据保真问题，并实现可追踪 source provenance、可控 raw episode fallback 和 evidence-level retrieval evaluation。

建议题目：

中文：**面向长期对话问答的证据保真 SimpleMem：压缩记忆、原文回退与可归因检索评测**

英文：**Evidence-Preserving SimpleMem: Measuring Compressed Memory, Raw-Episode Fallback, and Attributable Retrieval for Long-Term Conversational QA**

不要把以下内容写成“首次提出”：

- provenance-aware memory
- compressed memory + raw fallback
- hybrid retrieval
- query-adaptive retrieval
- graph memory

这些方向都有大量相关工作。更好的贡献表述是：

1. 你把 SimpleMem 的压缩记忆接回 LoCoMo 原始 turn-level evidence。
2. 你定义 evidence coverage、evidence recall@k、fallback trigger precision/recall、compression ratio vs evidence recall 等可测指标。
3. 你把错误拆成压缩丢证据、检索没召回、生成没用对三类。
4. 你在同一 LLM、同一数据、同一 token budget 下比较 compressed-only、raw-only、compressed+raw fallback、oracle fallback。

## 1. 三智能体讨论结论

### 1.1 探索智能体：仓库和模块结论

当前仓库已经不是单一 SimpleMem，而是长期记忆复现工作区：

- `baseline/SimpleMem/`
- `baseline/HiGMem/`
- `baseline/xMemory/`
- `baseline/LightMem/`
- `baseline/MemMachine/`
- `baseline/MAGMA/`
- `baseline/MemGAS/`
- `datasets/locomo/data/locomo10.json`
- `reproductions/successful/SimpleMem/2026-04-27_qwen3-8b_locomo10-full_16mem-30qa/`

已确认的关键事实：

- SimpleMem 当前 `MemoryEntry` 没有 `source_turn_ids`、`source_dialogue_ids`、`source_span`、`provenance` 字段。
- LoCoMo 数据有 evidence 字段，可用于 evidence-level evaluation。
- 本地 LoCoMo10 里 1986 个 QA 中约 1982 个有非空 evidence。
- 当前转换代码会把原始 `dia_id` 丢掉，改成连续整数；这会阻断 memory entry 到 gold evidence 的追踪。

最小研究闭环：

```text
LoCoMo dia_id
  -> Dialogue.source_turn_id
  -> MemoryEntry.source_turn_ids / source_window
  -> VectorStore 持久化 source ids
  -> Retriever 返回 retrieved_source_turn_ids
  -> evaluator 计算 evidence recall@k / MRR / nDCG
  -> fallback 用 source ids 展开 raw turns
```

### 1.2 调研智能体：领域地图

文献可以分成七组：

1. 基准与评测：LoCoMo、LongMemEval、BEAM、LoCoMo-Plus、MemoryAgentBench。
2. 压缩记忆：SimpleMem、LightMem、ReadAgent、MemoryBank。
3. 原文保真与 fallback：SmartSearch、MemMachine、D-Mem、EMem。
4. 层级 / 图 / 结构化记忆：xMemory、HingeMem、StructMem、HyperMem、MAGMA、Zep、A-MEM、MemOS。
5. 多 agent / 自演化 memory：MemMA、MemoryOS、Mem0。
6. 证据归因与引用评测：AIS、ALCE、ALiiCE、QASPER。
7. 检索基础设施：RAG、DPR、FiD、ColBERT、SPLADE、HyDE、RRF。

### 1.3 批评智能体：最重要的警告

最容易被质疑的点：

- LoCoMo 太短，很多样本已经可以被长上下文模型直接塞进去。
- 如果只报 answer F1，无法证明改的是 memory，而不是提示词、reranker 或上下文预算。
- 如果没有 raw-only、full-context、BM25、dense、hybrid、oracle evidence reader，对照不可信。
- “semantic lossless compression” 这个词风险很高；建议改成 evidence-preserving compression，并给出可测定义。
- raw fallback 已经很接近 ReadAgent / D-Mem，必须强调 SimpleMem-specific 的 evidence tracing 和 trigger evaluation。

## 2. 推荐研究问题

### RQ1：SimpleMem 压缩记忆到底丢了哪些证据？

指标：

- `compression evidence coverage`
- `evidence loss rate`
- `compressed answerability`
- `compression ratio vs evidence recall`

解释：

如果 gold evidence 在原文中存在，但没有任何 compressed memory 覆盖，那是写入/压缩阶段错误。

### RQ2：什么时候应该回退到 raw episode？

触发器只能使用非 oracle 信号：

- 检索分数
- score margin
- dense/sparse 分歧
- 实体/时间匹配缺口
- 问题类型预测
- LLM adequacy/self-check

必须比较：

- never fallback
- always fallback
- trigger fallback
- oracle fallback

指标：

- fallback precision
- fallback recall
- fallback rate
- extra tokens
- answer delta
- latency delta

### RQ3：证据级指标能否解释 answer-only 指标解释不了的失败？

指标：

- evidence recall@k
- MRR
- nDCG
- answer correctness conditioned on evidence retrieved
- answer correctness conditioned on evidence missing

价值：

这能拆出三类错误：

1. memory construction 丢证据。
2. retrieval 没召回。
3. reader/generator 没用对。

### RQ4：冲突和过期记忆如何裁决？

比较策略：

- compressed-priority
- raw-priority
- recency-priority
- evidence-priority

指标：

- conflict set answer F1
- conflict trigger rate
- correction rate
- error introduction rate

## 3. 必做 baseline

至少需要这些：

| baseline | 作用 |
|---|---|
| Full-context LLM | 当前数据是否已经可被长上下文直接解决 |
| Recent-window | 排除“只看最近几轮就够”的可能 |
| BM25 raw turn/chunk | 强关键词基线，尤其人名、日期、地点 |
| Dense raw retrieval | 向量检索基本线 |
| BM25 + dense + RRF | 标准 hybrid retrieval 线 |
| Dense + reranker | 排名瓶颈对照 |
| Summary-only / SimpleMem-only | 压缩记忆单独能力 |
| Raw-episode-only | 不压缩，只检索原文 |
| SimpleMem official/default | 复现主基线 |
| Proposed compressed + triggered fallback | 你的方法 |
| Oracle evidence reader | 给 gold evidence，只测 answer generator 上限 |
| Oracle fallback | 判断 trigger 离理想策略有多远 |

统一约束：

- 同一 reader LLM。
- 同一 embedding 模型。
- 同一数据 split。
- 同一 context budget。
- 同一 judge / metric 脚本。
- 报 accuracy、evidence recall、tokens、latency。

## 4. 论文与资料清单

### 4.1 第一层：必须精读

这些直接决定你的选题边界。

| 论文 | PDF | 创新点 | 优势 | 劣势 / 没做好的点 | 对当前项目的启发 |
|---|---|---|---|---|---|
| LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents, ACL 2024 | `papers/2402.17753_LoCoMo.pdf` | 构建长期多 session 对话记忆 benchmark，含 QA、事件摘要、多模态对话生成 | 当前仓库主数据来源；问题类型覆盖 single-hop、multi-hop、temporal、open-domain | 平均长度到 2026 已不算极长；社区对答案和 judge 可靠性有争议 | 必须读数据构建、evidence、问题类别和评测协议 |
| LongMemEval, ICLR 2025 | `papers/2410.10813_LongMemEval.pdf` | 把长期记忆拆成 indexing、retrieval、reading；测信息抽取、多 session 推理、时间、更新、拒答 | 比 LoCoMo 更强调 sustained assistant memory 和知识更新 | 部分设置仍可能被长上下文吃掉 | 可作为第二数据集或方法论对照 |
| BEAM / LIGHT, ICLR 2026 | `papers/2510.27246_BEAM_LIGHT.pdf` | 生成 500K 到 10M token 级长期对话 benchmark，并提出 LIGHT 记忆框架 | 解决 LoCoMo/LongMemEval 太短的问题 | 复现成本高，不适合作为第一阶段主实验 | 可做小子集或未来验证，开题时用来说明 LoCoMo 局限 |
| SimpleMem, 2026 arXiv | `papers/2601.02553_SimpleMem.pdf` | 语义结构化压缩、递归 consolidation、自适应检索 | 当前仓库主复现对象；工程基础已跑通 | 当前本地实现缺 source provenance，压缩是否保留证据难验证 | 你的所有改动应围绕它做受控实验 |
| EMem, 2025 arXiv | `papers/2511.17208_EMem.pdf` | EDU / event-centric memory，保留 source turn attribution，异构图关联 | 直接撞到 provenance + event-level memory | 如果你不读，会误以为 source attribution 是新点 | 必须明确差异：你是在 SimpleMem 上做证据保真评测和 fallback |
| ReadAgent, 2024 | `papers/2402.09727_ReadAgent.pdf` | gist memory + 需要细节时回查原文 | 与 raw fallback 极近 | 面向长文阅读，不是 LoCoMo 对话 memory | 证明 raw fallback 不是新点；借鉴 lookup trigger |
| D-Mem, 2026 arXiv | `papers/2603.18631_DMem.pdf` | dual-process memory：快速检索 + full deliberation fallback + gating | 正中 compressed/fast vs raw/full fallback 的主题 | LoCoMo 分数和设置需谨慎复核 | 你的 trigger 需要和 D-Mem 对照，不能只说“我也 fallback” |
| SmartSearch, 2026 arXiv | `papers/2603.15599_SmartSearch.pdf` | 原始对话上做 deterministic recall + rank fusion；强调 ranking beats structure | 对复杂结构化 memory 构成强挑战 | 依赖强检索/rerank pipeline，不一定是真 memory 写入系统 | 必须有 raw retrieval / reranker baseline |
| MemMachine, 2026 arXiv | `papers/2604.04853_MemMachine.pdf` | ground-truth-preserving memory，保留 raw episodes，contextualized retrieval | 强化“不要过度压缩”的路线 | 工程系统较重；论文分数需受控复现 | 支持你做 compressed-only vs raw-only vs hybrid |
| AIS: Measuring Attribution in NLG | `papers/2112.12870_AIS.pdf` | 定义 Attributable to Identified Sources | 给“可归因回答”提供正式评测框架 | 不是 memory 系统 | 用来定义 answer 是否被 retrieved evidence 支持 |
| ALCE, EMNLP 2023 | `papers/2305.14627_ALCE.pdf` | LLM citation generation benchmark 和自动引用评测 | 对 provenance/citation 评测很关键 | 任务是开放域 QA，不是长期 memory | 你的 evidence citation 指标可借鉴它 |
| ALiiCE, NAACL 2025 | `papers/2406.13375_ALiiCE.pdf` | positional fine-grained citation evaluation | 强调引用位置和细粒度证据 | 复杂度高，第一阶段不必完整复现 | 可用于扩展：answer span 到 source turn/span 的细粒度归因 |

### 4.2 第二层：结构化 / 层级 / 自适应 memory

| 论文 | PDF | 创新点 | 优势 | 劣势 / 没做好的点 | 对当前项目的启发 |
|---|---|---|---|---|---|
| xMemory, 2026 arXiv | `papers/2602.02007_xMemory.pdf` | 层级 memory：theme / semantic / episode / raw message | 说明长期记忆应支持逐层展开 | 复现成本中高；和 SimpleMem 代码结构不同 | 你的 compressed memory + raw window 可表述为轻量两层 memory |
| HingeMem, TheWebConf 2026 | `papers/2604.06845_HingeMem.pdf` | person/time/location/topic 边界触发 segmentation + query-adaptive retrieval | 与 LoCoMo 的人、时间、地点强相关 | 自适应检索不是新点 | 可借鉴 query-type routing 和 event boundary |
| StructMem, ACL 2026 | `papers/2604.21748_StructMem.pdf` | 保留 event-level bindings 和 cross-event connections | 对 temporal/multi-hop 有针对性 | 图/结构构建可能脆弱且成本高 | 你可做轻量 event/source structure，而不是完整图系统 |
| HyperMem, ACL 2026 | `papers/2604.08256_HyperMem.pdf` | hypergraph memory 捕获高阶关联 | 适合多事实共同决定答案的问题 | 实现复杂，第一阶段不建议追 | 用作相关工作，不作为复现首选 |
| MAGMA, ACL 2026 | `papers/2601.03236_MAGMA.pdf` | semantic/temporal/causal/entity 多图表示与策略遍历 | 可解释路径强 | 工程复杂，本仓库复现成本高 | 你的 evidence path 可作为简化可解释路径 |
| MemMA, 2026 arXiv | `papers/2603.18718_MemMA.pdf` | 多 agent 协调 construction/retrieval/utilization，并通过 probe QA 修复 memory | 把 downstream failure 反馈到 memory construction | 系统复杂，成本高 | 你可以借鉴 probe QA 做 compressed answerability |
| LightMem, ICLR 2026 | `papers/2510.18866_LightMem.pdf` | sensory / short-term / long-term 三阶段，sleep-time update | 效率曲线做得好 | 和 SimpleMem 主线有相似压缩/consolidation | 借鉴 token/call/latency 报告方式 |
| Lightweight LLM Agent Memory with SLMs, ACL 2026 | `papers/2604.07798_Lightweight_SLM_Memory.pdf` | 用小模型做 retrieval/writing/consolidation，固定检索预算 | 强调线上低延迟 | 与本地 Qwen/vLLM 设置不同 | 说明效率和模块化也是可写贡献 |
| Omni-SimpleMem, 2026 arXiv | `papers/2604.01007_OmniSimpleMem.pdf` | SimpleMem 多模态扩展 | 与 SimpleMem 系列强相关 | 多模态会放大工程风险 | 先读，但第一阶段建议只迁移文本检索思想 |

### 4.3 第三层：系统与产品化 memory

| 论文 | PDF | 创新点 | 优势 | 劣势 / 没做好的点 | 对当前项目的启发 |
|---|---|---|---|---|---|
| MemGPT, 2023 | `papers/2310.08560_MemGPT.pdf` | OS-style virtual context：main memory + archival memory | agent memory 经典起点 | 需要 agent 自主管理 memory，稳定性难控 | 用作背景，不是你直接对照重点 |
| MemoryBank, 2023 | `papers/2305.10250_MemoryBank.pdf` | 长期 user memory + forgetting / updating | 早期用户画像 memory 代表 | 与 LoCoMo 评测线不完全一致 | 用于介绍长期个性化 memory 起源 |
| A-MEM, 2025 | `papers/2502.12110_AMEM.pdf` | Zettelkasten 风格 agentic memory，动态链接和更新 | 强组织能力，常被后续论文引用 | LLM 调用和组织成本高 | 选作代表性 structured memory baseline |
| Mem0, 2025 | `papers/2504.19413_Mem0.pdf` | 生产化 scalable memory，含 graph variant | 工程影响大，评测覆盖 LoCoMo | 公开分数和竞品实现受争议 | 可读，但实验引用要写“作者报告” |
| Zep Temporal KG, 2025 | `papers/2501.13956_Zep.pdf` | temporal knowledge graph for agent memory | 对用户信息更新和时序冲突有启发 | 产品/系统色彩强；LoCoMo 对比曾有争议 | 对 conflict-aware memory 很有用 |
| MemoryOS of AI Agent, EMNLP 2025 | `papers/2506.06326_MemoryOS_of_AI_Agent.pdf` | STM/MTM/LTM 管理与热度机制 | 与 OS memory 管理类比清晰 | 概念多，落地成本较高 | 写 related work 时可用 |
| MemOS, 2025 arXiv | `papers/2507.03724_MemOS.pdf` | MemCube、provenance、versioning、memory scheduling | 把 memory 当系统资源管理 | 太大，不适合本科阶段照搬 | provenance/versioning 可作为设计依据 |

### 4.4 评测扩展和安全风险

| 论文 / 资料 | 文件 | 创新点 | 优势 | 劣势 / 没做好的点 | 对当前项目的启发 |
|---|---|---|---|---|---|
| LoCoMo-Plus, 2026 | `papers/2602.10715_LoCoMo_Plus.pdf` | 从 factual recall 扩展到隐式约束和 cognitive memory | 能暴露 LoCoMo 原始 QA 看不到的问题 | 还较新，需谨慎复核 | 可做未来验证 |
| MemoryAgentBench, 2025 | `papers/2507.05257_MemoryAgentBench.pdf` | 增量多轮交互评测 memory agent | 强调 retrieval、test-time learning、long-range understanding、forgetting | 与当前 LoCoMo 管线不同 | 支持你讨论 selective forgetting 和更新 |
| Hindsight Agent Memory Benchmark manifesto | `web/Hindsight_Agent_Memory_Benchmark_Manifesto.html` | 批评 LoCoMo/LongMemEval 只测聊天 recall 且太短 | 对开题风险很有帮助 | 公司博客，不能当正式论文证据 | 用作工程观点和风险提示 |
| Hindsight BEAM writeup | `web/Hindsight_BEAM_SOTA.html` | 强调 10M token 级别才真正压力测试 memory | 指出现有 benchmark 的 context-stuffing 问题 | 公司博客，有宣传成分 | 用作补充资料，不当主证据 |
| Zep critique of Mem0 LoCoMo evaluation | `web/Zep_Mem0_LoCoMo_critique.html` | 指出竞品 benchmark 实现和 LoCoMo 缺陷 | 说明 memory benchmark 很容易被实现细节影响 | 公司博客，利益相关 | 报告中提醒：所有外部 SOTA 只能写“作者报告” |
| Mem0 memory evaluation docs | `web/Mem0_Memory_Evaluation.html` | 展示生产系统如何看 LoCoMo / LongMemEval | 工程视角丰富 | 不是学术论文 | 参考指标和成本维度 |

### 4.5 检索基础，不要当作新贡献

| 论文 / 方法 | 文件 | 创新点 | 优势 | 对当前项目的启发 |
|---|---|---|---|---|
| RAG, 2020 | `papers/2005.11401_RAG.pdf` | parametric + non-parametric memory | 现代检索增强生成基础 | 你的 raw retrieval baseline 必须尊重 RAG 基本设定 |
| DPR, 2020 | `papers/2004.04906_DPR.pdf` | dense passage retrieval | 向量检索基础 | dense-only baseline |
| FiD, 2020 | `papers/2007.01282_FiD.pdf` | reader 融合多个 retrieved passages | strong reader 思路 | 检索到了多个 evidence 时如何读 |
| ColBERT, 2020 | `papers/2004.12832_ColBERT.pdf` | late interaction retrieval | 排名效果强 | SmartSearch/重排方向背景 |
| SPLADE, 2021 | `papers/2107.05720_SPLADE.pdf` | neural sparse retrieval | 兼具 lexical 和 learned signal | hybrid retrieval 背景 |
| HyDE, 2022 | `papers/2212.10496_HyDE.pdf` | hypothetical document expansion | 查询扩展基础 | query rewriting 不要当新点 |
| RRF | `web/RRF_Google_Research.html` | reciprocal rank fusion | 简单有效融合多个 ranker | BM25+dense 的标准融合方法 |
| QASPER, NAACL 2021 | `papers/2105.03011_QASPER.pdf` | 带 evidence 的论文 QA 数据集 | 证据标注和 QA 设计可借鉴 | 不是 memory 数据，但可启发 evidence evaluation |

## 5. 建议阅读顺序

第一周：

1. LoCoMo
2. SimpleMem
3. EMem
4. ReadAgent
5. D-Mem
6. SmartSearch
7. MemMachine
8. AIS / ALCE

第二周：

1. LongMemEval
2. BEAM
3. xMemory
4. HingeMem
5. StructMem
6. LightMem
7. MAGMA

第三周：

1. MemGPT
2. MemoryBank
3. A-MEM
4. Mem0
5. Zep
6. MemOS / MemoryOS
7. RAG / DPR / FiD / ColBERT / SPLADE / HyDE / RRF

## 6. 第一阶段实验路线

### Phase 0：复现实验基线锁定

目标：

- 固定 LoCoMo10、reader LLM、embedding、judge、context budget。
- 复跑 SimpleMem 原版。
- 添加 raw BM25、raw dense、raw hybrid、full-context、recent-window baseline。

输出：

- `baseline_table.md`
- `run_config.json`
- 每个 baseline 的 answer F1、tokens、latency。

### Phase 1：source provenance 接线

代码目标：

- 保留 LoCoMo `dia_id`。
- `Dialogue` 增加 `source_turn_id` / `session_id` / `turn_index`。
- `MemoryEntry` 增加 `source_turn_ids` / `source_window` / `source_dialogue_ids`。
- Vector store 写入和返回 provenance。
- Retriever 输出 source ids。

评测：

- memory evidence coverage。
- retrieval evidence recall@k。
- evidence MRR / nDCG。

### Phase 2：raw fallback

触发器从简单到复杂：

1. score threshold。
2. dense-sparse disagreement。
3. entity/time missing。
4. LLM adequacy check。
5. combined logistic/rule trigger。

对照：

- never fallback。
- always fallback。
- oracle fallback。
- triggered fallback。

### Phase 3：冲突与时间更新

先构造或筛选：

- 同一实体信息更新的问题。
- temporal / contradiction 类 QA。

比较：

- compressed-priority。
- raw-priority。
- recency-priority。
- evidence-priority。

### Phase 4：跨 benchmark 小验证

优先：

- LongMemEval-S 小子集。
- BEAM 小 token-tier 子集。
- LoCoMo-Plus 小子集。

目的不是追榜，而是证明方法不是只适配 LoCoMo10。

## 7. 论文写法建议

不要写：

> 本文提出一种全新的长期记忆系统，并达到 SOTA。

建议写：

> 本文以 SimpleMem 为代表性压缩记忆系统，在统一 LoCoMo10 协议下研究语义压缩记忆的证据保真问题。我们为压缩 memory entry 增加 source-level provenance，构建 evidence-level retrieval evaluation，并比较 compressed-only、raw-only 与 selective raw fallback 在不同问题类型和 token budget 下的权衡。

贡献点可以写成：

1. A provenance extension for SimpleMem that reconnects compressed memories to original dialogue evidence.
2. An evidence-level evaluation protocol that separates compression loss, retrieval miss, and reader failure.
3. A selective raw-episode fallback mechanism evaluated against never/always/oracle fallback under fixed budgets.
4. A controlled empirical analysis on LoCoMo10 and optional LongMemEval/BEAM subsets.

## 8. 文件清单

本目录包含：

- `agent_memory_survey.md`：本报告。
- `papers/`：38 篇论文 PDF。
- `web/`：网页资料和系统文档快照。
- `notes/`：预留给后续精读笔记。

## 9. 来源链接

核心来源：

- LoCoMo: https://arxiv.org/abs/2402.17753
- LongMemEval: https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
- BEAM: https://arxiv.org/abs/2510.27246
- SimpleMem: https://arxiv.org/abs/2601.02553
- EMem: https://arxiv.org/abs/2511.17208
- ReadAgent: https://arxiv.org/abs/2402.09727
- D-Mem: https://arxiv.org/abs/2603.18631
- SmartSearch: https://arxiv.org/abs/2603.15599
- MemMachine: https://arxiv.org/abs/2604.04853
- AIS: https://arxiv.org/abs/2112.12870
- ALCE: https://arxiv.org/abs/2305.14627
- ALiiCE: https://arxiv.org/abs/2406.13375
- Hindsight benchmark manifesto: https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark
- Zep critique of Mem0 LoCoMo evaluation: https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/

