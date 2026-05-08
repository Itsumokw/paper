# Keshav 三遍法阅读笔记

基于 S. Keshav "How to Read a Paper" 的三遍阅读法，对项目核心 baseline 论文进行结构化分析。

---

## SimpleMem (2601.02553)

**一句话总结**: 极简 LLM Agent 长期记忆系统，用 LLM 将对话压缩为 bullet-point 摘要，通过关键词精确匹配检索，在 LoCoMo 上达到或超越 Mem0、Zep 等复杂系统。核心论点："简洁即有效"。

**核心方法**: LLM 将对话 session 压缩为结构化摘要 → 存储为 bullet-point 列表 → 查询时用关键词精确匹配检索 → 拼接相关摘要作为上下文生成回答。

**关键数字**: LoCoMo F1 = 0.432（GPT-4o backbone）

**优点**: 极简设计，无需向量数据库/图结构；可解释性强；部署成本低。

**缺点**: 压缩不携带 provenance 元数据（来源、时间戳、说话人），无法追溯；原始对话一旦压缩即丢弃，压缩错误无法修正。

**与你方向的关联**: 暴露两个明确改进靶点——(1) 压缩摘要无 provenance 元数据；(2) 原始对话丢弃不可恢复。这正是 Provenance-aware SimpleMem 要解决的核心问题。

---

## MemGAS (ICLR 2026)

**一句话总结**: 通过多粒度嵌入 + 熵路由 + 图上 Personalized PageRank，实现对话记忆的关联检索，在四个长期记忆基准上优于单粒度基线。

**核心方法（三阶段）**:
1. **多粒度构建**: 每个 session 由 LLM 生成 summary 和 keywords，连同原始 session 文本和 turn 均值，构成四种粒度的嵌入（Contriever 编码）
2. **关联图构建**: 四种粒度节点展平为图，按时间顺序用 GMM 聚类筛选边，构建无向图
3. **检索选择**: query 计算各粒度相似度 → 熵值计算路由权重 → 加权融合 → Top-seed 节点做 PPR → 同一 session 4 节点分数求和排序

**优点**: 多粒度表示思路自然；熵路由 query-adaptive 无需额外训练；图关联建模了跨 session 语义连接；quickstart API 工程可用性强。

**缺点**: 图构建复杂度 O(N * mem_threshold) 扩展性存疑；GMM n_components=2 假设过强；依赖 LLM 预生成 summary/keyword 增加延迟和成本；无遗忘/衰减机制。

**与你方向的关联**: 多粒度表示可视为 provenance 增强——每条记忆附带 summary/keyword 溯源信息。SimpleMem 的 provenance 感知机制可引入图节点标注来源上下文。

---

## Omni-SimpleMem (2604.01007)

**一句话总结**: 用自动化科研管线 AutoResearchClaw 从朴素基线出发，自主发现多模态终身记忆系统 OMNI-SIMPLEMEM，在 LoCoMo 上 F1 从 0.117 提升至 0.598（+411%）。

**核心方法**:
1. **选择性摄入**: CLIP/VAD/Jaccard 轻量编码器过滤冗余多模态信号 → 封装为 Multimodal Atomic Units (MAU)，热存摘要+嵌入、冷存原始数据
2. **渐进式检索**: FAISS 稠密 + BM25 稀疏混合检索，set-union 合并，金字塔机制（摘要→全文→原始内容）按 token 预算逐层展开
3. **知识图谱增强**: LLM 抽取实体-关系三元组，查询时 h-hop 邻域扩展 + 距离衰减打分

**关键数字**: LoCoMo F1 = 0.598（GPT-4o），较 SimpleMem +38.4%；Mem-Gallery F1 = 0.797，+14.3%

**优点**: 自动化管线发现非直觉改进；附录极其详尽；开源代码。

**缺点**: 仅用 LoCoMo 单一子集做迭代开发，存在过拟合评估集风险；Mem-Gallery 和 LoCoMo 用了不同嵌入模型，公平性存疑；知识图谱实体抽取依赖 GPT-4o，成本未讨论。

**与你方向的关联**: MAU 结构天然支持 provenance 追踪——每个记忆单元携带时间戳、模态标签和结构链接。可在 structural links 字段中扩展来源链。关键挑战：冷热分离下 provenance 元数据需同步，金字塔展开每级都需传递 provenance 标注。

---

## ReMe

**一句话总结**: 将记忆"文件化+向量化"的双模 Agent 记忆框架，通过 ReAct 驱动的压缩与持久化机制，在 LoCoMo 上以 86.23 分大幅超越次优 81.55 分。

**核心方法（双路线）**:
1. **文件式 ReMeLight**: 记忆存储为 Markdown 文件（MEMORY.md + 每日日志），推理前自动执行：工具输出压缩（recent/old 分级截断）→ token 检查 → ReActAgent 结构化摘要压缩 → 异步持久化。检索：向量+BM25 混合（0.7:0.3）
2. **向量式 ReMe**: Personal/Procedural/Tool 三类记忆各自独立 Summarizer 和 Retriever，存入向量数据库

**关键数字**: LoCoMo Overall 86.23（超越 MemR3 81.55）；Multi Hop 82.98（次优 71.99）；HaluMem QA Accuracy 88.78

**优点**: 文件式记忆可读可编辑，用户可直接干预；上下文管理流水线完整（压缩/检查/持久化/检索）；多 benchmark 均表现优异。

**缺点**: Memory Integrity 在 HaluMem 上弱于 ProMem（67.72 vs 73.80），持久化可能丢失细节；检索权重固定 0.7:0.3 无自适应；压缩依赖 ReActAgent 调用 LLM 增加延迟和成本。

**与你方向的关联**: 文件式记忆天然可读但缺乏 provenance 链——"这条记忆是从哪段对话的哪个时间点提取的"。SimpleMem 若引入 provenance 追踪（source_dialog_id + timestamp + extraction_reason），可在 ReMe 的文件式架构上实现记忆溯源。ReMe 的 Memory Integrity 弱点恰好是 provenance-aware 设计可解决的核心问题。

---

## LoCoMo (2402.17753)

**一句话总结**: 首个系统评估 LLM 在超长对话（300+ turns）中长期记忆能力的 benchmark，揭示现有模型在时间推理和对抗性问题上与人类存在巨大差距。

**数据集设计**: 50 段超长对话（均 300 turns、9K tokens、最多 35 sessions），每个 agent 被赋予独特 persona 和时间事件图（最多 25 个因果关联事件，跨越 6-12 个月）。

**评估框架**: QA 任务（5 类推理）+ 事件摘要 + 多模态对话生成。QA 分为 single-hop（36%）、multi-hop（14.6%）、temporal（20.6%）、open-domain（3.9%）、adversarial（24.9%），使用 F1 partial match。

**主要发现**:
- 长上下文 LLM 和 RAG 比基础模型提升 22-66%，但仍落后人类 56%
- 时间推理落后人类 73%
- 对抗性问题上 GPT-3.5-16K 仅 2.1% F1，极易幻觉
- RAG 中 observations 作为检索单元效果最佳（top-5 比纯对话提升 5%）
- 长上下文模型在事件摘要上反而不如基础模型（下降 14%）

**作为 benchmark 的局限性**: 仅 50 段对话，统计显著性存疑；LLM 生成数据分布可能与真实对话存在系统性偏差；adversarial 占比高（24.9%）但 open-domain 仅 3.9%，类别分布不均衡。

**与你方向的关联**: LOCOMO 的 observations 本质上是 provenance-aware 的记忆抽象——每条 observation 可追溯到具体 turn ID。"长上下文不等于深度理解"直接支持 SimpleMem 的设计哲学：需要结构化的、可溯源的记忆表示。Adversarial 类别测试模型区分"可回答"与"不可回答"的能力，呼应了记忆系统需要置信度/来源判断的需求。
