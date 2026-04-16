# SimpleMem 后续工作深度调研与开题清单（扩展相关）

更新时间：2026-04-17  
时间窗：2026-01-05（SimpleMem 首发）至 2026-04-17  
证据范围：仅使用 `arxiv.org/abs`、`arxiv.org/html`、论文正文/表格/参考段落的一手信息

## 1. 分级规则（关联强度）

- `S`：直接在 SimpleMem 上扩展，或在主实验中把 SimpleMem 作为核心 baseline 且有正文证据。
- `A`：主实验未必直接“继承代码”，但明确把 SimpleMem 作为关键对照，或方法机制强依赖其核心设定。
- `B`：未直接对比 SimpleMem，但任务、数据、方法假设与其高度同构，且可迁移到当前仓库。
- `C`：仅背景相关，不建议第一批复现。

## 2. 决策矩阵（核心集）

| 论文 | 关联级别 | 是否出现 SimpleMem 文本证据 | 是否含 LoCoMo | 主指标 | 基线集合（正文可见） | 代码可用性 | 复现成本 | 可迁移到当前仓库的改动点 |
|---|---:|---|---|---|---|---|---|---|
| Omni-SimpleMem (2604.01007) | `S` | 是（相关工作+Baseline章节+主结果） | 是 | LoCoMo F1、Mem-Gallery F1/EM/BLEU | MemVerse, Mem0, SimpleMem, Claude-Mem, A-MEM, MemGPT | 有（GitHub） | 高 | 检索层：dense+sparse 混合检索；分层/渐进检索；多模态原子单元思想的文本化简版 |
| xMemory (2602.02007) | `B` | 否（全文检索未命中 `SimpleMem`） | 是 | LoCoMo/PerLTQA 上 F1、BLEU、ROUGE-L、token/query | Naive RAG, Nemori, LightMem, A-Mem, MemoryOS | 有（项目地址） | 中高 | 检索层：层级检索、代表性选择、不确定性驱动扩展；减少冗余证据 |
| EverMemOS (2601.02163) | `B` | 未在可解析正文中确认（保守处理） | 是 | LoCoMo、LongMemEval（SOTA声明） | 摘要未给完整基线表 | 有（GitHub） | 中高 | 写入层+检索层：trace→consolidation→recollection 生命周期；用户画像更新 |
| Mem-Gallery (2601.03515) | `B` | 否（未见 SimpleMem 文本证据） | 是（作为对照基准讨论） | F1/EM/BLEU 等多维度评测（13种系统） | 多 memory systems（含 A-Mem、MemoryOS 等） | 有（GitHub） | 中（作为评测集接入） | 评测层：补多模态长程记忆维度；可作为下一阶段外部验证集 |
| AgeMem (2601.01885) | `C` | 否 | 否 | 多长程 agent benchmark 任务表现 | 摘要未给完整基线表 | 未在摘要中明确代码链接 | 中 | 策略层思想可借鉴，但与当前 LoCoMo+SimpleMem 主线不对齐 |

## 3. 一手证据摘录（用于分级判定）

### Omni-SimpleMem（`S`）

- 元信息证据：`2026-04-01` 提交，`2026-04-02` 修订，明确是 SimpleMem 后续体系。  
  来源：<https://arxiv.org/abs/2604.01007>
- 正文证据（主实验直接对比）：主结果章节写明“against six baselines”，并在 Baseline 描述中明确列出 `SimpleMem`。  
  来源：<https://arxiv.org/html/2604.01007v2>

### xMemory（`B`）

- 元信息证据：`2026-02-02` 提交，`2026-04-11` v3，实验覆盖 LoCoMo 与 PerLTQA。  
  来源：<https://arxiv.org/abs/2602.02007>
- 正文证据：主表可见对照是 Naive RAG/Nemori/LightMem/A-Mem/MemoryOS，未显示 SimpleMem。  
  来源：<https://arxiv.org/html/2602.02007v3>

### EverMemOS（`B`）

- 元信息证据：`2026-01-05` 提交，`2026-01-09` v2，摘要声明 LoCoMo/LongMemEval 上 SOTA。  
  来源：<https://arxiv.org/abs/2601.02163>
- 正文证据状态：arXiv HTML 入口缺失，ar5iv 转换报错导致正文截断，故不将其判为 `A/S`（保守分级）。  
  来源：<https://ar5iv.labs.arxiv.org/html/2601.02163>

### Mem-Gallery（`B`）

- 元信息证据：`2026-01-07` 提交，13个 memory systems 的系统化评测框架。  
  来源：<https://arxiv.org/abs/2601.03515>
- 正文证据：明确讨论 LoCoMo 的覆盖边界，强调其对多模态长程记忆评测能力有限。  
  来源：<https://arxiv.org/html/2601.03515v1>

### AgeMem（`C`）

- 元信息证据：`2026-01-05` 提交，统一 LTM/STM 的 RL 管理框架。  
  来源：<https://arxiv.org/abs/2601.01885>
- 正文证据：研究主线是长程 agent 任务训练范式，不是 LoCoMo/SimpleMem 比较线。  
  来源：<https://arxiv.org/html/2601.01885v1>

## 4. 开题优先级排序（第一阶段）

评分公式：`优先级 = 关联强度 + 可复现性 + 可移植性 + 预期增益`  
其中：
- 关联强度：`S=10, A=8, B=5, C=2`
- 其余三项按 1~10 打分（越高越优先）

| 论文 | 关联强度 | 可复现性 | 可移植性（到当前仓库） | 预期增益 | 总分 |
|---|---:|---:|---:|---:|---:|
| Omni-SimpleMem | 10 | 6 | 8 | 9 | **33** |
| xMemory | 5 | 7 | 9 | 8 | **29** |
| EverMemOS | 5 | 6 | 7 | 7 | **25** |
| Mem-Gallery | 5 | 8 | 6 | 6 | 25 |
| AgeMem | 2 | 6 | 4 | 5 | 17 |

> 第一阶段 Top3：`Omni-SimpleMem`、`xMemory`、`EverMemOS`

## 5. Top3 最小复现实验单元（本仓库可执行）

### Top1: Omni-SimpleMem（先做“文本可迁移子集”）

- 目标：在不引入完整多模态管线的前提下，先迁移“检索策略”增益。
- 数据：`datasets/locomo/data/locomo10.json`
- 基线命令：
  - `python run_simplemem.py smoke5 --parallel-questions --test-workers 4`
- 代码改造位点：
  - `baseline/SimpleMem/core/hybrid_retriever.py`（混合检索+分层扩展）
  - 可选：`baseline/SimpleMem/models/memory_entry.py`（补可检索结构字段）
- 核心验证指标：
  - Overall F1、Category 1/3/4 F1、平均 retrieval time
- 失败判据：
  - `smoke5` 上 Overall F1 无提升，且延迟上涨超过 15%

### Top2: xMemory（优先做“检索去冗余”）

- 目标：把“decoupling→aggregation”落地成当前系统可用的检索重排。
- 数据：LoCoMo（同上）
- 基线命令：
  - `python run_simplemem.py smoke5 --parallel-questions --test-workers 4`
- 代码改造位点：
  - `baseline/SimpleMem/core/hybrid_retriever.py`
- 最小实现：
  - Top-k 后做语义去冗余（MMR/代表点选择）
  - 再按问题类型做二段扩展（时间/人物问题优先保留链式证据）
- 核心验证指标：
  - Category 3/4 F1 提升、`num_retrieved`下降或持平、token 成本不升
- 失败判据：
  - F1 无提升且检索条目显著膨胀

### Top3: EverMemOS（生命周期机制）

- 目标：验证“写入即组织”是否提升后续检索稳定性。
- 数据：LoCoMo（同上）
- 基线命令：
  - `python run_simplemem.py smoke5 --parallel-questions --test-workers 4`
- 代码改造位点：
  - `baseline/SimpleMem/core/memory_builder.py`（trace→consolidate）
  - `baseline/SimpleMem/core/hybrid_retriever.py`（scene 引导 recollection）
- 最小实现：
  - 在写入阶段增加轻量 consolidation（按主题/人物/时间窗口聚合）
- 核心验证指标：
  - 在 Category 1/4 上的 F1 稳定提升，且 retrieval 波动下降
- 失败判据：
  - F1 下降或 memory build 时间不可接受（>20%）

## 6. 对你当前仓库的落地建议（先后顺序）

1. 先做 `xMemory` 风格检索去冗余（改动最小、收益最可验证）。
2. 再做 `Omni-SimpleMem` 的渐进检索（先文本版，不立即做全多模态）。
3. 第三步再引入 `EverMemOS` 生命周期写入（改动更深，放在第二轮）。

---

## 附：本次调研局限

- arXiv 检索接口对 `SimpleMem` 关键词召回不稳定，已通过“核心集逐篇正文核查”兜底。
- EverMemOS 的 ar5iv HTML 转换异常，正文表格未完整提取；因此该条目保守打 `B`，避免误报 `A/S`。
