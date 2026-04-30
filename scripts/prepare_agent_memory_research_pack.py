#!/usr/bin/env python3
"""Build a local reading pack for LLM agent memory research."""

from __future__ import annotations

import csv
import json
import re
import shutil
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path("/home/stu0032/paper")
PACK_ROOT = ROOT / "runs" / "research_packs" / "agent_memory_20260429"
PDF_DIR = PACK_ROOT / "papers"
ARTICLE_DIR = PACK_ROOT / "articles"
NOTE_DIR = PACK_ROOT / "notes"


@dataclass
class Source:
    key: str
    title: str
    year: str
    category: str
    source_url: str
    download_url: str
    kind: str
    priority: str
    why_read: str
    innovation: str
    strengths: str
    weaknesses: str
    gaps: str
    relevance: str
    suggested_use: str


SOURCES: list[Source] = [
    Source(
        key="01_survey_memory_mechanism_llm_agents",
        title="A Survey on the Memory Mechanism of Large Language Model based Agents",
        year="2024",
        category="survey",
        source_url="https://arxiv.org/abs/2404.13501",
        download_url="https://arxiv.org/pdf/2404.13501",
        kind="pdf",
        priority="core",
        why_read="先建立术语和模块地图，避免把存储、检索、反思、更新混成一个贡献。",
        innovation="系统整理 LLM agent memory 的机制、结构和评测维度。",
        strengths="适合作为开题时的 taxonomy；能帮助给 SimpleMem/LightMem/xMemory/MemMachine 对齐模块。",
        weaknesses="综述覆盖到 2024，缺少 2025-2026 新系统和 LoCoMo 之后的争议。",
        gaps="需要补读最新系统论文，并用本地复现实验验证综述里的分类是否仍然成立。",
        relevance="是你后续写 related work 和方法分解的骨架。",
        suggested_use="第一天泛读，建立术语表和模块矩阵。",
    ),
    Source(
        key="02_locomo_acl2024",
        title="Evaluating Very Long-Term Conversational Memory of LLM Agents",
        year="2024 ACL",
        category="benchmark",
        source_url="https://aclanthology.org/2024.acl-long.747/",
        download_url="https://aclanthology.org/2024.acl-long.747.pdf",
        kind="pdf",
        priority="core",
        why_read="你的复现主数据集；不读清楚无法判断分数、category 和错误来源。",
        innovation="用 persona 和 temporal event graph 生成超长多会话对话，并设计 QA、总结、多模态生成任务。",
        strengths="公开、被大量 memory 系统采用；有 long-range temporal/causal QA。",
        weaknesses="规模仍不算很长；QA key 和 LLM judge 可靠性有争议；不少系统会针对 LoCoMo 调参。",
        gaps="需要 evidence-level 审计、人工核验子集和跨 benchmark 验证。",
        relevance="所有本地复现系统都围绕 LoCoMo；必须作为基准理解。",
        suggested_use="精读 dataset construction、QA category、evaluation 部分。",
    ),
    Source(
        key="03_longmemeval",
        title="LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory",
        year="2024 / ICLR 2025",
        category="benchmark",
        source_url="https://arxiv.org/abs/2410.10813",
        download_url="https://arxiv.org/pdf/2410.10813",
        kind="pdf",
        priority="core",
        why_read="补足 LoCoMo 对更新、拒答、时间推理和多会话交互的覆盖不足。",
        innovation="把长期记忆拆成信息抽取、多会话推理、时间推理、知识更新、abstention。",
        strengths="评测维度清楚；给出 indexing/retrieval/reading 阶段优化框架。",
        weaknesses="部分设置仍可被强长上下文和上下文管理策略利用。",
        gaps="需要更大规模、更强 evidence provenance 和真实在线写入设置。",
        relevance="LightMem 和 MemMachine 都重视这个 benchmark；适合验证模块泛化。",
        suggested_use="精读任务分类、错误分析和优化设计。",
    ),
    Source(
        key="04_perltqa",
        title="PerLTQA: A Personal Long-Term Memory Question Answering Dataset",
        year="2024",
        category="benchmark",
        source_url="https://arxiv.org/abs/2402.16288",
        download_url="https://arxiv.org/pdf/2402.16288",
        kind="pdf",
        priority="core",
        why_read="xMemory 使用的辅助 benchmark；更强调个性化长期记忆。",
        innovation="围绕个人长期信息构造 QA，区分 episodic 与 semantic memory。",
        strengths="适合测试个性化和长期事实整合，不完全依赖 LoCoMo。",
        weaknesses="与开放式 agent 行动、记忆污染、隐私问题仍有距离。",
        gaps="需要与 LoCoMo/LongMemEval 联合评估，避免单 benchmark 过拟合。",
        relevance="用于检验 xMemory 式层级检索是否能迁移。",
        suggested_use="作为第二评测集候选精读。",
    ),
    Source(
        key="05_membench",
        title="MemBench: Memorized Image Triggering Visual Question Answering",
        year="2025 ACL Findings",
        category="benchmark",
        source_url="https://aclanthology.org/2025.findings-acl.989/",
        download_url="https://aclanthology.org/2025.findings-acl.989.pdf",
        kind="pdf",
        priority="extended",
        why_read="提醒你 memory 不只是文本 recall，也涉及触发、观察者视角和多模态。",
        innovation="围绕记忆触发的 VQA 评测，区分记住事实和使用事实。",
        strengths="能拓宽 LoCoMo 文本 QA 之外的视角。",
        weaknesses="与当前四个文本复现系统的直接关系较弱。",
        gaps="如果短期只做 LoCoMo 模块，可以作为延伸阅读而非主线。",
        relevance="帮助思考未来是否扩展到 LoCoMo-V 或多模态 memory。",
        suggested_use="延伸阅读，暂不作为第一阶段实验目标。",
    ),
    Source(
        key="06_memorybench",
        title="MemoryBench: Towards Measuring Long-Term Memory Ability of LLM Agents",
        year="2025",
        category="benchmark",
        source_url="https://arxiv.org/abs/2510.17281",
        download_url="https://arxiv.org/pdf/2510.17281",
        kind="pdf",
        priority="core",
        why_read="关注持续交互和用户反馈，比 LoCoMo 的一次性 QA 更接近长期 agent。",
        innovation="把 memory 与 continual learning / user feedback 结合到评测中。",
        strengths="适合检验 memory update、遗忘和长期服务期表现。",
        weaknesses="复现成本和数据适配可能高于 LoCoMo。",
        gaps="需要看其任务是否能被你当前代码栈低成本接入。",
        relevance="对 update-aware memory 选题有直接启发。",
        suggested_use="精读评测协议，作为二期 benchmark。",
    ),
    Source(
        key="07_structmemeval",
        title="Evaluating Memory Structure in LLM Agents",
        year="2026",
        category="benchmark",
        source_url="https://arxiv.org/abs/2602.11243",
        download_url="https://arxiv.org/pdf/2602.11243",
        kind="pdf",
        priority="core",
        why_read="它专门问：memory 是否被组织成可用结构，而不只是召回事实。",
        innovation="用账本、todo、树等结构化任务测试 memory organization。",
        strengths="非常适合反驳单纯 top-k recall 分数；可检验 xMemory/graph memory 的真实价值。",
        weaknesses="场景较小，初始实验不一定足以证明系统排名。",
        gaps="可以借鉴任务思想，自建 LoCoMo 风格的结构化扰动集。",
        relevance="与你想做模块改进时的“结构是否有用”问题直接相关。",
        suggested_use="精读，尤其是任务设计。",
    ),
    Source(
        key="08_beam",
        title="BEAM: Beyond a Million Tokens, Benchmarking Long-Context Long-Term Memory",
        year="2025",
        category="benchmark",
        source_url="https://arxiv.org/abs/2510.27246",
        download_url="https://arxiv.org/pdf/2510.27246",
        kind="pdf",
        priority="core",
        why_read="如果只在 LoCoMo 上涨分，容易被质疑不是长期记忆；BEAM 用更长上下文施压。",
        innovation="把对话记忆压力扩展到百万乃至千万 token 级别。",
        strengths="能压制“把全文塞进上下文”的捷径，暴露扩展性问题。",
        weaknesses="实验成本更高，可能不适合第一阶段完整复现。",
        gaps="可以先做小规模子集或模拟 scaling 曲线。",
        relevance="用于验证你的模块是否真的 scalable。",
        suggested_use="精读问题设定和指标，作为扩展性论据。",
    ),
    Source(
        key="09_memgpt",
        title="MemGPT: Towards LLMs as Operating Systems",
        year="2023",
        category="system",
        source_url="https://arxiv.org/abs/2310.08560",
        download_url="https://arxiv.org/pdf/2310.08560",
        kind="pdf",
        priority="core",
        why_read="现代 agent memory 分层/虚拟上下文思想的代表。",
        innovation="把 LLM 视作管理虚拟上下文的操作系统，主动在 memory tiers 间搬运信息。",
        strengths="架构概念强，影响 Letta、MemoryOS/MemOS 和后续 memory layer。",
        weaknesses="依赖强模型自我管理；评测与当下 LoCoMo/LongMemEval 口径不同。",
        gaps="需要与低成本、本地模型、证据可溯源结合。",
        relevance="帮助理解 memory management policy，而不是只做检索。",
        suggested_use="精读 system design 和 multi-session chat 实验。",
    ),
    Source(
        key="10_generative_agents",
        title="Generative Agents: Interactive Simulacra of Human Behavior",
        year="2023 UIST",
        category="system",
        source_url="https://arxiv.org/abs/2304.03442",
        download_url="https://arxiv.org/pdf/2304.03442",
        kind="pdf",
        priority="core",
        why_read="memory stream + reflection + planning 的经典源头。",
        innovation="用自然语言 memory stream、reflection synthesis 和 planning 产生可信 agent 行为。",
        strengths="清楚展示 memory 如何影响长期行为，而不仅是 QA。",
        weaknesses="偏模拟环境，memory 检索和评测不够现代化。",
        gaps="可借鉴 reflection，但不要把简单反思摘要当成新贡献。",
        relevance="LoCoMo 的对话生成背景和 agent memory 概念来源之一。",
        suggested_use="泛读架构，重点看 memory/reflection ablation。",
    ),
    Source(
        key="11_memorybank",
        title="MemoryBank: Enhancing Large Language Models with Long-Term Memory",
        year="2023",
        category="system",
        source_url="https://arxiv.org/abs/2305.10250",
        download_url="https://arxiv.org/pdf/2305.10250",
        kind="pdf",
        priority="core",
        why_read="早期长期用户记忆系统，讨论遗忘曲线和人格建模。",
        innovation="引入长期 memory bank、重要性、强化/遗忘机制。",
        strengths="贴近 AI companion 和个性化长期对话。",
        weaknesses="现代证据召回、更新冲突和评测严谨性不足。",
        gaps="适合作为 baseline 思想，不足以支撑当下论文 novelty。",
        relevance="A-MEM/LightMem/LoCoMo 系统经常对照或继承这条线。",
        suggested_use="泛读，提取遗忘和用户画像概念。",
    ),
    Source(
        key="12_recursive_summarizing_dialogue_memory",
        title="Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models",
        year="2023",
        category="system",
        source_url="https://arxiv.org/abs/2308.15022",
        download_url="https://arxiv.org/pdf/2308.15022",
        kind="pdf",
        priority="extended",
        why_read="理解 summary-consolidation 的简单强 baseline。",
        innovation="递归摘要长对话，让 LLM 在有限上下文内保留长期信息。",
        strengths="方法简单，适合作为压缩记忆 baseline。",
        weaknesses="摘要会丢 source provenance、时间细节和少数关键证据。",
        gaps="你的改进若涉及压缩，必须证明比递归摘要更忠实或更可检索。",
        relevance="与 SimpleMem/LightMem 的压缩路线对照。",
        suggested_use="延伸阅读，作为 baseline 设计参考。",
    ),
    Source(
        key="13_longmem",
        title="LongMem: Augmenting Language Models with Long-Term Memory",
        year="2023 NeurIPS",
        category="model_memory",
        source_url="https://arxiv.org/abs/2306.07174",
        download_url="https://arxiv.org/pdf/2306.07174",
        kind="pdf",
        priority="extended",
        why_read="区分模型级 memory 与外部 memory layer。",
        innovation="把长期记忆作为模型增强机制，而不是纯外部数据库检索。",
        strengths="帮助思考 external memory 的边界。",
        weaknesses="与你当前四个系统的工程接口差异较大。",
        gaps="短期不适合作为复现改进主线。",
        relevance="related work 中可用于说明路线差异。",
        suggested_use="泛读。",
    ),
    Source(
        key="14_readagent",
        title="ReadAgent: A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts",
        year="2024",
        category="retrieval_reading",
        source_url="https://arxiv.org/abs/2402.09727",
        download_url="https://arxiv.org/pdf/2402.09727",
        kind="pdf",
        priority="core",
        why_read="gist memory + 原文回查，非常贴近“可溯源检索 + 原文回退”。",
        innovation="先用 gist 压缩长文，再在需要细节时回查原文。",
        strengths="直接对应压缩会丢细节的问题。",
        weaknesses="主要是阅读长文，不完全是在线 agent memory。",
        gaps="可以把其原文回查思想移植到 LoCoMo episodic evidence。",
        relevance="很适合做你的第一阶段原型灵感。",
        suggested_use="精读方法和回查策略。",
    ),
    Source(
        key="15_hipporag",
        title="HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models",
        year="2024 NeurIPS",
        category="retrieval_graph",
        source_url="https://arxiv.org/abs/2405.14831",
        download_url="https://arxiv.org/pdf/2405.14831",
        kind="pdf",
        priority="core",
        why_read="长期记忆与图检索结合的强代表。",
        innovation="借鉴 hippocampal indexing，用图结构和关联检索支持知识整合。",
        strengths="适合多跳关系和稀疏证据连接。",
        weaknesses="图构建成本、噪声和动态更新问题需要额外处理。",
        gaps="要证明图结构不是工程堆料，必须用 evidence chain coverage 等指标。",
        relevance="对时间/多跳重排和 evidence graph 有参考价值。",
        suggested_use="精读 retrieval 机制和消融。",
    ),
    Source(
        key="16_memorag",
        title="MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery",
        year="2024",
        category="retrieval_graph",
        source_url="https://arxiv.org/abs/2409.05591",
        download_url="https://arxiv.org/pdf/2409.05591",
        kind="pdf",
        priority="core",
        why_read="它把 RAG 改造成 memory-inspired discovery，和 xMemory 的 Beyond RAG 论点呼应。",
        innovation="用长程轻模型形成 global memory，再给检索提供 clues。",
        strengths="强调隐式信息需求和复杂检索，不只做 query-passage 相似度。",
        weaknesses="多模型调用成本较高；与在线 agent memory 的写入更新仍不同。",
        gaps="可借鉴 query clue，但要控制成本和公平预算。",
        relevance="对自适应检索和 query expansion 有启发。",
        suggested_use="精读 retrieval/cluing 部分。",
    ),
    Source(
        key="17_graphrag",
        title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        year="2024",
        category="retrieval_graph",
        source_url="https://arxiv.org/abs/2404.16130",
        download_url="https://arxiv.org/pdf/2404.16130",
        kind="pdf",
        priority="core",
        why_read="理解自动图索引、社区摘要和 global query 的代表方法。",
        innovation="从文本中抽取实体关系图和社区摘要，再执行 local/global search。",
        strengths="结构化索引和社区摘要对 memory organization 有参考价值。",
        weaknesses="面向文档集合，不是动态个人记忆；抽取错误会传播。",
        gaps="需要加 temporal validity、source provenance 和 online update。",
        relevance="Zep/HippoRAG/MemMachine 的图 memory 背景。",
        suggested_use="精读架构和失败模式。",
    ),
    Source(
        key="18_a_mem",
        title="A-MEM: Agentic Memory for LLM Agents",
        year="2025",
        category="system",
        source_url="https://arxiv.org/abs/2502.12110",
        download_url="https://arxiv.org/pdf/2502.12110",
        kind="pdf",
        priority="core",
        why_read="最直接的动态组织记忆网络 baseline，容易撞题。",
        innovation="借鉴 Zettelkasten，生成带属性的 note，动态链接并更新历史 memory。",
        strengths="组织结构比 flat vector memory 更强，LoCoMo 相关度高。",
        weaknesses="LLM 写入和更新成本高；provenance/错误更新风险需要关注。",
        gaps="若做 memory evolution，必须读它以避免重复。",
        relevance="与你考虑的模块改进直接竞争。",
        suggested_use="精读方法、prompt、消融和成本。",
    ),
    Source(
        key="19_mem0",
        title="Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory",
        year="2025",
        category="system",
        source_url="https://arxiv.org/abs/2504.19413",
        download_url="https://arxiv.org/pdf/2504.19413",
        kind="pdf",
        priority="core",
        why_read="生产型 memory layer 的关键论文，也是 LoCoMo 常见强 baseline。",
        innovation="抽取、合并、冲突处理和多 scope memory 结合。",
        strengths="工程完整，关注成本、latency 和可扩展性。",
        weaknesses="社区对部分评测口径和 LoCoMo 对比有争议。",
        gaps="需要复现/审计其评测设置，而不是只引用 SOTA 分数。",
        relevance="与 MemMachine/LightMem 对比很直接。",
        suggested_use="精读评测表、数据处理和成本分析。",
    ),
    Source(
        key="20_zep_temporal_kg",
        title="Zep: A Temporal Knowledge Graph Architecture for Agent Memory",
        year="2025",
        category="system_graph",
        source_url="https://arxiv.org/abs/2501.13956",
        download_url="https://arxiv.org/pdf/2501.13956",
        kind="pdf",
        priority="core",
        why_read="专门处理事实随时间变化，正好对应 update-aware memory。",
        innovation="用 temporal KG 表示实体、关系和事实有效期。",
        strengths="比普通 graph memory 更适合偏好改变、事实过期、矛盾覆盖。",
        weaknesses="实体消歧和时间边界依赖抽取质量；工程复杂度高。",
        gaps="需要把 temporal graph 的有效性放到 LoCoMo/LongMemEval 更新题中验证。",
        relevance="对 Cat2 temporal QA 和 future update benchmark 很关键。",
        suggested_use="精读 temporal validity 和 retrieval 部分。",
    ),
    Source(
        key="21_letta_filesystem_memory_blog",
        title="Benchmarking AI Agent Memory: Is a Filesystem All You Need?",
        year="2025",
        category="blog",
        source_url="https://www.letta.com/blog/benchmarking-ai-agent-memory",
        download_url="https://www.letta.com/blog/benchmarking-ai-agent-memory",
        kind="html",
        priority="core",
        why_read="用工程视角质疑复杂 memory 架构和 LoCoMo 排榜，非常适合防止误判。",
        innovation="把 filesystem/simple memory 作为强对照，强调 benchmark 可能测错东西。",
        strengths="指出评价与产品化之间的落差。",
        weaknesses="博客不是同行评审论文；需要结合正式论文和你自己的复现实验判断。",
        gaps="不要直接采信结论，拿它作为设计强 baseline 的提醒。",
        relevance="能防止你的改进只是在复杂化架构。",
        suggested_use="精读，提炼强简单 baseline。",
    ),
    Source(
        key="22_langmem_sdk_blog",
        title="LangMem SDK for Agent Long-Term Memory",
        year="2025",
        category="blog",
        source_url="https://blog.langchain.com/langmem-sdk-launch/",
        download_url="https://blog.langchain.com/langmem-sdk-launch/",
        kind="html",
        priority="extended",
        why_read="了解工业界如何把 semantic/episodic/procedural memory 产品化。",
        innovation="把 memory 抽取、更新和 LangGraph 集成做成 SDK。",
        strengths="有工程 API 和使用场景，适合理解落地约束。",
        weaknesses="不是研究论文；不会给严谨对比和消融。",
        gaps="适合作为系统设计参考，不应作为核心创新依据。",
        relevance="如果未来把模块接入 agent framework，可参考接口设计。",
        suggested_use="延伸阅读。",
    ),
    Source(
        key="23_memos",
        title="MemOS: A Memory OS for AI System",
        year="2025",
        category="system_os",
        source_url="https://arxiv.org/abs/2507.03724",
        download_url="https://arxiv.org/pdf/2507.03724",
        kind="pdf",
        priority="core",
        why_read="把 memory 作为可管理资源，对应你后续做 policy/budget 的方向。",
        innovation="提出 memory OS 抽象，强调 memory 的生命周期和调度管理。",
        strengths="系统视角强，适合写 future direction。",
        weaknesses="可能概念大于可复现实验细节。",
        gaps="需要回到 LoCoMo/LongMemEval 做可量化模块验证。",
        relevance="对 budget-controlled memory policy 有启发。",
        suggested_use="泛读到精读之间，重点看管理抽象。",
    ),
    Source(
        key="24_mirix",
        title="MIRIX: Multi-Agent Memory System for LLM-Based Agents",
        year="2025",
        category="system",
        source_url="https://arxiv.org/abs/2507.07957",
        download_url="https://arxiv.org/pdf/2507.07957",
        kind="pdf",
        priority="core",
        why_read="多 memory type + multi-agent controller 的强近邻工作，必须读以避免重复。",
        innovation="Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault 六类 memory 协同。",
        strengths="覆盖文本和多模态，报告 LoCoMo 强结果。",
        weaknesses="复杂度高，可能有工程堆料嫌疑；复现成本大。",
        gaps="若你只做一个模块，需证明比多 agent 复杂框架更轻且有效。",
        relevance="与你未来是否做 memory taxonomy/多 agent 管理直接相关。",
        suggested_use="精读体系结构和 LoCoMo 设置。",
    ),
    Source(
        key="25_h_mem",
        title="Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents",
        year="2025 / EACL 2026",
        category="system_hierarchy",
        source_url="https://arxiv.org/abs/2507.22925",
        download_url="https://arxiv.org/pdf/2507.22925",
        kind="pdf",
        priority="core",
        why_read="层级 memory 已经很多，读它能避免把“层级化”本身当贡献。",
        innovation="用分层抽象和更新支持长期推理效率。",
        strengths="和 xMemory/LightMem 处于同一竞争区间。",
        weaknesses="若缺少 provenance 或更新评测，仍可能只是层级摘要。",
        gaps="需要比较它与 xMemory、LightMem 的真正差异。",
        relevance="帮助判断你的层级改进是否有新意。",
        suggested_use="精读方法概念和实验对比。",
    ),
    Source(
        key="26_lightmem_iclr2026",
        title="LightMem: Lightweight and Efficient Memory-Augmented Generation",
        year="2025 / ICLR 2026",
        category="replicated_system",
        source_url="https://arxiv.org/abs/2510.18866",
        download_url="https://arxiv.org/pdf/2510.18866",
        kind="pdf",
        priority="core",
        why_read="你的四个复现系统之一；必须弄清 sensory/short/long-term 设计和效率指标。",
        innovation="topic-aware sensory/short-term memory 与 sleep-time offline update 降低在线成本。",
        strengths="效率指标强，和本地 Qwen 复现实验直接相关。",
        weaknesses="对证据忠实性、cat5、judge 稳定性和跨系统公平口径需要谨慎。",
        gaps="可以从 evidence recall、原文回退和统一 budget 角度改进。",
        relevance="本地 LightMem 结果已跑出，是第一批深入对照对象。",
        suggested_use="精读主方法、LoCoMo/LongMemEval 设定和 ablation。",
    ),
    Source(
        key="27_lightweight_llm_agent_memory_slm",
        title="Lightweight LLM Agent Memory with Small Language Models",
        year="2026",
        category="system_efficiency",
        source_url="https://arxiv.org/abs/2604.07798",
        download_url="https://arxiv.org/pdf/2604.07798",
        kind="pdf",
        priority="extended",
        why_read="同名/近似 LightMem 但路线不同，强调用小模型管理 memory。",
        innovation="用 SLM 执行检索、写入和巩固，降低成本。",
        strengths="对本地小模型复现很有参考价值。",
        weaknesses="需要警惕与 ICLR LightMem 名称混淆；评测要仔细核对。",
        gaps="若下载或论文不可访问，可作为后续补读。",
        relevance="你当前用 Qwen2.5-3B，本地低成本方向相关。",
        suggested_use="延伸阅读。",
    ),
    Source(
        key="28_xmemory",
        title="Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation",
        year="2026",
        category="replicated_system",
        source_url="https://arxiv.org/abs/2602.02007",
        download_url="https://arxiv.org/pdf/2602.02007",
        kind="pdf",
        priority="core",
        why_read="你的复现系统之一；它明确批评 top-k similarity 在 agent memory 中的失效。",
        innovation="message-episode-semantic-theme 层级，使用 sparsity-semantics objective 做 split/merge。",
        strengths="对冗余检索、多证据覆盖和 token efficiency 有清晰假设。",
        weaknesses="层级构建成本、统计可靠性和 cat5/评测口径仍需审计。",
        gaps="可以在 evidence coverage、uncertainty expansion、token budget 上做深入消融。",
        relevance="最适合作为自适应检索/展开原型的基础。",
        suggested_use="精读方法、objective、retrieval ablation。",
    ),
    Source(
        key="29_memmachine",
        title="MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents",
        year="2026",
        category="replicated_system",
        source_url="https://arxiv.org/abs/2604.04853",
        download_url="https://arxiv.org/pdf/2604.04853",
        kind="pdf",
        priority="core",
        why_read="你的复现系统之一；它强调保留完整 episodic ground truth，是可溯源路线代表。",
        innovation="保留完整 episode，使用 contextualized retrieval 和 adaptive retrieval agent。",
        strengths="减少抽取式 memory 的信息丢失；更适合 provenance 和原文回退。",
        weaknesses="Neo4j/episode 扩展带来工程复杂度；官方高分依赖强模型与优化 prompt。",
        gaps="需要在本地 Qwen 复现下拆解 retrieval-stage vs answer-stage 贡献。",
        relevance="你当前正在跑；后续模块可以围绕它做 source grounding 对照。",
        suggested_use="精读检索优化、LoCoMo/LongMemEvalS 和成本表。",
    ),
    Source(
        key="30_memori",
        title="Memori: Persistent Memory Layer for Efficient Context-Aware LLM Agents",
        year="2026",
        category="system",
        source_url="https://arxiv.org/abs/2603.19935",
        download_url="https://arxiv.org/pdf/2603.19935",
        kind="pdf",
        priority="extended",
        why_read="近邻长期记忆系统，强调 triples + summaries 和 API 层可移植。",
        innovation="把语义三元组和摘要结合成持久 memory layer。",
        strengths="工程化清晰，适合对比不同 memory schema。",
        weaknesses="若主要依赖抽取，仍会有 provenance 和错误更新问题。",
        gaps="作为 related work 和工程对照，不一定是第一阶段核心。",
        relevance="与 MemMachine/LightMem 同一竞品空间。",
        suggested_use="延伸阅读。",
    ),
    Source(
        key="31_simplemem",
        title="A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents",
        year="2026",
        category="replicated_system",
        source_url="https://arxiv.org/abs/2601.02553",
        download_url="https://arxiv.org/pdf/2601.02553",
        kind="pdf",
        priority="core",
        why_read="你的四个复现系统之一；它是“简单但强”的直接基准。",
        innovation="用 event-centric structured memory 和较短上下文取得强结果。",
        strengths="简单、可解释、效率好；是避免复杂化过度的重要 baseline。",
        weaknesses="可能仍有压缩失真、source grounding 不足和 judge 口径问题。",
        gaps="你的新模块必须先证明比 SimpleMem 的简单路线更值得复杂度。",
        relevance="本地已完整跑完，应作为第一对照。",
        suggested_use="精读方法、数据口径和评测表。",
    ),
    Source(
        key="32_autonomous_agent_memory_survey",
        title="Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers",
        year="2026",
        category="survey",
        source_url="https://arxiv.org/abs/2603.07670",
        download_url="https://arxiv.org/pdf/2603.07670",
        kind="pdf",
        priority="core",
        why_read="覆盖到 2026 初，能补上 LightMem/xMemory/MemMachine 时代的新趋势。",
        innovation="从 compression、RAG stores、reflection、virtual context、policy-learned management 等机制家族重组领域。",
        strengths="适合快速定位前沿空白。",
        weaknesses="综述结论需要回到具体系统和实验验证。",
        gaps="用它检查你的 related work 是否漏掉最新 work。",
        relevance="作为开题和 research map 的最新综述。",
        suggested_use="第一周精读。",
    ),
    Source(
        key="33_self_rag",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        year="2023",
        category="adaptive_retrieval",
        source_url="https://arxiv.org/abs/2310.11511",
        download_url="https://arxiv.org/pdf/2310.11511",
        kind="pdf",
        priority="extended",
        why_read="不是 agent memory 论文，但对“何时检索、如何批评证据”很重要。",
        innovation="用 reflection tokens 让模型自适应决定检索、评价证据和生成。",
        strengths="对 adaptive retrieval 和 retrieval quality checking 有启发。",
        weaknesses="需要训练模型，不一定适合你的本地复现直接实现。",
        gaps="可借鉴控制思想，先做 inference-time 轻量版本。",
        relevance="服务于自适应检索/展开模块。",
        suggested_use="延伸阅读，重点看 reflection/control 机制。",
    ),
    Source(
        key="34_reflexion",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        year="2023",
        category="agent_learning",
        source_url="https://arxiv.org/abs/2303.11366",
        download_url="https://arxiv.org/pdf/2303.11366",
        kind="pdf",
        priority="extended",
        why_read="反思型 episodic memory 的基础，但不应被简单重复。",
        innovation="把任务反馈写成 verbal reflection，作为后续试验的 memory。",
        strengths="对 procedural/reflective memory 很有启发。",
        weaknesses="更多是 task learning，不是长期对话 QA。",
        gaps="若做 reflective memory，要证明它在 LoCoMo/LongMemEval 中解决具体失败模式。",
        relevance="用于理解反思记忆，不作为主复现对象。",
        suggested_use="泛读。",
    ),
    Source(
        key="35_mem2actbench",
        title="Mem2ActBench: Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents",
        year="2026",
        category="benchmark",
        source_url="https://arxiv.org/abs/2601.19935",
        download_url="https://arxiv.org/pdf/2601.19935",
        kind="pdf",
        priority="extended",
        why_read="把 memory 评价从回答问题推到行动和工具调用。",
        innovation="测试 agent 是否能主动利用长期 memory 选择工具和填参数。",
        strengths="能避免只优化 QA 分数。",
        weaknesses="与当前 LoCoMo 四系统复现差距较大。",
        gaps="作为未来工作方向，短期不必完整接入。",
        relevance="提醒最终科研贡献最好不止 QA recall。",
        suggested_use="延伸阅读。",
    ),
]


AGENT_DISCUSSION = r"""
# 三智能体讨论纪要

## 第一轮：探索智能体

- 四个复现系统可拆成：记忆构建、存储、检索、压缩/反思、重排、回答、评测。
- SimpleMem 是简单强 baseline，适合防止复杂化过度。
- LightMem 强调三阶段 memory 和效率，适合比较 compression/consolidation。
- xMemory 的核心价值在于反对 flat top-k，做层级解耦和聚合。
- MemMachine 的价值在于保留原始 episode，适合 provenance 和原文回退。
- 最值得先做的原型方向：
  1. 可溯源检索 + 原文回退；
  2. 自适应检索/展开；
  3. 时间/多跳专用重排。

## 第一轮：调研智能体

- 给出 30 篇公开可访问论文/文章，从 survey、benchmark、系统、RAG/graph、博客五类覆盖。
- 核心必读集中在：LoCoMo、LongMemEval、SimpleMem、LightMem、xMemory、MemMachine、Mem0、A-MEM、Zep、MIRIX、MemGPT、ReadAgent、HippoRAG、GraphRAG。
- 延伸阅读覆盖：LangMem、Self-RAG、Reflexion、Mem2ActBench、多模态和行动型 memory。

## 第一轮：批评建议智能体

- 最大风险：把论文做成 LoCoMo prompt/rerank 调参工程。
- 已拥挤方向：chunk/summary/compression、hierarchical memory、graph memory、LLM reranker、memory type taxonomy、LoCoMo 排榜。
- 真痛点：
  - 评测可信度；
  - provenance/source grounding；
  - 更新、冲突、遗忘；
  - abstention 和不该记的问题；
  - 多跳 evidence chain coverage；
  - 在线增量 ingest；
  - 安全与记忆污染；
  - 跨 benchmark 泛化。

## 统筹结论

第一阶段不建议追求“再涨 LoCoMo 2 个点”。更稳的科研问题是：

> 现有长期对话记忆系统在压缩、层级聚合或图检索过程中会丢失/错配证据。我们能否在固定 token/API/latency 预算下，设计一个 evidence-faithful、可原文回退、自适应展开的 memory retrieval 层，并用 evidence recall、chain coverage、source attribution、answer correctness 共同证明它有效？

这个问题可以接在已有复现基础上，不必重写所有系统；可先在 xMemory 或 MemMachine 上做原型，再把 LightMem/SimpleMem 作为对照。
"""


def safe_filename(name: str, suffix: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return f"{slug}{suffix}"


def download(url: str, target: Path, retries: int = 3) -> tuple[bool, str]:
    if target.exists() and target.stat().st_size > 1024:
        return True, "already_exists"

    headers = {
        "User-Agent": "Mozilla/5.0 research-pack-downloader/1.0",
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                data = response.read()
            if len(data) < 1024:
                raise RuntimeError(f"downloaded file too small: {len(data)} bytes")
            target.write_bytes(data)
            return True, f"downloaded_attempt_{attempt}"
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = repr(exc)
            time.sleep(1.5 * attempt)
    return False, last_error


def markdown_table(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(rows[0]) + " |"]
    out.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for row in rows[1:]:
        out.append("| " + " | ".join(cell.replace("\n", "<br>") for cell in row) + " |")
    return "\n".join(out)


def build_readme(manifest: list[dict]) -> str:
    core = [m for m in manifest if m["priority"] == "core"]
    extended = [m for m in manifest if m["priority"] != "core"]
    rows = [["#", "标题", "优先级", "年份", "类型", "本地文件", "下载状态"]]
    for idx, item in enumerate(manifest, 1):
        rows.append(
            [
                str(idx),
                f"[{item['title']}]({item['source_url']})",
                item["priority"],
                item["year"],
                item["category"],
                item.get("local_file", ""),
                item.get("download_status", ""),
            ]
        )

    paper_sections = []
    for idx, item in enumerate(manifest, 1):
        paper_sections.append(
            textwrap.dedent(
                f"""
                ### {idx}. {item['title']}

                - 年份/类型：{item['year']} / {item['category']} / {item['priority']}
                - 链接：{item['source_url']}
                - 本地文件：`{item.get('local_file', '未下载')}`
                - 为什么读：{item['why_read']}
                - 创新点：{item['innovation']}
                - 优势：{item['strengths']}
                - 劣势/风险：{item['weaknesses']}
                - 没做到/空白：{item['gaps']}
                - 与当前复现关系：{item['relevance']}
                - 建议用法：{item['suggested_use']}
                """
            ).strip()
        )

    return textwrap.dedent(
        f"""
        # LLM Agent Memory 调研资料包

        生成日期：2026-04-29  
        目标：在 SimpleMem / LightMem / xMemory / MemMachine 的 LoCoMo 复现基础上，为下一步模块改进准备文献、技术路线和批判性问题。

        ## 先读路线

        1. **评测地基**：LoCoMo、LongMemEval、PerLTQA、StructMemEval、BEAM。先搞清楚到底在测什么，避免只追 LoCoMo 排榜。
        2. **本地四系统**：SimpleMem、LightMem、xMemory、MemMachine。读它们的构建、检索、重排、回答、成本表。
        3. **近邻强 baseline**：Mem0、A-MEM、Zep、MIRIX、H-MEM、MemOS。重点看哪些方向已经拥挤。
        4. **可溯源/检索启发**：ReadAgent、HippoRAG、MemoRAG、GraphRAG、Self-RAG。为 evidence-faithful retrieval 和原文回退找技术材料。
        5. **基础历史**：MemGPT、Generative Agents、MemoryBank、Reflexion。用于 related work，不要把这些老概念重复包装成新贡献。

        ## 最建议先做的科研切口

        **主切口：evidence-faithful memory retrieval。**

        目标不是“LoCoMo 再涨几分”，而是在固定预算下让 memory 系统返回可验证、可回退到原文、覆盖完整证据链的上下文。建议指标：

        - answer correctness：F1 / judge
        - evidence recall@k：gold evidence 是否被召回
        - chain coverage：多跳证据链是否完整
        - source attribution accuracy：回答能否追溯到正确 speaker/time/turn
        - redundancy rate：检索上下文是否重复
        - cost：ingestion tokens、query tokens、LLM calls、latency、storage

        ## 下载状态总览

        - 核心条目：{len(core)}
        - 延伸条目：{len(extended)}
        - 下载成功：{sum(1 for m in manifest if m.get('download_ok'))}
        - 下载失败：{sum(1 for m in manifest if not m.get('download_ok'))}

        {markdown_table(rows)}

        ## 每篇详解

        {chr(10).join(paper_sections)}
        """
    ).strip() + "\n"


def main() -> int:
    if PACK_ROOT.exists():
        shutil.rmtree(PACK_ROOT)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for source in SOURCES:
        suffix = ".pdf" if source.kind == "pdf" else ".html"
        target_dir = PDF_DIR if source.kind == "pdf" else ARTICLE_DIR
        target = target_dir / safe_filename(source.key, suffix)
        ok, status = download(source.download_url, target)
        item = asdict(source)
        item["download_ok"] = ok
        item["download_status"] = status
        item["local_file"] = str(target.relative_to(PACK_ROOT)) if ok else ""
        manifest.append(item)

    (PACK_ROOT / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (PACK_ROOT / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    (PACK_ROOT / "README.md").write_text(build_readme(manifest), encoding="utf-8")
    (NOTE_DIR / "agent_discussion.md").write_text(AGENT_DISCUSSION.strip() + "\n", encoding="utf-8")

    zip_path = PACK_ROOT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in PACK_ROOT.rglob("*"):
            archive.write(path, path.relative_to(PACK_ROOT.parent))

    print(f"PACK_ROOT={PACK_ROOT}")
    print(f"ZIP={zip_path}")
    print(f"downloaded={sum(1 for m in manifest if m['download_ok'])}/{len(manifest)}")
    failures = [m for m in manifest if not m["download_ok"]]
    if failures:
        print("FAILURES:")
        for item in failures:
            print(f"- {item['key']}: {item['download_status']} :: {item['download_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
