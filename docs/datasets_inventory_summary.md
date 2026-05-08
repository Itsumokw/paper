# datasets 目录数据集盘点

记录日期：2026-05-07

说明：`datasets/DATASET_COLLECTION_INDEX.md` 是索引文件，不计入数据集。当前 `datasets` 目录下共有 7 个数据集目录。

| 数据集 | 语种 | 本地路径 | 内容/字段特点 | 体量 | 优点 | 缺点 | 当前用途建议 |
|---|---|---|---|---|---|---|---|
| LoCoMo | 英语/基准 | `datasets/locomo` | 长期多 session 对话、QA、evidence、event summary、observation、session summary、多模态线索 | 10 samples；272 sessions；5,882 turns；1,986 QA | 最接近目标格式；已有 QA/evidence；可直接作为 LoCoMo-style 转换模板 | 样本数很少；主要是英文；作为扩展数据不够大 | 作为目标格式、F1 口径和 evidence 设计参照 |
| PerLTQA | 中文/英文 | `datasets/PerLTQA` | personal long-term memory QA；profile、social relationship、events、dialogues、QA、memory anchors | 中文：141 个 memory profiles；32 个 QA 主体；8,593 QA；3,145 events；英文 v2 同样 8,593 QA | 中文长期记忆 QA 最直接；已有 QA/answer/memory anchor；评测成本低 | 不是自然连续多轮聊天流，更像 memory bank + QA | 中文长期记忆 QA 主体，可对齐 LoCoMo F1 |
| DuLeMon | 中文 | `datasets/DuLeMon` | persona-grounded dialogue；`bot_persona`、`user_said_persona`、`user_no_said_persona`、conversation；含 self/both 两种设置 | self：24,500 条；both：3,001 条；合计 27,501 条 | 中文 persona/memory 对话体量大；适合补 persona QA | 不是 LoCoMo 式长期多 session；没有现成 QA/evidence；文件显示编码需要注意 | 中文对话材料和 persona QA 补充 |
| CareCall-Memory | 韩语/英文机翻 | `datasets/CareCall-Memory` | multi-session care call；dialogue、memory、summary；韩语 sample + 英文自动翻译 full | 韩语 sample：10 episodes / 45 sessions；英文机翻 full：770 episodes / 3,581 sessions | 结构最像长期记忆；已有 memory/summary；补 QA 成本低 | 韩语原始完整版需要申请；英文版有机翻噪声；场景偏老人关怀 | 等待韩语原版；当前可用英文 full 做结构实验 |
| OPELA | 韩语 | `datasets/OPELA` | 韩语 persona-user 长对话；persona/user text、persona_summary、user_summary、self-disclosure、empathy、engage、active 等标签 | 560 conversations；15-335 turns；平均 31.44 turns；69 条 >= 40 turns；15 条 >= 60 turns | 开源直下；韩语原生长对话；带 summary 和心理属性标签；适合补 QA | 不是严格 multi-session；许可证 CC BY-NC-SA 4.0；需要自己补 QA/evidence | 韩语长对话主候选，CareCall 原版拿不到时优先用它 |
| Japanese Long-term Chat / LAC + JMSC | 日语 | `datasets/japanese-long-term-chat` | LAC 真实长期 Slack 异步聊天；JMSC 日语版 multi-session chat；含 dialogue/persona TSV；已生成 UTF-8 副本 | LAC：815 utterances / 4 rooms；JMSC：8,820 utterances / 197 pair ids | 日语真实长期聊天；JMSC 有 persona 和 session 结构；UTF-8 副本已处理 | LAC 公开部分小；无 QA/evidence；原始文件为 CP932/Shift-JIS 编码 | 日语 LoCoMo-style 扩展候选 |
| deL1L2IM | 德语 | `datasets/deL1L2IM` | 德语学习者与母语者长期 IM；TEI-P5 XML；含 timestamp、speaker、message | 9 个 chat XML；约 4,548 条 message 标签；另有 manual | 德语真实长期 IM，跨时间属性强 | 规模小；偏语言学习场景；XML 解析成本高；无 QA/evidence | 德语长期对话补充，适合小规模迁移实验 |

## 总体建议

| 用途 | 推荐数据集 |
|---|---|
| 目标格式参照 | LoCoMo |
| 中文 QA 主体 | PerLTQA |
| 中文对话/persona 补充 | DuLeMon |
| 韩语长期记忆最优候选 | CareCall-Memory 原始韩语版，等待申请 |
| 韩语长对话可用主候选 | OPELA |
| 日语扩展 | Japanese Long-term Chat / LAC + JMSC |
| 德语扩展 | deL1L2IM |

如果下一步要开始做格式转换，建议顺序是：

1. 先解析 LoCoMo，固定统一目标 schema。
2. 转 PerLTQA，因为它已有 QA/answer/memory anchor。
3. 转 OPELA，手动或半自动补韩语 QA/evidence。
4. 转 Japanese LAC/JMSC。
5. 最后处理 deL1L2IM，因为 XML 清洗成本最高。
