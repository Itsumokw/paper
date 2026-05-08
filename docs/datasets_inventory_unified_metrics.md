# datasets 目录统一体量口径

记录日期：2026-05-07

## 统一统计标准

为了避免不同数据集各说各的体量，本文后续建议统一使用下面 5 个指标：

| 指标 | 定义 |
|---|---|
| 独立样本数 | 一个完整用户/人物/对话单元。LoCoMo 中是 sample，OPELA/DuLeMon 中是 conversation，PerLTQA 中是 memory profile，deL1L2IM 中是 pair/chat file |
| 原生 session 数 | 数据集中明确给出的 session、day、date block 或时间段。没有原生 session 时记为 N/A |
| turn / utterance 数 | 模型能读取的一条发言。不同数据源里可能叫 turn、message、utterance、dialogue line |
| QA 数 | 已经带 question + answer 的评测问题数。没有 QA 的数据集记为 0 |
| 平均长度 | `turn / utterance 数 ÷ 独立样本数`，用于粗略比较每个样本的长短 |

注意：

- OPELA 同时有 `total_turn` 和 `total_sent`，这里用 `total_turn` 作为统一 turn 口径；`total_sent` 只作为补充。
- Japanese Long-term Chat 目录下包含 LAC 和 JMSC 两个子集，所以在表中拆开统计。
- PerLTQA 不是自然聊天流，但有大量 memory dialogue blocks 和 QA，因此用 memory profile 作为独立样本数。

## 统一体量表

| 数据集 | 语种 | 独立样本数 | 原生 session 数 | turn / utterance 数 | QA 数 | 平均长度 | 已有记忆线索 | 转 LoCoMo-style 成本 |
|---|---|---:|---:|---:|---:|---:|---|---|
| LoCoMo | 英语 | 10 | 272 | 5,882 | 1,986 | 588.20 | QA evidence、event summary、observation、session summary | 极低，目标格式本身 |
| PerLTQA | 中文/英文 | 141 memory profiles | 4,961 date blocks | 25,246 | 8,593 | 179.05 | profile、social relationship、events、dialogues、memory anchors | 低，已有 QA，只需格式对齐 |
| DuLeMon | 中文 | 27,501 conversations | N/A | 448,977 | 0 | 16.33 | persona facts 224,266 条 | 中，需补 QA/evidence |
| CareCall-Memory | 韩语 sample / 英文 full | 770 episodes | 3,581 sessions | 86,250 | 0 | 112.01 | memory items 10,699；summary items 10,308 | 低，但韩语 full 需申请；英文 full 有机翻噪声 |
| OPELA | 韩语 | 560 conversations | N/A | 17,607 turns | 0 | 31.44 | persona/user summary；self-disclosure、empathy、engage、active 标签 | 低-中，需补 QA/evidence |
| Japanese LAC | 日语 | 4 rooms | 84 room-days | 815 | 0 | 203.75 | speaker、dayid、真实聊天文本 | 中，需补 QA/evidence，公开部分小 |
| Japanese JMSC | 日语 | 197 pairs | 735 sessions | 8,820 | 0 | 44.77 | persona、session id、summary label | 中，需补 QA/evidence |
| deL1L2IM | 德语 | 9 pair files | 40 unique dates | 4,548 | 0 | 505.33 | speaker、timestamp、真实 IM 文本 | 中-高，需 XML 解析和补 QA |

## 按统一口径看谁“体量大”

| 排名口径 | 结果 |
|---|---|
| 独立样本数最多 | DuLeMon：27,501 conversations |
| turn/utterance 总量最大 | DuLeMon：448,977 turns |
| 已有 QA 最多 | PerLTQA：8,593 QA |
| 单样本最长 | LoCoMo：平均 588.20 turns；deL1L2IM：平均 505.33 messages |
| 韩语长对话最适合主用 | OPELA：560 conversations，平均 31.44 turns，最长 335 turns |
| 最接近目标格式 | LoCoMo 和 CareCall-Memory |
| 中文最适合直接评测 | PerLTQA |

## 论文里建议的统一描述方式

后续写论文时，不要只写“体量大/体量小”，建议统一这样写：

> 本文从独立样本数、session 数、turn/utterance 数和 QA 数四个维度统计各数据源规模。其中，独立样本数表示一个完整用户或对话单元，turn/utterance 数表示模型实际可读取的发言数量，QA 数表示可直接用于 F1 评测的问题数量。对于没有显式 session 划分的数据集，本文不强行构造 session，而是在转换阶段根据任务需要进一步切分。

## 当前结论

如果比较“原始对话体量”，DuLeMon 最大；如果比较“已有 QA 评测体量”，PerLTQA 最大；如果比较“LoCoMo-style 结构接近度”，LoCoMo 和 CareCall-Memory 最强；如果比较“韩语长对话可用性”，OPELA 当前最稳。
