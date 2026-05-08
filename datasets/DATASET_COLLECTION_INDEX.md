# LoCoMo-style 长期记忆数据集本地索引

记录日期：2026-05-07

本目录用于集中保存 LoCoMo-style 长期记忆研究可用的数据源。当前已把中文数据集、LoCoMo、韩语 CareCall-Memory、日语 Japanese Long-term Chat / LAC、德语 deL1L2IM 放在同一个 `datasets` 目录下。

## 本地目录总览

| 语种 | 数据集 | 本地路径 | 状态 | 关键文件 | 备注 |
|---|---|---|---|---|---|
| 英语/基准 | LoCoMo | `datasets/locomo` | 已下载 | `data/locomo10.json` | 官方 snap-research/locomo 仓库，包含 10 个长期对话样本及 QA/summary 标注 |
| 中文 | PerLTQA | `datasets/PerLTQA` | 已存在 | `Dataset/zh`, `Dataset/en`, `Dataset/en_v2` | 中文 personal long-term memory QA 候选，已带 QA，适合作中文长期记忆 QA 主体 |
| 中文 | DuLeMon | `datasets/DuLeMon` | 已存在 | `DuLeMon.zip`, `DuLeMon/` | 中文 persona/dialogue 数据，可作为中文对话材料补充 |
| 韩语 | CareCall-Memory | `datasets/CareCall-Memory` | 部分已下载 | `data/carecall-memory-sample.json`, `data/carecall-memory_en_auto_translated.json` | 韩语原始完整版需要填 NAVER 表单申请；仓库直接公开韩语 sample 和英文机翻完整版 |
| 韩语 | OPELA | `datasets/OPELA` | 已下载 | `data/oplea_open_data.csv` | 韩语长对话 persona/empathy/long-term memory 数据，CC BY-NC-SA 4.0，可作为 CareCall-Memory 拿不到时的韩语主候选 |
| 日语 | Japanese Long-term Chat / LAC | `datasets/japanese-long-term-chat` | 已下载 | `lac-public-dialogue.tsv`, `jmsc-public-dialogue.tsv`, `jmsc-public-persona.tsv` | 原始 TSV 为 CP932/Shift-JIS 编码；已额外生成 UTF-8 副本 |
| 日语 | Japanese Long-term Chat / LAC UTF-8 copy | `datasets/japanese-long-term-chat/utf8` | 已生成 | `lac-public-dialogue.tsv`, `jmsc-public-dialogue.tsv`, `jmsc-public-persona.tsv` | 后续处理建议优先使用这个 UTF-8 目录 |
| 德语 | deL1L2IM | `datasets/deL1L2IM` | 已下载并解压 | `deL1L2IM.zip`, `extracted/transformation/Tei-P5/teip5-chat/*.xml` | TEI-P5 XML 格式，包含德语学习者与母语者的长期 IM 对话 |

## 新下载数据的来源与规模核对

| 数据集 | 来源 | 本地核对结果 | LoCoMo-style 迁移状态 |
|---|---|---|---|
| CareCall-Memory | https://github.com/naver-ai/carecall-memory | 韩语 sample：10 episodes / 45 sessions；英文机翻完整版：770 episodes / 3,581 sessions | 结构最接近：已有 multi-session dialogue、memory、summary；补 QA/answer/evidence 即可。若要使用韩语原始完整版，需要先申请 |
| OPELA | https://github.com/smilegate-ai/OPELA | 已下载 `oplea_open_data.csv`；560 conversations；turn 范围 15-335；平均 31.44 turns；69 条 >= 40 turns，15 条 >= 60 turns | 韩语长对话开源备选；已有 persona/user 全文、persona summary、user summary 和心理属性标签，补 QA/answer/evidence 成本较低 |
| Japanese Long-term Chat / LAC | https://github.com/nttcslab/japanese-long-term-chat | LAC public dialogue：815 utterances / 4 rooms；JMSC：8,820 utterances / 197 pair ids | LAC 是真实长期异步聊天，适合补 QA/evidence；JMSC 可作为日语 multi-session persona 补充 |
| deL1L2IM | https://orbilu.uni.lu/handle/10993/20579 | 已下载 `deL1L2IM.zip` 并解压；chat XML 文件 9 个；本地检出约 4,548 条 message 标签 | 德语中较适合长期对话迁移，但需要先从 TEI XML 抽取为 session/turn 格式，再补 QA/evidence |
| LoCoMo | https://github.com/snap-research/locomo | 已补齐官方仓库；`data/locomo10.json` 存在 | 作为目标格式和 F1 评测口径参照 |

## CareCall-Memory 重要说明

CareCall-Memory 的公开 GitHub 仓库中，韩语原始完整版没有直接匿名下载链接。README 明确说明：

- 韩语原始数据需要先填写申请表：https://naver.me/5zovK7N5
- 仓库内直接可用的是韩语 sample 和英文自动机翻完整版。
- 英文机翻版与原始数据统计一致，但包含机器翻译噪声。

因此，当前本地状态是：

- 可直接实验：英文机翻完整版 `carecall-memory_en_auto_translated.json`
- 可查看韩语结构：韩语 sample `carecall-memory-sample.json`
- 如论文必须使用韩语原文：需要补做 NAVER 表单申请

## 后续推荐处理顺序

1. 以 LoCoMo 的 `locomo10.json` 为目标格式，先写统一转换脚本。
2. 第一个转换对象选 CareCall-Memory，因为它已经有 `dialogue + memory + summary`。
3. 如果 CareCall-Memory 韩语原始完整版暂时拿不到，优先转换 OPELA，使用 `datasets/OPELA/data/oplea_open_data.csv`。
4. 第二个非英语转换对象选 Japanese LAC，使用 `datasets/japanese-long-term-chat/utf8/lac-public-dialogue.tsv`。
5. 第三个非英语转换对象选 deL1L2IM，先解析 TEI XML，再按日期或 XML 内部 chat block 划分 session。
6. 中文部分优先用 PerLTQA 作为 QA 主体，DuLeMon 作为补充对话材料。
