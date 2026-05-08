# 韩语 LoCoMo-style 长对话记忆数据集候选

记录日期：2026-05-07

筛选标准：

- 必须是韩语相关数据。
- 优先选择开源可下载、可直接拿到对话文本的数据。
- 不强制要求已有 QA/evidence，但应能低成本补充 question、answer、evidence。
- 重点关注长对话、persona、memory、summary、self-disclosure、empathy、user facts 等 LoCoMo-style 迁移要素。

| 推荐级别 | 数据集 | 链接 | 本地状态 | 本身包含什么 | 优点 | 问题 | LoCoMo-style 迁移判断 |
|---|---|---|---|---|---|---|---|
| S | OPELA | https://github.com/smilegate-ai/OPELA | 已下载到 `datasets/OPELA` | 560 条韩语长对话；15-335 turns；平均 31.44 turns；字段含 `persona_text_all`、`user_text_all`、`persona_summary`、`user_summary`、self-disclosure、empathy、engaging、active 等标签 | 当前最推荐的韩语长对话主候选；开源可直接下载；对话较长；summary 和心理属性标签很适合设计记忆 QA | 不是严格 multi-session，更像 persona-user 长单次对话；许可证为 CC BY-NC-SA 4.0，只适合非商业研究 | 很适合。可围绕 persona/user summary、自我披露、情绪、偏好、经历补 QA/evidence |
| A | CareCall-Memory | https://github.com/naver-ai/carecall-memory | 已下载公开部分到 `datasets/CareCall-Memory` | 韩语 sample；英文机翻完整版；multi-session dialogue、memory、summary | 结构最像 LoCoMo-style 长期记忆；已有 memory/summary | 韩语原始完整版需要申请；公开可直接用的是韩语 sample 和英文机翻版 | 理论最优，但韩语完整版受限；拿到原版后可作为韩语长期记忆主数据 |
| A- | NLPBada Korean Persona Chat v2 | https://huggingface.co/datasets/NLPBada/korean-persona-chat-dataset-v2 | 未下载 | `session_dialog`、`session_persona`，MIT | 转 QA 最省事；persona facts 已经整理好 | 对话偏短，不满足“长对话主数据” | 适合做 persona QA 补充，不建议替代 OPELA 做长对话主数据 |
| A- | NLPBada Korean Persona Chat v1 | https://huggingface.co/datasets/NLPBada/korean-persona-chat-dataset | 未下载 | `session_dialog`、`session_persona`，约 10k rows，MIT | 规模更大；字段干净 | 质量可能比 v2 更杂，对话长度有限 | 适合扩充样本量，作为 v2 或 OPELA 的补充 |
| B+ | KMI | https://github.com/hjkim811/KMI | 未下载 | 韩语心理咨询/动机访谈对话，主题和 MI 标签 | 适合构建情绪、原因、目标、改变意愿类 memory QA | 场景偏心理咨询，不是普通日常长对话 | 适合做 emotional/goal memory 子集 |
| B+ | Korean Role Playing | https://huggingface.co/datasets/huggingface-KREW/korean-role-playing | 未下载 | 韩语角色扮演多轮对话，角色/persona/relationship 信息 | 规模较大，可直接下载，角色一致性强 | 角色扮演味较重，真实用户长期记忆弱 | 可做 character/persona memory 补充 |
| B | KorEmpatheticDialogues | https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues | 未下载 | 韩语共情对话，train/valid/test JSON | 适合情绪状态、事件原因、支持需求 | 缺 persona/memory，长期属性弱 | 只适合做情绪记忆辅助子集 |
| B | XPersona Korean | https://github.com/HLTCHKUST/Xpersona | 未下载 | Persona-Chat 的韩语扩展 | 经典 persona chat，多语言对照，容易引用 | 翻译/迁移数据，不是原生长期韩语聊天 | 可做 persona QA baseline 子集 |
| B- | CareCall-Corpus | https://github.com/naver-ai/carecall-corpus | 未下载 | 韩语老人关怀对话 turn 数据 | 和 CareCall-Memory 同源；可直接下载 | 不是 memory 版本，没有 memory/summary 字段 | 可抽取健康、饮食、作息类事实后补 QA |

## 当前建议

如果必须选择一个韩语长对话主数据集，优先使用 OPELA。

原因：

1. 它可以直接下载，不依赖申请。
2. 它本身是韩语原生长对话，不是英文翻译。
3. 它有 persona/user summary、自我披露、共情、吸引力、主动性等标签，适合设计记忆类 QA。
4. 虽然它不是严格 multi-session，但对话长度和记忆要素都比普通 persona chat 更适合 LoCoMo-style 迁移。

推荐组合：

| 用途 | 数据集 |
|---|---|
| 韩语长对话主数据 | OPELA |
| 韩语长期记忆最优候选 | CareCall-Memory 原始韩语版，等待申请结果 |
| Persona QA 补充 | NLPBada Korean Persona Chat v2 / v1 |
| 情绪与目标记忆子集 | KMI |

## OPELA 本地核对结果

| 项目 | 数值 |
|---|---|
| 本地路径 | `datasets/OPELA` |
| 关键文件 | `datasets/OPELA/data/oplea_open_data.csv` |
| 文件大小 | 约 5.6 MB |
| conversations | 560 |
| columns | 38 |
| min turns | 15 |
| max turns | 335 |
| average turns | 31.44 |
| conversations >= 40 turns | 69 |
| conversations >= 60 turns | 15 |
| license | CC BY-NC-SA 4.0 |

关键字段：

- `doc_id`
- `persona_name_original`
- `total_turn`
- `total_sent`
- `total_minutes`
- `persona_text_all`
- `user_text_all`
- `persona_summary`
- `user_summary`
- `labeler_self`
- `labeler_empathy`
- `labeler_engage`
- `labeler_active`
