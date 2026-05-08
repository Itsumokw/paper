# LoCoMo-style 多语种长期记忆数据集 A 级候选

记录日期：2026-05-06

筛选标准：

- 只保留 A / A- 级候选。
- 排除 B 类、C 类候选。
- 优先选择本身已经具备跨时间、多 session、个人记忆或长期对话结构的数据集。
- 目标用途是转化为 LoCoMo-style F1 评测数据：补充 QA、answer、evidence，而不是从零构造长期对话。

| 推荐级别 | 语种 | 数据集 | 原链接 | 本身包含什么 | 优点 | 主要问题 | 一个 sample 的内容形态 | 需要补充什么 | 适合用途 |
|---|---|---|---|---|---|---|---|---|---|
| A | 韩语 | CareCall-Memory | https://github.com/naver-ai/carecall-memory | 多 session 韩语关怀通话；对话 turns；长期 memory；session summary | 最接近长期个人记忆数据，本身已经包含 memory 和 summary，迁移成本最低 | 场景偏老人关怀，开放域程度不如 LoCoMo | `user_id + session_id + dialogue turns + memory + summary`。例如某次通话中用户提到女儿来看望自己，memory 中记录“用户有女儿”“女儿今天来访” | QA、标准答案、evidence turn/session 标注 | 中文/多语种扩展论文中最稳的韩语长期记忆 QA 主体 |
| A- | 韩语 | MSPD: Korean Multi-Session Personalized Dialogue | https://aclanthology.org/2023.acl-industry.68.pdf | 多 episode、多 session、韩语个性化对话；用户偏好、日常话题、情感交互 | 规模大，天然是 multi-session personalized dialogue，适合生成跨 session QA | 数据获取和许可需要确认；是否直接可下载要进一步核验 | `episode_id + sessions[] + speaker utterances + persona/personalized context`。例如 session 1 提到用户喜欢登山，session 4 再提到周末去了山上 | QA、answer、evidence；必要时抽取/规范化 memory facts | 作为 CareCall-Memory 的规模补充，适合做韩语开放域长期个性化对话 |
| A- | 日语 | Japanese Long-term Chat / LAC | https://aclanthology.org/2024.lrec-main.322.pdf | 60 对日本人，8 周 Slack 异步聊天，约 71,244 utterances；真实长期聊天 | 真实跨时间聊天，关系和话题自然发展，非常接近长期记忆场景 | 不天然带 QA/evidence；数据访问方式需要确认 | `pair_id + week/session + timestamp + speaker + utterance`。例如第 1 周聊工作安排，第 5 周再次提到同事或生活事件 | QA、answer、evidence；必要时抽取事件线和人物关系 | 日语长期自然对话的首选候选，适合作为 LoCoMo-style 日语扩展 |

## 建议优先级

1. CareCall-Memory：最先做，因为它已经有 memory/summary 字段，最容易转成 QA 评测。
2. Japanese Long-term Chat / LAC：第二优先，因为它是真实跨 8 周长期聊天，论文故事最顺。
3. MSPD：第三优先，规模和 multi-session 属性很好，但需要先确认数据是否容易获取。

## 论文表述建议

本文不直接混合普通多语种短对话，而是筛选本身具备跨时间、多 session 或长期个人记忆结构的数据集，并将其统一转换为 LoCoMo-style 长期记忆评测格式。转换后的样本包含 sessions、memory facts、question、answer 和 evidence，从而可以使用 F1 口径评测不同 memory 方法在多语种长期对话场景下的表现。
