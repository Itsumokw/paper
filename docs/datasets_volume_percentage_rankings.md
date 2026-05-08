# datasets 体量百分比排行榜

记录日期：2026-05-07

说明：每个排行榜都以该口径下最大的数据库为 100%，其他数据集按比例换算。由于“体量”可以有不同含义，本文建议主排行榜使用 `turn / utterance 总量`，因为它最接近模型实际要处理的文本规模。

## 主排行榜：按 turn / utterance 总量

| 排名 | 数据集 | turn / utterance 数 | 相对最大体量 |
|---:|---|---:|---:|
| 1 | DuLeMon | 448,977 | 100.00% |
| 2 | CareCall-Memory | 86,250 | 19.21% |
| 3 | PerLTQA | 25,246 | 5.62% |
| 4 | OPELA | 17,607 | 3.92% |
| 5 | Japanese JMSC | 8,820 | 1.96% |
| 6 | LoCoMo | 5,882 | 1.31% |
| 7 | deL1L2IM | 4,548 | 1.01% |
| 8 | Japanese LAC | 815 | 0.18% |

## 辅助排行榜：按独立样本数

| 排名 | 数据集 | 独立样本数 | 相对最大体量 |
|---:|---|---:|---:|
| 1 | DuLeMon | 27,501 | 100.00% |
| 2 | CareCall-Memory | 770 | 2.80% |
| 3 | OPELA | 560 | 2.04% |
| 4 | Japanese JMSC | 197 | 0.72% |
| 5 | PerLTQA | 141 | 0.51% |
| 6 | LoCoMo | 10 | 0.04% |
| 7 | deL1L2IM | 9 | 0.03% |
| 8 | Japanese LAC | 4 | 0.01% |

## 辅助排行榜：按原生 session / date block 数

说明：DuLeMon 和 OPELA 没有原生 session 划分，因此不参与这个榜单。

| 排名 | 数据集 | session / date block 数 | 相对最大体量 |
|---:|---|---:|---:|
| 1 | PerLTQA | 4,961 | 100.00% |
| 2 | CareCall-Memory | 3,581 | 72.18% |
| 3 | Japanese JMSC | 735 | 14.82% |
| 4 | LoCoMo | 272 | 5.48% |
| 5 | Japanese LAC | 84 | 1.69% |
| 6 | deL1L2IM | 40 | 0.81% |

## 辅助排行榜：按已有 QA 数

| 排名 | 数据集 | QA 数 | 相对最大体量 |
|---:|---|---:|---:|
| 1 | PerLTQA | 8,593 | 100.00% |
| 2 | LoCoMo | 1,986 | 23.11% |
| 3 | DuLeMon | 0 | 0.00% |
| 4 | CareCall-Memory | 0 | 0.00% |
| 5 | OPELA | 0 | 0.00% |
| 6 | Japanese LAC | 0 | 0.00% |
| 7 | Japanese JMSC | 0 | 0.00% |
| 8 | deL1L2IM | 0 | 0.00% |

## 结论

| 判断问题 | 结论 |
|---|---|
| 原始对话文本规模谁最大？ | DuLeMon 是 100%，远大于其他数据集 |
| 最适合直接做 QA 评测的是谁？ | PerLTQA 是 100%，LoCoMo 约 23.11% |
| 最接近长期多 session 结构的是谁？ | CareCall-Memory 在 session 规模上达到 PerLTQA 的 72.18%，且已有 memory/summary |
| 韩语长对话主数据谁最稳？ | OPELA 虽然 turn 总量只有 DuLeMon 的 3.92%，但它是当前最稳的韩语原生长对话数据 |
| LoCoMo 为什么仍然重要？ | LoCoMo 体量不大，但格式最标准，应该作为 schema 和 F1 评测口径模板 |
