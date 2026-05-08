# Prompts 模板库

来源：[awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)

| # | 文件 | 名称 | 用途 |
|---|------|------|------|
| 1 | `01_zh2en.md` | 中转英 | 中文草稿 → 英文 LaTeX（含审稿人视角自查） |
| 2 | `02_en2zh.md` | 英转中 | 英文 LaTeX → 中文直译（快速理解论文） |
| 3 | `03_zh2zh.md` | 中转中 | 中文草稿 → 中文学术规范段落（Word 适配） |
| 4 | `04_shrink.md` | 缩写 | 微幅缩减 5-15 词，保留全部信息 |
| 5 | `05_expand.md` | 扩写 | 微幅扩写 5-15 词，补充隐含逻辑 |
| 6 | `06_polish_en.md` | 表达润色（英文） | 深度润色至顶会出版水准 |
| 7 | `07_polish_zh.md` | 表达润色（中文） | 克制润色，保留作者风格 |
| 8 | `08_logic_check.md` | 逻辑检查 | 终稿红线审查：前后矛盾、术语不一致 |
| 9 | `09_deai_latex.md` | 去 AI 味（LaTeX） | 去除 AI 写作痕迹，注入人味 |
| 10 | `10_figure_caption.md` | 生成图标题 | 中文描述 → 英文 Figure caption |
| 11 | `11_table_caption.md` | 生成表标题 | 中文描述 → 英文 Table caption |
| 12 | `12_experiment_analysis.md` | 实验分析 | 实验数据 → LaTeX 分析段落 |
| 13 | `13_reviewer.md` | Reviewer 视角 | 模拟顶会审稿，找致命问题 |
| 14 | `14_chart_recommend.md` | 实验绘图推荐 | 数据 → 最佳图表类型 + 设计规范 |

## 使用方式

在 Claude Code 中直接引用即可，例如：
- "用 `docs/prompts/01_zh2en.md` 的模板处理这段中文草稿"
- "用 `docs/prompts/12_experiment_analysis.md` 分析以下实验结果"
- "用 `docs/prompts/13_reviewer.md` 审一遍我的论文"
