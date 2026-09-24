# Jev span extraction 调研与可行性探针

日期：2026-09-23。仅调研与小范围验证，未替换 Task 2 baseline，未运行完整评测。

## 工具与版本

- [Stanza 原始论文](https://aclanthology.org/2020.acl-demos.14/)发表于 2020 年；当前 [v1.14.0](https://github.com/stanfordnlp/stanza/releases/tag/v1.14.0) 于 2026-07-15 发布。工具版本日期不代表每个语言模型都重新训练过。
- 核对 [1.14.0 官方资源清单](https://github.com/stanfordnlp/stanza-resources/blob/main/resources_1.14.0.json)：英语、中文、日语、俄语、乌克兰语具备 tokenize/POS/depparse；未找到鞑靼语 tt。此前全六语覆盖的说法需纠正。
- [GLiNER2 论文](https://aclanthology.org/2025.emnlp-demos.10/)发表于 2025 年。当前已有 [GLiNER2.5 Multi](https://huggingface.co/fastino/gliner2.5-multi-v1)，模型仓库创建于 2026-08-14，287M 参数，采用 start/end boundary 架构。它是可直接提出 span 的抽取模型，不是普通分词器；模型卡的 multilingual 标签不能证明六种目标语言、尤其鞑靼语的方面/观点抽取质量。

## 社区与官方证据

1. [官方预解析值抽取 cookbook](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md)：代码/外部工具提候选，Jev Choice 选择，代码还原原文。单个 Choice 最多 255 个选项。
2. [jev-extract](https://github.com/dangquan1402/jev-extract)：直接实现 token start Choice → conditional end Choice，也有整段 span Choice、句子定位等模式。查阅时仓库 HEAD 为 `6a00749e36c513eff12b36679f1d3629e4f3da1a`。
   - 作者的 [50 条合成样本比较](https://github.com/dangquan1402/jev-extract/blob/6a00749e36c513eff12b36679f1d3629e4f3da1a/benchmarks/results/ALL_MODES_REPORT.md)：顺序起止位置 EM 76%，完整 span Choice 78%，后者输入费用约五倍。此报告使用 jev-latest；另一个明确使用 jev-1.13.0 的运行记录顺序方案为 78%。不能把不同运行混为固定成绩。
   - [同轮节省候选实验](https://github.com/dangquan1402/jev-extract/blob/6a00749e36c513eff12b36679f1d3629e4f3da1a/benchmarks/results/SPAN_CHEAP_REPORT.md)：顺序方案 74%，整段候选 78%，先定位句子再选 span 78%；对应 50 条样本费用 $0.00421/$0.02135/$0.00493。作者自报结果，未独立复现；也不是多语言 ABSA 评测。
   - 重排、shape gate、top-k Noul 等复杂方案在该小数据集上没有胜过简单顺序抽取。不应据此加入额外校验链。
3. [jeveryword](https://github.com/jkrup/jeveryword)：编号 token 后选择起止位置；另一种用法是逐词分类再合并。作者明确声明仅在少量合成消息上试过，未完成 benchmark。可借鉴接口和编号方式，不能当成性能证明。

## 本项目小探针

- 固定 `jev-1.13.0`，开发集每种语言文件前两条，共 12 条；日语 hotel，其余 restaurant。俄/鞑靼/乌克兰语存在对应翻译样本，不能视为 12 个独立语义案例。
- 每句只抽取最左侧一个显式方面词。先选择起始 token，再在起点之后选择终点；终点选项展示对应完整 span。原文和 token 编号是模型输入；gold 仅在返回之后用于比对。
- 中日文按非空白字符编号，其他语言按 Unicode 词及标点编号。未安装或测试 Stanza/GLiNER，不构成分词器比较。
- 24 次请求，共 61,438 input tokens；按 $0.042/M 估算约 $0.00258。12/12 返回可还原原文的位置；6/12 完全匹配最左侧 gold aspect，也均为任一 gold aspect 的匹配。
- 失败包括：方面词边界过短/过长、将具体配料当作方面、选择了未纳入该句 gold 的其他评价对象。不是坐标格式失败。
- **这不是 Task 2 cF1，也不是完整 aspect recall**：没有抽取所有方面、观点、关系或 VA，没有测空答案及隐式方面处理，没有与现有 baseline 做同任务对照。
- [汇总](../reports/span_probe_20260923/summary.json)。本地 `reports/span_probe_20260923/cache/` 保存探针脚本与完整请求响应；含原文，按现有规则忽略。

## 下一步建议

先比较两个简单抽取方案：带编号的 token 逐词 BIO 标注（可一批多题）与顺序 start/end Choice。后者需支持多个 span 和停止选项；Task 2 还需允许 NULL 方面，并在方面条件下抽取观点，避免无条件笛卡尔积。先看 span 和 aspect–opinion pair F1，再看完整 cF1。

代码负责字符偏移映射；Jev 只做离散 Choice，不用 Score 预测整数下标，也不要求它数 Unicode 字符。选择终点时限定 end >= start。超过 255 选项再按句处理；短句实验暂不引入层层回退、复核或重排。

当前更值得优先验证的是抽取表示与标注方式，不能仅凭工具年份或本探针认定某一方案优于现有 baseline。
