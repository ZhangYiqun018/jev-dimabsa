# Task 2 few-shot：证据、表示与最小实验

2026-09-23。两位 subagent 分别调研抽取实证、Jev 样例格式；主代理核查本地实现和示例选择研究。只调研，没有新 API 实验，没有修改抽取实现。

**结论：值得验证，优先让真实训练样例演示当前那一道边界选择，而不是直接沿用 Task 1 的文本/VA 样例。** 最可能帮助的是标注边界与对应关系；没有证据可以预告本项目提高多少分。

## 当前并非严格 zero-shot

`jev/extraction.py` r3 已在 state 放入两条英中人造样例，包括文本、token 表和 aspect/opinion 列表。r2 还带 BIO 标签，r3 删除了标签。因此当前成绩不能作为严格 0-shot，历史 r2/r3 差异也不能解释成 few-shot 增益。

当前规则要求保留程度/否定修饰词。真实标注可能有更细的边界惯例；若调整这条规则，应在新对照的所有实验臂一致调整，避免只有 few-shot 臂换规则。

## 来源与证据范围

- **直接 ASTE 实证：**[Sentiment Analysis in the Era of Large Language Models: A Reality Check](https://arxiv.org/abs/2305.15005)，原始 v1 的 Table 2/4、§3/4.6。ChatGPT 在 Rest14 的 triplet micro-F1 从 0-shot 40.04 到 1/5/10-shot 的 44.92±3.53 / 50.75±5.93 / 54.11±2.98；Rest15 为 33.51 → 47.30 / 49.99 / 48.11，收益并不单调。K-shot 指**每个情感类别 K 个样例**，不是总共 K 个；few-shot 为三次运行均值及标准差，论文每数据集评测上限为 500 条，这几个 ASTE 测试集均小于上限（Rest14 492 条、Rest15 322 条），因此使用完整测试集。它是英语、离散情感、生成式 ASTE，不是 Jev Choice 或 DimABSA VA，不能据此预告我们的分数。
- **抽取选样实证：**[GPT-NER](https://arxiv.org/abs/2304.10428)，Table 2，不加额外验证阶段的完整英语 CoNLL2003 测试：随机样例 72.62 F1、句向量检索 84.36、实体级检索 89.97。实体级检索依赖监督训练的 NER 模型，不能当作免费技巧；比较的是检索方式，不是 0-shot 与 few-shot。论文关于生成 BIO 标签数不对齐的问题也不能直接套到 Jev 的逐 token Choice。
- [Takoyaki，SemEval 2026](https://aclanthology.org/2026.semeval-1.219/)：官方英语 Task2 第一的系统采用真实训练示例检索及边界处理；其日语 BM25 对比随机样例的 +5.46 cF1 消融属于 **Task3**，不能称为 Task2 或 Jev 的增益。完整方案还有多路检索/集成，不能归功于 few-shot 一个因素。
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)：在所研究任务中，相似示例检索优于随机选样。支持把检索作为候选策略，但不是六语言 ABSA/Jev 的证明。
- [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633)：用 LM 给候选示例打分并训练检索器，应用于语义解析。说明选样可以学习，也说明其方法有额外准备成本；本项目暂不引入训练检索器。
- [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837)：在其分类/多选设置中，输入分布、标签空间和格式贡献很大，错误标签有时也不显著伤害结果。**不能外推成抽取边界样例可以随意标错**；此处应保留真实标注。
- [Jev state](https://docs.typesafe.ai/concepts/state.md)、[Choice](https://docs.typesafe.ai/primitives/choice.md)、[Noul](https://docs.typesafe.ai/primitives/noul.md)：支持把示例放进结构化 state；同一请求的各题独立判断。没有专用训练接口，示例字段本身不是特殊 API 功能。社区 jev-extract/jeveryword 尚未找到抽取 few-shot 的受控消融。

## 按真实决策形式组织样例

下表是建议设计，不是已经被社区证明的最优格式。

| 抽取形式 | 一份演示应包含 | 限制 |
|---|---|---|
| BIO | 原文、同一 tokenizer 的编号、完整 BIO 标签 | 长；即使演示标签序列，查询仍然独立预测 token，不会自动获得联合解码 |
| SE start | 原文、token 表、已抽历史/cursor、start 或 none | 仅演示第一个实体不能教会剩余抽取和停止 |
| SE end | 原文、token 表、已知 start、end | 同请求的 end 不能读取 start 新产生的答案；条件化 end 仍需后续请求 |
| 完整跨度边界选择 | 原文、类型、粗片段、完整候选、正确 ID 与文本 | 最贴近现有边界错误；单选不能恢复合并在一起的多个正确短语 |
| aspect 条件 opinion | 原文、指定 aspect、候选 opinion、是否成立 | 用 Noul 则示范布尔标签；用逐次 Choice 则示范选项和历史，不能只展示集合却假定 API 自动多选 |

NULL aspect 表示评价目标隐含；none 表示该选择题没有合适候选，二者不能混用。Noul 演示用标注得到的 true/false，不人为制造概率，不附 Task 1 的 VA 分数或标定结果。

## 训练样例怎样构造

- 只使用同语料 train 的完整标注。复用已有规范化文本去重思路，排除正在评测的 trial/dev 重复文本；不新建审计框架，也不使用查询 gold 做检索或排序。
- 对边界 Choice，可以将 train 的 gold 跨度先做轻微扰动得到粗片段，再按与查询相同的规则扩展候选。不必先调用 Jev 为整个 train 生成错误。它是可控演示材料，不是对真实 BIO 错误分布的精确模拟。
- 不要让粗片段永远就是 gold、正确答案永远位于固定 ID/中间位置。按边界稳定排序并选取不同修正方向的样例即可，不做额外顺序鲁棒性测试框架。
- 如果展示 none，必须确保当前候选组不存在符合该问题的正确答案。若同组存在多个正确短语，不能随意挑一个制造单选标签；应换合适样例或明确对应的锚点语义。
- 一条文本派生多道题不等于多个独立 shots。记录来源文本数、演示决策数及序列化长度，跨 BIO/SE/span 比较时尤其不能只看“3-shot”。
- 固定样例先覆盖有关的边界差异；如果尝试动态检索，只依赖查询原文、已预测锚点/目标，不依赖查询答案。固定检索规则后，每条文本得到不同例子是正常检索，不等于按评测结果反复挑例子。

## 最小实验建议

1. 先锁定完整跨度边界选择实现、候选来源与统一规则，复用已有 BIO 缓存。
2. 在小规模 trial 比较 **真正 0-shot** 与 **3 条不同 train 文本的固定样例**。样例放共享 state，独立边界题批量发送；不附 VA、整套抽取过程或长推理。旧 r3 仅作历史参考，不把它称为 0-shot。
3. 若样例有效，再固定 3-shot 数量，比较固定样例与同语料简单 BM25 检索。先不做 embedding 模型、训练检索器、复杂多样性重排或 shot 数量扫描。不同例子长度会影响成本，实际记录 usage。
4. 冻结选择后，在同一 dev 子集比较 pair F1、精确候选覆盖、修回数、误改数和输入 tokens。dev 已被多次使用，只能称开发比较。不在本轮跑 test。

不同时展开四种抽取形式 × 多种选样 × 多种 shot 数量。候选缺失时 few-shot 不能让 Choice 输出候选之外的正确短语；边界覆盖与选择能力要分别报告。
