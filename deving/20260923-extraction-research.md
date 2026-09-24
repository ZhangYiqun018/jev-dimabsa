# Task 2 抽取方案调研与本地诊断

日期：2026-09-23。三个 subagent 分别核查 Jev 社区、候选生成与官方强队方案；主代理汇总并核对现有缓存。没有新调用付费 API，没有跑 test，没有修改抽取实现。

**建议：优先验证完整短语候选的边界选择，再考虑 aspect 条件化 opinion 抽取。** 当前独立 BIO 和单调 SE 的一次边界决策会丢掉正确短语；换分词器本身尚不是最有证据的突破口。社区没有已经证明能解决本任务的现成 Jev 抽取器。

## 当前错误说明什么

使用 [r3 对比](20260923-compact-se-vs-bio.md) 的同一批 96 条 dev，按每条文本内大小写不敏感的 surface string 去重。以下是读取缓存后的诊断，不是新预测结果。

| 原始候选 | Gold 数 | 精确命中 | 未命中但与某候选存在子串关系 | 未命中且无子串关系 |
|---|---:|---:|---:|---:|
| BIO aspect | 154 | 125 | 29 | 0 |
| BIO opinion | 181 | 116 | 53 | 12 |
| SE aspect | 154 | 122 | 23 | 9 |
| SE opinion | 181 | 96 | 60 | 25 |

子串关系指 gold 包含预测或预测包含 gold，不代表定位到同一次出现，也不证明可自动修正。它只说明边界变体值得优先检验。将 BIO 与 SE 候选直接取并集，gold pair 覆盖率从 BIO 的 109/201（54.2%）升到 123/201（61.2%），收益有限，且需付两套抽取成本。

## 不调用模型的候选覆盖诊断

所有数值均来自同一批 96 dev；没有使用 gold 生成候选，gold 只用于事后计算覆盖。这里的覆盖率是候选上界，**不是模型准确率、F1 或官方 cF1**。该 dev 已反复使用，不是独立评测。

### 从全文枚举短跨度

沿用 `jev/extraction.py:tokenize`：中日文主要按字符，其余主要按词和标点。枚举连续 1 至 L 个 token 的原文片段，保留标点，大小写不敏感去重；同一候选池暂不区分 aspect/opinion，NULL aspect 另加。表中候选数不含 NULL。

| 最大宽度 L | Aspect 覆盖 | Opinion 覆盖 | Pair 覆盖 | 每文本候选均值 | 最大候选数 | 候选数 >255 的文本 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 152/154 | 163/181 | 181/201（90.0%） | 81.7 | 366 | 4 |
| 6 | 153/154 | 177/181 | 195/201（97.0%） | 118.4 | 553 | 7 |
| 8 | 154/154 | 180/181 | 200/201（99.5%） | 151.5 | 736 | 12 |

L=12 时此样本全部 gold 边界可表示。说明当前 tokenizer 并没有把此样本的正确边界挡在候选空间外；不能据此推断它对所有语言、所有数据都足够好。

### 在 BIO 已抽片段附近扩缩边界

将缓存中的字符 offsets 映射回 token 边界，每个原片段的起终点分别移动 −r 至 +r 个 token，保留合法、非空连续跨度，按类型去重。没有额外长度、标点或语义过滤，保留原候选，NULL aspect 另加。

| 半径 r | Aspect 覆盖 | Opinion 覆盖 | Pair 覆盖 | 每文本 aspect/opinion 候选均值 | 全部潜在 pair 数 |
|---|---:|---:|---:|---:|---:|
| 0 | 125/154 | 116/181 | 109/201（54.2%） | 3.26 / 2.93 | 1,491 |
| 1 | 140/154 | 147/181 | 153/201（76.1%） | 21.55 / 20.02 | 60,575 |
| 2 | 152/154 | 162/181 | 178/201（88.6%） | 48.05 / 44.50 | 314,119 |

r=3 的 pair 覆盖仅再升至 181/201（90.0%），潜在配对进一步增加。不能把扩展后的所有候选直接做笛卡尔积打分。局部扩展也救不回完全没被识别的表达。

复现数据位置：`reports/extraction_comparison_20260923/dev_{bio,pointer}/` 中的 summary 和 ignored cache；按 summary 的 corpus、ID 找官方 dev 文本。缓存键使用 SHA256(corpus + ID + Text) 前 16 位，不能只按 ID 合并，因为不同语料可能重名。

## 社区和论文提供的证据

1. **Jev 官方建议过量提候选，再让模型选择。** [Pre-parsed extraction cookbook](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md) 强调 over-find、原文复制与 none；[Jev 1.13 能力说明](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) 支持减少间接推理。它们是设计指导，没有本任务成绩。多答案需要独立判断或多次选择，不能把单个 Choice 的 top-k 概率解释成多个实体独立存在的概率。

2. **jev-extract 支持完整 span 选择，但只是单答案抽取。** [固定版本同轮报告](https://github.com/dangquan1402/jev-extract/blob/6a00749e36c513eff12b36679f1d3629e4f3da1a/benchmarks/results/SPAN_CHEAP_REPORT.md) 在 50 条合成英文 QA 上：SE 74% EM/$0.00421，全文 span 78%/$0.02135，先句或子句再 span 78%/$0.00493。后者值得借鉴局部候选设计，不能原样只选一个子句用于多方面任务。smart 候选有截断风险，不能当作高召回保证；四个百分点也只有两题。

3. **jeveryword 的并行 SE 主要是成本方向。** [源码](https://github.com/jkrup/jeveryword/blob/6fed1518bbb84cc4865401630f37c01dc3decc10/src/extract.mjs) 使用公共 token 表、简短 ID 选项、并行猜端点及必要时条件化重问。作者没有系统 benchmark；未知多个目标时，两端点可能指向不同跨度，因此不作为主要提分方案。

4. **Span-ASTE 提供“跨度→分别筛选→关系”的任务相关依据。** [ACL 2021 论文](https://aclanthology.org/2021.acl-long.367/) 与 [代码](https://github.com/chiayewken/Span-ASTE) 处理多词方面/观点及跨度交互，并用剪枝控制枚举成本。可以借鉴结构，但其监督训练结果不能转移成 Jev 的预期分数。

5. **官方强队也专门处理边界及示例选择。** [Takoyaki 论文](https://aclanthology.org/2026.semeval-1.219/) §3.3、4.3、4.4 研究 BM25 检索示例和从预测/gold partial overlap 挖掘边界修正规则。[官方总报告](https://aclanthology.org/2026.semeval-1.452/) Table 7 确认其英语 Task2 强结果。其日语 BM25 +5.46、边界修正 +1.92 cF1 的消融属于 **Task3**，不能当作我们 Task2 的预期收益。对我们可迁移的是同语言训练例和局部边界候选，不是照搬整套规则挖掘流程。

6. **条件化抽取有传统 ASTE 研究依据。** [BMRC，AAAI 2021](https://arxiv.org/abs/2103.07665) 与 [作者代码](https://github.com/chenshaowei57/BMRC) 先问有哪些方面，再问给定某个方面有哪些观点，也研究反向流程。[RoBMRC，NAACL 2022](https://aclanthology.org/2022.naacl-main.20/) 讨论端点配对及概率组合问题。可借鉴 A→O 的问法，但这些是监督训练的传统 ASTE 系统，没有六语言 Jev/DimABSA 的效果保证。

7. **强生成模型仍有边界错误。** [TeamLasse](https://aclanthology.org/2026.semeval-1.273/) 用 Qwen2.5-32B 联合抽 Aspect–Opinion pairs，再用 XLM-R large 预测 VA；官方 Task2 日语第二（0.5694）。论文指出 `very friendly` 与 `friendly` 等 strict-match 差异。不能把“所有程度词都保留”当作跨语料统一正确规则，应该以对应训练集的真实标注为依据。

## 分词器与现成抽取器的取舍

- [Stanza](https://github.com/stanfordnlp/stanza) 当前资源覆盖英语、中文、日语、俄语、乌克兰语的分词/POS/依存，但没有 Tatar 对应资源。分词/POS 可以减少无效候选；仅取名词或形容词会漏多词短语、否定和程度成分，因此不能用作未经覆盖测量的硬过滤。
- [GLiNER2.5 Multi](https://huggingface.co/fastino/gliner2.5-multi-v1) 是可用原文 offsets 的约 287M 参数候选抽取器，适合作独立候选源对照。其卡片没有证明 Tatar 或本任务 aspect/opinion 的效果。引入它会增加本地模型依赖与推理成本；当前不必先下载或训练才能验证边界假设。

## 建议的小实验顺序

1. **先做局部完整跨度选择。** 保持已有 BIO 输出作为锚点；每个片段枚举 ±2 token 边界变体，单组最多 25 个，保留原跨度和 none，让 Jev 比较完整短语。先独立验证这个改动；少量同语言 train 检索示例作为后续单独对照，避免同时改多个因素。单个原片段可能合并了多个正确表达，单选修复解决不了这种情况，必须在结果中单独识别，不能宣称候选 88.6% 就是该方法可达到的召回。
2. **若主要仍漏 opinion，再试 aspect 条件化候选判断。** 先筛 aspect，在完整 opinion 候选上询问是否评价这个目标，并保留 NULL 目标。多项可成立时批量独立判断；不要按每个 aspect 重跑整句 BIO。它是我们针对 Task2 的改造，尚无 Jev 社区实证。
3. **若局部方法覆盖不足，再引入全文短跨度或 GLiNER 候选。** 先独立筛 aspect/opinion 再关系判断；不能直接对全文候选全配对。分词/POS 用来降低候选数量，而不是假设换一个新 tokenizer 就能提分。

只在小规模 trial 上选一个方案，再用相同 dev 子集比较现有 BIO；不跑 test，不展开多个复杂变体。报告 exact pair F1、precision/recall、候选覆盖和实际 input-token usage。已有 dev 不能再称独立验证；方法成熟后才需另选未用于调参的数据。

255 限制针对每个 Choice 的选项；有 none 时最多 254 个跨度。Noul 多问题不受这个“单题选项”限制，但请求大小和 token 成本仍存在。题数减少不等于输入成本降低。没有新增 preflight、一致性验证框架或大范围 unit tests。
