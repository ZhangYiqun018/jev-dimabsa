# Jev / DimABSA：进展复核、社区调研与优化设计

> 后续实施采用用户确认的简化范围：重试与续跑修复、train 校准、dev 选择和冻结 test。没有执行本文较宽方案中的额外 preflight、特征扩展或 routing。运行入口见 `tools/calibrate_st1.py`；最终实验见 `logs/0005-st1-supervised-calibration.md`。

日期：2026-09-23。代码基线：`6a43695`。范围：TypeSafe Jev / System One 在 DimABSA Track A Subtask 1 的连续情感评分。本文区分本地实测、外部作者实测和待验证设计。

**建议主线：保留 Jev 的语义判断能力，在本地学习它到数据集 VA 标尺的映射；先验证简单校准，再增加语义特征。** 目前的优先级应是评估协议 → 校准 → arousal 特征 → 示例优化 → 按需升级模型。继续扩大 shots 的收益证据，弱于已有预测的离线校准证据。

## 1. 已完成什么，哪些基础值得保留

已阅读 README、全部四篇实验日志、客户端、rubric、few-shot、两个 runner、scorer wrapper、泄漏审计和 probe；检查了预测文件、运行元数据、汇总文件及 vendored 官方评估实现。现有方案是每句一次请求，每个去重后的 aspect 同时问 V/A 两个 Score，9 级输出加 1 映射到 VA；gold 保留重复标注，预测按小写 aspect 匹配，符合官方计分行为。

| 已有 test 系统 | RMSE_VA | 历史输入费用 |
|---|---:|---:|
| zero-shot | 2.4708 | $0.4588 |
| first-k 3-shot | 2.1721 | $0.6700 |
| valence-stratified 3-shot | 2.1628 | $0.6752 |
| valence-stratified 5-shot | 2.1309 | $0.7445 |
| valence-stratified 9-shot | 2.0736 | $0.8936 |

计分：16,186 个 gold 条目、9,658 句，micro `sqrt(mean(error_V² + error_A²))`，`--do_norm` 关闭。费用为历史成功响应 token 按 $0.042/Mtok 估算，不能等同于包含所有失败请求的实付账单。

有价值的现有设计：轻量客户端、同句问题并行、结构化 state、固定示例、文本重叠排除、预测追加写入、模型返回版本记录、官方 scorer 不改动。这些基础可以继续使用。

## 2. 本次新增的离线证据

### 2.1 Arousal 是主要误差来源，但两维都需要标尺校准

逐条对齐现有 test 预测与 gold，没有重新请求 Jev，也没有在 test 上拟合任何参数：

| 系统 | RMSE_V | RMSE_A | V 平均误差 | A 平均误差 | A 占总平方误差 |
|---|---:|---:|---:|---:|---:|
| zero-shot | 1.1609 | 2.1811 | +0.2907 | −1.7144 | 77.92% |
| first-k 3-shot | 1.1300 | 1.8550 | +0.3013 | −1.3289 | 72.94% |
| stratified 9-shot | 1.1194 | 1.7454 | +0.2976 | −1.1974 | 70.86% |

9-shot 的 A 平均预测为 4.9108，gold 为 6.1083。按语料，`eng_laptop` 的 A 偏差 −2.025，`eng_restaurant` −1.837，`jpn_finance` −1.609。`zho_finance` 的 A 只占约 49.5% 平方误差，说明也不能对所有语料只修 A。

这些数值说明存在偏移，但不能直接拿 test 偏差作为补偿常数。后续参数必须从训练/校准分区学习。

### 2.2 简单校准已经出现强信号

利用现有 **zero-shot dev** 缓存，做了探索性的五折 OOF 验证：每个语料独立拟合，同一归一化句子固定同折，所有 aspect 和重复 gold 一起进入该折；按 gold 条目权重拟合。分折规则为 `SHA256(normalized Text) % 5`。每个验证折只使用另外四折的标签拟合。

| 方法 | dev OOF RMSE_VA | RMSE_V | RMSE_A |
|---|---:|---:|---:|
| 原始 zero-shot | 2.3789 | 1.2140 | 2.0459 |
| 每语料预测拟合分区的 V/A 均值 | 1.4916 | 1.3109 | 0.7115 |
| 每语料、每维只加偏移 | 1.5947 | 1.1139 | 1.1412 |
| 每语料、每维拟合截距与斜率 | **0.8614** | **0.5858** | **0.6315** |

线性方法为 `clip(intercept + slope * raw_score, 1, 9)`，每语料总共四个拟合参数。这里是普通最小二乘，没有神经网络训练。

覆盖 3,267 个 gold 条目。对十个语料的 affine OOF 预测分别调用了未修改的官方 scorer；将输出四舍五入到两位后，复算与官方打印精度一致。表中使用未舍入校准值；原始缓存本身只有两位小数。

解释：V 的相关性虽高，数值尺度仍明显不对；A 的标签在单一语料内较集中，简单均值已相当强。只加偏移会保留过大的波动，所以应把均值收缩和斜率校准放在优先级很高的位置。正斜率线性映射在不触发裁剪时不会改善 PCC，因此降低 RMSE 不代表模型获得了更好的情绪区分能力。

**限制：0.8614 是 dev 上的探索性 OOF，不是新的 test 成绩，不可与 test 2.0736 或论文 test 基线直接比较。** 尚无重复划分置信区间、没有验证同一方法对 9-shot 的效果，也没有建立全新未触碰的 holdout。俄/鞑/乌 dev 各只有 56 个文本组，小语料参数需做收缩。跨语料联合训练时，还必须把平行译文放进相同分组；本次各语料分别拟合，没有跨语言共享标签。

复现：

```bash
.venv/bin/python tools/analyze_st1_design.py \
  --out reports/design_diagnostics_2026-09-23.json --verify-official
```

脚本只做本地读取与离线计分。报告保存聚合统计、输入文件 SHA256 和分析脚本 SHA256，不保存数据集句子。原始预测文件不变。

## 3. 需要收紧的实验结论与工程细节

1. **Test 已参与多次方案选择。** 现有 best arm 属于在公开 test 上比较后选出的探索结果。后续冻结训练/校准/开发协议；公开 test 保留为最终参考，但不能恢复其“从未看过”的地位。需要严格无偏结论时，增加新 holdout 或独立样本。
2. **Shots sweep 不是纯数量消融。** `_select_stratified` 随 n 改变分箱边界，从而替换部分示例。元数据证实 `jpn_finance` 的 5-shot 与 9-shot 只共享 3 条，3-shot 与 5-shot 只共享 2 条。数量、内容、顺序均可能贡献收益。不能据此断言“选择几乎无影响、数量才重要”。
3. **单次两种取样的差异不能给所有选择策略下上界。** −0.0093 仅说明本次两个固定 3-shot 集合的 micro 差异很小；不排除二维覆盖、相关示例或其他集合有更大作用。
4. **单条预测波动不是 RMSE 显著性阈值。** 不能将约 0.04 的单项输出波动直接当成全数据集 RMSE 的噪声门槛。应做相同输入重跑，以及按句子/平行译文簇的配对 bootstrap。
5. **“first-k”只是取样规则接近论文。** 当前每条训练记录最多保留一个 aspect，而论文提供完整任务示例格式；本项目 3/9-shot 与论文 one-shot 也不是相同示例预算。统一称任务成绩对照，另加真正同 k、同示例信息量的受控实验。论文模型完整名称是 Kimi K2 Thinking。
6. **HTTP 529 实际没有被客户端重试。** `RETRYABLE_STATUS` 不含 529，与日志多次 overload 对应。官方 SDK 默认重试 5xx，可补 529、jitter、请求级 timeout 处理及总重试预算。重试次数、失败与成功 token 分开记。
7. **批量 runner 首次调用强制 `--restart`。** 重启 driver 会删除同名已有预测；与单语料 runner 的续跑语义不一致。应默认续跑，只有显式 restart 才覆盖，并用新的 run_id 保留历史。
8. **进程中断后缺少完整身份和费用凭证。** Meta 只在结束写出；预测存在但 meta 不存在时仍可续跑。应先持久化 manifest，按响应追加 journal，定期原子 checkpoint。恢复时检查 data hash、实际 example payload hash、完整问题 hash、模型版本和运行配置；未知身份时隔离旧文件。
9. **持续失败仍有统计缺口。** `run_all_st1.py` 尝试三次后返回 error，成功的部分调用 token 仍未纳入汇总；顶层 `main()` 即使有 error 也返回 0。需明确区分成本可统计与成绩不可发布：gold 未覆盖完整时不要发布完整 test 分数，但保留已发生用量并返回失败状态。
10. **API 返回了未被保留的信息。** `Answer` 已支持 probabilities/confidence，但 runner 只存格式化 VA。应另存原始 score、概率分布、confidence、响应模型、latency、attempts、usage。官方格式仅用于导出；分布方差和原始精度留给校准及误差诊断。

## 4. 社区与官方实践：证据及适用边界

调研从官方文档索引、GitHub 社区目录和 Hacker News 检索发现项目，再阅读下面的一手 README、实验报告及方法说明。外部数字均为作者报告，本次未重跑其付费实验。社区项目大多仍很新，跨任务迁移必须在 DimABSA 复验。

| 来源与证据性质 | 实际观察或设计 | 对本项目的启发 | 边界 |
|---|---|---|---|
| [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)，官方，9 月 17 日审阅 | 明确说明 Score 数值校准较弱；长无关 state、间接指令和冲突 criteria 会损伤判断 | 保留语义级别，数值映射交给本地；缩短并明确 aspect 作用域 | 不意味着 Score 不能做情感回归，只意味着 `1 + score` 不是充分校准 |
| [Autoresearch feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)，官方可复现 cookbook | 2,000 条品酒评价，800 条 held-out：直接评分 RMSE 2.15；18 个问题特征为 1.87，38 个问题为 1.77，后接 CatBoost | Jev 做语义特征提取，本地监督回归可以优于直接读 Score | 80–100 的酒评分任务，不能照搬其 RMSE 或预期增益；先做更小特征集 |
| [jev-calibration-audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md)，独立实测与原始响应 | 16 个同 state 问题对比单问题：confidence 平均变化 0.008，答案翻转 0.4%；50 次完全相同请求产生 15 种答案 | 同句 V/A 和辅助问题继续合并；噪声实验重复相同 payload，不注入 uid | 主要是 Choice/Noul 及特定语料，不能当作九级 VA Score 的无干扰证明；其“confidence 未说明”描述已落后于现官方文档 |
| [jev-orderby-bench](https://github.com/yodablocks/jev-orderby-bench)，独立实测 | 360 行对照中，把 40 行合并 state 后 Boolean inversion 从 0.036 左右变为 0.171；另一组 306 个电商 graded pairs 的 Score inversion 为 0.254 | 保持一条目标 review 一个 state；同文本多问题与多文本混装不是同一优化；分别检查排序与标尺 | 单 seed、有限任务；不能据此断言所有小 batch 都失效 |
| [Janus](https://github.com/FirasSX914/Janus/blob/main/RESEARCH.md)，独立实测与数据 | 两组各 500 条分类数据；Banking77 上阈值 sweep 可获益，换数据后最优阈值和是否值得 cascade 都改变 | confidence 只作为待验证特征；升级阈值用本任务校准集学习 | sweep 最佳点不是未经选择偏差的部署收益，分类阈值也不能直接搬到回归 |
| [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md)，小规模配对实测 | 3 对交替运行，优化 runtime 后中位时长 9.450→7.092 秒，请求 22→17；操作和目标在同请求并行预测 | 减少不必要 round trip，计入端到端成本，并用外部验证确认成功 | 浏览器任务、样本很小，作者给出 sign-test p=0.25；不证明 DimABSA 会更准或快 25% |
| [devagrawal09/jev-review](https://github.com/devagrawal09/jev-review)，开源架构案例 | 先选证据，再判机制和严重性；阈值留在代码中 | 多 aspect 歧义严重时，可研究“候选文本片段选择→评分” | 缺少同任务受控指标，只能作设计参考；二阶段请求会增加成本 |
| [JevBench](https://github.com/fstandhartinger/jevbench/blob/main/RESULTS-v1.2.md)，独立跨模型评测 | 534 个冻结决策，综合分包含能力、校准、速度、成本 | 保留质量/成本/时延分别汇报；开源替代模型若引入，要重新评估 | 综合名次不是 DimABSA 回归能力排名；部分自部署 latency 使用假设修正 |

官方 [Score](https://docs.typesafe.ai/primitives/score) 说明输出是级别期望；[Confidence](https://docs.typesafe.ai/confidence) 说明 confidence 来自概率分布形状，并非额外独立的“预测正确率”。九级分布上的方差可以作为特征，但它也不是人类 VA 标注的已校准方差。

[Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions) 的 12.2× 成本、10× 时延来自特定长文本 13 问题示例，相对于顺序单问。本项目已经同句合并 V/A，不能把这份收益再次计入预期。

## 5. 推荐架构

```text
训练数据 ── 文本/译文分组与重叠排除 ── 固定示例库
                                           │
目标 review + aspect ── Jev 同句并行评分 ── 原始响应缓存
                                           │
                       V/A + 分布统计 + 可选语义特征
                                           │
                    本地校准器 / 小型监督回归器
                                           │
                       clip [1,9] → 两位小数导出
                                           │
                            未修改的官方 scorer
```

### 第一层：低复杂度标尺校准，最高优先级

对 V/A 分别比较：语料均值、只加偏移、线性回归、向语料均值收缩、向全局参数收缩的语料线性回归。样本少时优先共享或正则化参数；不要一开始堆高阶曲线。Isotonic 放到线性明显存在非线性残差以后。

保留两条成绩轨道：**Jev prompt-only** 与 **Jev + supervised calibration**。第二条利用了额外标签，即使不微调 Jev，也不能写成同预算的 few-shot 结果。记录用于拟合的标签数、语料、Jev 版本、prompt 与示例版本。

主实验在 train 划出分组 calibration 子集，其他训练样本构造 prompt 示例；固定 dev 用于选择少量方案。任何用于拟合校准器的样本均不能作为自己的 few-shot 示例；固定示例库与 calibration 分区应彻底分离。若使用完整 train 做交叉拟合，逐折排除对应验证折，且部署时示例构造保持一致。

为匹配当前最好 arm，补齐 9-shot dev/calibration 预测。不能把 zero-shot 上拟合的斜率直接套到 9-shot 输出。先比较 `zero + calibration` 与 `9-shot + calibration`，再决定额外示例是否值得。

### 第二层：只针对校准后的残差增加语义特征

先加 6–8 个明确、aspect-scoped 的问题，例如：情绪激活、愤怒/挫败、兴奋/喜悦、焦虑、平静满足、情绪是否仅隐含、转折冲突、目标情感证据是否明确。问题直接读 `review_to_score`；不能假定同次请求中 A 问题能读取 V 问题的答案。

用 V/A 原始 score、级别概率方差、少量语义特征训练 Ridge；确有额外收益再试小型 CatBoost。对每个候选做去特征消融。增加特征的条件是改善固定 dev 的残差及跨分组验证，而不是提高训练拟合度。A 的验证基线必须包含“该语料均值”，防止复杂模型不如常数预测。

若多 aspect 句子仍明显更差，再试候选分句 + Choice/Noul 相关性过滤；候选从目标文本确定，不能引用 gold opinion。先量化真实误差再支付第二次请求。

### 第三层：示例与 rubric 优化

- **纯数量实验**：预先固定一条示例序列，保证 E3 ⊂ E5 ⊂ E9 ⊂ E16，旧示例顺序不变。少量固定种子区分集合偶然性。
- **选择策略实验**：固定 k=9 与 prompt，比较 first-k、V-only、V/A 二维覆盖；二维策略只用训练标签，采用分位覆盖/实际可用候选，避免空极端区间。加入维度明确的近邻检索作为后续 arm，而非一次更改所有策略。
- **Rubric 实验**：固定 examples/state，比较现 rubric 与简洁的情绪激活锚点。当前 valence 高端的“enthusiastic/superlatives”和低端“harsh”等词可能混入 arousal；这是待检验假设。A 的“ordinary”也应尽量改为可判断的情绪描述。
- 本地语言 rubric 只做受控候选；不能凭“母语一定更好”扩大十套提示词维护成本。

数据论文 [2601.23022v3](https://arxiv.org/html/2601.23022v3) 的 4.1、4.4 与附录 D/H 支持 first-k 协议及 few-shot 校准作用，并报告其受测模型大约 32-shot 趋于平台；这不是 Jev 的最佳 k 结论。

### 第四层：按需升级，低优先级

只有校准后仍存在可识别的困难样本、且第二模型能互补时，再做 Jev→较强模型的 cascade。使用分布统计、特征冲突和语料信息预测残差风险，在独立 calibration 分区定阈值；报告调用比例、RMSE 与总成本曲线。

每个 aspect 必须给出合法 VA，不能通过丢弃低 confidence 样本改善官方分数。不要直接照搬 0.7 阈值，也不要默认多次相同调用取平均能纠正系统偏差。暂不需要换自部署模型、训练大型模型或扩到 ST2/ST3。

## 6. 执行顺序与验收标准

| 顺序 | 实验/改动 | 主要变量 | 验收与停止条件 |
|---|---|---|---|
| P0 | 固定版本、manifest、原始响应、续跑和 529 | 工程可靠性 | 不覆盖旧 run；中断可恢复；返回模型不混合；失败用量不丢；完整覆盖才计正式成绩 |
| P1 | train 分组 calibration + 当前 dev；zero 与 9-shot | 校准映射 | 比较均值/偏移/线性/收缩；按句子聚类的 ΔRMSE 区间；小语料不可大幅退化 |
| P2 | 对已校准模型增加 6–8 个特征 | 语义信息 | 超过同一 dev 的 affine 基线才保留；逐维误差、PCC 和成本同时报告 |
| P3 | 固定 k 选择策略；嵌套 k sweep | 示例内容或数量，每次一个 | 只有新增收益稳定且成本可接受时扩展到 16-shot；不要同时改 rubric |
| P4 | 局部 residual routing | 第二模型调用比例 | 优于最佳单模型方案的质量/成本折中才采用 |

先用每语料最多 100 句固定 dev 子集排除无效方案，对 finalists 跑完整 dev。测模型随机性时选固定样本原样重复 3 次；其作用是量化 API 变动，不替代按样本分组 bootstrap。正式比较可做 2,000 次配对 cluster bootstrap，同时保留每语料结果；平行译文用共同来源簇，避免重复计独立证据。经大量选择后的 dev 区间仍不是无偏 test 区间。

建议预先约定的实用门槛：新增复杂度应在完整 dev 带来至少 0.02 的 RMSE_VA 改善，且配对区间支持方向；这是工程取舍阈值，不是从 0.04 单项波动推导的统计事实。若简单线性已占主要收益，应停在小模型方案。

成本：本次诊断新增 Jev API 花费为零。历史 9-shot 每句平均约 2,203 输入 token，因此新增 2,000 句同类请求的粗估约 $0.185；只适用于相近 prompt 和文本长度，特征问题及重试会增加用量。下一轮应先测 100 句真实 tokens，再按 `Σ input_tokens / 1e6 × 当前单价` 估算并设置预算。当前 [Models](https://docs.typesafe.ai/models) 仍列 $0.042/Mtok、每请求 64k、state 加最长问题 32k；速率限制会调整，不宜把并发 10 视为永远最佳。

## 7. 交付与结论

本次交付：本设计文档、`tools/analyze_st1_design.py`、`reports/design_diagnostics_2026-09-23.json`。没有修改生产预测流程，也没有启动新的付费 Jev 实验。外部案例用于提出候选方案，不作为本项目已取得的收益。

**下一轮最有价值的实验，是在严格分组、示例与校准样本隔离的协议下，比较 zero-shot/9-shot 的监督线性校准。** 已有 dev OOF 结果足以将它排在更多 shots、复杂提示词、模型级联之前；最终收益仍需冻结方案后验证。
