# BIO r3：完整 Task 2 dev + VA 官方评测

> **2026-09-23 清理后**：评测入口改为基于 `jev/task2.py` 的整理版 [`tools/evaluate_bio_r3_dev.py`](../tools/evaluate_bio_r3_dev.py)，抽取器精简为 `jev/extraction.py` 的 `BIOExtractor`。缓存中全部 1,368 条 BIO r3 抽取逐请求回放一致；断网重跑无新请求，预测文件逐字节相同，官方分数只有 1e-16 量级的浮点求和差异。下文的执行指纹 `7f55a4ed2a74` 指清理前源码，原脚本存于 [`archive/tools/evaluate_bio_r3_dev.py`](archive/tools/evaluate_bio_r3_dev.py)。

冻结历史 BIO r3（原合成示例、原规则、原题目、阈值 0.65），不使用 BM25 新配置。所有八语料完整 dev，不排除曾用于小实验的评论。先抽取 AO，再只对接受的配对调用现有 Jev Score V/A 题目，16 对/批；沿用 Task1 zero-shot train-fitted shrink 参数，裁剪 [1,9]、保留两位小数。参数曾在 Task1 dev 上选择方法类型，但本轮不拟合、不选择标定。

主指标为冻结 shrink 后官方 cF1；同时用同一批原始 Score 缓存离线报告未标定 cF1，以区分 VA 标定贡献。AO F1 使用既有大小写不敏感 surface-set 指标。官方 cF1 使用未修改的官方 scorer，逐语料单独调用。

不输入 query gold；不重复做 pair 判断；case-insensitive 同一配对仅输出一次，避免官方 scorer 的重复预测惩罚。NULL aspect 延续 r3，NULL opinion 仍不支持。历史 dev96 抽取缓存按原源码指纹及 corpus/ID/Text 复用。新 API 请求逐次缓存，可中断恢复。最多 8 并发。

```bash
python3 -m unittest tests.test_bio_r3_va
python3 tools/evaluate_bio_r3_dev.py
```

本轮仅 dev；与官方 test 排名不能直接等同。所有数据原文、预测、API 请求和 scorer 日志在忽略的 cache 中。开发结果不写入 README 官方 test 主表，不推送 GitHub。

## 完整结果

全部 **1,344/1,344** 条完成，无缺失记录。模型 `jev-1.13.0`，抽取指纹 `41ea479bf861`，本轮执行指纹 `7f55a4ed2a74`。原 dev96 抽取记录全部复用。

| 指标（八语料等权 Macro） | 分数 |
|---|---:|
| AO exact surface-set F1 | **47.14%** |
| 官方 scorer categorical F1（VA 完美时的上限） | **47.13%** |
| 原始 VA 的官方 cF1 | **39.02%** |
| 冻结 shrink 标定后的官方 cF1（主指标） | **43.51%** |
| 旧词典方案，同一完整 dev 的 cF1 | **30.68%** |

标定将 cF1 提高 **4.49 个百分点**；BIO r3 完整流程比旧词典流程提高 **12.82 个百分点**。这些比较使用同一完整 dev，不涉及跨 split 比较。旧词典基线与本轮除抽取外还有 pair 阈值、NULL 支持、VA 请求组织等差异，因此不能将 12.82 点全部归因为 BIO 标签形式本身。

Micro：AO surface-set F1 45.48%，TP1254/FP1555/FN1451；官方 categorical F1 45.47%，TP1254/FP1555/FN1453；标定后的官方 cF1 42.44%。官方 scorer 保留日语 gold 中额外的 2 条重复 AO 标注，surface-set 诊断会去重，故两种 AO 计数略有差异。官方数据与 scorer 均未改动，所有正式 cF1 均以官方结果为准。

### 逐语料完整 dev

| 语料 | 评论数 | AO F1（surface-set） | 原始 VA cF1 | 标定 VA cF1 | 旧词典 cF1 | cF1 差值（百分点） |
|---|---:|---:|---:|---:|---:|---:|
| 英文餐厅 | 200 | 75.25% | 60.74% | **69.38%** | 47.08% | +22.30 |
| 英文笔电 | 200 | 66.02% | 52.25% | **60.64%** | 34.14% | +26.50 |
| 中文餐厅 | 300 | 45.82% | 38.90% | **43.59%** | 18.93% | +24.66 |
| 中文笔电 | 300 | 28.13% | 23.74% | **26.72%** | 13.62% | +13.10 |
| 日语酒店 | 200 | 19.14% | 15.36% | **17.85%** | 38.18% | **−20.33** |
| 俄语餐厅 | 48 | 45.54% | 38.76% | **41.63%** | 31.28% | +10.35 |
| 鞑靼语餐厅 | 48 | 48.98% | 41.36% | **44.44%** | 31.83% | +12.61 |
| 乌克兰语餐厅 | 48 | 48.21% | 41.06% | **43.83%** | 30.41% | +13.42 |
| **Macro** | **1,344** | **47.14%** | **39.02%** | **43.51%** | **30.68%** | **+12.82** |

七个语料提升，一个下降。日语是明确短板：即使当前 AO 的 VA 全部正确，官方 categorical F1 也只有 19.08%，无法靠标定恢复到旧词典的 38.18%。中文笔电的抽取上限同样较低。本轮没有按语料挑选最佳系统并拼接一个新结果。

全量 Macro AO F1 47.14%，此前 dev96 的 45.71%，差 +1.43 点，但各语料波动明显：此前日语样本 29.27%，完整集 19.14%；俄语此前 24.49%，完整集 45.54%。这说明小样本汇总不足以稳定刻画各语料水平，不应把变化当成代码提升。

### 官方分数中的大致位置

来源：[赛事最终报告 Table 7](https://aclanthology.org/2026.semeval-1.452/)。下列是对官方 **test** 八语料成绩自行计算的 Macro，不是比赛发布的总排名。

| 官方 test 系统 | 八语料平均 cF1 |
|---|---:|
| PAI | 57.73% |
| PALI | 57.50% |
| nchellwig | 56.55% |
| Takoyaki | 56.20% |
| TeleAI | 55.66% |
| TeamLasse | 53.43% |
| kevinyu66 | 51.48% |
| AILS-NTUA | 50.16% |
| Habib university | 47.15% |
| Scmhl5 | 41.95% |
| ICT-NLP | 40.98% |
| 官方 Kimi-K2 Thinking baseline | 38.59% |
| 官方 Qwen3-14B baseline | 28.75% |

本次 **dev 43.51%** 的数值位于 Scmhl5 / ICT-NLP 与 Habib university 之间，属于中后段分数区间；与头部约 56%–58% 尚有明显距离。因为 split 不同，只能作为档次参照，不能宣称名次、已经击败某队、或 test 能保持 43.51%。英文 dev 分数较高，也不能据此宣称达到英文 test SOTA。

保持当前 AO 输出不变，VA 完美时本次 dev Macro 最高约 47.13%，标定后的 43.51% 距这个上限还差约 3.62 点。因此本批数据中，继续改 VA 的提升空间有明确上限；这个上限不是未来 test 的保证或限制。

## 成本与实现范围

- 新成功请求 **4,988** 次，HTTP attempts **4,990**；2 次额外尝试的未返回 usage 不计入已知 token 成本。
- 新输入 **13,139,734 tokens**，按 $0.042 / 百万输入估算 **$0.551869**。
- 其中抽取/配对输入 11,356,974；VA 输入 1,782,760。
- 复用历史 96 条抽取的 712,052 tokens 后，完整逻辑输入 **13,851,786 tokens**。无训练或重新拟合标定开销。
- 本轮新抽取中 1 条文本触发既有 r3 的 200-token 分块，仍处理全部块；不声称能够恢复跨块跨度。
- 一个针对性单测通过：配对阈值边界、大小写去重、VA 标定裁剪及不传入 query gold。官方 scorer 对八语料 raw/calibrated 共 16 个文件成功评分，无 scorer warning。
- 结果留在开发记录；未改变生产默认命令、未将 dev 分数填入 README test 表，未 commit/push。

## 结果文件

- [冻结协议](../reports/bio_r3_full_dev_20260923/protocol.json)
- [逐语料官方结果、AO 诊断、成本](../reports/bio_r3_full_dev_20260923/summary.json)
- [与旧完整 dev 的对照](../reports/bio_r3_full_dev_20260923/comparison.json)
- [评测脚本](../tools/evaluate_bio_r3_dev.py)
- [离线汇总](../tools/summarize_bio_r3_dev.py)

离线重新汇总：`python3 tools/summarize_bio_r3_dev.py`。原始预测及官方评分 stdout 在忽略的 cache 中，可直接复算，无需重新调用 Jev。
