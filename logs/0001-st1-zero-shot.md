# 0001 — Subtask 1, zero-shot

**2026-09-22** · `jev-1.13.0` · test split · RMSE_VA, official scorer, `--do_norm` off

## Configuration

- Request: `state` = the bare sentence; one call per sentence; two `Score` questions per
  aspect (valence, arousal), 9 levels each, `VA = 1 + score`.
- Question wording after the pipeline (BERT) starter kit: `What valence given the aspect "X"?`
  plus the official definition sentence.
- No in-context examples.
- Concurrency 10. Rubric fingerprint `0a2014ccee57`.

## Result

| corpus | n | RMSE_VA | PCC_V | PCC_A |
|---|---:|---:|---:|---:|
| eng_restaurant | 1504 | 2.6048 | 0.9203 | 0.5388 |
| eng_laptop | 1421 | 2.9967 | 0.8840 | 0.3710 |
| zho_restaurant | 1929 | 2.1988 | 0.8501 | 0.3849 |
| zho_laptop | 1673 | 1.9910 | 0.8965 | 0.5277 |
| zho_finance | 2354 | 2.4773 | 0.8237 | 0.4481 |
| jpn_hotel | 1092 | 2.4749 | 0.9400 | 0.5878 |
| jpn_finance | 1302 | 3.2797 | 0.9155 | 0.2765 |
| rus_restaurant | 1637 | 2.1613 | 0.9149 | 0.4918 |
| tat_restaurant | 1637 | 2.3989 | 0.8350 | 0.4173 |
| ukr_restaurant | 1637 | 2.1665 | 0.9076 | 0.4874 |
| **micro (N=16,186)** | 9658 | **2.4708** | 0.8888 | 0.4531 |

RMSE is micro, weighted by gold entry count. The `PCC` columns are per corpus in each row and
the mean of those ten values in the micro row — Pearson does not pool the way RMSE does.

Against published baselines on the same split: Kimi-K2
zero-shot 2.3849 (ahead on 3/10 corpora), GPT-5 mini zero-shot 2.7439 (ahead on 9/10),
Kimi-K2 one-shot 1.8873, Qwen3-14B QLoRA 2.1841.

## Cost

10,923,776 input tokens · **$0.4588** at $0.042/Mtok. Output tokens are not billed.

## Artifacts

`reports/pred_st1_<corpus>_test.jsonl` and `.meta.json`, `reports/st1_test_summary.json`.
Prediction files are gitignored — they embed review text.

## Observed

- `zho_finance` lost one record to HTTP 529 `system_overloaded` and was resumed; `usage_total`
  in the metadata is the sum of both runs. Reading only the last run's usage understates the
  corpus by 1.43M tokens.
- Jev is non-deterministic: repeated identical calls differ by roughly 0.04 per dimension.
  No repeat runs were made, so no error bars are reported.
- `PCC_V` (0.82–0.94) is high while `PCC_A` (0.28–0.59) is not. Arousal is the weaker
  dimension on every corpus.
