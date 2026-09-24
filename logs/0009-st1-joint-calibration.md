# 0009 — Task 1 (DimASR): joint V/A calibration

**2026-09-24** · **`jev-1.13.0`** · **official test, fresh run** · **micro RMSE_VA**

## Configuration

Jev predictions are the 9-shot stratified runs of log [0005](0005-st1-supervised-calibration.md)
(same requests and cached responses for the train calibration sample, dev and test); only
the post-processing changes. Code: [`tools/run_task1.py`](../tools/run_task1.py).

- **Joint calibration.** Per corpus, ridge regression of (gold V, gold A) on
  [V, A, |V − 5|, V × A] of the Jev scores (standardised), clipped to [1, 9]. The penalty
  λ ∈ {0.1, 1, 3, 10, 30, 100, 300} is chosen by grouped 5-fold CV on the 0005 train
  calibration sample (256 groups per corpus; same groups and folds as 0005). Chosen λ: 0.1–10
  ([`parameters.json`](../reports/task1/parameters.json)).
- **Candidates compared on dev**, all fitted on the same train sample: 0005 shrink or joint
  calibration, each with the 9 fixed examples or with 9 BM25-retrieved train examples per
  review (one explicit aspect per retrieved review; the review's own text and dev/test texts
  excluded).
- **Selection rule (from 0005).** The best dev candidate replaces the 0005 system only if it
  is at least 0.02 better and the paired cluster-bootstrap 95% interval (2,000 draws) is
  below zero.

## Result

Dev (3,267 entries):

| Candidate | Dev RMSE_VA |
|---|---:|
| fixed examples, shrink (0005) | 0.9077 |
| **fixed examples, joint (selected)** | **0.8572** |
| BM25 examples, shrink | 0.9126 |
| BM25 examples, joint | 0.8613 |

Selected improvement 0.0506, 95% interval [−0.0626, −0.0395]
([`dev_summary.json`](../reports/task1/dev_summary.json),
[`selection.json`](../reports/task1/selection.json)).

Test, **fresh run**: all 9,658 test requests sent again through the 0005 runner (same
examples and request configuration, checked by digest; no response cache), every response
`jev-1.13.0`, scored with the frozen parameters by the unchanged official scorer
(`--do_norm` off, 16,186 entries; [`test_rerun_summary.json`](../reports/task1/test_rerun_summary.json)).

| Corpus | N | 0005 (shrink) | **0009 (joint, fresh run)** | Official best (team) |
|---|---:|---:|---:|---:|
| eng_restaurant | 1504 | 1.2903 | **1.2163** | 1.1035 (LogSigma) |
| eng_laptop | 1421 | 1.3364 | **1.2086** | 1.2408 (LogSigma) |
| jpn_hotel | 1092 | 0.6997 | **0.6454** | 0.5561 (TeleAI) |
| jpn_finance | 1302 | 0.7562 | **0.7296** | 0.6581 (TeleAI) |
| rus_restaurant | 1637 | 1.3763 | **1.3290** | 1.2190 (PAI) |
| tat_restaurant | 1637 | 1.4903 | **1.4604** | 1.5294 (PAI) |
| ukr_restaurant | 1637 | 1.3846 | **1.3464** | 1.1888 (PAI) |
| zho_restaurant | 1929 | 1.0577 | **0.9591** | 0.9256 (ICT-NLP) |
| zho_laptop | 1673 | 0.7721 | **0.7611** | 0.6103 (TeleAI) |
| zho_finance | 2354 | 0.6345 | **0.5823** | 0.4841 (HUS@NLP-VNU) |
| **Micro** | 16186 | 1.1199 | **1.0645** | — |

Micro RMSE_V 0.7076, RMSE_A 0.7952. Official per-corpus scores:
[ACL Anthology](https://aclanthology.org/2026.semeval-1.452/), Table 6. Reconstructed micro
aggregates (`√(Σ N_c · RMSE_c² / Σ N_c)`) of the 14 teams with all ten corpora, best first:
PAI ≈ 1.0663, TeleAI ≈ 1.0737, PALI ≈ 1.1340, HUS@NLP-VNU ≈ 1.1368, Habib University ≈ 1.1467.

The frozen parameters were first scored on the cached 0005 test responses: 1.0639
([`test_summary.json`](../reports/task1/test_summary.json)); per corpus the fresh run differs
by −0.0014 to +0.0039. Individual Jev scores differ by 0.050 on average between the two sets
of responses (8.9% identical, mean shift −0.0004).

## Cost

Fresh test run: 9,658 requests, 21.3M input tokens (≈ $0.89 at $0.042/M). The BM25 candidate
used 9.1M input tokens on the train sample and dev (≈ $0.38).

## Artifacts

- [`test_rerun_summary.json`](../reports/task1/test_rerun_summary.json) (fresh run) and
  [`test_summary.json`](../reports/task1/test_summary.json) (cached responses), official scorer per corpus.
- Fresh responses: ignored `reports/calibration_20260923/cache/*/test_rerun_s9.jsonl`.
- Frozen parameters and selection: [`parameters.json`](../reports/task1/parameters.json),
  [`selection.json`](../reports/task1/selection.json).
- BM25 predictions and requests: ignored `reports/task1/cache/`.

## Observed

- The joint calibration lowers test RMSE on all ten corpora; micro −0.0554 (−4.9%), dev −5.6%.
- Its test micro is the lowest reconstructed aggregate of the 14 teams with all ten corpora,
  0.0018 below PAI (0.0024 on the cached responses). The competition ranks each
  corpus, not this aggregate, and without the teams' predictions no paired test of that
  difference is possible.
- Per corpus its RMSE is lower than the official best on eng_laptop and tat_restaurant, and
  higher on the other eight (by 0.03–0.16).
- BM25-retrieved examples did not help on dev: with joint calibration better on six corpora,
  worse on four, and 0.0041 worse overall.
- Task 1 test had been scored for logs 0001–0005 before this configuration was chosen; the
  choice here used train and dev only. The frozen parameters were scored on test twice, on the
  cached responses and on the fresh run, with no change in between.
