# 0009 — Task 1 (DimASR): joint V/A calibration

**2026-09-24** · **`jev-1.13.0`** · **official test, run once for this configuration** · **micro RMSE_VA**

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

Test (unchanged official scorer, `--do_norm` off, 16,186 entries):

| Corpus | N | 0005 (shrink) | **0009 (joint)** | Official best (team) |
|---|---:|---:|---:|---:|
| eng_restaurant | 1504 | 1.2903 | **1.2150** | 1.1035 (LogSigma) |
| eng_laptop | 1421 | 1.3364 | **1.2088** | 1.2408 (LogSigma) |
| jpn_hotel | 1092 | 0.6997 | **0.6440** | 0.5561 (TeleAI) |
| jpn_finance | 1302 | 0.7562 | **0.7289** | 0.6581 (TeleAI) |
| rus_restaurant | 1637 | 1.3763 | **1.3304** | 1.2190 (PAI) |
| tat_restaurant | 1637 | 1.4903 | **1.4613** | 1.5294 (PAI) |
| ukr_restaurant | 1637 | 1.3846 | **1.3425** | 1.1888 (PAI) |
| zho_restaurant | 1929 | 1.0577 | **0.9585** | 0.9256 (ICT-NLP) |
| zho_laptop | 1673 | 0.7721 | **0.7600** | 0.6103 (TeleAI) |
| zho_finance | 2354 | 0.6345 | **0.5815** | 0.4841 (HUS@NLP-VNU) |
| **Micro** | 16186 | 1.1199 | **1.0639** | — |

Micro RMSE_V 0.7068, RMSE_A 0.7951. Official per-corpus scores:
[ACL Anthology](https://aclanthology.org/2026.semeval-1.452/), Table 6. Reconstructed micro
aggregates (`√(Σ N_c · RMSE_c² / Σ N_c)`) of the 14 teams with all ten corpora, best first:
PAI ≈ 1.0663, TeleAI ≈ 1.0737, PALI ≈ 1.1340, HUS@NLP-VNU ≈ 1.1368, Habib University ≈ 1.1467.

## Cost

No new requests for the selected system. The BM25 candidate used 9.1M input tokens on the
train sample and dev (≈ $0.38 at $0.042/M).

## Artifacts

- [`test_summary.json`](../reports/task1/test_summary.json) (official scorer per corpus).
- Frozen parameters and selection: [`parameters.json`](../reports/task1/parameters.json),
  [`selection.json`](../reports/task1/selection.json).
- BM25 predictions and requests: ignored `reports/task1/cache/`.

## Observed

- The joint calibration lowers test RMSE on all ten corpora; micro −0.0560 (−5.0%), dev −5.6%.
- Its test micro is 0.0024 below PAI's reconstructed aggregate. The competition ranks each
  corpus, not this aggregate, and without the teams' predictions no paired test of that
  difference is possible.
- Per corpus its RMSE is lower than the official best on eng_laptop and tat_restaurant, and
  higher on the other eight (by 0.03–0.15).
- BM25-retrieved examples did not help on dev: with joint calibration better on six corpora,
  worse on four, and 0.0041 worse overall.
- Task 1 test had been scored for logs 0001–0005 before this configuration was chosen; the
  choice here used train and dev only, and test was scored once.
