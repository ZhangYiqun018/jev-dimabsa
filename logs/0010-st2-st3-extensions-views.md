# 0010 — Tasks 2 and 3: opinion extensions, BM25 views and rival features

**2026-09-28** · **`jev-1.13.0`** · **official test, run once** · **macro cF1**

## Configuration

Builds on the frozen 0007 pipeline (same BIO r3 extraction, lattice, checks, V/A lines) and,
for Task 3, on the frozen 0008 category stage (same train-fitted weights). Code:
[`tools/run_task2.py`](../tools/run_task2.py), [`jev/extend.py`](../jev/extend.py),
[`jev/retrieval.py`](../jev/retrieval.py), [`jev/rerank.py`](../jev/rerank.py),
[`tools/run_task3.py`](../tools/run_task3.py).

1. **Opinion extensions.** Every lattice opinion candidate is extended by up to 3 tokens to
   the left and 2 to the right (no punctuation inside, at most 12 tokens). The
   example-conditioned span check judges each new variant; variants with p ≥ 0.5 are paired
   with the lattice aspects and get the BIO r3 pair Noul and the pair check. On dev this
   raises the share of gold opinions among the candidates from 0.70 to 0.85 (zho_laptop)
   and from 0.75–0.77 to 0.86–0.89 (rus, tat, ukr).
2. **BM25 views.** The span and pair checks are repeated with 4 train examples retrieved by
   character-trigram BM25 and by word BM25 (jieba for Chinese; extraction tokens otherwise),
   next to the 0007 character-bigram examples.
3. **Reranker features added:** extension flag; mean and minimum check logit across the three
   views; each check logit minus that of the best overlapping rival pair. Same model family,
   L2, candidate minimum and threshold as 0007; refitted on full dev
   ([`reranker.json`](../reports/task2_v2/reranker.json)).

Tried on dev and not adopted (development record on the `dev` branch): joint V/A
calibration of the pair scores (+0.03), per-corpus thresholds (≤ +1.5 on single corpora,
unstable), a Jev Choice among competing spans with 8 paired examples (no macro gain), and
16 instead of 4 examples per check (span-check AUC +0.007 to +0.023).

## Result

Unchanged official scorer. Dev is 5-fold out-of-fold by record.

| Corpus | Task 2 dev 0007 → 0010 | **Task 2 test** 0007 → **0010** | Task 3 dev 0008 → 0010 | **Task 3 test** 0008 → **0010** |
|---|---:|---:|---:|---:|
| eng_restaurant | 76.35 → 77.82 | 68.76 → **68.21** | 73.36 → 74.86 | 63.94 → **63.48** |
| eng_laptop | 67.67 → 69.95 | 61.86 → **62.25** | 39.69 → 41.52 | 37.22 → **37.38** |
| zho_restaurant | 57.89 → 58.80 | 48.64 → **50.21** | 54.89 → 56.26 | 44.99 → **46.51** |
| zho_laptop | 37.84 → 39.74 | 38.72 → **39.43** | 32.03 → 33.28 | 31.06 → **31.88** |
| jpn_hotel | 53.58 → 53.62 | 49.43 → **50.03** | 42.36 → 41.89 | 37.17 → **37.59** |
| rus_restaurant | 53.75 → 53.86 | 50.75 → **51.26** | 49.75 → 50.90 | 46.62 → **46.80** |
| tat_restaurant | 51.59 → 56.16 | 46.44 → **45.09** | 49.67 → 52.23 | 42.78 → **42.03** |
| ukr_restaurant | 53.03 → 53.72 | 48.87 → **50.24** | 48.87 → 49.66 | 45.22 → **46.84** |
| **Macro** | 56.46 → **57.96** | 51.68 → **52.09** | 48.83 → **50.07** | 43.62 → **44.06** |

Task 2 test perfect-VA F1 56.55 (0007: 56.12). Task 3 category accuracy on matched test
pairs 0.60–0.93, as in 0008.

Official overview ([ACL Anthology](https://aclanthology.org/2026.semeval-1.452/), Tables 7
and 8), macro over teams with all eight corpora: Task 2 — TeamLasse 53.43, this system
52.09, kevinyu66 51.48 (top three: PAI 57.73, PALI 57.50, nchellwig 56.55); Task 3 —
TeamLasse 44.33, this system 44.06, AILS-NTUA 40.63 (top three: PALI 49.20, Takoyaki
48.03, nchellwig 47.19).

## Cost

This round, Tasks 2 and 3: 131.0M input tokens (≈ $5.50 at $0.042/M), of which dev
development ≈ 25.4M and the test run (extension checks and pairs, two extra views, V/A of
the new selections, Task 3 categories) ≈ 105.6M.

## Artifacts

- Metrics: [`reports/task2_v2/`](../reports/task2_v2/) and
  [`reports/task3_v2/`](../reports/task3_v2/) (`dev_summary.json`, `test_summary.json`).
- Requests and responses: ignored `reports/task2/cache/` and `reports/task3/cache/`; the V/A
  and Task 3 per-record caches are keyed by the selected pairs.

## Observed

- Test kept about a third of the dev gain: Task 2 +0.41 against +1.50 on dev, Task 3
  +0.44 against +1.24.
- tat_restaurant rose 4.57 on dev (102 gold pairs) and fell 1.35 on test; zho_restaurant
  and ukr_restaurant gained most on test (+1.57, +1.37).
- The largest remaining Task 2 gaps to the leaders are zho_laptop (39.43 vs 53.08) and
  zho_restaurant (50.21 vs 56.38).
