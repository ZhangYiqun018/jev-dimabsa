# 0007 — Task 2 (DimASTE): BIO lattice candidates + Jev checks + reranker

**2026-09-24** · **`jev-1.13.0`** (all 65,229 test requests returned it) · **official test, run once** · **macro cF1**

## Configuration

Only review text reaches Jev; annotations are read by the official scorer and, for train
and dev, by the fitting steps below. Code: [`tools/run_task2.py`](../tools/run_task2.py).

1. **BIO r3 extraction** ([`jev/extraction.py`](../jev/extraction.py)). Deterministic tokens
   (CJK characters, other words, punctuation). One Choice per token and role (aspect, opinion)
   over B/I/O, 48 per request, two invented examples in the state; argmax decode; one Noul
   per aspect × opinion span pair. Unchanged from the development reference.
2. **Lattice candidates** ([`jev/lattice.py`](../jev/lattice.py)). Per role: the argmax spans,
   every span of up to 12 tokens whose BIO-marginal score (start × continuation × end
   probability) is at least 0.2, the train edge-affix variant of each
   ([`jev/spans.py`](../jev/spans.py)), and literal train-lexicon matches
   ([`jev/triplets.py`](../jev/triplets.py)). One BIO r3 Noul pair question per
   non-overlapping aspect × opinion candidate pair. NULL aspects only for corpora whose
   train split has at least 5% of them and which are not English (README: no implicit
   aspects in English test), i.e. `jpn_hotel`.
3. **Example-conditioned checks** ([`jev/checks.py`](../jev/checks.py)). The state holds the
   review and 4 BM25 (character-bigram) same-corpus train reviews with their annotated
   phrases ([`jev/retrieval.py`](../jev/retrieval.py)); train texts that occur in dev or
   test are never retrieved. One Noul per candidate span ("exactly one annotated phrase
   of this role?") and one per candidate pair with lattice probability ≥ 0.2.
4. **Reranker** ([`jev/rerank.py`](../jev/rerank.py)). Logistic regression, L2 5, one model
   per language group (English, Chinese, Japanese, Russian/Tatar/Ukrainian), over every
   candidate pair with lattice probability ≥ 0.3. Features: the three Noul answers of
   steps 1–2, the lexicon baseline's Noul answer where that pair was asked (log
   [0006](0006-st2-lexicon-pair-baseline.md)), the two checks, BIO span scores,
   argmax/affix membership, train annotation counts, train edge-affix log-odds, lengths,
   competing boundary variants, distance, per-corpus intercepts, per-group slopes.
   Fitted on all full-dev candidates ([`reranker.json`](../reports/task2/reranker.json)).
5. **Selection.** Score order; a pair whose aspect and opinion both overlap a kept pair is
   dropped; stop at score 0.25.
6. **V/A.** The Task 1 valence and arousal Score questions per kept pair, 16 pairs per
   request, then one line per dimension and corpus fitted on 1,000 Task 2 **train** gold
   pairs ([`va_calibration.json`](../reports/task2/va_calibration.json)), clipped to [1, 9].

The configuration, threshold and feature set were chosen on full dev (development record
on the `dev` branch) and frozen before this single test run.

## Result

Unchanged official scorer. Dev is 5-fold out-of-fold by record (dev is the reranker's
training data). *Perfect-VA F1* is the categorical F1 of the same pairs, i.e. cF1 with
exact V/A. PAI, PALI and AILS-NTUA are the systems listed with all eight corpora in the
official overview ([arXiv:2604.07066](https://arxiv.org/abs/2604.07066), Table 10); per-corpus
winners also include Takoyaki (eng_restaurant 70.21, eng_laptop 63.66) and TeleAI
(jpn_hotel 58.37).

| Corpus | Dev cF1 | **Test cF1** | Test perfect-VA F1 | PAI | PALI | AILS-NTUA |
|---|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 76.35 | **68.76** | 74.13 | 69.03 | 69.28 | 65.18 |
| eng_laptop | 67.67 | **61.86** | 67.37 | 61.69 | 62.42 | 53.11 |
| zho_restaurant | 57.89 | **48.64** | 51.82 | 56.38 | 56.34 | 50.42 |
| zho_laptop | 37.84 | **38.72** | 40.72 | 53.06 | 53.08 | 46.46 |
| jpn_hotel | 53.58 | **49.43** | 51.96 | 56.82 | 56.66 | 50.21 |
| rus_restaurant | 53.75 | **50.75** | 56.33 | 57.93 | 57.24 | 49.88 |
| tat_restaurant | 51.59 | **46.44** | 52.19 | 49.08 | 48.28 | 38.74 |
| ukr_restaurant | 53.03 | **48.87** | 54.40 | 57.87 | 56.71 | 47.25 |
| **Macro** | **56.46** | **51.68** | 56.12 | 57.73 | 57.50 | 50.16 |

Earlier Jev system on the same test split: 27.71 ([0006](0006-st2-lexicon-pair-baseline.md)).
Official baselines: Kimi-K2 Thinking one-shot 38.59, Qwen3-14B QLoRA 28.75.

## Cost

Test: 65,229 requests, 129.5M input tokens, ≈ $5.44 at $0.042/M input tokens (known usage).
Dev stages used by this configuration: BIO r3 11.4M, lattice 9.9M, checks 5.3M, V/A
calibration on train 5.5M, dev V/A ≤ 5.3M input tokens (≈ $1.5). The lexicon baseline
runs are accounted in 0006.

## Artifacts

- Metrics: [`test_summary.json`](../reports/task2/test_summary.json),
  [`dev_summary.json`](../reports/task2/dev_summary.json).
- Frozen parameters: [`reranker.json`](../reports/task2/reranker.json),
  [`va_calibration.json`](../reports/task2/va_calibration.json).
- Predictions and every request/response are in the ignored `reports/task2/` prediction
  files and `cache/` (they contain dataset text). With a complete cache,
  `tools/run_task2.py test --cache-only` reproduces the test predictions byte for byte.

## Observed

- Test is 4.78 points below the dev estimate. The largest drops are zho_restaurant
  (−9.25), eng_restaurant (−7.59) and eng_laptop (−5.81); zho_laptop is +0.88.
- On dev, 61–63% of zho_laptop gold pairs are in the candidate set (eng_restaurant 91%,
  jpn_hotel 77%).
- V/A costs 4.44 points of macro test cF1 (perfect-VA F1 56.12 vs 51.68).
- `jpn_hotel` test repeats 23 train sentences (tools/leakage_audit.py); the BM25
  retriever excludes train texts that occur in dev or test.
