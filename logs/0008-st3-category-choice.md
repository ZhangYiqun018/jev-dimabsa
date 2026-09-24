# 0008 — Task 3 (DimASQP): categories for the frozen Task 2 pairs

**2026-09-24** · **`jev-1.13.0`** (all 17,267 requests returned it) · **official test, run once** · **macro cF1**

## Configuration

Task 3 dev and test are the Task 2 reviews in the same order; their gold is the Task 2
gold plus a Category (checked on dev: identical once Category is removed). Pairs and V/A
are the frozen Task 2 predictions ([0007](0007-st2-lattice-reranker.md); dev out-of-fold);
only the category is new. Code: [`tools/run_task3.py`](../tools/run_task3.py),
[`jev/categories.py`](../jev/categories.py).

1. **Jev Choice per kept pair** over the corpus's train categories (12–121 options), 16 pairs
   per request. Each option is described by its entity and a one-line attribute meaning
   (SemEval-2016 ABSA scheme). The state holds the review, 4 BM25 same-corpus train reviews
   with their (aspect, opinion, category) annotations, and a glossary of the 3 most frequent
   train aspects per category.
2. **Train lookups**: P(category | lower-case aspect) and P(category | lower-case opinion),
   each smoothed towards the corpus prior, plus the prior itself.
3. **Combiner**: conditional logit over categories, one 6-weight vector per language group
   (English, Chinese, Japanese, Russian/Tatar/Ukrainian), fitted on about 1,000 gold pairs
   per corpus from train with leave-one-out counts and retrieval that excludes the
   record's own text ([`category_weights.json`](../reports/task3/category_weights.json)).
   Dev was not used for fitting.

Train reviews whose text occurs in dev or test are excluded from every statistic, example
and glossary. The attribute definitions, the opinion lookup and the combined variant were
chosen on dev (development record on the `dev` branch) and frozen before this run.

## Result

Unchanged official scorer (`-t 3`). *Category accuracy* is over predicted pairs whose
aspect and opinion match a gold pair; Task 3 cF1 ≈ Task 2 cF1 × this accuracy. Teams are
those with all eight corpora in the official overview
([ACL Anthology](https://aclanthology.org/2026.semeval-1.452/), Table 8).

| Corpus | Dev cF1 | **Test cF1** | Test category accuracy | Official best (team) | PALI | Takoyaki |
|---|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 73.36 | **63.94** | 0.929 | 65.14 (Takoyaki) | 63.95 | 65.14 |
| eng_laptop | 39.69 | **37.22** | 0.603 | 42.27 (Takoyaki) | 37.93 | 42.27 |
| zho_restaurant | 54.89 | **44.99** | 0.925 | 55.21 (NYCU Speech Lab) | 53.57 | 49.66 |
| zho_laptop | 32.03 | **31.06** | 0.802 | 48.24 (NYCU Speech Lab) | 43.19 | 37.45 |
| jpn_hotel | 42.36 | **37.17** | 0.753 | 42.52 (PALI) | 42.52 | 40.86 |
| rus_restaurant | 49.75 | **46.62** | 0.919 | 55.99 (PAI) | 54.96 | 51.30 |
| tat_restaurant | 49.67 | **42.78** | 0.920 | 47.36 (Takoyaki) | 44.43 | 47.36 |
| ukr_restaurant | 48.87 | **45.22** | 0.925 | 54.37 (PAI) | 53.07 | 50.19 |
| **Macro** | **48.83** | **43.62** | — | — | 49.20 | 48.03 |

Macro cF1 of the 9 teams with all eight corpora (computed from Table 8): PALI 49.20,
Takoyaki 48.03, nchellwig 47.19, ALPS-Lab 45.81, TeamLasse 44.33, AILS-NTUA 40.63, TeleAI
31.26, Scmhl5 30.65, Habib University 22.07. Baselines
([arXiv:2601.23022](https://arxiv.org/abs/2601.23022), Table 3): Llama-3.3-70B fine-tuned
38.62, GPT-OSS-120B fine-tuned 37.27, Kimi-K2 Thinking one-shot 26.95, Qwen3-14B QLoRA 14.51.

## Cost

Test: 5,669 requests, 30.3M input tokens, ≈ $1.27 at $0.042/M input tokens. Development
(two iterations: without and with attribute definitions): train samples 40.8M, dev 11.5M
input tokens (≈ $2.2).

## Artifacts

- Metrics: [`test_summary.json`](../reports/task3/test_summary.json),
  [`dev_summary.json`](../reports/task3/dev_summary.json) (all four variants).
- Frozen parameters: [`category_weights.json`](../reports/task3/category_weights.json).
- Predictions, requests and responses are in ignored `reports/task3/` prediction files and
  `cache/` (they contain dataset text). With a complete cache,
  `tools/run_task3.py test --cache-only` reproduces the test predictions.

## Observed

- Test keeps 84.4% of the Task 2 test cF1 (43.62 / 51.68); dev kept 86.5%.
- Category errors on the laptop and hotel corpora are mostly in the attribute: on dev the
  entity is right for 91–95% of matched pairs (first dev iteration).
- Scoring the dev gold as a prediction gives cF1 1.0 on every corpus; the Task 2 dev pairs
  with gold categories reproduce Task 2 dev cF1 except jpn_hotel (−0.30), where a few gold
  pairs carry two categories and one is output.
