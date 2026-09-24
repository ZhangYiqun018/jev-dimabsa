# Development experiments

Trial/dev probes, unfinished approaches and method comparisons. These are
development measurements, not benchmark results: nothing here goes into the README
test tables, and finalized experiments belong in [`logs/`](../logs/README.md).
Raw requests, predictions and dataset text stay in ignored `reports/**/cache/`.

## Task 2 status

| | Macro | Split |
|---|---:|---|
| Lexicon baseline ([log 0006](../logs/0006-st2-lexicon-pair-baseline.md)) | cF1 27.71% | official test |
| Lexicon baseline, same pipeline | cF1 30.68% | full dev |
| BIO r3 ([record](20260923-bio-r3-full-dev.md)) | cF1 43.51% | full dev; not run on test |
| **Lattice reranker** ([record](20260924-task2-plan.md)) | **cF1 56.46% / 51.68%** | full dev (5-fold CV) / official test |

BIO r3 is the development reference: [`jev/extraction.py`](../jev/extraction.py)
(`BIOExtractor`) plus [`tools/evaluate_bio_r3_dev.py`](../tools/evaluate_bio_r3_dev.py).
Weak corpora on full dev: `jpn_hotel` 17.85%, `zho_laptop` 26.72%. With the current
AO pairs, perfect VA would give 47.13%, so extraction, not VA, bounds the score.

Every later variant was compared against BIO r3 on small, repeatedly used samples
and none replaced it. **trial24** has 6 reviews each from eng/zho restaurant, zho
laptop and rus restaurant (39 gold pairs, no Japanese). **dev96** has 12 per corpus
across all eight (201 gold pairs) and is part of full dev. Differences of a few
points on these samples are within resampling noise.

## Records

AO F1 is the exact, case-insensitive surface-pair F1 without VA; it is not official
cF1. Code for retired experiments is in [`archive/`](archive/README.md).

| Record | What was tried | Result (macro) | Outputs |
|---|---|---|---|
| [Span probe](20260923-span-probe.md) | Leftmost aspect by start/end Choice; tool survey | 6/12 exact | `reports/span_probe_20260923/` |
| [BIO / SE trial-dev](20260923-extraction-trial-dev.md) | Per-token BIO vs start/end Choice, r1→r2 | dev96 AO: BIO 46.57%, SE 42.86%, lexicon 27.53% | `reports/extraction_iteration_20260923/` |
| [Aspect count](20260923-aspect-count.md) | Predict the number of aspects directly | dev96 64.58% vs always-one 67.71% | `reports/aspect_count_20260923/` |
| [Compact SE vs BIO](20260923-compact-se-vs-bio.md) | r3 for both, shared format | dev96 AO: **BIO r3 45.71%**, SE 42.26% | `reports/extraction_comparison_20260923/` (dev_bio cache reused by full dev) |
| [Extraction research](20260923-extraction-research.md) | Candidate-coverage diagnostics from caches, literature | no API calls | — |
| [Few-shot research](20260923-extraction-fewshot-research.md) | Evidence and design for extraction examples | no API calls | — |
| [Combined full-span](20260923-combined-extraction.md) | All spans + BM25 examples + conditioned relations | dev96 AO 33.09% | `reports/extraction_combined_20260923/` |
| [Conditioned SE](20260923-conditioned-se-trial.md) | Aspect → opinion start/end | trial24 AO 27.84% | `reports/conditioned_se_20260923/` |
| [Anchored span Choice](20260923-anchored-span-choice.md) | Real-example BIO; full-span Choice | trial24 AO 24.37%; Choice stopped at 11/24 | `reports/span_choice_20260923/` |
| [Rule rollback](20260923-bio-rule-rollback.md) | Real examples with old rules | trial24 AO 38.47%, below gate | `reports/bio_rule_rollback_20260923/` |
| [BM25 two-shot](20260923-bio-bm25.md) | BIO r3 with 2 retrieved train examples | trial24 58.33%, dev96 44.67% | `reports/bio_bm25_20260923/` |
| [**BIO r3 full dev**](20260923-bio-r3-full-dev.md) | Full dev, pair VA, official scorer | **cF1 43.51%**, AO 47.14% | `reports/bio_r3_full_dev_20260923/` |
| [BM25 50-shot](20260923-bio-bm25-50.md) | Three retrieval variants, 50 examples; 2-of-3 vote | trial24 53.10–54.94%; vote 54.13% | `reports/bio_bm25_50_20260923/` |
| [**Task 2 plan and reranker**](20260924-task2-plan.md) | Decoding rules, lattice pairing, variant Choice, cross-validated reranker; frozen test run | dev CV cF1 56.46; **test 51.68** | `reports/task2_dev_20260924/`, `reports/task2_test_20260924/` |

Historical references on trial24 / dev96: BIO r3 AO 61.21% / 45.71%.

## Maintained commands

```bash
# Offline summaries of the full dev run (no API calls with a complete cache)
.venv/bin/python tools/evaluate_bio_r3_dev.py
.venv/bin/python tools/summarize_bio_r3_dev.py

.venv/bin/python -m unittest tests.test_extraction tests.test_bio_r3_va
```

`evaluate_bio_r3_dev.py` makes fresh Jev requests only for requests missing from
`reports/bio_r3_full_dev_20260923/cache/calls/`; cached responses are reused only
when model, state and questions are identical.
