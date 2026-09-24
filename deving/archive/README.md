# Archived Task 2 experiment code

Code from retired Task 2 extraction experiments, moved here verbatim during the
2026-09-23 cleanup. Subdirectories mirror the paths the code ran from
(`archive/jev/x.py` was `jev/x.py`, and so on).

**Not maintained and not runnable in place.** Imports, `ROOT` and source
fingerprints still assume the original locations, and several scripts fingerprint
their own source path, so a rerun would miss its cache and make fresh,
nondeterministic API calls. The saved summaries under `reports/` are the record
of each experiment; this code documents how they were produced.

| Archived file | Experiment record |
|---|---|
| `jev/extraction.py` | Original extractor: BIO and start/end (SE) modes, revisions 1–3. SHA-256 prefix `41ea479bf861`; the maintained [`jev/extraction.py`](../../jev/extraction.py) keeps only BIO r3 and replays its cached requests identically. [trial/dev](../20260923-extraction-trial-dev.md), [SE vs BIO](../20260923-compact-se-vs-bio.md) |
| `tools/iterate_extraction.py` | Runner for the two records above and the combined arm |
| `jev/combined_extraction.py`, `tools/combined_experiment.py`, `tests/test_combined_extraction.py` | [Full-span candidates + retrieved demonstrations](../20260923-combined-extraction.md) |
| `jev/conditioned_se.py`, `tools/probe_conditioned_se.py`, `tests/test_conditioned_se.py` | [Aspect-conditioned SE](../20260923-conditioned-se-trial.md) |
| `jev/span_choice.py`, `tools/probe_span_choice.py`, `tools/diagnose_span_choice.py`, `tools/summarize_span_choice.py`, `tests/test_span_choice.py` | [Fixed-example BIO / full-span Choice](../20260923-anchored-span-choice.md) |
| `tools/probe_bio_rule_rollback.py`, `tests/test_bio_rule_rollback.py` | [Rule rollback](../20260923-bio-rule-rollback.md) |
| `tools/probe_bio_bm25.py`, `tools/summarize_bio_bm25.py`, `tests/test_bio_bm25.py` | [BM25 two-shot](../20260923-bio-bm25.md) |
| `jev/retrieval_variants.py`, `tools/probe_bio_bm25_50.py`, `tools/summarize_bio_bm25_50.py`, `tools/ensemble_bio_bm25_50.py`, `tests/test_retrieval_variants.py`, `tests/test_bm25_ensemble.py` | [BM25 50-shot and ensemble](../20260923-bio-bm25-50.md) |
| `tools/probe_aspect_count.py`, `tests/test_aspect_count.py` | [Aspect count](../20260923-aspect-count.md) |
| `tools/evaluate_bio_r3_dev.py`, `tests/test_extraction.py` | Pre-cleanup versions of the maintained BIO dev entry point and extractor tests (SE tests included) |

`tools/diagnose_span_choice.py:selected` defines the trial24/dev96 samples:
records sorted by `SHA256("20260923:" + corpus + ":" + ID)`, dev excluding trial text.
