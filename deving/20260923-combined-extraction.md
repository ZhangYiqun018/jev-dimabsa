# Development — full-span candidates with retrieved demonstrations

> **Archived (2026-09-23 cleanup):** this experiment's code was moved verbatim to [`archive/`](archive/README.md). It is unmaintained and not runnable in the current layout; commands below record how it was run. The `reports/` summaries are the record.

## Frozen design

This is one combined trial/dev experiment, not a component ablation. It combines
full-text span candidates, same-corpus BM25 demonstrations, permissive type
screening, and aspect-conditioned opinion decisions. It does not use BIO/SE
predictions, VA calls, calibration, or test evaluation.

- Model `jev-1.13.0`; dataset `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- Same deterministic 24 trial / 96 dev records as the r3 comparison; dev excludes
  trial text. Process corpora round-robin so partial runs do not cover only the
  first languages.
- Enumerate every contiguous 1–8-token span using the existing tokenizer. Deduplicate
  case-insensitive surfaces and preserve all occurrence offsets. No top-n cutoff.
- Retrieve three distinct train reviews from the same corpus using BM25
  (`k1=1.5`, `b=.75`): character bigrams for Chinese/Japanese, existing word units
  for other languages. Exclude normalized trial/dev text from training candidates.
- Each retrieved review supplies up to two positive and two negative decisions per
  stage, derived from its full annotations. Type examples prioritize boundary-near
  negatives; relation examples prioritize mismatched genuine opinions, then boundary
  negatives. No invented probabilities or VA labels. Both stages use the same source
  reviews but different demonstrations.
- Independent aspect/opinion Noul screening, threshold 0.35 for both. Batch at most
  48 questions with shared state; retain overlaps and all surviving candidates.
- For each retained aspect plus NULL, ask independent Noul decisions on retained
  opinions, including both exact boundary validity and the relationship. No additional
  verification stage. NULL opinions remain unsupported and count as missed gold pairs.
- Trial selects the final relation threshold from 0.50 / 0.65 / 0.80 by macro pair F1;
  ties choose the higher threshold. Freeze before dev. No screening-threshold search.
- Shared cumulative limit: 5,500,000 reported input tokens, approximately $0.231
  at $0.042/M, leaving margin under the user's approximate $0.25 budget for in-flight
  requests. Up to three requests can be in flight; unsuccessful unreported usage is
  not observable. Every completed API response is cached, even for unfinished records.

## Run

Both splits must use the same output directory to share the cumulative budget and
the frozen trial selection:

```bash
python3 tools/iterate_extraction.py --split trial --mode combined --per-corpus 6 \
  --out reports/extraction_combined_20260923 --concurrency 3 --max-input-tokens 5500000
python3 tools/iterate_extraction.py --split dev --mode combined --per-corpus 12 \
  --out reports/extraction_combined_20260923 --concurrency 3 --max-input-tokens 5500000
```

The combined mode selects its threshold internally; the legacy `--threshold` and
`--revision` options do not configure this arm. Cached requests are identified by
model, configuration, source fingerprint, exact examples and payload. Changed code
or examples cause fresh calls; this is not a free historical rerun guarantee.

## Reporting

Report exact case-insensitive surface pair F1 without VA, not official cF1. Compare
to historical BIO/SE at their frozen 0.65 threshold on the same completed records.
Record candidate coverage before/after screening, candidate/pair counts, per-stage
usage and missing records. Incomplete runs have no full macro result; any partial
metric is explicitly marked. This repeatedly used dev subset is exploratory.

The implementation adds three focused tests: Unicode/repeated-offset preservation,
retrieval exclusions and demonstration labels, and multi-aspect/multi-opinion/NULL
handling without reading query gold. Together with the three existing extractor
tests, all six passed before the API run. No preflight framework or broad test suite
was added. Production Task 2 and the README result tables are unchanged.

## Results

All 24 trial and 96 dev records completed. Returned model `jev-1.13.0`; source
fingerprint `82772f80e102`. No missing records, retries, or budget interruption.
Trial threshold scores were 0.2245 / 0.3141 / 0.3591 at 0.50 / 0.65 / 0.80;
**0.80 was frozen before dev**. Historical BIO/SE retain their original 0.65.

| Split | BIO r3 macro pair F1 | SE r3 macro pair F1 | Combined macro pair F1 |
|---|---:|---:|---:|
| trial, 24 records | 0.6121 | 0.4798 | 0.3591 |
| dev, 96 records | 0.4571 | 0.4226 | 0.3309 |

| Dev metric | BIO r3 | Combined |
|---|---:|---:|
| Pair precision, micro | 0.5000 | 0.2868 |
| Pair recall, micro | 0.4328 | 0.3881 |
| Pair F1, micro | 0.4640 | 0.3298 |
| TP / FP / FN | 87 / 87 / 114 | 78 / 194 / 123 |
| Candidate-pair coverage after screening | 0.5423 | 0.8458 |
| Explicit aspect candidate recall | 0.8117 | 0.9545 |
| Explicit opinion candidate recall | 0.6409 | 0.8729 |
| Input tokens | 712,052 | 3,264,903 |
| Estimated USD | 0.029906 | 0.137126 |

Before screening, the full dev candidate pool covers **200/201 gold pairs (99.5%)**.
After screening, 170 remain reachable; relation decisions accept only 78. Among
123 missed gold pairs, 7 lack only an aspect, 23 lack only an opinion, 1 lacks both,
and **92 have both candidates but are rejected**. All 3 NULL-aspect pairs are missed.

| Corpus | Combined pair F1 |
|---|---:|
| eng_restaurant | 0.6316 |
| eng_laptop | 0.4706 |
| zho_restaurant | 0.3038 |
| zho_laptop | 0.2609 |
| jpn_hotel | 0.2373 |
| rus_restaurant | 0.2456 |
| tat_restaurant | 0.2973 |
| ukr_restaurant | 0.2000 |

## Cost and candidate volume

| Run | Screening input | Relation input | Total input | Calls | Estimated USD |
|---|---:|---:|---:|---:|---:|
| trial | 225,119 | 116,666 | 341,785 | 173 | 0.014355 |
| dev | 1,954,351 | 1,310,552 | 3,264,903 | 1,294 | 0.137126 |
| **Total** | **2,179,470** | **1,427,218** | **3,606,688** | **1,467** | **0.151481** |

Dev generated 14,542 surface candidates; screening retained 514 aspect candidates
and 959 opinion candidates, producing 8,588 conditioned pair questions. The largest
review had 736 candidates, and the largest retained cross-product had 1,664 pairs.
The 48-question batches handled these without truncation. Dev input cost is about
4.59 times the historical BIO run. No additional API calls were made for diagnostics.

## Interpretation and decision

**This combination did not meet either improvement criterion. Do not replace BIO
or the production baseline with it.** Candidate generation and permissive screening
improve coverage, but the subsequent independent relation/boundary decisions lose
many valid pairs and keep too many invalid variants. Full coverage alone did not
translate into extraction quality.

A post-hoc surface diagnostic classifies the 194 false positives as:

- 130 have aspect and opinion surfaces each equal to, containing, or contained in
  the corresponding surfaces of at least one gold pair, with at least one mismatch;
- 1 combines two exact gold surfaces into the wrong relation;
- 63 are other cases under this heuristic.

Substring matches are not semantic adjudication and can refer to different occurrences.
Nevertheless, the large first group supports treating competing phrase boundaries
explicitly, rather than assuming independent Noul decisions will choose one correct
variant. A compact boundary Choice is a possible next experiment, **not implemented
or evaluated here**. No dev-driven threshold change was made.

This experiment changes several factors at once. It does **not** establish that
few-shot, retrieval, or aspect conditioning individually hurt performance. The result
is only evidence against this particular combined pipeline and prompts on this small,
previously used dev set.

## Artifacts

- [Trial summary](../reports/extraction_combined_20260923/trial_combined_n6.json)
- [Frozen threshold and configuration](../reports/extraction_combined_20260923/selection.json)
- [Dev summary](../reports/extraction_combined_20260923/dev_combined_n12.json)
- [Post-hoc dev diagnostic](../reports/extraction_combined_20260923/dev_diagnostic.json)
- Implementation: [combined extractor](archive/jev/combined_extraction.py),
  [experimental runner](archive/tools/combined_experiment.py),
  [three focused tests](archive/tests/test_combined_extraction.py).

Raw text, examples, predictions and responses remain only under ignored
`reports/extraction_combined_20260923/cache/`. No README result-table edits,
promotion to `logs/`, or GitHub push were made.
