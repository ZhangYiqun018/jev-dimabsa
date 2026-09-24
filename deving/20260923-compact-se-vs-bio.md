# Development — compact SE versus BIO

> **Archived (2026-09-23 cleanup):** this experiment's code was moved verbatim to [`archive/`](archive/README.md). It is unmaintained and not runnable in the current layout; commands below record how it was run. The `reports/` summaries are the record.

## Frozen comparison protocol

- `jev-1.13.0`; dataset snapshot `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- Same deterministic 24 trial records and 96 dev records as the initial comparison.
- Both arms use revision 3: identical tokenizer, shared extraction rules and synthetic
  examples, NULL-aspect handling, pair questions and threshold 0.65. The threshold is
  retained from the earlier trial selection, not retuned on dev.
- Shared examples contain the same text, tokens, aspect and opinion lists, with the
  BIO-specific example labels removed for both arms. Both arms are called afresh.
- Revised SE: numbered positions with local context (two tokens on either side),
  end conditioned on start, and an explicit history of extracted spans and positions.
  Cursors advance strictly; stop on `none` or token exhaustion, without the old
  eight-span cap. The history does not backtrack or repair earlier boundary choices.
- BIO mechanics unchanged: independent B/I/O decisions per type and token, then merge.
- No counting stage, reranking, extra verification, VA scoring, or test evaluation.
- Same runner and concurrency (3 per arm). Report exact pair F1, precision/recall,
  candidate recall, questions, calls and input-token costs. API scheduling is not a
  controlled latency benchmark.
- This remains a small development comparison on already-used dev data, not an
  independent test result or a mature experiment.

## Runs

```bash
python3 tools/iterate_extraction.py --split trial --mode bio --revision 3 \
  --per-corpus 6 --threshold .65 --out reports/extraction_comparison_20260923/trial_bio
python3 tools/iterate_extraction.py --split trial --mode pointer --revision 3 \
  --per-corpus 6 --threshold .65 --out reports/extraction_comparison_20260923/trial_pointer
python3 tools/iterate_extraction.py --split dev --mode bio --revision 3 \
  --per-corpus 12 --threshold .65 --out reports/extraction_comparison_20260923/dev_bio
python3 tools/iterate_extraction.py --split dev --mode pointer --revision 3 \
  --per-corpus 12 --threshold .65 --out reports/extraction_comparison_20260923/dev_pointer
```

## Results

Returned model: `jev-1.13.0`. Extractor fingerprint: `41ea479bf861`.

Pair F1 is a case-insensitive, exact surface-string metric without VA, not official cF1.

| Split | BIO macro pair F1 | Revised SE macro pair F1 |
|---|---:|---:|
| trial | 0.6121 | 0.4798 |
| dev | 0.4571 | 0.4226 |

| Dev metric | BIO | Revised SE |
|---|---:|---:|
| Pair precision (micro) | 0.5000 | 0.5407 |
| Pair recall (micro) | 0.4328 | 0.3632 |
| Pair F1 (micro) | 0.4640 | 0.4345 |
| Candidate-pair recall | 0.5423 | 0.4328 |
| API calls | 247 | 658 |
| Total questions, including pair decisions | 5,875 | 1,535 |
| Extraction input tokens | 593,906 | 1,042,039 |
| Pair-decision input tokens | 118,146 | 68,380 |
| Total input tokens | 712,052 | 1,110,419 |
| Estimated USD | 0.029906 | 0.046638 |

| Corpus | BIO pair F1 | Revised SE pair F1 |
|---|---:|---:|
| eng_restaurant | 0.8235 | 0.8400 |
| eng_laptop | 0.6857 | 0.6471 |
| zho_restaurant | 0.4776 | 0.4151 |
| zho_laptop | 0.3333 | 0.2857 |
| jpn_hotel | 0.2927 | 0.1081 |
| rus_restaurant | 0.2449 | 0.2273 |
| tat_restaurant | 0.4912 | 0.4889 |
| ukr_restaurant | 0.3077 | 0.3684 |

## Cost

| Run | Input tokens | Estimated USD |
|---|---:|---:|
| trial bio | 97,151 | 0.004080 |
| trial pointer | 148,680 | 0.006245 |
| dev bio | 712,052 | 0.029906 |
| dev pointer | 1,110,419 | 0.046638 |
| **Total** | **2,068,302** | **0.086869** |

At $0.042/M reported input tokens; output free. Cached results and local scoring add no API cost.

## Observations and decision

- BIO has 87 correct pairs, 87 false positives and 114 false negatives; SE has 73/62/128. SE is more precise but retrieves fewer correct pairs.
- For SE, 74 missed gold pairs lack only an exact opinion candidate, 19 lack only an aspect, 21 lack both and 14 are rejected during pairing. BIO counts are 53/22/17/22 respectively. Missing exact candidates include boundary mismatches.
- SE asks fewer questions but has more candidate descriptions per position question and repeats the review, examples and growing history over sequential requests. Total extraction input remains higher; fewer questions did not imply lower billed token usage in this implementation.
- Neither arm triggered chunking or a span-count limit. SE history does not repair earlier merges or omissions, and monotonically advancing cursors still cannot recover a skipped boundary.
- SE wins on English restaurant and Ukrainian restaurant, while BIO wins on the other six sampled corpora. Japanese remains weak, especially for SE. NULL-aspect extraction remains weak for both.
- Keep BIO as the current development reference. This run does not justify replacing it with the revised SE. It also does not establish that all SE designs are inferior: this is one implementation and one run on a small, previously used dev subset.
- The earlier r2 results are historical context, not a controlled ablation: this round also neutralized the shared example format and made fresh nondeterministic calls for both arms.
- Three focused extraction tests passed, including a ten-span r3 case verifying that history snapshots are stable, local descriptions are bounded, and extraction does not stop at eight.

## Artifacts

- [BIO dev summary](../reports/extraction_comparison_20260923/dev_bio/dev_bio_r3_n12.json)
- [SE dev summary](../reports/extraction_comparison_20260923/dev_pointer/dev_pointer_r3_n12.json)
- [BIO trial summary](../reports/extraction_comparison_20260923/trial_bio/trial_bio_r3_n6.json)
- [SE trial summary](../reports/extraction_comparison_20260923/trial_pointer/trial_pointer_r3_n6.json)
- Raw requests/responses are under each run's ignored `cache/`. Code: [extractor](archive/jev/extraction.py), [runner](archive/tools/iterate_extraction.py).
