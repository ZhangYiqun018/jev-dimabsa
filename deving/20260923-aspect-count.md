# Development — Direct aspect-count prediction

> **Archived (2026-09-23 cleanup):** this experiment's code was moved verbatim to [`archive/`](archive/README.md). It is unmaintained and not runnable in the current layout; commands below record how it was run. The `reports/` summaries are the record.

**Date** 2026-09-23 · **Model returned** `jev-1.13.0` · **Splits** trial/dev · **Metric** exact count accuracy

## Configuration

- Dataset snapshot: `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- All 98 Track A trial records from four corpora. Dev reuses the 96 fixed records from the earlier extraction development run, 12 per corpus across eight corpora; that selection excludes all trial text.
- One zero-shot Choice question per review, options `0` through `32`. The question was frozen before running either split. No tokenization, span extraction, examples, pair decisions or VA calls.
- Count distinct case-insensitive explicit Aspect surface strings in the gold. Repeated mentions or several opinions for the same surface string count once; NULL aspects do not count. An explicit aspect paired with a NULL opinion still counts. This is not the number of triplets, categories or occurrence spans.
- Only raw review text is sent. Gold is read for comparison after prediction. Observed gold counts range from 0 to 8, all within the fixed choice range.
- Prompt fingerprint: `8e7085055972`. Raw requests, responses and usage are cached. No changes were made to the SE/BIO extractors in this experiment.

## Result

| Metric | Trial, n=98 | Dev, n=96 |
|---|---:|---:|
| Exact count correct | 68 / 98 | 62 / 96 |
| **Exact accuracy** | **69.39%** | **64.58%** |
| Under-count | 7 (7.14%) | 4 (4.17%) |
| Over-count | 23 (23.47%) | 30 (31.25%) |
| Mean absolute count error | 0.4184 | 0.5833 |
| Within one of gold | 93.88% | 86.46% |
| Always predict one: accuracy | 72.45% | 67.71% |
| Accuracy on gold count >=2 | 11 / 21 (52.38%) | 25 / 31 (80.65%) |

Dev by corpus:

| Corpus | Correct / 12 | Accuracy | Under | Over |
|---|---:|---:|---:|---:|
| English restaurant | 10 | 83.33% | 0 | 2 |
| English laptop | 9 | 75.00% | 1 | 2 |
| Chinese restaurant | 4 | 33.33% | 1 | 7 |
| Chinese laptop | 6 | 50.00% | 0 | 6 |
| Japanese hotel | 9 | 75.00% | 0 | 3 |
| Russian restaurant | 9 | 75.00% | 1 | 2 |
| Tatar restaurant | 7 | 58.33% | 1 | 4 |
| Ukrainian restaurant | 8 | 66.67% | 0 | 4 |

## Cost

194 completed requests, 198 HTTP attempts including retries. Successful responses report **130,233 input tokens**, estimated **$0.005470** at $0.042/M input tokens. Failed attempts have no returned usage and are not included in that estimate. No test-split requests were made.

## Observed

- Overall accuracy is below the always-one baseline on both splits. Most gold examples contain one aspect; overall accuracy alone hides behavior on multi-aspect examples.
- On dev's 65 one-aspect records, 37 counts are correct and 28 are too large. These 28 records account for 28 of the 30 dev over-counts.
- On dev's 31 multi-aspect records, 25 counts are correct, four are too small and two too large. Three of the four under-counts have gold count three; the other has gold count six.
- Correct counts do not establish correct aspect identity or boundaries. This experiment returns only a number, so it cannot identify which phrase was incorrectly added or omitted.
- No count-conditioned extractor was run. These results do not measure end-to-end extraction quality or latency improvements from parallel slots.
- Trial and dev differ in language/domain composition and annotation conventions; their accuracies are not directly comparable. Some multilingual records are translated counterparts. Dev was already used by earlier experiments.
- One focused unit test passed, covering repeated aspect strings, case, multiword terms, NULL targets and implicit opinions in the trial schema.

## Artifacts

- [Summary and per-record counts](../reports/aspect_count_20260923/summary.json).
- [Probe runner and exact prompt](archive/tools/probe_aspect_count.py).
- Full requests/responses remain in ignored `reports/aspect_count_20260923/cache/`.

```bash
python3 tools/probe_aspect_count.py --out reports/aspect_count_20260923
```

The command uses the earlier extraction development run's saved dev ID selection. Existing cached responses are reused; a new output directory makes fresh API calls.
