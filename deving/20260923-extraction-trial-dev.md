# Development — Trial/dev information extraction with BIO and start/end Choice

> **Archived (2026-09-23 cleanup):** this experiment's code was moved verbatim to [`archive/`](archive/README.md). It is unmaintained and not runnable in the current layout; commands below record how it was run. The `reports/` summaries are the record.

**Date** 2026-09-23 · **Model returned** `jev-1.13.0` · **Splits** trial/dev only · **Metric** exact aspect–opinion pair F1, without VA

## Configuration

- Dataset snapshot: `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, as recorded in `data-version.json`.
- Trial: six records each from English restaurant, Chinese restaurant/laptop and Russian restaurant, 24 total. Dev: 12 records per corpus across all eight corpora, 96 total. Sort by SHA256 of `20260923:<corpus>:<ID>` and select the first N. Dev excludes text appearing anywhere in the four trial files.
- Inference receives only ID/Text. Gold is used for evaluation and trial iteration, never included in inference inputs. Two hand-written, invented boundary examples are included in revision 2; no example was copied from the dataset.
- Both methods use the same deterministic units: individual CJK characters, Unicode words elsewhere, and separate punctuation. Offsets index the original text; whitespace is preserved when restoring spans. Stanza/GLiNER were not used.
- **BIO**: independently label each token as beginning/inside/outside an aspect, and beginning/inside/outside an opinion; merge contiguous spans. Orphan I starts a span. Up to 48 questions per API batch.
- **Pointer**: choose the next start, then its conditional end, for aspects and opinions. Continue until `none`; eight spans per type per 200-token chunk. The extra start query records whether the cap omitted another span. End choices display the resulting original-text slice. No input text was truncated.
- Both methods form aspect × opinion candidates plus NULL-aspect candidates, then use one Noul decision per pair. No VA calls, calibration, second-pass verification or reranking. This experiment does **not** yet use aspect-conditioned opinion extraction.
- Revision 1 used the generic extraction instructions. Revision 2 added boundary guidance and two synthetic examples, based on observed trial errors. Extraction fingerprints: r1 `b94799e261a4`, r2 `cfcabdeb8d35`.
- After revision 2, compare pair thresholds 0.50/0.65/0.80 using cached trial probabilities. Both methods selected 0.65 by corpus-macro pair F1. Freeze both prompts and threshold before dev.
- Baseline comparison reuses existing dev predictions from experiment 0006 for exactly the selected IDs. No new baseline API calls.

## Result

F1 uses per-record sets of lowercased `(Aspect, Opinion)` surface strings, requiring exact phrase boundaries. Gold has no character offsets, so occurrence-offset accuracy is not measured. Every gold pair, including NULL fields, stays in the denominator. This is **not the official VA-weighted cF1**.

Trial corpus-macro pair F1, on the same 24 records:

| Method | r1, threshold .50 | r2, threshold .50 | r2, threshold .65 |
|---|---:|---:|---:|
| BIO | 0.3910 | 0.4421 | 0.4985 |
| Start/end Choice | 0.3044 | 0.4675 | 0.4963 |

Frozen dev comparison, 12 records per corpus:

| Corpus | Existing lexicon baseline | BIO r2 | Start/end r2 |
|---|---:|---:|---:|
| English restaurant | 0.5965 | 0.8571 | 0.8333 |
| English laptop | 0.3077 | 0.7222 | 0.6111 |
| Chinese restaurant | 0.1616 | 0.5152 | 0.3396 |
| Chinese laptop | 0.1053 | 0.2051 | 0.4118 |
| Japanese hotel | 0.4000 | 0.3415 | 0.2424 |
| Russian restaurant | 0.2326 | 0.3478 | 0.2791 |
| Tatar restaurant | 0.1935 | 0.4211 | 0.4000 |
| Ukrainian restaurant | 0.2051 | 0.3158 | 0.3111 |
| **Macro** | **0.2753** | **0.4657** | **0.4286** |

Dev micro metrics, pooling counts across the 96 records:

| Metric | Existing baseline | BIO r2 | Start/end r2 |
|---|---:|---:|---:|
| Pair precision | 0.2344 | 0.5146 | 0.5248 |
| Pair recall | 0.3184 | 0.4378 | 0.3682 |
| Pair F1 | 0.2700 | 0.4731 | 0.4327 |
| Aspect F1 in retained pairs | 0.5508 | 0.6528 | 0.6468 |
| Opinion F1 in retained pairs | 0.4144 | 0.6024 | 0.5178 |

BIO has TP/FP/FN = 88/83/113; pointer 74/67/127; baseline 64/209/137. Before pair filtering, candidate-pair recall is 0.5373 for BIO and 0.4428 for pointer.

## Cost

Estimated from reported input tokens at $0.042/M; output free. Each trial row below includes all 24 records; each dev row all 96. Cached evaluation and threshold selection do not make API calls.

| Run | API calls | Input tokens | Estimated USD |
|---|---:|---:|---:|
| Trial BIO r1 | 48 | 79,158 | 0.003325 |
| Trial pointer r1 | 133 | 94,525 | 0.003970 |
| Trial BIO r2 | 48 | 99,452 | 0.004177 |
| Trial pointer r2 | 121 | 122,149 | 0.005130 |
| Dev BIO r2 | 244 | 710,763 | 0.029852 |
| Dev pointer r2 | 690 | 1,194,258 | 0.050159 |
| **Total** | **1,284** | **2,300,305** | **0.096613** |

No fine-tuning, GPU inference or VA scoring was performed. These costs exclude the earlier baseline and the preceding 12-record feasibility probe.

## Observed

- Both new methods exceeded the old baseline in seven of eight sampled corpora; both declined on Japanese hotel. This is a small dev comparison, not a test or SOTA result. No per-language combination of methods was selected and rescored as a new system.
- Of BIO's 113 missed gold pairs, 20 lacked only an exact aspect candidate, 53 lacked only an exact opinion candidate, 20 lacked both, and 20 had both candidates but were rejected by the pair threshold. This counts boundary mismatches as missing exact candidates.
- The dev sample has three NULL-aspect gold pairs. BIO matched none and predicted 16 false NULL pairs; pointer matched one and predicted 13 false NULL pairs.
- The shared trial files include `Opinion=NULL` (including both fields NULL); Task 2 dev files do not. Neither new method predicts NULL opinions, so those trial cases remain false negatives. Trial and dev scores are not directly comparable.
- Pointer hit its eight-span cap on two dev records: Chinese laptop `6697182:S005` (aspect), and Russian restaurant `14407:3_25` (opinion). Results retain these bounded predictions; this is not an unlimited pointer decoder. BIO has no corresponding span-count cap. No sampled record needed 200-token chunking.
- Individual BIO decisions still produced fragmented spans; pointer still merged some neighboring opinions. Complete surface extraction did not follow automatically from valid position outputs.
- Some language corpora contain translated counterparts, so records across languages are not all independent observations. Dev has also been used by earlier project experiments.
- Two focused tests passed: exact Unicode/whitespace reconstruction with repeated phrases, and multi-pair pointer extraction with a NULL target and a poison gold field excluded from requests.

## Artifacts

- Implementation: [`jev/extraction.py`](archive/jev/extraction.py); runner: [`tools/iterate_extraction.py`](archive/tools/iterate_extraction.py).
- Trial selection and frozen threshold: [`selection.json`](../reports/extraction_iteration_20260923/selection.json).
- Dev [BIO summary](../reports/extraction_iteration_20260923/dev_bio_r2/dev_bio_r2_n12.json) and [pointer summary](../reports/extraction_iteration_20260923/dev_pointer_r2/dev_pointer_r2_n12.json).
- Each run's ignored `cache/` holds original-offset spans, probabilities, full requests/responses, usage and limit flags. Raw dataset text is not committed.

Reproduce the frozen dev arms in this workspace (requires the existing 0006 dev prediction cache for comparison):

```bash
python3 tools/iterate_extraction.py --split dev --mode bio --revision 2 \
  --per-corpus 12 --threshold .65 --out reports/extraction_iteration_20260923/dev_bio_r2
python3 tools/iterate_extraction.py --split dev --mode pointer --revision 2 \
  --per-corpus 12 --threshold .65 --out reports/extraction_iteration_20260923/dev_pointer_r2
```

Responses resume when the source fingerprint matches the cached run. The extractor has
since been revised, so running the current code may make fresh calls even in these
directories; consult the saved summaries for the historical measurements. New calls
are potentially nondeterministic. The original Task 2 baseline command is unchanged.
