# 0002 — Subtask 1, 3-shot calibration

**2026-09-22** · `jev-1.13.0` · test split · RMSE_VA, official scorer, `--do_norm` off

## Configuration

Everything from [0001](0001-st1-zero-shot.md) except:

- `state` becomes an object: `labelled_examples` (3 records) + `review_to_score`.
- 3 in-context examples, the first 3 train records of the **same corpus**, frozen for the
  whole run and recorded by ID in the run metadata.
- Two clauses added to `instructions`: the target sentence is named as `` `review_to_score` ``,
  and the examples are marked as calibration only.
- Question wording otherwise unchanged. Rubric fingerprint `bcf1bcac3b09`.

Placement was not assumed. `tools/probe_fewshot.py` compared no-examples, structured state,
and a single concatenated text block on one sentence; the noise floor from a repeat call was
0.083 mean absolute deviation. Structured state moved the answer by 0.267 and reduced error
against gold; the text block moved it by 0.165 and increased error. Structured state was used.

## Result

| corpus | n | 3-shot | zero-shot | Δ | PCC_V | PCC_A |
|---|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 1504 | 2.4426 | 2.6048 | −0.1621 | 0.9168 | 0.5409 |
| eng_laptop | 1421 | 2.7153 | 2.9967 | −0.2814 | 0.8886 | 0.4219 |
| zho_restaurant | 1929 | 2.0736 | 2.1988 | −0.1252 | 0.8498 | 0.4060 |
| zho_laptop | 1673 | 1.8503 | 1.9910 | −0.1407 | 0.8894 | 0.5357 |
| zho_finance | 2354 | 2.2657 | 2.4773 | −0.2116 | 0.8136 | 0.4444 |
| jpn_hotel | 1092 | 2.1906 | 2.4749 | −0.2842 | 0.9437 | 0.6292 |
| jpn_finance | 1302 | 2.5357 | 3.2797 | −0.7440 | 0.9155 | 0.3323 |
| rus_restaurant | 1637 | 1.8165 | 2.1613 | −0.3448 | 0.9255 | 0.5493 |
| tat_restaurant | 1637 | 1.9505 | 2.3989 | −0.4484 | 0.8809 | 0.4962 |
| ukr_restaurant | 1637 | 1.8441 | 2.1665 | −0.3224 | 0.9192 | 0.5394 |
| **micro (N=16,186)** | 9658 | **2.1721** | 2.4708 | **−0.2987** | 0.8943 | 0.4895 |

All 10 corpora improved. Against published baselines on the same split: 7/10 corpora now beat
Kimi-K2 zero-shot (2.3849), one more than at zero-shot; still 0/10 against Kimi-K2 one-shot
(1.8873). Micro is 0.0121 ahead of Qwen3-14B QLoRA (2.1841) and 0.0169 behind GPT-5 mini
one-shot (2.1552).

## Cost

15,951,880 input tokens · **$0.6700** at $0.042/Mtok — 1.42× the zero-shot run, measured.

## Artifacts

`reports/pred_st1_<corpus>_test_s3.jsonl` and `.meta.json`,
`reports/st1_test_shots_comparison.json`. Prediction files are gitignored.

## Observed

- `eng_restaurant` and `ukr_restaurant` each lost one record to HTTP 529 and were resumed.
  Because `run_st1.py` exits 1 when any record failed, the driver recorded both as `error` rows
  with empty messages and dropped their tokens from the tally it printed — it printed `$0.5394`
  for a run that cost `$0.6700`. The numbers in this entry were computed from the prediction
  files and their metadata, not from that tally, so they were unaffected; the two rows in
  `reports/st1_test_summary_s3.json` were wrong and have been recomputed from the same
  artifacts. The driver was fixed afterwards, in the entry for [0003](0003-st1-few-shot-n3-stratified.md).
- The example set is frozen per corpus; `n=3` selects the first 3 train records and, for every
  corpus, the leak filter excluded nothing at that depth.
- `PCC_A` rose on 9/10 corpora (mean 0.4531 → 0.4895) but remains far below `PCC_V`
  (mean 0.8888 → 0.8943). Arousal is still under-predicted.
- `jpn_finance` improved most in absolute terms (−0.7440) and is still the worst corpus.
- The change is not single-variable: besides adding examples, `state` changed from a string to
  an object and two clauses were added to `instructions`. The probe isolated the structured
  state at 0.267 mean absolute deviation against a 0.083 noise floor, but no run separates
  "example content" from "structured state" on RMSE.
