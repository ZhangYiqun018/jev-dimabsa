# 0003 — Subtask 1, 3-shot with valence-stratified examples

**2026-09-22** · `jev-1.13.0` · test split · RMSE_VA, official scorer, `--do_norm` off

## Configuration

Everything from [0002](0002-st1-few-shot-n3.md) except how the 3 examples are picked.

- `--example-selection stratified`: the 1–9 scale is cut into 3 equal-width bands
  (1.00–3.67, 3.67–6.33, 6.33–9.00) and the earliest non-leaking train record in each band is
  taken, at most one example per record. Frozen for the whole run, recorded by ID in the
  metadata — same as 0002.
- Question wording, `state` shape, and `n=3` are unchanged.
- Rubric fingerprint `7822495a335f`.

0002's `first-k` picks the first 3 train records, which is also the official protocol
(*"the first k samples in the training set"*). On 8/10 corpora those 3 land above 5.0:

| corpus | first-k valences | stratified valences |
|---|---|---|
| eng_restaurant | 6.75, 7.83, 7.50 | 2.33, 5.00, 6.75 |
| eng_laptop | 7.12, 5.50, 5.00 | 3.30, 5.50, 7.12 |
| zho_restaurant | 4.00, 6.25, 4.75 | 3.50, 4.00, 6.62 |
| zho_laptop | 6.00, 6.83, 6.00 | 3.50, 6.00, 6.83 |
| zho_finance | 6.17, 6.25, 6.00 | 3.33, 6.17, 6.50 |
| jpn_hotel | 6.33, 3.50, 6.75 | 3.50, 6.33, 6.75 |
| jpn_finance | 6.00, 6.00, 3.00 | 3.00, 6.00, 7.00 |
| rus_restaurant | 8.30, 5.00, 4.62 | 3.20, 5.00, 8.30 |
| tat_restaurant | 8.30, 5.00, 4.62 | 3.20, 5.00, 8.30 |
| ukr_restaurant | 8.30, 5.00, 4.62 | 3.20, 5.00, 8.30 |

## Result

| corpus | n | stratified | first-k | Δ | PCC_V | PCC_A |
|---|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 1504 | 2.4893 | 2.4426 | +0.0467 | 0.9190 | 0.5226 |
| eng_laptop | 1421 | 2.7383 | 2.7153 | +0.0230 | 0.8902 | 0.3830 |
| zho_restaurant | 1929 | 2.0436 | 2.0736 | −0.0300 | 0.8504 | 0.4063 |
| zho_laptop | 1673 | 1.9417 | 1.8503 | +0.0914 | 0.8891 | 0.5291 |
| zho_finance | 2354 | 2.2635 | 2.2657 | −0.0022 | 0.8140 | 0.4391 |
| jpn_hotel | 1092 | 2.1916 | 2.1906 | +0.0009 | 0.9434 | 0.6273 |
| jpn_finance | 1302 | 2.4463 | 2.5357 | −0.0894 | 0.9147 | 0.2942 |
| rus_restaurant | 1637 | 1.7618 | 1.8165 | −0.0547 | 0.9240 | 0.5538 |
| tat_restaurant | 1637 | 1.9126 | 1.9505 | −0.0380 | 0.8703 | 0.4992 |
| ukr_restaurant | 1637 | 1.7837 | 1.8441 | −0.0604 | 0.9215 | 0.5460 |
| **micro (N=16,186)** | 16186 | **2.1628** | 2.1721 | **−0.0093** | | |

6/10 corpora better, 4/10 worse. The micro delta is −0.0093, which is 3% of the −0.2987 that
adding examples at all is worth (0001 → 0002); per-corpus deltas range −0.0894 to +0.0914 and
partly cancel.

## Cost

16,076,712 input tokens · **$0.6752** at $0.042/Mtok — 1.008× the `first-k` arm.

## Artifacts

`reports/pred_st1_<corpus>_test_s3_stratified.jsonl` and `.meta.json`,
`reports/st1_test_summary_s3_stratified.json`. Prediction files are gitignored.

## Observed

- **The micro-level noise floor of this metric was not measured**, so −0.0093 is not
  distinguishable from zero here. Nothing in this run separates "stratified selection is
  slightly better" from "stratified selection is slightly worse" from "the two are the same".
  What the run does show is that the *choice* of examples contributes at most a few thousandths,
  while their *presence* contributes 0.30.
- `zho_finance` lost one record to a retryable error and `run_st1.py` exited 1. The driver
  treated any nonzero exit as a dead corpus, so it recorded an `error` row with an empty message
  (the runner prints per-record errors to stdout, and the driver read stderr) and dropped the
  corpus's 2,274,292 tokens from the printed tally, which read `$0.5797`. The record was resumed
  and every number above comes from re-scoring and re-summing afterwards. The driver was then
  fixed to treat exit 1 as a partial result, resume once, and score what is on disk; its row in
  `reports/st1_test_summary_s3_stratified.json` was recomputed from the same artifacts and now
  carries the true token count. The table and `$0.6752` were already correct and are unchanged.
- `rus_restaurant`, `tat_restaurant` and `ukr_restaurant` are parallel translations with
  identical gold, and they select identical examples under both strategies — three sites that
  vary together, not three independent observations.
- `PCC_A` is lowest on `jpn_finance` (0.2942), the corpus with the largest gain.
- Rubric fingerprint changed from 0002's `bcf1bcac3b09` because the fingerprint covers
  `jev/fewshot.py`, which this change edited. The `first-k` example IDs were re-checked against
  0002 and are unchanged, so the `first-k` column above is the 0002 predictions re-scored, not a
  re-run.
