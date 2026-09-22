# 0004 — Subtask 1, example-count sweep at n = 3, 5, 9

**2026-09-22** · `jev-1.13.0` · test split · RMSE_VA, official scorer, `--do_norm` off

## Configuration

Everything from [0003](0003-st1-few-shot-n3-stratified.md) except `--shots`. Valence-stratified
selection throughout, so n is the only knob that moves. Rubric fingerprint `7822495a335f`.

At `n` the 1–9 scale is cut into `n` equal-width bands and the earliest non-leaking train record
in each is taken, at most one example per record. Bands narrow as n grows — 2.67 wide at n=3,
0.89 at n=9 — and the train valence distribution is concentrated in the middle, so at n=9 the
bands at both ends are empty for two corpora:

| corpus | bands occupied at n=9 | examples chosen outside a band |
|---|---|---|
| zho_laptop | 7/9 (bands 1.00–1.89 and 8.11–9.00 empty) | 6.00, 6.17 |
| zho_finance | 6/9 (bands 1.00–1.89, 1.89–2.78, 8.11–9.00 empty) | 6.25, 5.67, 5.50 |

Those are filled by the file-order top-up, so for these two corpora the n=9 set is part
band-spanning and part first-records-in-file. The other 8 corpora occupy all 9 bands. At n=5,
`zho_finance` occupies 3/5 bands and the other 9 corpora occupy 5/5.

## Result

| corpus | n | n=3 | n=5 | n=9 | 5−3 | 9−5 |
|---|---:|---:|---:|---:|---:|---:|
| eng_restaurant | 1504 | 2.4893 | 2.4909 | 2.4196 | +0.0016 | −0.0713 |
| eng_laptop | 1421 | 2.7383 | 2.6831 | 2.6957 | −0.0552 | +0.0126 |
| zho_restaurant | 1929 | 2.0436 | 2.0669 | 2.0327 | +0.0233 | −0.0342 |
| zho_laptop | 1673 | 1.9417 | 2.0263 | 1.9673 | +0.0846 | −0.0590 |
| zho_finance | 2354 | 2.2635 | 2.0611 | 1.9676 | −0.2024 | −0.0935 |
| jpn_hotel | 1092 | 2.1916 | 2.1222 | 2.0253 | −0.0694 | −0.0969 |
| jpn_finance | 1302 | 2.4463 | 2.4514 | 2.2125 | +0.0051 | −0.2389 |
| rus_restaurant | 1637 | 1.7618 | 1.7517 | 1.7473 | −0.0101 | −0.0044 |
| tat_restaurant | 1637 | 1.9126 | 1.9092 | 1.8997 | −0.0034 | −0.0095 |
| ukr_restaurant | 1637 | 1.7837 | 1.7635 | 1.7733 | −0.0202 | +0.0098 |
| **micro (N=16,186)** | 16186 | **2.1628** | **2.1309** | **2.0736** | **−0.0319** | **−0.0573** |

`PCC` for the n=5 and n=9 arms:

| arm | RMSE_VA | PCC_V | PCC_A |
|---|---:|---:|---:|
| n=3 (0003) | 2.1628 | 0.8937 | 0.4801 |
| n=5 | 2.1309 | 0.8952 | 0.4909 |
| n=9 | 2.0736 | 0.8960 | 0.4922 |

`n=9` is the first Jev arm to beat GPT-5 mini one-shot (2.1552); it remains 0.1863 behind
Kimi-K2 one-shot (1.8873).

## Cost

| arm | input tokens | cost |
|---|---:|---:|
| n=3 | 16,076,712 | $0.6752 |
| n=5 | 17,725,728 | $0.7445 |
| n=9 | 21,275,864 | $0.8936 |

## Artifacts

`reports/pred_st1_<corpus>_test_s{5,9}_stratified.jsonl` and `.meta.json`,
`reports/st1_test_summary_s{5,9}_stratified.json`, `reports/run_s5_s9_stratified.log`.

## Observed

- **The n=5 gain is one corpus.** Removing `zho_finance` from the micro average leaves
  −0.0027 instead of −0.0319; that corpus moved −0.2024 and carries the most gold entries
  (2354), so it dominates the weighted average. Across the other nine, the change is not
  measurable. 6/10 corpora improved, 4/10 regressed.
- **The n=9 gain is spread.** Removing the largest mover (`jpn_finance`, −0.2389) leaves
  −0.0395 of the −0.0573. 8/10 corpora improved over n=5.
- No record failed and no run was resumed: both arms completed in one pass with zero `ERROR`
  lines, and the totals the driver printed equal the sums over the metadata.
- Both arms used the driver fix from [0003](0003-st1-few-shot-n3-stratified.md); the printed
  cost totals are complete.
- Leak filtering drops 26 candidates for `jpn_hotel`, 2 for `tat_restaurant` and 1 for
  `eng_restaurant` regardless of n — the filter runs on the candidate pool, before selection.
- `rus_restaurant`, `tat_restaurant` and `ukr_restaurant` select identical examples at every n
  (parallel translations with identical gold), so they are one observation reported three times.
- These are single runs. Jev is non-deterministic at roughly 0.04 per dimension, and the
  micro-level noise floor of this metric was not measured, so the −0.0027 above is the only
  delta here that can be called indistinguishable from zero on its own terms; the rest are not
  tested for significance either.
