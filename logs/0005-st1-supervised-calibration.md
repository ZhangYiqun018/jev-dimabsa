# 0005 — Supervised train calibration, frozen dev selection and test

**Date:** 2026-09-23 · **Returned model:** `jev-1.13.0` · **Rubric fingerprint:** `3a2a9aee82a9`

**Metric:** official ST1 `RMSE_VA`, `--do_norm` off; micro weighting by all gold annotations.

## Configuration

- Arms: zero-shot and the frozen historical stratified 9-shot examples. Example gold scores were unchanged; calibration transforms model outputs only.
- Seed: 20260923. Each corpus uses 256 training text groups, excluding the frozen examples and dev/test ID or normalized-text overlaps.
- Russian, Tatar and Ukrainian translations share selected source groups and five-fold assignments. Repeated text stays in one group.
- Calibration uses 2,563 records / 4,664 VA annotations per arm, with 2,048 distinct groups after combining parallel translations. The two arms use the same training records.
- Candidates: raw, train mean, offset, shrinkage, nonnegative affine. Coefficients are fitted per corpus and dimension on train only. Shrinkage uses `mean_gold + alpha * (raw - mean_raw)`, with alpha in `{0, .25, .5, .75, 1}` selected by five-fold grouped train CV.
- Dev selects one method type per arm. Within 0.02 RMSE of the best, prefer raw → mean → offset → shrink → linear. Retain calibration only if improvement is at least 0.02 and the paired cluster bootstrap 95% interval is below zero (2,000 draws).
- Both arms selected shrinkage. Parameters and request configuration were frozen before test. No dev refitting.
- Test: 9,658 texts / 16,186 gold annotations per arm. Outputs are clipped to [1,9], rounded to two decimals and checked against the unchanged official scorer for every corpus and arm. Bootstrap intervals use unrounded predictions and shared source clusters for parallel translations.
- Concurrency: 10. The existing rubric and frozen example payloads were used throughout.

## Training calibration sample

| Corpus | Text groups | Records | VA annotations |
|---|---:|---:|---:|
| eng_restaurant | 256 | 257 | 397 |
| eng_laptop | 256 | 257 | 377 |
| zho_restaurant | 256 | 257 | 356 |
| zho_laptop | 256 | 256 | 498 |
| zho_finance | 256 | 256 | 673 |
| jpn_hotel | 256 | 256 | 456 |
| jpn_finance | 256 | 256 | 425 |
| rus_restaurant | 256 | 256 | 494 |
| tat_restaurant | 256 | 256 | 494 |
| ukr_restaurant | 256 | 256 | 494 |

## Dev results

Micro RMSE_VA, N = 3,267; full precision model scores before export rounding.

| Arm | Raw | Mean | Offset | Shrink | Linear | Selected |
|---|---:|---:|---:|---:|---:|---|
| 0-shot | 2.3812 | 1.5209 | 1.6116 | 0.9111 | 0.8991 | shrink |
| 9-shot | 2.0413 | 1.5209 | 1.5939 | 0.9077 | 0.8897 | shrink |

## Frozen test results

| Corpus | Zero raw | Zero + shrink | 9-shot raw | 9-shot + shrink |
|---|---:|---:|---:|---:|
| eng_restaurant | 2.6075 | 1.0783 | 2.4137 | 1.2903 |
| eng_laptop | 2.9967 | 1.3324 | 2.6954 | 1.3364 |
| zho_restaurant | 2.1972 | 1.0606 | 2.0302 | 1.0577 |
| zho_laptop | 1.9902 | 0.7629 | 1.9672 | 0.7721 |
| zho_finance | 2.4768 | 0.6286 | 1.9682 | 0.6345 |
| jpn_hotel | 2.4785 | 0.7187 | 2.0261 | 0.6997 |
| jpn_finance | 3.2828 | 0.7771 | 2.2155 | 0.7562 |
| rus_restaurant | 2.1606 | 1.4463 | 1.7507 | 1.3763 |
| tat_restaurant | 2.4066 | 1.6265 | 1.8984 | 1.4903 |
| ukr_restaurant | 2.1657 | 1.4759 | 1.7726 | 1.3846 |
| **Micro, N = 16,186** | **2.4720** | **1.1395** | **2.0731** | **1.1199** |

Paired cluster bootstrap intervals are within-arm calibration minus raw:

| Arm | Dev ΔRMSE 95% interval | Test ΔRMSE 95% interval |
|---|---|---|
| 0-shot | [-1.5166, -1.4229] | [-1.3645, -1.3004] |
| 9-shot | [-1.1767, -1.0882] | [-0.9803, -0.9258] |

Per-dimension RMSE, bias and per-corpus PCC are in the JSON summaries. Undefined PCC for constant predictions is stored as `null`.

## Cost

Known returned input tokens × $0.042/M; failed requests without returned usage may incur unknown charges.

| Phase | Arm | Completed records | Input tokens | Estimated USD |
|---|---|---:|---:|---:|
| calibration | 0-shot | 2,563 | 2,802,662 | $0.1177 |
| calibration | 9-shot | 2,563 | 5,562,035 | $0.2336 |
| dev | 0-shot | 1,768 | 2,056,599 | $0.0864 |
| dev | 9-shot | 1,768 | 4,016,549 | $0.1687 |
| test | 0-shot | 9,658 | 10,923,776 | $0.4588 |
| test | 9-shot | 9,658 | 21,275,864 | $0.8936 |
| **Total** | | **27,978** | **46,637,485** | **$1.9588** |

## Artifacts

- Entry point: [`tools/calibrate_st1.py`](../tools/calibrate_st1.py). Run `dev` then `test`; existing successful predictions are reused on resume.
- [`samples.json`](../reports/calibration_20260923/samples.json): selected IDs, group membership and example IDs.
- [`parameters.json`](../reports/calibration_20260923/parameters.json), [`selection.json`](../reports/calibration_20260923/selection.json): fitted coefficients and frozen selection.
- [`dev_summary.json`](../reports/calibration_20260923/dev_summary.json), [`test_summary.json`](../reports/calibration_20260923/test_summary.json), [`usage.json`](../reports/calibration_20260923/usage.json).
- Dataset-bearing inputs, raw responses and official-format exports are local in ignored `reports/calibration_20260923/cache/`.

## Observed

- Calibration reduced test RMSE in all ten corpora for both arms.
- The calibrated test difference between arms is 0.0196 RMSE in favor of 9-shot. The intervals above compare each arm against its own raw output; they do not test the difference between arms.
- Calibration uses additional supervised training labels. These scores are a different setting from the prompt-only baseline table and published few-shot systems.
- Test data had already been examined in earlier experiments; this is not a previously untouched holdout.
- The first test process stopped while parsing undefined PCC for a constant prediction. The parser was fixed, and cached predictions were reused when scoring resumed. No calibration parameters or candidate selection changed.
- Recorded successful responses required 2 additional client retry attempts in total. All requested records completed.
- Three focused unit tests passed: transient retries; resume/accounting/incomplete-batch behavior; calibration math, group isolation and constant-output scorer parsing.
