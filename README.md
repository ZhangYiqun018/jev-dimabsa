# Jev on DimABSA — Subtask 1 experiments

**[TypeSafe Jev](https://typesafe.ai)** (System One) baselines and supervised score calibration for
[DimABSA](https://github.com/DimABSA/DimABSA2026), the dimensional aspect-based sentiment
analysis task from SemEval-2026 Task 3.

DimABSA replaces categorical polarity with continuous **valence–arousal (VA) scores on a
1.00–9.00 scale**. Jev's [`Score`](https://docs.typesafe.ai/primitives/score) primitive returns
a continuous probability-weighted position on a described scale, so the task maps onto it
without any text generation or output parsing.

No GPU and no model fine-tuning. The latest experiment fits a small local calibration
function using training labels, then applies it to Jev predictions.

## Latest result — supervised calibration

On the official ST1 test split, **9-shot + shrink calibration reaches 1.1199 RMSE_VA**;
**zero-shot + shrink calibration reaches 1.1395**. Both improve on their matched raw
outputs in all ten corpora.

Official scorer, `--do_norm` **off**, micro average over **16,186 gold annotations**.
`RMSE_VA = √mean(Δvalence² + Δarousal²)` on the 1–9 scale; **lower is better**.

| Arm | Raw RMSE_VA | After shrink calibration | Test inference cost |
|---|---:|---:|---:|
| Zero-shot | 2.4720 | **1.1395** | $0.4588 |
| 9-shot, valence-stratified | 2.0731 | **1.1199** | $0.8936 |

Most of the improvement comes from calibration. After calibration, 9-shot leads by
**0.0196 RMSE** at about **1.95×** the test inference cost. The complete calibration,
dev and test experiment cost approximately **$1.9588**, based on returned input usage.

These results use **additional supervised training labels**: 256 text groups per corpus,
2,563 training records and 4,664 VA annotations per arm. Coefficients are fitted on train;
dev selects the method; parameters are frozen before test. Few-shot examples retain their
original gold scores—only model outputs are calibrated. Test had already been examined
in earlier experiments, so it is not a previously untouched holdout.

See [experiment 0005](logs/0005-st1-supervised-calibration.md) for per-corpus results,
confidence intervals, parameters and costs, and [the calibration protocol](#calibration-protocol)
for the selection procedure.

## Historical baselines — without local score calibration

The Jev rows below are earlier runs. Published systems use different supervision settings;
the new supervised calibration results above are reported separately.

| System | RMSE_VA |
|---|---|
| Kimi-K2, one-shot | 1.8873 |
| **Jev, 9-shot, valence-stratified** | **2.0736** |
| GPT-5 mini, one-shot | 2.1552 |
| **Jev, 3-shot** | **2.1721** |
| Qwen3-14B, QLoRA fine-tuned | 2.1841 |
| Kimi-K2, zero-shot | 2.3849 |
| **Jev, zero-shot** | **2.4708** |
| GPT-5 mini, zero-shot | 2.7439 |

Published figures are from the dataset paper, [arXiv:2601.23022](https://arxiv.org/abs/2601.23022),
Table 3. The Jev 3-shot run uses the first three eligible training records; stratified
9-shot uses examples spanning the valence scale. In that sweep, stratified 3-, 5- and
9-shot achieved 2.1628, 2.1309 and 2.0736 respectively. Full configurations and results
are in [experiments 0001–0004](logs/).

## Dataset

The data belongs to the DimABSA organizers and is **not redistributed here**. Download it
yourself — the terms below are quoted from the competition rules.

> - Datasets should only be used for scientific or research purposes.
> - Any other use is explicitly prohibited.
> - Datasets must not be redistributed or shared with third parties.
> - Interested parties should be directed to the official website.

**Download:** <https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset>

Place it at `vendor/DimABSA2026/`, which is where the code expects it and which `.gitignore`
excludes. The read-only evaluation script is vendored from that repository and is never
modified; `scoring/score.py` only shells out to it.

Papers: [arXiv:2601.23022](https://arxiv.org/abs/2601.23022) (Track A dataset) ·
[arXiv:2601.21483](https://arxiv.org/abs/2601.21483) (Track B) ·
[arXiv:2604.07066](https://arxiv.org/abs/2604.07066) (task overview)

## Layout

```
jev/
  client.py     Jev HTTP client — no third-party deps; retries; never logs the key
  rubrics.py    the 9-level valence/arousal scales and the question wording
  fewshot.py    in-context example selection and leak filtering
  data.py       jsonl loading, prediction de-duplication
runners/
  run_st1.py        one corpus, one split
  run_all_st1.py    every corpus, then score each
scoring/
  score.py          wraps the official metrics script, unmodified
tools/
  leakage_audit.py  train/dev/test overlap report
  probe_fewshot.py  how calibration examples are delivered to the API
  calibrate_st1.py  train calibration, dev selection, frozen test evaluation
logs/               one file per experiment
```

## Running it

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy
export TYPESAFE_API_KEY=...          # the client also reads ~/.zshrc

# latest experiment: fit on train, select on dev, then evaluate frozen test
.venv/bin/python tools/calibrate_st1.py dev --out reports/calibration_reproduction
.venv/bin/python tools/calibrate_st1.py test --out reports/calibration_reproduction

# raw baseline: one corpus
.venv/bin/python runners/run_st1.py \
  --data vendor/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_test_task1.jsonl \
  --out reports/pred.jsonl --shots 3 --concurrency 10

# raw baseline: every corpus, scored against the official script
.venv/bin/python runners/run_all_st1.py --split test --shots 3 --concurrency 10
```

An API key comes from the [TypeSafe console](https://console.typesafe.ai/); the model is served
at `https://api.typesafe.ai/v1/systemone`. Primitive and request-shape docs:
<https://docs.typesafe.ai/introduction>.

`numpy` supports local calibration; `scipy` is used by the official scorer. Predictions are
appended as each sentence completes, so an interrupted run resumes by ID; the client retries
rate limits, 529 and timeouts with backoff. Each prediction also stores raw scores,
probabilities, confidence, returned model and usage in `_jev`. The default model is
`jev-1.13.0`. Resume checks the saved request configuration; historical files without
that configuration remain readable but require a new output path for new runs.
`--restart` explicitly discards an existing run. Incomplete batch runs exit nonzero.

## Few-shot examples

Examples come only from the **train** split of the same corpus. They are frozen per corpus: the
same records are used for every request in a run, recorded by ID in the run metadata, and never
re-picked because a prediction came out badly. `--shots N` sets how many (default 3; `0` is the
zero-shot control).

`--example-selection` picks which:

- `first-k` (default) — the first N train records, matching the official rule *"the first k
  samples in the training set"*. This is the arm to compare against the published baselines.
- `stratified` — the earliest non-leaking record in each equal-width band of the 1–9 scale, so
  the examples span the scale instead of clustering wherever the file happens to start.

Both take at most one example per record. The two arms differ by 0.0093 micro RMSE — see
[0003](logs/0003-st1-few-shot-n3-stratified.md).

`tools/leakage_audit.py` found that ID sets are disjoint across splits but **sentence text is
not** — `jpn_hotel` reuses 23 sentences between train and test, `tat_restaurant` 2, and
`eng_restaurant` 1 between train and dev. Example selection therefore drops any train record
whose normalised text or ID also occurs in the split being evaluated. At `--shots 3` this
drops nothing; at larger `k` it matters.

## Known limits

- **Jev is not deterministic.** Repeated calls on identical input differ by roughly 0.04 per
  dimension. This single-call variation is not a significance threshold for aggregate RMSE.
- In the uncalibrated baseline, arousal stays under-predicted even with examples. `PCC_A` ≈ 0.49 against `PCC_V` ≈ 0.89
  (Pearson correlation per dimension; higher is better, 1 is perfect) — Jev orders valence well
  and arousal badly. Per-corpus values are in the logs.
- Only Subtask 1 is implemented.

## Calibration protocol

This compares zero-shot and frozen stratified 9-shot requests. Each corpus uses 256
training text groups (seed 20260923), excluding examples and dev/test overlaps.
Russian, Tatar and Ukrainian translations share sample groups and CV folds.
Per-corpus, per-dimension candidates are raw, train mean, offset, shrinkage and
nonnegative affine calibration; shrinkage uses five-fold train CV. Dev chooses one
method type per arm, preferring simpler methods within 0.02 RMSE. Calibration is
retained only with at least 0.02 improvement and a paired cluster bootstrap 95%
interval below zero (2,000 draws). Parameters are frozen before test; test exports
are checked with the unchanged official scorer.

Archived results live in `reports/calibration_20260923/`. The commands above write a fresh
run to `reports/calibration_reproduction/`, since raw caches are not distributed.
Dataset-bearing inputs, responses
and exports stay in ignored `cache/`; sample IDs, parameters, selection and summaries
are separate JSON files. Rerunning resumes completed requests. Use `--out` with a new
directory for another experiment.

The three focused unit tests cover transient retries, resume/accounting and calibration
math/group isolation:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
