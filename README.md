# Jev on DimABSA — Task 1 & Task 2 baselines

<p align="center">
  <a href="#running-it"><strong>Quick start</strong></a> &nbsp;·&nbsp;
  <a href="#task-1-results">Task 1 results</a> &nbsp;·&nbsp;
  <a href="#task-2-results">Task 2 results</a> &nbsp;·&nbsp;
  <a href="#dataset">Dataset</a> &nbsp;·&nbsp;
  <a href="logs/README.md">Experiments</a> &nbsp;·&nbsp;
  <a href="#known-limits">Limitations</a>
</p>

A **[TypeSafe Jev](https://typesafe.ai)** (System One) baseline for
[DimABSA](https://github.com/DimABSA/DimABSA2026), the dimensional aspect-based sentiment
analysis task from SemEval-2026 Task 3. Task 1 scores given aspects; Task 2 extracts
aspect–opinion pairs and scores their sentiment.

DimABSA replaces categorical polarity with continuous **valence–arousal (VA) scores on a
1.00–9.00 scale**. Jev's [`Score`](https://docs.typesafe.ai/primitives/score) primitive returns
a continuous probability-weighted position on a described scale, so the task maps onto it
without any text generation or output parsing.

No GPU and no fine-tuning. One API key and about an hour.

<a id="task-1-results"></a>

## Result — Subtask 1 (DimASR), official test split

Official scorer, `--do_norm` **off**. Micro average, weighted by gold entry count (N = 16,186),
best first.

`RMSE_VA` is the root mean squared error of the predicted valence–arousal pair against gold,
`√(mean(Δvalence² + Δarousal²))`, in the same units as the 1–9 scale. **Lower is better; 0 is
perfect.** It is not a per-dimension error — both dimensions are pooled under one root.

| System | RMSE_VA |
|---|---|
| [PAI](https://aclanthology.org/2026.semeval-1.193/), Qwen3-32B LoRA + Sinkhorn adaptation† | ≈1.0663 |
| [TeleAI](https://aclanthology.org/2026.semeval-1.233/), Qwen2.5-7B LoRA regression + calibration† | ≈1.0737 |
| **Jev, 9-shot + shrink calibration\*** | **1.1199** |
| **Jev, zero-shot + shrink calibration\*** | **1.1395** |
| [ICT-NLP](https://aclanthology.org/2026.semeval-1.131/), multilingual XLM-R large ensemble† | ≈1.1592 |
| Kimi-K2, one-shot | 1.8873 |
| **Jev, 9-shot, valence-stratified** | **2.0736** |
| GPT-5 mini, one-shot | 2.1552 |
| **Jev, 3-shot** | **2.1721** |
| Qwen3-14B, QLoRA fine-tuned | 2.1841 |
| Kimi-K2, zero-shot | 2.3849 |
| **Jev, zero-shot** | **2.4708** |
| GPT-5 mini, zero-shot | 2.7439 |

\* Supervised calibration using 256 training text groups per corpus; fitted on train,
selected on dev, and frozen before test. Few-shot example scores stay unchanged.
Test was used in earlier experiments. Full protocol and results: [0005](logs/0005-st1-supervised-calibration.md).

† Official ST1 dataset winners: PAI (Russian, Tatar, Ukrainian), TeleAI (both Japanese
corpora, Chinese laptop), ICT-NLP (Chinese restaurant). Scores are approximate micro
aggregates reconstructed from the [official overview](https://aclanthology.org/2026.semeval-1.452/),
Table 6: `√(Σ N_c × RMSE_c² / Σ N_c)`. The competition ranks each corpus, not this aggregate.

Our best run is **0.0536 RMSE** behind PAI on this aggregate; matching it requires a further
**4.8%** reduction. Methods, per-corpus results and next steps: [SOTA comparison](docs/sota-comparison-2026-09-23.md).

Published baselines: [arXiv:2601.23022](https://arxiv.org/abs/2601.23022), Table 3.
Earlier experiments, per-corpus scores and costs: [`logs/`](logs/).

<a id="task-2-results"></a>

## Result — Subtask 2 (DimASTE), official test split

cF1, **higher is better**. Macro mean across all eight corpora (6,690 test texts).

| System | cF1 (macro) |
|---|---|
| Kimi-K2 Thinking, one-shot | 0.3859 |
| Qwen3-14B, QLoRA fine-tuned | 0.2875 |
| **Jev, training lexicon + pair decisions + transferred shrink\*** | 0.2771 |

\* Training-vocabulary spans, Jev pair classification and VA scoring, with frozen Task 1
zero-shot calibration. No Task 2 tuning. Official baseline scores come from the
[organizers’ report](https://aclanthology.org/2026.semeval-1.452/), Table 7; macro means
are computed here. Per-corpus scores, limitations and costs: [0006](logs/0006-st2-lexicon-pair-baseline.md).

A token-level BIO extractor with Jev pair decisions is under development. It has been
evaluated on dev only, so it is not in this table; see [`deving/`](deving/README.md).

## Dataset

The data belongs to the DimABSA organizers and is **not redistributed here**. Download it
yourself — the terms below are quoted from the competition rules.

> - Datasets should only be used for scientific or research purposes.
> - Any other use is explicitly prohibited.
> - Datasets must not be redistributed or shared with third parties.
> - Interested parties should be directed to the official website.

**Download:** [pinned official snapshot](https://github.com/DimABSA/DimABSA2026/tree/bdc93be1224106ae7d3eb95739c02a76ed4ae8a1/task-dataset)
(commit `bdc93be12241`).

Place it at `vendor/DimABSA2026/`, which is where the code expects it and which `.gitignore`
excludes. To fetch this exact version:

```bash
git clone https://github.com/DimABSA/DimABSA2026.git vendor/DimABSA2026
git -C vendor/DimABSA2026 checkout bdc93be1224106ae7d3eb95739c02a76ed4ae8a1
```

The read-only evaluation script is vendored from that repository and is never
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
  triplets.py   Task 2 baseline: training-lexicon candidates, pair decisions and VA
  extraction.py Task 2 development extractor: per-token BIO Choice + Noul pair decisions
  task2.py      Task 2 shared helpers: AO diagnostics, request cache, pair VA, official cF1
runners/
  run.py           unified --task 1|2 inference and official scoring
  execution.py     shared concurrency, resume and usage accounting
  run_st1.py        one corpus, one split
  run_all_st1.py    every corpus, then score each
scoring/
  score.py          wraps the official metrics script, unmodified
tools/
  leakage_audit.py  train/dev/test overlap report
  probe_fewshot.py  how calibration examples are delivered to the API
  calibrate_st1.py  train calibration, dev selection, frozen test evaluation
  evaluate_bio_r3_dev.py   Task 2 BIO extractor on full dev, official scorer
  summarize_bio_r3_dev.py  offline comparison of that run with the baseline
logs/               one file per mature experiment
deving/             development records; archive/ holds retired experiment code
docs/               project status/handoff and Task 1 design notes
```

## Running it

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy
export TYPESAFE_API_KEY=...          # the client also reads ~/.zshrc

# fit calibration on train, select on dev, and save parameters
.venv/bin/python tools/calibrate_st1.py dev --out reports/calibration_reproduction

# apply the saved calibration to test predictions, export and score both arms
.venv/bin/python tools/calibrate_st1.py test --out reports/calibration_reproduction

# view raw and calibrated test scores (zero-shot and 9-shot)
cat reports/calibration_reproduction/test_summary.json

# raw Task 1 baseline, one corpus (or --corpus all)
.venv/bin/python runners/run.py --task 1 --corpus eng_restaurant --split test \
  --shots 3 --out reports/st1_reproduction --concurrency 5

# Task 2 baseline: all eight corpora, transferred Task 1 calibration included
.venv/bin/python runners/run.py --task 2 --split test \
  --out reports/st2_reproduction --concurrency 5
```

Both tasks use `jev-1.13.0`; the dataset/scorer snapshot is pinned in
[`data-version.json`](data-version.json). Outputs resume by ID. The original Task 1
commands remain supported. Task 2 method and official baselines: [0006](logs/0006-st2-lexicon-pair-baseline.md).

An API key comes from the [TypeSafe console](https://console.typesafe.ai/); the model is served
at `https://api.typesafe.ai/v1/systemone`. Primitive and request-shape docs:
<https://docs.typesafe.ai/introduction>.

`numpy` supports local calibration; `scipy` is used by the official scorer. Predictions are
appended as each sentence completes, so an interrupted run resumes by ID; the client retries
rate limits, 529 and timeouts with backoff. Each prediction also stores raw scores,
probabilities, confidence, returned model and usage in `_jev`. The default model is
`jev-1.13.0`. Resume checks the saved request configuration; historical files without
that configuration remain readable but require a new output path for new runs.
The original `run_st1.py --restart` explicitly discards an existing run; use a new
`--out` directory for a fresh unified run. Incomplete batch runs exit nonzero.

## Task 1 few-shot examples

Examples come only from the **train** split of the same corpus. They are frozen per corpus: the
same records are used for every request in a run, recorded by ID in the run metadata, and never
re-picked because a prediction came out badly. `--shots N` sets how many (`0` is the zero-shot control). The unified runner
defaults to 0; the original Task 1 runners retain their default of 3.

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
- The Task 2 baseline uses a training vocabulary to propose explicit spans, so unseen terms and implicit
  aspects/opinions cannot be extracted. Its transferred Task 1 calibration is a starting
  point, not a Task 2-tuned model. Subtask 3 is not implemented.
