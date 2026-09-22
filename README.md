# Jev on DimABSA — Subtask 1 baseline

A **[TypeSafe Jev](https://typesafe.ai)** (System One) baseline for
[DimABSA](https://github.com/DimABSA/DimABSA2026), the dimensional aspect-based sentiment
analysis task from SemEval-2026 Task 3.

DimABSA replaces categorical polarity with continuous **valence–arousal (VA) scores on a
1.00–9.00 scale**. Jev's [`Score`](https://docs.typesafe.ai/primitives/score) primitive returns
a continuous probability-weighted position on a described scale, so the task maps onto it
without any text generation or output parsing.

No GPU and no fine-tuning. One API key and about an hour.

## Result — Subtask 1 (DimASR), official test split

Official scorer, `--do_norm` **off**. Micro average, weighted by gold entry count (N = 16,186),
best first.

`RMSE_VA` is the root mean squared error of the predicted valence–arousal pair against gold,
`√(mean(Δvalence² + Δarousal²))`, in the same units as the 1–9 scale. **Lower is better; 0 is
perfect.** It is not a per-dimension error — both dimensions are pooled under one root.

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

Baseline figures are from the dataset paper, [arXiv:2601.23022](https://arxiv.org/abs/2601.23022),
Table 3. Three in-context examples move Jev **−0.2987** and take it from 3/10 to 7/10 corpora
ahead of Kimi-K2 zero-shot.

The two Jev rows shown are the official protocol — the first 3 records of the training set — and
the best arm found, which picks examples to span the 1–9 scale and uses 9 of them. The remaining
gain is in the example count: **−0.0319** from 3 to 5, of which one corpus accounts for all but
0.0027, then a further **−0.0573** from 5 to 9, which is spread across the corpora. At 9 examples
Jev passes GPT-5 mini one-shot and sits 0.1863 behind Kimi-K2 one-shot. How the examples are
chosen barely matters; how many there are does.

Per-corpus numbers, correlation metrics, the full sweep, and run configuration:
[`logs/`](logs/).

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
logs/               one file per experiment
```

## Running it

```bash
uv venv .venv && uv pip install --python .venv/bin/python scipy
export TYPESAFE_API_KEY=...          # the client also reads ~/.zshrc

# one corpus
.venv/bin/python runners/run_st1.py \
  --data vendor/DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_test_task1.jsonl \
  --out reports/pred.jsonl --shots 3 --concurrency 10

# every corpus, scored against the official script
.venv/bin/python runners/run_all_st1.py --split test --shots 3 --concurrency 10
```

An API key comes from the [TypeSafe console](https://console.typesafe.ai/); the model is served
at `https://api.typesafe.ai/v1/systemone`. Primitive and request-shape docs:
<https://docs.typesafe.ai/introduction>.

`scipy` is needed only by the official scorer, for its Pearson correlation. Predictions are
appended as each sentence completes, so an interrupted run resumes by ID; the client retries
rate limits and 5xx with backoff.

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
  dimension. Differences below that are not evidence of anything.
- Arousal stays under-predicted even with examples. `PCC_A` ≈ 0.49 against `PCC_V` ≈ 0.89
  (Pearson correlation per dimension; higher is better, 1 is perfect) — Jev orders valence well
  and arousal badly. Per-corpus values are in the logs.
- Only Subtask 1 is implemented.
