# Jev on DimABSA — Subtask 1 baseline

A **TypeSafe Jev** (System One) baseline for [DimABSA](https://github.com/DimABSA/DimABSA2026),
the dimensional aspect-based sentiment analysis task from SemEval-2026 Task 3.

DimABSA replaces categorical polarity with continuous **valence–arousal (VA) scores on a
1.00–9.00 scale**. Jev's `Score` primitive returns a continuous probability-weighted position
on a described scale, so the task maps onto it without any text generation or output parsing.

No GPU and no fine-tuning. One API key and about an hour.

## Result — Subtask 1 (DimASR), official test split

RMSE_VA, lower is better, official scorer with `--do_norm` **off**. Micro average is weighted
by gold entry count (N = 16,186).

| System | RMSE_VA |
|---|---|
| GPT-5 mini, zero-shot | 2.7439 |
| **Jev, zero-shot** | **2.4708** |
| Kimi-K2, zero-shot | 2.3849 |
| **Jev, 3-shot** | **2.1721** |
| Qwen3-14B, QLoRA fine-tuned | 2.1841 |
| GPT-5 mini, one-shot | 2.1552 |
| Kimi-K2, one-shot | 1.8873 |

Baseline figures are from the dataset paper, [arXiv:2601.23022](https://arxiv.org/abs/2601.23022),
Table 3. Three in-context examples move Jev **−0.2987** and take it from 3/10 to 7/10 corpora
ahead of Kimi-K2 zero-shot. It remains **+0.2847** behind Kimi-K2 one-shot.

Per-corpus numbers and run configuration: [`logs/`](logs/).

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

`scipy` is needed only by the official scorer, for its Pearson correlation. Predictions are
appended as each sentence completes, so an interrupted run resumes by ID; the client retries
rate limits and 5xx with backoff.

## Few-shot examples

Examples come only from the **train** split of the same corpus — the official rule is *"the
first k samples in the training set"*. They are frozen per corpus: the same records are used
for every request in a run, recorded by ID in the run metadata, and never re-picked because a
prediction came out badly.

`tools/leakage_audit.py` found that ID sets are disjoint across splits but **sentence text is
not** — `jpn_hotel` reuses 23 sentences between train and test, `tat_restaurant` 2, and
`eng_restaurant` 1 between train and dev. Example selection therefore drops any train record
whose normalised text or ID also occurs in the split being evaluated. At `--shots 3` this
drops nothing; at larger `k` it matters.

## Known limits

- **Jev is not deterministic.** Repeated calls on identical input differ by roughly 0.04 per
  dimension. Differences below that are not evidence of anything.
- Arousal stays under-predicted even with examples (`PCC_A` ≈ 0.49 against `PCC_V` ≈ 0.89).
- Only Subtask 1 is implemented.
