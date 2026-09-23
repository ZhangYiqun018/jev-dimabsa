# 0006 — Task 2 (DimASTE): training lexicon + Jev pair decisions

## Scope and frozen setup

Task 2 receives **only review text** and predicts `(Aspect, Opinion, VA)` triplets.
Eight corpora are available; neither finance corpus belongs to this task.
The unchanged official scorer reports continuous F1 (**cF1**, higher is better).
Aspect and opinion must both match before VA closeness receives credit.

- Task 1 checkpoint before development: local commit `41b3c89`.
- Model: `jev-1.13.0`, explicitly requested; returned versions recorded in summaries.
- Data/scorer: [content-addressed snapshot](../data-version.json), SHA-256 `980b1f8e015afafb8d59f272445616f3960140095d42ad44e6a209ebcc4ca6c5`.
  The 55 recorded dataset/scorer files match upstream commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1` byte-for-byte.
  This revision was identified from the official archive; the local vendor directory itself is not a separate Git checkout.
- No model training, threshold sweep, extraction tuning, or Task 2 calibration fitting.
- All settings fixed before full dev/test runs; these splits are for reporting, not selecting configurations.

## Official baseline

Source: [organizers' final overview](https://aclanthology.org/2026.semeval-1.452/), §5.2 and **Table 7**, Track A Subtask 2 **test**.
Kimi-K2 Thinking uses one-shot prompting; Qwen3-14B uses per-corpus QLoRA.
The transcribed numbers are in [official_baselines.json](../reports/st2_baseline_20260923/official_baselines.json).
Do not compare dev results against these test baselines or confuse cF1 with Task 1 RMSE.

## Baseline method

1. Build separate aspect and opinion vocabularies from each corpus's training annotations.
   Exclude training texts overlapping dev/test using the existing text normalization convention.
   Ignore `NULL`: this initial extractor covers explicit spans.
2. Find literal vocabulary spans in the target text. Preserve its original surface strings.
   Use word boundaries for space-delimited scripts and substring matching for Chinese/Japanese;
   prefer the longest vocabulary match at a shared start position.
3. Form distinct aspect–opinion pairs, excluding identical strings. Jev receives the review
   text and narrow questions, **never the input file's gold Triplet, VA or aspect list**.
4. For each candidate pair, ask a Noul relation question and the existing two VA Score questions.
   Batch at most 16 pairs per request; accept pairs with Noul probability ≥ 0.5.
5. Apply the frozen **Task 1 zero-shot shrink parameters** for that corpus to the pair's V/A,
   clip to [1,9], and serialize two decimals. This is a **transferred calibration baseline**:
   the parameters were fitted on aspect-level scores, not pair-conditioned Task 2 scores.
   There are no Task 2 few-shot demonstrations.

This starting point exploits Task 1's scale-calibration experience without introducing a
second generative model. Its main limitation is extraction recall: unseen training-vocabulary
terms cannot be recovered, and literal/longest-span matching can miss annotation boundaries.
It can also produce false pairs when common aspect/opinion terms merely co-occur. No claim
is made that the inherited calibrator is optimal for pair-conditioned predictions.
Historical Task 1 test inspection also means this is not a completely untouched research holdout.

## Implementation and reproduction

`runners/execution.py` owns shared concurrent execution, JSONL resume and per-row usage.
`jev/triplets.py` owns Task 2 candidate generation and questions. `runners/run.py` selects the
task, iterates corpora, writes summaries and calls the unchanged official scorer.
The existing Task 1 runners and calibration commands remain supported.

```bash
.venv/bin/python runners/run.py --task 2 --split dev \
  --out reports/st2_baseline_reproduction --concurrency 5
.venv/bin/python runners/run.py --task 2 --split test \
  --out reports/st2_baseline_reproduction --concurrency 5

# Same entry point, Task 1 raw inference:
.venv/bin/python runners/run.py --task 1 --corpus eng_restaurant --split dev \
  --shots 0 --out reports/st1_reproduction
```

Re-run the same command to resume missing records. Failed corpora are not scored.
Raw predictions and request metadata stay in ignored `cache/` directories because they
contain dataset text; summaries contain only metrics and accounting.

## Validation and execution notes

- The three pre-existing focused tests pass after extracting the shared executor.
- One new test covers text-only inference, word-boundary matching, held-out text exclusion,
  pair rejection, transferred calibration, multi-batch usage and empty predictions.
- Both tasks completed three-record real API runs through the unified command and official scorer.
  These are plumbing checks, not reported benchmark results.
- Early full-run requests returned HTTP 403 / 1010. An explicit identifying User-Agent
  (`jev-dimabsa/0.1 (research baseline)`) restored service; successful rows were resumed.
- HTTP 520 was observed on a small number of requests and added to the existing transient
  retry statuses. Completed records were retained; only missing records were resumed.
- Execution used concurrency 5 for dev and initially 5, then 10 on resume for test;
  concurrency does not change the frozen prompts or extraction parameters.
- Known usage excludes unreported charges from failed requests or partially completed records.

## Results — full official dev/test splits

Unchanged official scorer; cF1, higher is better. The final row is the unweighted
arithmetic mean across eight corpora, not an official overall competition rank.

| Corpus | Jev dev | Jev test | Kimi-K2 Thinking test | Qwen3-14B test |
|---|---:|---:|---:|---:|
| eng_restaurant | 0.4708 | 0.3845 | 0.4920 | 0.4483 |
| eng_laptop | 0.3414 | 0.2983 | 0.4424 | 0.3827 |
| jpn_hotel | 0.3818 | 0.3561 | 0.3464 | 0.1622 |
| rus_restaurant | 0.3128 | 0.2719 | 0.4242 | 0.3341 |
| tat_restaurant | 0.3183 | 0.2718 | 0.3577 | 0.2020 |
| ukr_restaurant | 0.3041 | 0.2854 | 0.4220 | 0.3099 |
| zho_restaurant | 0.1893 | 0.1804 | 0.3529 | 0.2509 |
| zho_laptop | 0.1362 | 0.1684 | 0.2494 | 0.2099 |
| **Macro mean** | **0.3068** | **0.2771** | **0.3859** | **0.2875** |

## Usage and artifacts

- Dev: 1,344 texts, 10,833,670 input / 853,061 output tokens; estimated input cost $0.4550.
- Test: 6,690 texts, 43,997,780 input / 3,431,350 output tokens; estimated input cost $1.8479.
- Assumed input price: $0.042/M tokens, as in the preceding experiments. These are known-token estimates, not billed totals; failed/partial request charges may be missing.
- Returned model: `jev-1.13.0`; rubric fingerprint: `9877fbd18a68`.
- Scores: [dev summary](../reports/st2_baseline_20260923/task2_dev_summary.json), [test summary](../reports/st2_baseline_20260923/task2_test_summary.json).
- Aggregate and usage: [run_summary.json](../reports/st2_baseline_20260923/run_summary.json).
- This accounting excludes smoke runs and the pre-existing Task 1 calibration expense. No Task 2 GPU training was performed.
