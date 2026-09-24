# Experiment log

One file per finalized experiment, numbered in the order they were recorded. Exploratory
development iterations are kept on the `dev` branch (`deving/`), not here.

**What goes in an entry:** the configuration that was run, the numbers that came out, what it
cost, and any observed fact that affects how the numbers should be read. Nothing else.

**What does not:** hypotheses, plans, interpretations, or anything not directly observed.
Those belong in the conversation or the next experiment, not here.

Every entry states the model version returned by the API and the rubric fingerprint. New
prediction files also retain model and raw response metadata; older runs rely on sidecars.

## Template

```markdown
# NNNN — short title

**Date** · **Model** · **Split** · **Metric**

## Configuration
- every knob that was set

## Result
per-corpus table, then the aggregate

## Cost
tokens and dollars, from `usage_total`

## Artifacts
where the predictions and summary live

## Observed
facts only — anything a reader needs to not misread the table
```

## Index

| # | Experiment | Test score | Cost |
|---|---|---|---|
| [0001](0001-st1-zero-shot.md) | Subtask 1, zero-shot | 2.4708 | $0.4588 |
| [0002](0002-st1-few-shot-n3.md) | Subtask 1, 3-shot calibration | 2.1721 | $0.6700 |
| [0003](0003-st1-few-shot-n3-stratified.md) | Subtask 1, 3-shot, valence-stratified examples | 2.1628 | $0.6752 |
| [0004](0004-st1-example-count-sweep.md) | Subtask 1, example-count sweep n = 3, 5, 9 | 2.1309 / 2.0736 | $0.7445 / $0.8936 |
| [0005](0005-st1-supervised-calibration.md) | Supervised shrink calibration, zero / 9-shot | 1.1395 / 1.1199 | $1.9588 |
| [0006](0006-st2-lexicon-pair-baseline.md) | Subtask 2, lexicon + pair decisions + transferred shrink | cF1 0.2771 (macro) | $2.3029 (dev + test) |
| [0007](0007-st2-lattice-reranker.md) | Subtask 2, BIO lattice candidates + example-conditioned checks + reranker | cF1 0.5168 (macro) | ≈ $5.44 test + ≈ $1.5 dev |
| [0008](0008-st3-category-choice.md) | Subtask 3, category Choice + train lookups on the Subtask 2 pairs | cF1 0.4362 (macro) | ≈ $1.27 test + ≈ $2.2 dev |

Task 1 scores are micro RMSE_VA (lower is better); Tasks 2 and 3 are macro cF1 (higher is better).
