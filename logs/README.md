# Experiment log

One file per experiment, numbered in the order they were run.

**What goes in an entry:** the configuration that was run, the numbers that came out, what it
cost, and any observed fact that affects how the numbers should be read. Nothing else.

**What does not:** hypotheses, plans, interpretations, or anything not directly observed.
Those belong in the conversation or the next experiment, not here.

Every entry states the model version returned by the API and the rubric fingerprint, because
both change results and neither is recoverable from the prediction files.

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

| # | Experiment | RMSE_VA (micro) | Cost |
|---|---|---|---|
| [0001](0001-st1-zero-shot.md) | Subtask 1, zero-shot | 2.4708 | $0.4588 |
| [0002](0002-st1-few-shot-n3.md) | Subtask 1, 3-shot calibration | 2.1721 | $0.6700 |
