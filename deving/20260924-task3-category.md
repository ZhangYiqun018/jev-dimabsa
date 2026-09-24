# Task 3 category stage: dev iterations (2026-09-24)

Plan: add a category to every frozen Task 2 pair, develop on dev, and run test once only
if the projected test macro (Task 2 test cF1 × dev retention, per corpus) reaches 44.3
(TeamLasse). Combiner weights are always fitted on a train sample (leave-one-out counts,
retrieval without the record's own text); dev only compares designs. Dev pairs are the
Task 2 5-fold out-of-fold predictions.

Checks before any Jev call: the dev gold scored as a prediction gives Task 3 cF1 1.0 on all
corpora (IDs aligned by line); Task 2 dev pairs with gold categories reproduce Task 2 dev
cF1 (jpn_hotel −0.30: a few gold pairs have two categories). A probe confirmed one Choice
over all 121 eng_laptop categories returns a full distribution (~2.8k input tokens per
question).

## Iteration 1: bare labels ("laptop — quality")

| Variant | Dev macro cF1 | Projected test |
|---|---:|---:|
| lookup (aspect) | 41.22 | 37.62 |
| Jev | 48.33 | 44.16 |
| Jev + aspect lookup | 48.52 | 44.33 |
| Jev + aspect + opinion lookup | 48.84 | 44.63 |

Matched-pair accuracy was lowest on eng_laptop (0.56) and jpn_hotel (0.77). Errors were
mostly attributes: entity accuracy 0.91 / 0.94 / 0.95 on eng_laptop / jpn_hotel /
zho_laptop against attribute accuracy 0.62 / 0.81 / 0.87 (e.g. portability → design
features, quality → general). A separate entity/attribute question would not target this,
so the opinion lookup (no API calls) and attribute definitions were tried instead.
Train + dev: 18.0M + ~5.8M input tokens.

## Iteration 2: one-line attribute definitions in the Choice options

| Variant | Dev macro cF1 | Projected test |
|---|---:|---:|
| lookup (aspect + opinion) | 43.02 | 39.30 |
| Jev | 49.00 | 44.79 |
| Jev + aspect lookup | 48.79 | 44.59 |
| **Jev + aspect + opinion lookup (frozen)** | **48.83** | **44.63** |

The Jev variants are within 0.2 points; rus/tat/ukr dev have 48 reviews each, so one or two
pairs move a corpus by 1–2%. The combined variant was kept as planned; it is best on the
three weakest corpora, which have the largest dev sets. Train + dev: 22.8M + ~5.8M input
tokens.

## Test (run once)

Macro cF1 **43.62** (projected 44.63), retention 84.4% of Task 2 test (dev 86.5%). Log
[0008](../logs/0008-st3-category-choice.md).
