# Development — direct aspect-conditioned SE on trial

> **Archived (2026-09-23 cleanup):** this experiment's code was moved verbatim to [`archive/`](archive/README.md). It is unmaintained and not runnable in the current layout; commands below record how it was run. The `reports/` summaries are the record.

## Protocol

One frozen trial-only probe of direct aspect → opinion extraction. Same 24 trial
reviews (6 per corpus) as the earlier comparisons; no dev/test inference, VA,
calibration, candidate screening, separate relation filter, or threshold selection.
Model `jev-1.13.0`; dataset `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.

1. Select the next explicit aspect start, then its end conditioned on the start.
2. Given this aspect, select matching opinion start/end pairs until `none` or token
   exhaustion. Restart from token zero for each aspect, allowing opinions before
   the target and shared opinions across targets.
3. Continue with the next aspect. Finally run an opinion pass for `NULL` aspect.
4. Copy spans from original offsets and deduplicate surface pairs. Every extracted
   pair is an output; no learned confidence or probability-product cutoff.

Use the existing tokenizer and local endpoint context (radius 2), explicit history,
and monotonically advancing `end+1` cursors. Thus overlapping spans within one pass
and repair of skipped/merged spans remain unsupported. There is no eight-span cap.
The probe supports at most 254 query tokens (plus `none` gives 255 options), explicitly
errors beyond that rather than truncating. The selected trial reviews have at most
27 tokens, so this limit does not remove any target records.

Three same-corpus BM25 train reviews are retrieved per query, using the same
retriever as the combined experiment. Normalized trial/dev text is excluded from
the training pool. Each SE step receives corresponding examples: start decisions
with prior history/cursor and stop examples; end decisions with a known start and
the gold end. For conditioned opinions, training examples name a particular aspect;
the NULL pass gets NULL examples (which may all demonstrate stopping). Explicit
target examples choose the train aspect with greatest lexical overlap with the
query's predicted aspect, ties by occurrence order.

Training annotations contain surface strings, not occurrence offsets. Examples use
the first exact, token-aligned occurrence; labels that cannot be aligned are omitted.
Some end steps can have fewer than three usable examples. This is not a solution to
training occurrence ambiguity. NULL opinions remain unsupported and count as misses.
The raw query passed to the extractor contains only ID and Text.

## Run and verification

```bash
python3 tools/probe_conditioned_se.py --per-corpus 6 \
  --out reports/conditioned_se_20260923 --max-input-tokens 1500000 --concurrency 3
```

Responses are cached per request for resume. The cumulative 1.5M reported-input
token limit is about $0.063 before in-flight requests, within the remaining margin
of the previous experiment's $0.25 budget. Raw text and examples remain in ignored
`cache/`; shareable summaries contain only IDs, metrics and usage.

Two focused tests passed: multi-target cursor reset / shared opinions / NULL and
query-gold exclusion; and exact train-span alignment with stage-specific start,
end and stop demonstrations. No additional preflight or verification pipeline.

## Metric interpretation

Use the same case-insensitive exact surface pair F1 as previous probes, without VA;
not official cF1. Exported `probability=1` is only a compatibility marker for an
emitted pair, not model confidence. There is no score threshold to optimize.

The shared evaluator's `candidate_pair_recall` equals final recall for this method
because only directly extracted pairs exist. Its `pair_rejected` category means
both surfaces appeared somewhere but their conditioned pair was not extracted;
it does not imply an additional filter. Compare old BIO/SE at their frozen 0.65
threshold on exactly the same completed records. This is a joint change of
conditioning and few-shot format, not an isolated ablation of either component.

## Results

All 24 records completed, with returned model `jev-1.13.0` and source fingerprint
`1699b1f9db02`. No prompt or threshold adjustments were made after seeing results.

| Trial method | Macro pair F1 | Micro precision | Micro recall | Micro pair F1 |
|---|---:|---:|---:|---:|
| BIO r3, historical | **0.6121** | 0.6471 | 0.5641 | 0.6027 |
| Independent SE r3 + pair filter, historical | 0.4798 | 0.5484 | 0.4359 | 0.4857 |
| Full-span combined pipeline, historical | 0.3591 | 0.2885 | 0.3846 | 0.3297 |
| **Direct conditioned SE + retrieved examples** | **0.2784** | **0.2683** | **0.2821** | **0.2750** |

The new run emits 41 unique pairs: **11 TP, 30 FP, 28 FN** against 39 gold pairs.
Two gold pairs have unsupported NULL opinions; they remain in the denominator.
Eleven predictions have NULL aspect, of which only one is correct (10 false positives).

| Corpus | Direct conditioned SE pair F1 |
|---|---:|
| eng_restaurant | 0.0909 |
| zho_restaurant | 0.5000 |
| zho_laptop | 0.2500 |
| rus_restaurant | 0.2727 |

This trial subset includes English, Chinese and Russian only, not all six languages.

## Error interpretation

- Explicit aspect recall falls from the old SE's 21/25 (84%) to **16/25 (64%)**.
  Conditioned opinions cannot recover a target that was never correctly extracted.
- Explicit opinion recall falls from 23/36 (63.9%) to **20/36 (55.6%)**. Conditioning
  and the new examples did not improve exact opinion coverage in this run.
- Of 28 missed pairs: 7 lack only an exact aspect surface, 14 lack only an opinion,
  4 lack both, and 3 have both surfaces somewhere but no corresponding emitted link.
- NULL target handling produces 10 of the 30 false positives. Removing them alone
  would not resolve the aspect/opinion omissions; no post-hoc NULL filter was added.
- Serial SE still commits to individual boundaries and cannot revisit an early error.
  The direct pipeline avoids the previous candidate cross-product but does not make
  boundary decisions or stopping inherently more reliable.

This changes both the decomposition and the examples relative to historical SE.
The result does not isolate whether conditioning, demonstration construction, prompt
wording or their interaction caused the drop. It is evidence against this concrete
implementation, not proof that all conditional SE or few-shot extraction fails.

## Cost and decision

216 completed API calls, 219 HTTP attempts; **299,521 reported input tokens**, about
**$0.012580**. Usage from unsuccessful attempts is not reported by the API. No budget
stop occurred. Stage totals:

| Stage | Input tokens |
|---|---:|
| Aspect start | 81,539 |
| Aspect end | 37,555 |
| Conditioned opinion start | 134,084 |
| Conditioned opinion end | 46,343 |

Historical independent SE used 148,680 tokens on these 24 reviews. The direct method
has fewer conceptual stages but roughly **2.01 times** the input cost, because each
target restarts opinion extraction and repeats retrieved examples. Together with the
previous combined trial/dev experiment ($0.151481), these new runs total approximately
$0.164061, still below that experiment's $0.25 budget envelope.

**Do not promote this implementation or run dev automatically.** The user requested
trial validation; this run provides no positive signal for expanding it. Preserve
BIO as the current development reference. No new corrective layers or extra API
experiments were added, and no GitHub push was made.

Artifacts: [summary and same-record references](../reports/conditioned_se_20260923/trial_conditioned_se_n6.json),
[extractor](archive/jev/conditioned_se.py), [trial runner](archive/tools/probe_conditioned_se.py),
[focused tests](archive/tests/test_conditioned_se.py). Raw examples/requests remain only in
the ignored cache directory. README main tables and mature `logs/` are unchanged.
