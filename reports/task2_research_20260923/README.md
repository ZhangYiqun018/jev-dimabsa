# Task 2 improvement research — intermediate results (2026-09-23)

Backup of a multi-agent offline research run on the Task 2 BIO r3 caches. No Jev
requests were made and no repo code was changed to produce any of it.

Everything is under the ignored `cache/` directory: the agents quote short span
fragments from reviews, so these files are dataset-derived and must not be
committed or shared. The live working copy the run reads from is `/tmp/t2/`;
this copy exists because `/tmp` can be cleared.

| File | Content |
|---|---|
| `cache/diag_*.json` | Five offline diagnostics: post-processing simulations, train annotation conventions, Jev representation/cost, experiment history and noise, lexicon/BIO complementarity |
| `cache/run2_error-anatomy.json` | Sixth diagnostic: per-error anatomy of BIO r3 on full dev |
| `cache/run2_*.json` (four others) | Experiment proposals per angle: decode-first, representation, linking-hybrid, protocol-roadmap |
| `cache/candidates.json` | Nine merged candidate experiments plus what was dropped at merge |
| `cache/verdicts/C*_{quant,skeptic}.json` | Independent verification verdicts per candidate: quantitative re-check and methodology skeptic |
| `cache/resume_args.json` | Which candidates and verdicts were already finished, for resuming the run |

Headline measured numbers (all optimistic full-dev estimates, dev has been
reused repeatedly): span boundaries cause 1,167 of 1,451 missed pairs; only 21 of
1,555 false positives are a wrong link between two correct spans; NULL-aspect
pairs are 175 false positives with zero true positives in the seven corpora whose
dev gold has no NULL; train-derived edge-affix rules move jpn_hotel cF1 from
17.85% to about 30%. The ranked plan is written up separately once the run
finishes.
