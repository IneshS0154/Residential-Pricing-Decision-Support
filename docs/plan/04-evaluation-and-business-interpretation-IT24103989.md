# Plan: Evaluation & Business Interpretation

**Member:** IT24103989 — Gunarathna H.M.S.S
**Proposal phases covered:** Phase 5 (Evaluation), Phase 6 (Decision and Delivery)
**Runs:** Fourth/last, starting once trained models (including the quantile models) exist —
though the leakage-verification task can and should start as soon as IT24102584's pipeline is
in a reviewable state, in parallel with modelling.

**Depends on:** trained models and decomposition (IT24102616), the preprocessing pipeline
(IT24102584), EDA segment findings (IT24103124).
**Blocks:** nothing downstream — this is the final synthesis step, feeding the report/recommendation.

## Task checklist (in order)

1. **Verify leakage prevention independently** — check IT24102584's pipeline is fit inside CV
   training folds only, with nothing fit on the full dataset pre-split. Do this yourself, don't
   just trust the pipeline owner's account — that's the point of the separation of duties (see
   `CLAUDE.md` and `docs/preprocessing_feature_log.md`). Do this as early as the pipeline is
   usable, not at the end.
2. **Evaluate point accuracy** — RMSE (transformed scale), MAE (currency), MAPE, reported with
   dispersion across CV folds, never from a single split.
3. **Evaluate interval calibration** — empirical coverage against the nominal level (e.g. 80%),
   reported *together with* mean interval width (never one without the other — see
   `CLAUDE.md`). If coverage is materially off from nominal, that's a finding, not a footnote.
4. **Analyze segment error** — break error down by price band, neighborhood, and quality level.
   Cross-reference with IT24103124's segment-structure EDA findings — this is where the
   secondary lens (Property Segment Analysis) earns its place.
5. **Run temporal validation** — train on earlier `YrSold`, test on later, to check performance
   survives a shift in market conditions. If this disagrees with the primary CV result, report
   the disagreement as a finding — don't resolve it in favor of the more flattering number.
6. **Select the routing threshold** — examine the trade-off between share of the portfolio
   routed to `AUTO` and the pricing error realized on that share. Justify the chosen operating
   point with numbers, not intuition.
7. **Select the final model** — using validation results and business requirements (accuracy +
   interval honesty + segment robustness), not accuracy alone.
8. **Interpret results for the valuation team** — translate metrics into plain implications.
9. **Develop the recommendation** — a concrete operating rule: route X% of the portfolio to
   `AUTO`, keep the rest with human appraisers, expect Y pricing error on the automated share.
10. **Identify limitations** — dataset generalizability, market period, absence of market-context
    variables, interval-calibration dependence, limits of the decomposition, association vs.
    causation. The proposal's "Initial Limitations" section is the starting checklist — extend
    it with what you actually observed.
11. **Responsible AI** — state plainly that the output is decision support, not a decision-maker;
    a property should never be priced solely off the model without human review.
12. **Worked demonstration** — produce the full valuation report output (estimate + interval +
    decomposition + routing flag) for 3–4 representative properties, including at least one
    routed to `ESCALATE`.

## Deliverables

- Evaluation code and results in [`src/evaluation/`](../../src/evaluation/).
- Segment error breakdown, temporal validation result, routing threshold justification.
- Final model selection writeup.
- Limitations and responsible-use section.
- Worked demonstration on 3–4 properties.

## Reminder from CLAUDE.md

- Coverage and interval width are always reported together.
- Metrics need dispersion across folds, not a single number from a single split.
