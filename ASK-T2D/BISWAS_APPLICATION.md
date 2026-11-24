# Biswas Fast Grants 2026 Cycle 2 — ASK-T2D draft application

**Draft for investigator and institutional approval, not submitted.** Tier: $25,000; duration: 12 months. Updated 24 November 2026.

## Project title
ASK-T2D: When Is Another Measurement Worth Collecting for Diabetes-Range Screening?

## One-sentence summary
A real NHANES pilot found a tiny, uncertain advantage from AI-selected body measurements. We will test when selective screening helps, fails, or wastes resources.

## The question / idea
- Can AI select useful measurements at the same testing budget?
- Our NHANES pilot found only a tiny advantage over random selection.
- Does direct expected-loss modelling improve this?
- When is measuring everyone better?

## What we'll do
- Build on our published NHANES pilot.
- Compare equal-budget random, risk, uncertainty and AI policies.
- Test direct Brier-gain, missingness and calibration across periods.
- Publish code, uncertainty intervals and failures.

## Why now / why this team
- Our real NHANES pilot (6,280 test records) exposed an unresolved measurement-choice problem.
- PI offers open AI research code; clinical/statistical verification is budgeted.

## Budget
- Analyst $15k; clinical/statistical review $3k; compute $1k; independent rerun $1.5k; open methods/release $1.5k; university indirect $3k. Total $25k.

## Outputs
- Open NHANES benchmark, reproducible code and analysis protocol.
- Published positive, negative and failure results.

## Brief bio (PI approval pending)
- Dr Abdul Hameed, Professor of Computer Science, UMT, Pakistan.
- AI research and reproducible model assessment.
- Q-TRUST: public code/results; manuscript under review.
- ASK-T2D: executed NHANES pilot and public analysis code.

## Real prior-data experiment
- Development: 4,076 eligible complete records; 118 HbA1c >=6.5%.
- 2017–Mar2020 test: 6,280 eligible complete records; 214 HbA1c >=6.5%.
- The development-only rule selected waist circumference.
- Brier: baseline 0.03274065; random 50% 0.03239020; ASK-T2D 50% 0.03236725; universal 100% 0.03233700.
- Brier difference ASK-T2D minus one prespecified random allocation -0.00002295; bootstrap 95% CI [-0.00022651, +0.00016690] includes zero.
- **No statistically clear benefit over random has been demonstrated.** The next study is needed to evaluate direct expected-Brier gain and test when the value of selective measurement disappears.

These are retrospective cross-sectional exploratory model results, not clinical validation or diagnosis. The analysis uses complete cases, a single HbA1c threshold, no survey-weighted inference and no external Pakistani cohort.

## Primary proposed scientific endpoint
Temporal-test paired difference in overall Brier loss for a direct expected Brier-risk acquisition policy versus the mean of at least 200 random equal-budget allocations at 50% optional acquisition, with bootstrap and randomisation uncertainty, calibrated models, and locked evaluation procedures. Universal measurement is a higher-cost reference, not an equal-cost comparator.

## Required checks before submitting
- Confirm UMT's eligibility to receive a charitable gift (foreign equivalency or eligible US 501(c)(3) intermediary).
- Confirm a named clinical/biostatistical adviser, approved duties and fees.
- Obtain PI signoff on biography and selected research outputs.
- Confirm budget/institutional support and all field character limits in Airtable.
- One-page month-six update and Foundation acknowledgement in resulting publications are required.

## Scientific provenance
- [Actual NHANES results](results/pilot_v2_results.json)
- [Pilot source](pilot_v2.py)
- [Measured findings and interpretation](PRELIMINARY_RESULTS.md)
- [Foundation application instructions](https://www.biswasfamilyfoundation.org/science/fast-grants)
