# ASK-T2D — Biswas Fast Grants 2026 Cycle 2

**Competitive draft, not submitted.** Requested $25,000 / 12 months. Updated 24 Nov 2026.

## Project title
ASK-T2D: When Does AI-Guided Measurement Add Value in Metabolic Screening?

## One-sentence summary
A real NHANES pilot found no clear benefit over random test selection. We will identify when AI-guided measurement helps, when it fails, and what it costs.

## The question / idea
- Can AI predict when one added body measurement improves screening reliability?
- Our 6,280-record NHANES pilot did not beat random selection convincingly.
- When does targeted collection beat measuring everyone?

## What you'll do
- Estimate expected Brier-risk reduction using nested development data.
- Compare random, risk, uncertainty, fixed and universal strategies.
- Test costs, missing data, calibration and a new locked holdout.
- Publish failures and code.

## Why now / why this team
- Executed NHANES pilot reveals a real null/uncertain result.
- PI has open AI research software; independent clinical/statistical review is budgeted.

## Budget
- Research analyst $15k; clinical/statistical review $3k; compute $1k; independent reproduction $1.5k; open release $1.5k; indirect $3k. Total $25k.

## Public outputs
- Open, versioned benchmark, code and protocol.
- A decision map of when targeted measurement helps or fails.

## Brief bio (PI sign-off required)
- Dr Abdul Hameed, Professor of Computer Science, UMT, Pakistan.
- AI modelling, evaluation and reproducible research.
- Q-TRUST: public code/results; manuscript under review.
- ASK-T2D: executed NHANES pilot and public code.

## Evidence-backed scientific premise
An actual NHANES pilot used 4,076 complete development records (2015–2016; 118 HbA1c-range positives) and 6,280 later-period records (2017–Mar 2020; 214 positives). The variance-proxy acquisition policy chose waist circumference for exactly half the later-period cohort. Its Brier loss was 0.032367 vs 0.032390 for **one** matched-count random allocation; bootstrap interval for the difference [-0.00022651, +0.00016690] contains zero. Universal measurement gave 0.032337 with twice the acquisition count. The results do **not** establish advantage over random allocation or patient benefit.

The funded research will compare a directly estimated expected reduction in Brier loss with random (>=200 seeds), highest risk, uncertainty, simple fixed policy, and universal acquisition, with cost and prediction accuracy separately reported. Development-only nested cross-fitting will govern policy fitting.

**Validation correction:** 2017–March 2020 NHANES has already informed the revised proposal; it is *exploratory* and cannot now be described as an untouched confirmatory set. Any new confirmation must use a previously unused, harmonized cycle (for example 2013–14 if compatible) under a preregistered analysis. Otherwise results remain exploratory.

**Critical limitations:** single HbA1c lab-range outcome, not diagnosis; complete cases; no survey-weighted population inference, clinical recruitment, patient-facing product or confirmed cost saving.

## Source and reviewer audit
- [Real executed NHANES metrics](results/pilot_v2_results.json)
- [Public pilot interpretation](PRELIMINARY_RESULTS.md)
- [Reproducible code](pilot_v2.py)
- [Experimental 200-seed code](pilot_v3.py): **no verified result yet**
- [Foundation instructions](https://www.biswasfamilyfoundation.org/science/fast-grants)

## Outstanding before submission
Institutional gift eligibility (foreign-equivalency determination or eligible US charitable intermediary), named clinical/biostatistical reviewer with approval, institutional budget and PI sign-off. The applicant may select *unsure* for gift eligibility where appropriate; it must never be falsely stated as confirmed. If funded, one-page report in month 6 and acknowledgment of the Foundation in research publications are required.
