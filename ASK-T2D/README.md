# ASK-T2D | Evidence-led AI measurement acquisition for metabolic screening

**Verified status (24 Nov 2026):** A real CDC NHANES exploratory analysis **has been executed**. The variance-proxy acquisition policy showed no statistically clear Brier improvement over one prespecified matched-count random allocation. **Do not claim clinical validation or proven cost savings.**

## Read the evidence first
- [Executed NHANES v2 JSON](results/pilot_v2_results.json)
- [Method and interpretation](PRELIMINARY_RESULTS.md)
- [Reproduction script](pilot_v2.py)
- [Extended 200-random-allocation analysis script](pilot_v3.py) (results not verified as of this update)
- [Evidence-led grant application draft](BISWAS_APPLICATION.md)

## Measured exploratory pilot
Development: NHANES 2015–2016, 4,076 complete adult records, 118 HbA1c-range positives. Later-period exploratory test: NHANES 2017–March 2020, 6,280 complete adult records, 214 positives. Single HbA1c>=6.5% defines a *laboratory-range outcome*, not confirmed diabetes. The body-measure selection fixed on development data was waist circumference.

| Policy | Body-measure acquisition | Brier loss (lower better) |
|---|---:|---:|
| Age/sex only | 0% | 0.03274065 |
| Random allocation | 50% | 0.03239020 |
| Variance-proxy ASK-T2D | 50% | 0.03236725 |
| Universal waist acquisition | 100% | 0.03233700 |

Paired ASK-T2D minus random Brier difference = **-0.00002295**, participant bootstrap 95% CI **[-0.00022651, +0.00016690]**. It crosses zero. This does not demonstrate a meaningful advantage over random allocation.

## Important scientific guardrails
- The 2017–March 2020 dataset has **already informed hypothesis formation**, so it cannot serve as an untouched confirmatory set for the revised research. The funded study will preregister a distinct historical-cycle holdout (subject to variable harmonization) or describe validation as exploratory if no untouched sample is secured.
- The current policy ranks conditional **predictive variance**, which is *not* direct expected Brier-risk gain. A direct benefit estimator is proposed, not completed.
- Complete-case analysis, no survey-weighted inference, no clinical follow-up, no external Pakistan cohort and no validated financial test costs.
- No clinical claims, human-subject recruitment, or patient-facing decision support.
- The v3 200-allocation run must not be cited until its data files and result commit are verified.

## Reproduce
```bash
pip install pandas numpy scipy scikit-learn requests
python ASK-T2D/pilot_v2.py
```

This repository is a public method and transparency artifact supporting an **unsubmitted** proposal to the Biswas Family Foundation, not proof of a funded award.
