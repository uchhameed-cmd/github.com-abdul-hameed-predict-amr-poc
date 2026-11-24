# ASK-T2D | Executed CDC NHANES pilot (24 November 2026)

**Status:** Real public-use CDC NHANES XPT data were downloaded and analyzed in GitHub Actions; the committed JSON is the definitive computational output. This is a retrospective, cross-sectional *methodological* pilot, **not** medical diagnosis, a clinical study, or proof of benefit to patients.

## Locked data and outcome
- Development NHANES 2015–2016: 4,076 complete cases after restricting to adults >=20 who self-report no diagnosed diabetes (DIQ010=2), with HbA1c and both optional body measures observed. Of these, 118 had HbA1c >=6.5% (2.895%).
- Temporal test NHANES 2017–March 2020 pre-pandemic: 6,280 complete cases, including 214 HbA1c >=6.5% (3.408%).
- HbA1c supplies only a laboratory-reference outcome, not a predictor or a confirmed clinical diagnosis.
- Candidate optional tests BMI or waist; the development-only policy selected **waist circumference**.

## First executed benchmark
At half the cohort measured (3,140 participants), all policies used the same *number* of optional measurements, although no real monetary costs were measured.

| Policy | Optional measurement rate | Brier loss (lower better) | AUROC |
| --- | ---: | ---: | ---: |
| Age/sex baseline | 0% | 0.03274065 | 0.64437 |
| Highest predicted risk | 50% | 0.03248799 | 0.69003 |
| One random allocation | 50% | 0.03239020 | 0.70800 |
| Conditional variance proxy (ASK-T2D) | 50% | 0.03236725 | 0.70243 |
| Universal measurement | 100% | 0.03233700 | 0.72720 |

**Paired Brier difference, ASK-T2D minus the prespecified random allocation:** -0.00002295 (participant-level bootstrap 95% CI [-0.00022651, +0.00016690]). Interval crosses zero. **No reliable improvement over random acquisition established.** Universal measurement achieved the best observed Brier score, but required twice as many optional acquisitions.

## What the result does and does not establish
- Confirms end-to-end data merging, reproducible modelling and a later-period locked comparison **can be performed**.
- Does **not** establish that variance-based acquisition outperforms matched-budget random allocation.
- Does not test direct expected Brier-risk reduction, validated conditional distributions, clinical benefit, independent hospital populations, decision cost, or survey-weighted population inference.
- Complete-case restriction may introduce selection bias; results apply to these analytic cohorts only.
- A second 200-randomisation sensitivity run is under development; do not cite its metrics until independently verified.

## Reproduction
```bash
pip install pandas numpy scipy scikit-learn requests
python ASK-T2D/pilot_v2.py
```
The runner downloads CDC XPT files directly; raw records are not committed. Output is saved as `ASK-T2D/results/pilot_v2_results.json`.

[Computed JSON](results/pilot_v2_results.json) | [Analysis source](pilot_v2.py) | [GitHub Actions](../../actions/workflows/ask-t2d-nhanes.yml)

Research supported by an exploratory proposal *under preparation* for the Biswas Family Foundation Fast Grants Cycle 2; no grant has been awarded or approved.
