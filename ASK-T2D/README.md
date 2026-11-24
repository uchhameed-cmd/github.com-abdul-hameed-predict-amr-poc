# ASK-T2D | Exploratory NHANES Diabetes Screening Benchmark

**Status (24 November 2026):** Public analysis code is hosted here. The actual NHANES experiment has **not** completed, and there are **no verified NHANES accuracy or calibration results** yet. This is research software, not a clinical product.

## Purpose
Assess whether adding BMI or waist circumference to a baseline age-and-sex prediction model improves discrimination and calibration for a cross-sectional HbA1c >= 6.5% reference outcome, among NHANES adults aged 20+ reporting no prior diabetes diagnosis (DIQ010 == 2).

## Design
- Development: NHANES 2015–2016.
- Locked temporal evaluation: NHANES 2017–March 2020 pre-pandemic. These periods do not overlap.
- HbA1c (LBXGH) is only an *outcome reference*, never a model feature; it cannot independently diagnose a patient.
- Candidate measurements: BMI and waist circumference. Baseline: age, reported sex.
- A single global extra measurement is selected by validation Brier score, never by temporal test results.
- Report AUROC, average precision, Brier score, log loss and cohort counts.
- The model currently does **not** do person-specific acquisition or compare true measurement costs; these are future grant aims.

## Reproduce
Install Python 3.11+ and:
```bash
pip install -r ASK-T2D/requirements.txt
python ASK-T2D/nhanes_pilot.py
```
Data is downloaded from public CDC NHANES XPT endpoints into an ignored local data directory. Results are written to results/nhanes_results.json on successful execution.

Alternatively open **Actions > ASK-T2D NHANES (manual) > Run workflow** and download the generated artifact if the run succeeds.

## Integrity, limitations and authorship
This exploratory benchmark does not use NHANES survey weights for population inference, has not been externally clinically validated, and cannot be used in patient care. Do not present unexecuted analysis as preliminary diagnostic performance. This project is being prepared as open research software supporting the proposed Biswas Family Foundation Fast Grants application. Institutional affiliation and investigator details must be approved by the investigator before they are represented as an institutional endorsement.
