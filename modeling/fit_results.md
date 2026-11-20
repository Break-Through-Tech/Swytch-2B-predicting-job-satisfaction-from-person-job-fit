# Person-job fit results (Gallup, Track A)

**Question:** does person-job fit add predictive value for job satisfaction beyond background characteristics?

**Answer:** yes. Adding seven fit features raises test ROC-AUC by about **+0.10 to +0.11** in all three algorithms, and every 95% confidence interval is well above zero. Most of that gain, however, comes from the answers the fit scores are built from, especially what the job gives. The matching formula itself adds only +0.005 beyond those answers. See [Limitations](#limitations).

## Contents

- [Files](#files)
- [Fit features](#fit-features)
- [Results](#results)
- [Which fit features matter](#which-fit-features-matter)
- [Deeper checks](#deeper-checks)
- [Limitations](#limitations)
- [How to reproduce](#how-to-reproduce)

## Files

| File | What it does |
|---|---|
| `notebooks/gallup_eda/engineer_fit_features.py` | Joins Q18, Q19, Q33H and Q34 from the Gallup file by `ENTITY_ID`, builds the fit features, and saves `output/gallup_fsu_fit_model.csv`. The output has 18,381 rows and 444 columns, the original columns are unchanged, and the file is tracked with Git LFS. |
| `notebooks/gallup_eda/test_engineer_fit_features.py` | Unit tests for every formula and for the merge checks. |
| `modeling/gallup_fit_models.py` | Trains logistic regression, random forest and Keras models: background only vs background + fit. Computes all metrics, the paired bootstrap confidence intervals and permutation importance. |
| `modeling/gallup_fit_scoring_comparison.py` | Track B: compares the Outline's two-sided fit formulas with one-sided "unmet need" scores and with the raw answers. |
| `modeling/output/fit/*.csv` | All metrics, improvements, importance scores and test-set predictions. |

## Fit features

Only four Gallup questions ask what a person **wants**, so fit can only be built from them:

- Q19: work-from-home days they'd like
- Q33H: interest in advancement
- Q32A–C: influence they believe they **should** have
- Q28: hours/pay preference

All other job questions, such as Q33A–J, Q25 and Q36, describe the job as it is.

### Where Q31/Q32 come from

The codebook labels them only "Compensation", "Working conditions" and "New technologies". The full wording is in the [American Job Quality Study report](https://www.jff.org/wp-content/uploads/2025/10/AmericanJobQualityReport_Final_10142025.pdf):

- **Q31** (p. 45): "How much influence do you, or your coworkers collectively, **have** on the following issues?"
- **Q32:** "How much influence **should** you … have?"

The report calls the difference the **"voice gap"** and notes that larger voice gaps go with lower job satisfaction (p. 29, citing Diaz-Linhart et al., 2024). The data agree: 8,135 people say they have *no* influence over pay, but only 1,390 say they *should* have none.

### The seven features

Each score runs from 0 to 1, where 1 is a perfect match.

| Feature | Formula | Scored for | Satisfied: big mismatch (≤ 0.5) | Satisfied: perfect match |
|---|---|---:|---:|---:|
| `remote_fit` | 1 − \|Q19 − Q18\| / 7. Q19 = 8 ("not possible") is NaN | 80.9% | 39.7% | 70.6% |
| `advancement_fit` | Outline rule: 1 = wants it and offered, or doesn't want it and not offered; 0 = the opposite pairs | 60.3% | 47.7% | 68.0% |
| `voice_fit_pay` | 1 − \|Q32A − Q31A\| / 3 | 91.0% | 49.9% | 74.2% |
| `voice_fit_conditions` | 1 − \|Q32B − Q31B\| / 3 | 92.4% | 40.4% | 73.8% |
| `voice_fit_tech` | 1 − \|Q32C − Q31C\| / 3 | 90.3% | 46.1% | 73.5% |
| `hours_fit` | 1 if Q28 = "same hours, same money", else 0 | 99.1% | 55.8% | 69.1% |
| `advancement_onet_fit` | 1 − \|interest in advancement − O\*NET Recognition\|, both scaled 0–1 | 85.8% | 58.3% | 74.6% (n = 437) |

Missing-value codes are never treated as numbers:

- −98 (no response)
- Q19 = 8 (work from home not possible)
- Q33H = 6 or 7 (not sure / not applicable)
- Q34 = 3 (not applicable)
- Q31/Q32 = 5 (not applicable)

Each fit feature gets a missing indicator in the models, so "could not be scored" is kept separate from a typical score.

### Gallup–O\*NET fit (Outline step 3C)

Only one valid pair exists: **interest in advancement (Q33H)** vs **O\*NET Recognition** (`onet_wv3val_mean`). O\*NET's Recognition work value describes jobs that offer advancement, leadership and prestige, on a 1–7 extent scale.

Pairs I did **not** build, and why:

- **Q33A (freedom) vs O\*NET Independence:** Q33A is the autonomy a person *experiences*, not the autonomy they want. As the Outline warns, this would compare two job descriptions, not a person with a job.
- **Q32 (influence wanted):** O\*NET has nothing about influence over pay or technology.
- **Q19 (work-from-home days wanted):** O\*NET has no direct telework item, and Q18 already gives each person's actual days.

O\*NET values are also averages for 25 broad occupation groups, because Gallup Q6 is that coarse.

## Results

All models share the same setup:

- the same 18,381 respondents and the same target (Q4 ≥ 7; 63.9% satisfied);
- the same 80/20 stratified split with seed 42 (14,704 training rows, 3,677 test rows);
- class-balanced training;
- preprocessing fit on training data only.

The background preprocessing is the same as in `gallup_baseline.py`, so the reference numbers match the team's earlier runs exactly: LR baseline 0.6271, LR + O\*NET 0.6296, RF + O\*NET 0.6144.

### Test ROC-AUC

| Feature set | Logistic regression | Random forest | Keras NN |
|---|---:|---:|---:|
| Background only (8 features) | 0.6271 | 0.6233 | 0.6282 |
| + remote & advancement fit (Outline core) | 0.6843 | 0.6781 | 0.6874 |
| + all Gallup fit (adds voice and hours) | 0.7299 | 0.7256 | 0.7299 |
| **+ all fit incl. O\*NET advancement** | **0.7308** | **0.7292** | **0.7374** |
| + 407 O\*NET features (reference) | 0.6296 | 0.6144 | 0.6195 |

### AUC improvement over background only

These are 95% paired bootstrap confidence intervals with 2,000 draws.

| Comparison | Logistic regression | Random forest | Keras NN |
|---|---:|---:|---:|
| Remote & advancement fit | +0.057 (+0.043, +0.071) | +0.055 (+0.040, +0.069) | +0.059 (+0.045, +0.074) |
| All Gallup fit | +0.103 (+0.086, +0.120) | +0.102 (+0.083, +0.120) | +0.102 (+0.084, +0.119) |
| **All fit** | **+0.104 (+0.087, +0.120)** | **+0.106 (+0.087, +0.124)** | **+0.109 (+0.092, +0.127)** |
| 407 O\*NET features | +0.003 (−0.003, +0.008) | −0.009 (−0.024, +0.005) | −0.009 (−0.019, +0.002) |

The 5-fold cross-validated logistic regression AUC tells the same story:

| Feature set | 5-fold CV AUC |
|---|---:|
| Background | 0.626 ± 0.005 |
| + Outline core fit | 0.676 ± 0.004 |
| + all fit | 0.724 ± 0.003 |
| + O\*NET | 0.628 ± 0.007 |

### Other metrics: background only vs all fit

Precision, recall, F1 and balanced accuracy use a 0.5 threshold. The confusion matrix is listed as TN, FP, FN, TP.

| Algorithm | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced acc. | Confusion matrix (TN, FP, FN, TP) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Logistic regression | background | 0.627 | 0.738 | 0.712 | 0.594 | 0.648 | 0.585 | 764, 564, 954, 1395 |
| Logistic regression | + fit | 0.731 | 0.813 | 0.771 | 0.697 | 0.732 | 0.666 | 843, 485, 712, 1637 |
| Random forest | background | 0.623 | 0.740 | 0.713 | 0.579 | 0.639 | 0.583 | 780, 548, 990, 1359 |
| Random forest | + fit | 0.729 | 0.816 | 0.767 | 0.720 | 0.743 | 0.667 | 815, 513, 658, 1691 |
| Keras NN | background | 0.628 | 0.741 | 0.713 | 0.620 | 0.664 | 0.590 | 742, 586, 892, 1457 |
| Keras NN | + fit | 0.737 | 0.822 | 0.781 | 0.690 | 0.733 | 0.674 | 874, 454, 728, 1621 |

All three fit models clear the team's own target of **F1 ≥ 0.70**, scoring 0.73–0.74. The background-only models reach 0.64–0.66.

### Model settings

- **Random forest:** tuned with the same search as `gallup_random_forest.py` (8 settings, 3-fold CV). The best setting for the fit models was 200 trees, max depth 10, and min leaf size 5–10.
- **Keras network:**
  - layers: Dense 64 → Dropout 0.3 → Dense 32 → Dropout 0.3 → sigmoid, with L2 regularization of 1e-4;
  - training: Adam, class weights, and early stopping on validation AUC using 15% of the training rows;
  - the best epoch was 11–62;
  - seeds are fixed and op determinism is on, so reruns give identical numbers.

## Which fit features matter

Permutation importance measures the average ROC-AUC drop when one column is shuffled on the test set (20 repeats), using the "all fit" model.

| Feature | Logistic regression | Random forest | Keras NN |
|---|---:|---:|---:|
| `advancement_fit` | 0.027 | 0.026 | 0.029 |
| `voice_fit_conditions` | 0.024 | 0.023 | 0.028 |
| `remote_fit` | 0.024 | 0.019 | 0.025 |
| `hours_fit` | 0.018 | 0.014 | 0.018 |
| Q51 (age) | 0.016 | 0.014 | 0.017 |
| Q40 (earnings) | 0.015 | 0.007 | 0.011 |
| `voice_fit_tech` | 0.008 | 0.009 | 0.010 |
| `voice_fit_pay` | 0.006 | 0.013 | 0.008 |
| `advancement_onet_fit` | 0.001 | 0.007 | 0.007 |
| **All 7 fit features together** | **0.169** | **0.179** | **0.180** |
| **All 8 background features together** | 0.045 | 0.039 | 0.049 |

The four most useful fit features are the same in every model:

1. Career advancement
2. Influence over working conditions
3. Remote work
4. Hours/pay

These are the candidates for the "top work-value factors" slide.

## Deeper checks

### 1. Does the fit formula add anything beyond the answers it is built from?

Logistic regression, same split:

| Model | ROC-AUC | Δ (95% CI) |
|---|---:|---:|
| Background + "want" answers only (Q19, Q33H, Q32A–C) | 0.690 | +0.063 (+0.048, +0.078) vs background |
| Background + "have" answers only (Q18, Q34, Q31A–C) | 0.732 | +0.105 (+0.087, +0.122) vs background |
| Background + all raw want/have answers, including Q28 | 0.767 | +0.139 (+0.120, +0.157) vs background |
| Raw answers + all 7 fit features | 0.771 | **+0.005 (+0.002, +0.008)** vs raw answers |

What the job gives predicts more than what the person wants. Entering the raw answers directly predicts better than the fit scores, which squeeze each pair into one number. The match itself adds a small amount beyond the answers, but it is reliably above zero.

### 2. Two-sided fit vs unmet need (Track B)

The Outline formulas count "getting more than you want" as the same mismatch as "getting less". In the data only one side hurts:

| Situation | Share satisfied |
|---|---:|
| Work from home: gets more days than wanted | 66.6% |
| Work from home: exact match | 70.6% |
| Work from home: wants 2+ more days | 50.6% |
| Working conditions: has as much influence as they should | 73.8% |
| Working conditions: has more influence than needed | 67–82% |
| Working conditions: 3 levels less influence than they should | 30.4% |
| Advancement: doesn't want it and not offered (Outline scores this **1 = match**) | **32.3%**, the lowest group |
| Advancement: doesn't want it but offered (Outline scores this 0) | 53.1% |
| Advancement: wants it and offered | 73.9% |
| Advancement: wants it and not offered | 43.4% |

So I also scored fit one-sidedly as **unmet need**: how far the job falls short of what the person wants. The comparison uses logistic regression on the same split:

| Fit scoring | ROC-AUC | 5-fold CV AUC |
|---|---:|---:|
| Two-sided (Outline formulas) | 0.731 | 0.724 |
| **Unmet need (one-sided)** | **0.741** | **0.734** |
| Both | 0.747 | 0.740 |
| Raw answers | 0.767 | 0.760 |

Unmet need beats the two-sided formulas by **+0.010 (95% CI +0.003, +0.017)**. For Swytch, this means matching should focus on avoiding a shortfall, not on avoiding any difference at all.

## Limitations

1. **Same-survey bias.** Fit and satisfaction come from the same person in the same questionnaire, so mood can colour both answers. The direction of cause is also unclear: satisfied workers may *report* having more influence.
2. **Most of the gain is "what the job gives", not "fit".** The have-only answers add +0.105, and the matching formula adds only +0.005 beyond the raw answers. The defensible claim is: "Survey measures of fit improve prediction by about 0.10 AUC, mainly because they capture whether the job delivers what people want." The claim that "matching itself" drives satisfaction is not supported.
3. **The advancement alignment rule is flawed for people who don't want advancement.** Not wanting promotion and not being offered it scores as a perfect match, yet that group is the least satisfied (32%). The Outline anticipated this; the unmet-need version avoids it.
4. **Coverage.**
   - `advancement_fit` can't be scored for 40% of people: neutral, not sure, or not applicable.
   - `remote_fit` can't be scored for 19%, mostly jobs that can't be done from home.
   - Missing indicators keep these people in the models, but their fit is unknown.
5. **O\*NET is coarse.** Gallup gives only 25 broad occupation groups, so O\*NET values are group averages. Only one valid O\*NET preference pair exists, and it adds 0.001–0.008 AUC.
6. **No survey weights.** Results describe this sample, not all U.S. workers. Gallup's `WEIGHT` column is not in the modeling dataset.
7. **A single train/test split.** The bootstrap CIs cover test-sample noise. The 5-fold CV results agree, but RF and Keras were checked on one split only.
8. **Prediction, not causation.** This is one survey at one point in time.

## How to reproduce

Run these from the repository root.

1. Install Git LFS once:
   ```bash
   brew install git-lfs
   ```
   ```bash
   git lfs install
   ```
2. Download the large data files:
   ```bash
   git lfs pull
   ```
3. Put `JFF_formatted_public_release_values.csv` from the team Drive in `data/raw/`.
4. Run the tests:
   ```bash
   cd notebooks/gallup_eda && python -m unittest test_engineer_fit_features && cd ../..
   ```
5. Build the fit dataset:
   ```bash
   python notebooks/gallup_eda/engineer_fit_features.py
   ```
6. Train the models (about 2.5 minutes):
   ```bash
   python modeling/gallup_fit_models.py
   ```
7. Run the Track B comparison (about 20 seconds):
   ```bash
   python modeling/gallup_fit_scoring_comparison.py
   ```

TensorFlow may print `E0000 … use_unbounded_threadpool` messages during Keras training. They are harmless.
