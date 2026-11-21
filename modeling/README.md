# Gallup + O*NET Job Satisfaction Modeling

## Overview

This folder contains machine learning experiments for the SWYTCH Job Satisfaction Prediction project, developed through Break Through Tech AI Studio.

The goal is to investigate whether occupational characteristics from O*NET improve predictions of employee job satisfaction when combined with Gallup survey information.

See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for the Gallup and GSS pipeline architecture.

We compare logistic regression and random forest models, evaluate predictive performance, and analyze the contributions of Gallup and O*NET features.

## Dataset

The modeling dataset is `notebooks/gallup_eda/output/gallup_fsu_pre_model.csv`.

It contains:
- 18,381 respondents
- 407 O*NET occupational features
- Gallup survey background variables
- A binary target variable, `satisfied`

The target is derived from Gallup question Q4, with values of 7 or higher classified as satisfied.

The models use eight Gallup background variables and, where applicable, 407 O*NET features.

**Data availability:** The modeling dataset is approximately 128 MB and is not currently tracked in Git. It must be obtained separately before running the scripts. The complete dataset-generation process has not yet been documented.

## Modeling Approach

We use an 80/20 stratified training and test split with `random_state=42`.

Preprocessing includes:
- Median imputation for missing numerical values
- Most-frequent imputation for missing categorical values
- One-hot encoding for categorical variables
- Numerical standardization for logistic regression

The models are evaluated using ROC-AUC, balanced accuracy, F1-score, and confusion matrices.

## Model Results

| Model | Test ROC-AUC |
|---|---:|
| Gallup Baseline Logistic Regression | 0.6271 |
| Gallup + O*NET Logistic Regression | **0.6297** |
| Tuned Gallup + O*NET Logistic Regression | 0.6290 |
| Gallup + O*NET Random Forest | 0.6144 |

The Gallup + O*NET logistic regression model achieved the highest test ROC-AUC.

However, the improvement over the Gallup-only baseline was small. A paired bootstrap analysis estimated an ROC-AUC improvement of approximately 0.0026, with a 95% confidence interval of (-0.0025, 0.0080).

Because this interval includes zero, the analysis does not establish a statistically clear improvement from adding O*NET features.

## Feature Importance

Permutation importance was evaluated using a validation subset of the training data, keeping the test set separate.

### Individual Feature Importance

The most influential individual features were Gallup variables, particularly Q51 and Q40.

Individual O*NET features generally showed much smaller permutation importance scores.

### Grouped Feature Importance

To examine the contribution of correlated occupational features, we also performed grouped permutation importance by shuffling all features within each group together.

| Feature Group | Number of Features | Mean ROC-AUC Decrease |
|---|---:|---:|
| Gallup Background | 8 | 0.1100 |
| O*NET Occupational | 407 | 0.0038 |

The results suggest that the trained model relies substantially more on Gallup background features than on O*NET occupational characteristics.

Grouped permutation importance measures model reliance, not causal relationships or the maximum predictive value a feature group could provide.

## Python Scripts

| Script | Purpose |
|---|---|
| `gallup_baseline.py` | Trains the Gallup-only logistic regression baseline |
| `gallup_onet_model.py` | Trains logistic regression using Gallup and O*NET features |
| `gallup_onet_tuned.py` | Tunes logistic regression hyperparameters using cross-validation |
| `gallup_random_forest.py` | Trains and tunes a random forest classifier |
| `compare_gallup_models.py` | Compares baseline and combined models using paired bootstrap analysis |
| `gallup_feature_importance.py` | Calculates individual and grouped permutation importance |
| `gallup_fit_models.py` | Track A: background vs background + person-job fit features (logistic regression, random forest, Keras), with paired bootstrap CIs. Results: [`fit_results.md`](fit_results.md) |
| `gallup_fit_scoring_comparison.py` | Track B: two-sided fit formulas vs one-sided "unmet need" scores vs raw answers |
| `gallup_heldout_validation.py` | 5-fold group-held-out validation for unseen occupations, industries, and occupation-industry cells |
| `gallup_full_models.py` | Adds the rest of the Gallup survey in tiers (profile → want/have fit → job conditions → job appraisals), with tuned logistic regression, gradient boosting, random forest, Keras and an ensemble. Results: [`full_model_results.md`](full_model_results.md) |

The fit scripts need `notebooks/gallup_eda/output/gallup_fsu_fit_model.csv`, built by `python notebooks/gallup_eda/engineer_fit_features.py`. `gallup_full_models.py` also reads the raw Gallup file `data/raw/JFF_formatted_public_release_values.csv`. Its tests run from this folder with `python -m unittest test_gallup_full_models`.

## Running the Models

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run scripts from the repository root:

```bash
python modeling/gallup_baseline.py
python modeling/gallup_onet_model.py
python modeling/gallup_onet_tuned.py
python modeling/gallup_random_forest.py
python modeling/compare_gallup_models.py
python modeling/gallup_feature_importance.py
python modeling/gallup_heldout_validation.py
```

The required modeling dataset must be present at the path described above.

Some scripts save prediction files, importance rankings, and visualizations in `modeling/output/`.

## Limitations and Future Work

- The observed predictive improvement from O*NET features is small and not statistically clear.
- O*NET attributes represent occupational characteristics rather than individual-level preferences.
- Approximate occupation matching may introduce uncertainty.
- The random respondent-level train/test split does not establish performance on entirely unseen occupations.
- Further work could investigate person-job fit features, improved occupational matching, and alternative validation strategies.
- The dataset-generation process should be documented to support full reproducibility.

## Conclusion

These experiments establish an initial machine learning baseline for predicting job satisfaction using Gallup survey information and O*NET occupational characteristics.

Although adding O*NET features produced a small improvement in logistic regression ROC-AUC, the current results suggest that Gallup background variables provide most of the predictive signal used by the model.

Future work will focus on developing more meaningful measures of person-job fit and evaluating whether these features improve predictive performance.