# Full-survey models (Gallup)

**Question:** the fit models reach a test ROC-AUC of about 0.73 using 8 background answers and 7 fit scores. The Gallup file has 132 question columns (68 numbered questions). How high can prediction go if we use the rest of the survey properly, and which information gets it there?

**Answer:**

- **Best model: ROC-AUC 0.866 [95% CI 0.854, 0.878] and F1 0.850.** This is the ensemble on tier E: background, profile, want/have fit, job conditions and job appraisals.
- **Best pre-hire model: ROC-AUC 0.781 [0.766, 0.797] and F1 0.810.** This is tier C: only what a matching platform could ask the person and the employer before a hire.
- **Quote the pre-hire result as about 0.77–0.78.** Across five random 80/20 splits, tier C logistic regression ranges from 0.753 to 0.780 (mean 0.768). The main split is the highest of the five. Tier E ranges from 0.848 to 0.864 (mean 0.858).
- **Both clear 0.70 on ROC-AUC, F1 and PR-AUC.**
  - The pre-hire model's balanced accuracy is 0.645 at the F1-optimal cut-off, which favours recall. At a cut-off equal to the base rate (0.64), it is 0.71.
  - Compared with the team baseline (tier A logistic regression, 0.628), the best model gains **+0.238 AUC** [0.217, 0.257].
- **The gain comes from the features, not the algorithm.** At every tier, logistic regression, gradient boosting, random forest and Keras land within 0.007 of each other.
- **The top three work-value factors** are:
  1. belonging and respect (Relationships);
  2. fair pay (Working Conditions);
  3. a tie between feeling supported (Support) and job security (Working Conditions).

  #1 and #2 hold under both importance methods. See [Top work-value factors](#top-work-value-factors).

## Contents

- [Files](#files)
- [What "0.70" should mean](#what-070-should-mean)
- [What goes into each tier](#what-goes-into-each-tier)
- [Method](#method)
- [Results](#results)
- [Top work-value factors](#top-work-value-factors)
- [What this means for Swytch](#what-this-means-for-swytch)
- [Checks](#checks)
- [Limitations](#limitations)
- [How to reproduce](#how-to-reproduce)

## Files

| File | What it does |
|---|---|
| `modeling/gallup_full_models.py` | Builds the tiers, tunes and trains every model, and computes metrics, bootstrap CIs, permutation importance, the leakage screen and subgroup AUCs. |
| `modeling/test_gallup_full_models.py` | 17 unit tests. They cover the tier and leakage rules, answer-code cleaning, out-of-fold job averages, threshold choice, importance, the data merge and a model smoke test. |
| `modeling/gallup_full_robustness.py` | Robustness checks: drop-and-refit factor importance, the want vs have split of the pre-hire tier, five random splits and a short check-in survey. |
| `modeling/verify_full_results.py` | Independent check: recomputes every saved metric, CI and gain with its own AUC code, and re-derives the split, target and data facts from the raw file. |
| `modeling/output/full/full_model_metrics.csv` | Every model × tier, with all metrics and tuned hyperparameters. |
| `modeling/output/full/full_tier_gains.csv` | Paired AUC differences with 95% CIs. |
| `modeling/output/full/full_factor_importance.csv` | Grouped (work-value factor) and single-answer permutation importance. |
| `modeling/output/full/full_univariate_screen.csv` | Test AUC of each answer on its own (leakage screen). |
| `modeling/output/full/full_subgroup_auc.csv` | Test AUC by gender, age, education, employer type and race/ethnicity. |
| `modeling/output/full/full_test_predictions.csv` | Test-set probabilities for every model. |

## What "0.70" should mean

- **ROC-AUC (the challenge's metric):** 0.70 is a meaningful bar, because a model with no skill scores 0.50.
- **F1:** a model that predicts "satisfied" for everyone already gets **F1 = 0.780** on this test set, because 63.9% of people are satisfied. So F1 ≥ 0.70 is passed with no skill at all.

Every F1 below should therefore be read against 0.780:

| Model | F1 | Above "always satisfied" |
|---|---:|---:|
| Team background (tier A) | 0.781 | +0.001 (no real skill) |
| Pre-hire model (tier C) | 0.810 | +0.030 |
| Best model (tier E) | 0.850 | +0.070 |

ROC-AUC, PR-AUC (no-skill = 0.639) and balanced accuracy (no-skill = 0.50) are the honest headline metrics.

## What goes into each tier

Each tier adds information that is one step further from what a job-matching platform could know before someone starts a job.

Every survey answer is used as a category, so special codes are never read as points on a scale:
- 6 = "not sure";
- 7 = "not applicable";
- -98 = "no response";
- -99 = "not selected".

Blank answers mean the question was not asked (for example, second-job items). They get their own "not asked" category.

| Tier | Adds | Gallup questions | Known before hire? |
|---|---|---|---|
| A | Team background | Q8 org size, Q9 employer type, Q14 tenure, Q16 shift, Q17 days worked, Q20 hours, Q40 earnings, Q51 age | Yes |
| B | Full profile and job facts | **Person:** marital status, education, gender, Hispanic, race (Q61), born in U.S., children (Q54), working adults (Q53), certification (Q58), census division. **Job:** occupation (Q6), industry (Q7), contractor (Q10), arrangement (Q12), pay type (Q15), worker type, union (Q30), paid time off (Q26), health-insurance source (Q55), retirement plan (Q56), number of jobs (Q5), most/fewest hours (Q21, Q22) | Yes |
| C | Want/have pairs and the 7 fit scores | Q18/Q19 remote, Q33H/Q34 advancement, Q31A–C/Q32A–C voice, Q28 hours, plus `remote_fit`, `advancement_fit`, `voice_fit_*`, `hours_fit`, `advancement_onet_fit` | Mostly. Ask the person what they want, and the employer what the job offers. |
| D | Job conditions the worker reports | Q23 overtime, Q24 schedule notice, Q25A–D schedule input, Q27 work-family interference, Q29 injury, Q35A–D training, Q37A–I discrimination, Q38A–C harassment, Q39 monitoring | Partly. An employer could describe some of these. |
| E | How the worker appraises the job | Q33A freedom, Q33B variety, Q33C learning, Q33D pace, Q33E enough time, Q33F supported, Q33G belong, Q33I paid fairly, Q33J safe, Q36 respected, Q41 job security | No. Only someone in the job can answer. |
| F | General well-being (**not usable**) | Q1 life satisfaction, Q2 happiness, Q3 health, Q42 worthwhile, Q43 anxiety, Q59 finances | No. Tier F is only a ceiling and leakage check: these items are outcomes that move with job satisfaction, not information about the job. |

**Never used:**
- Q4 (the target) and Q47 (satisfaction with a second job);
- IDs, dates and free text;
- survey design fields (`SAMPLE`, `MODE`);
- `STATE` (census division is used instead);
- Q52/Q57: 78% missing (-98), and `MARITAL_STATUS`/`EDUCATION` match them 100% wherever they were answered, while covering 98.6%+ of people;
- sexual orientation, health conditions and arrest/incarceration (Q64, Q66–Q68): sensitive and not about the job;
- second-job details (Q44–Q50).

The unit tests check that Q4, Q47 and the well-being items never reach tiers A–E.

## Method

- **Same setup as every earlier model:**
  - 18,381 respondents;
  - satisfied = Q4 ≥ 7;
  - 80/20 stratified split with `random_state=42`.
  - Tier A logistic regression (0.628) matches the team baseline (0.627).
- **Tuning on the training set only.** Each model is tuned by 5-fold cross-validated ROC-AUC:
  - logistic regression: L2 penalty, C chosen from 6 values;
  - gradient boosting: scikit-learn `HistGradientBoostingClassifier` with native categorical splits, 20 random settings;
  - random forest: 500 trees, 8 random settings;
  - Keras: the Track A network (64 → 32, dropout, early stopping).
- **Ensemble:** the mean predicted probability of logistic regression, gradient boosting and Keras. The choice of members barely matters. At tier E, adding the random forest gives 0.867, and logistic regression + boosting alone gives 0.866.
- **F1 threshold:**
  - chosen to maximise F1 on 5-fold out-of-fold predictions for the training set;
  - then applied unchanged to the test set.

  The test set never influences a threshold or a hyperparameter.
- **Uncertainty:**
  - 95% CIs come from 2,000 bootstrap resamples of the test set;
  - every model uses the same resamples, so model differences are paired.
- **Survey weights:**
  - models are trained unweighted;
  - test ROC-AUC is also reported with Gallup's `WEIGHT`, to check that results hold for the weighted U.S. worker population.
- **Importance:**
  - permutation importance on the test set, 20 shuffles;
  - survey items are grouped into work-value factors, and each factor is labelled with the closest O\*NET work value.
- **No new dependencies.** Only scikit-learn, TensorFlow/Keras, pandas and numpy are used, all of which are already in `requirements.txt`.

## Results

### Test ROC-AUC by tier

| Tier | Answers | Logistic regression | Gradient boosting | Gain over previous tier (LR) [95% CI] |
|---|---:|---:|---:|---:|
| A Team background | 8 | 0.628 | 0.631 | — |
| B + profile and job facts | 46 | 0.662 | 0.668 | +0.034 [0.022, 0.046] |
| C + want/have fit | 64 | 0.780 | 0.780 | +0.118 [0.101, 0.133] |
| D + job conditions | 89 | 0.811 | 0.810 | +0.031 [0.023, 0.039] |
| E + job appraisals | 100 | 0.863 | 0.865 | +0.052 [0.043, 0.061] |
| *F + well-being (not usable)* | *106* | *0.898* | *0.902* | *+0.035 [0.029, 0.042]* |

Every gain is positive in 100% of bootstrap draws. The biggest single step is the want/have fit tier (+0.118). This agrees with the earlier fit work: what the job gives relative to what the person wants carries real signal.

### Headline models

| | Pre-hire (tier C) ensemble | Best (tier E) ensemble | No-skill reference |
|---|---:|---:|---:|
| ROC-AUC [95% CI] | **0.781** [0.766, 0.797] | **0.866** [0.854, 0.878] | 0.500 |
| ROC-AUC, survey-weighted | 0.778 | 0.856 | 0.500 |
| ROC-AUC, training out-of-fold | 0.772 | 0.863 | — |
| PR-AUC | 0.854 | 0.916 | 0.639 |
| F1 (threshold from training) | **0.810** | **0.850** | 0.780 |
| Precision / recall | 0.720 / 0.926 | 0.793 / 0.916 | 0.639 / 1.000 |
| Balanced accuracy | 0.645 | 0.747 | 0.500 |
| Brier score (lower is better) | 0.179 | 0.141 | 0.231 |

The test, out-of-fold and survey-weighted AUCs agree within about 0.01. Five different random splits give the same picture (logistic regression):

| Tier | Five splits (seeds 1, 7, 42, 123, 2026) | Mean (SD) |
|---|---|---:|
| A | 0.609, 0.610, 0.623, 0.623, 0.636 | 0.620 (0.011) |
| C | 0.753, 0.762, 0.780, 0.767, 0.778 | 0.768 (0.011) |
| E | 0.848, 0.860, 0.863, 0.857, 0.864 | 0.858 (0.006) |

The main split (seed 42) is the most favourable of the five for tier C. The pre-hire result is therefore best stated as about 0.77–0.78.

Balanced accuracy depends on the cut-off. The F1-optimal cut-off labels most people satisfied (recall 0.93), so the pre-hire balanced accuracy is 0.645. At a cut-off equal to the training base rate (0.64), it is 0.71 for tier C and 0.78 for tier E.

### All algorithms tie

| Tier | Logistic regression | Gradient boosting | Random forest | Keras | Ensemble |
|---|---:|---:|---:|---:|---:|
| C | 0.780 | 0.780 | 0.773 | 0.775 | 0.781 |
| E | 0.863 | 0.865 | 0.862 | 0.862 | 0.866 |

- **The ensemble's edge is tiny.** It beats the best single model by only +0.001 [-0.002, +0.004] at tier E.
- **Single algorithms differ by at most 0.007.** The largest gap is the random forest at tier C.
- **The signal is mostly additive.** Tuned boosting chose small trees (7 leaves) at tiers A–D and larger ones (31 leaves) at E–F. It never beat logistic regression by more than 0.006 AUC.
- **Practical takeaway:** logistic regression is just as good, and far easier to explain. For slides, use it, or the ensemble if the last 0.003 matters.

## Top work-value factors

These are grouped permutation-importance results for the tier E ensemble on the test set. Each row is the AUC lost when all of that factor's answers are shuffled together.

| Rank | Factor (closest O\*NET work value) | Items | AUC drop |
|---:|---|---:|---:|
| 1 | **Belonging & respect** (Relationships): Q33G, Q36 | 2 | 0.049 |
| — | Background, profile & job facts (not a work value) | 46 | 0.020 |
| 2 | **Fair pay** (Working Conditions): Q33I | 1 | 0.008 |
| 3 | **Feeling supported** (Support): Q33F | 1 | 0.008 |
| 4 | Job security (Working Conditions): Q41 | 1 | 0.007 |
| 5 | Learning, variety & training (Achievement) | 5 | 0.007 |
| 6 | Work-life balance & hours fit (Working Conditions) | 3 | 0.006 |
| 7 | Advancement (Recognition) | 5 | 0.004 |
| 8 | Remote-work fit (Working Conditions) | 3 | 0.004 |
| 9 | Voice over pay, conditions, technology (Independence) | 9 | 0.003 |
| 10–13 | Workload & pace, autonomy & schedule control, discrimination & harassment, physical safety | — | ≤ 0.002 |

**Second method: drop and refit.** Permutation importance shares credit between correlated items. As a check, each factor was dropped entirely and logistic regression refit at tier E (C = 0.01, test AUC 0.863). The AUC lost is the factor's unique contribution (paired bootstrap, 1,000 draws):

| Factor | Unique AUC loss [95% CI] |
|---|---:|
| Belonging & respect | 0.0068 [0.0029, 0.0104] |
| Background, profile & job facts (not a work value) | 0.0048 [0.0010, 0.0086] |
| Fair pay | 0.0040 [0.0020, 0.0059] |
| Job security | 0.0034 [0.0014, 0.0054] |
| Work-life balance & hours fit | 0.0028 [0.0011, 0.0046] |
| Remote-work fit | 0.0025 [0.0010, 0.0039] |
| Feeling supported | 0.0018 [0.0001, 0.0037] |
| Learning, variety & training | 0.0015 [-0.0003, 0.0033] |
| Advancement | 0.0015 [0.0001, 0.0028] |
| Voice; workload & pace; discrimination & harassment; autonomy; safety | ≤ 0.0005, CIs include 0 |

How to read the two tables:

- **#1 and #2 are robust.** Belonging & respect and fair pay are the top two work-value factors under both methods.
- **#3 is a tie between feeling supported and job security.**
  - Support ranks third when shuffled, but adds little once belonging is in the model. "Feeling supported" (Q33F) and "belong" (Q33G) are strongly correlated (Spearman 0.71), and each alone predicts almost equally well (test AUC 0.776 vs 0.778).
  - Job security ranks fourth when shuffled and third when dropped.
  - Ranks 2–5 in the permutation table overlap within their shuffle-to-shuffle SDs (about 0.0015), so their order is not reliable.
- **The unique contributions are small.** They are 0.002–0.007 per factor, because the appraisal items overlap heavily. Dropping one factor lets the others make up most of the loss.
- **Discrimination and harassment rank low, but they matter a lot.**
  - They are common: 24% of workers report any discrimination (Q37) and 20% report harassment (Q38).
  - They come with large drops in satisfaction: 43% vs 70% satisfied for discrimination, and 45% vs 69% for harassment.
  - They rank low only because their signal overlaps with belonging and respect, which the model already has. For example, 18% of those reporting discrimination disagree that they are treated with respect, against 3% of everyone else.
  - So low rank here means "redundant given the other answers", not "unimportant".

**Pre-hire view (tier C ensemble):**

| Rank | Factor | AUC drop |
|---:|---|---:|
| 1 | Voice over pay, conditions, technology (Q31/Q32 + `voice_fit_*`) | 0.073 |
| 2 | Advancement (Q33H/Q34 + fit) | 0.061 |
| 3 | Remote-work fit | 0.015 |
| 4 | Work-life / hours fit | 0.011 |

The background block alone scores 0.057. Drop-and-refit agrees on the order: voice 0.026, advancement 0.026, background 0.021, remote 0.009, hours 0.006.

**Want vs have in the pre-hire tier.** In Gallup, both sides of each pair are reported by the worker. Splitting them shows which side carries the signal (logistic regression, C tuned by 5-fold CV, paired gains over tier B):

| Model | Test AUC | Gain over B [95% CI] |
|---|---:|---:|
| B: profile and job facts | 0.662 | — |
| B + what the person wants (Q19, Q33H, Q32A–C, Q28) | 0.720 | +0.059 [0.047, 0.070] |
| B + what the job gives (Q18, Q34, Q31A–C) | 0.745 | +0.083 [0.068, 0.098] |
| B + both sides (raw answers) | 0.775 | +0.113 [0.097, 0.128] |
| C: both sides + 7 fit scores | 0.780 | +0.118 [0.101, 0.133] |

- Both sides are needed.
- The fit formulas add only +0.005 [0.001, 0.008] beyond the raw answers.

## What this means for Swytch

1. **Pre-hire information reaches about 0.77–0.78 AUC.** That is a 0.15 jump over the team background.
   - Most of the gain needs both sides of the match. What the person wants (remote days, advancement, voice) gets to 0.72. Adding what the job actually offers gets to 0.78.
   - This is the number that applies to job matching. It is an upper bound: here the "have" side is worker-reported.
2. **The jump to 0.87 needs the person's own experience of the job.** That means whether they feel supported, belong, are paid fairly and feel secure. Only someone already in the job can answer these.
3. **Averages for "this kind of job" don't replace that experience.**
   - **The experiment:** we gave the tier C model the average appraisals of *other* workers in the same occupation × industry. This is computed out-of-fold, so nobody sees their own answer. It is roughly the information a review site could provide.
   - **The result:** it adds **+0.000** [-0.000, +0.001] for logistic regression and +0.000 [-0.001, +0.002] for boosting. On top of that, the person's own reports of job conditions and appraisals (tiers D + E together) add +0.083 [0.070, 0.095].
   - **Why:** Gallup gives only 25 occupation groups × 19 industries, so these cells are too coarse to tell one employer from another. Employer-level culture data (for example, the employer's own engagement survey) would be the thing to test next.
4. **A short check-in after hire works almost as well.** Five items alone reach test AUC **0.843** [0.829, 0.856]: belong, respected, paid fairly, job security and supported. The full tier E model reaches 0.866.
   - Logistic regression, C tuned by 5-fold CV.
   - The four items without "supported" reach 0.838.
   - All 11 appraisal items reach 0.848.

## Checks

**Leakage screen.**
- No usable answer is a near-copy of the target. The strongest single answers are "belong" (test AUC 0.778 alone) and "supported" (0.776).
- The only item stronger than those is life satisfaction (Q1, 0.809), which is excluded.
- Tier F shows what including well-being would do: +0.035, up to about 0.90. We report it only to show the ceiling, and do not use it.
- Full list: `full_univariate_screen.csv`.

**Subgroups (tier E ensemble).** Test AUC is 0.83–0.89 in every group with at least 100 people. Two gaps are worth noting:

| Group | Test AUC | Compared with | Gap [95% bootstrap CI] |
|---|---:|---:|---:|
| Women | 0.846 | Men: 0.886 | -0.040 [-0.064, -0.017] |
| Black workers (n = 510) | 0.835 | Others: 0.871 | -0.036 [-0.072, -0.000] |

- **The gap for women is clear.** Its interval excludes zero.
- **The gap for Black workers is borderline.** Its interval just reaches zero because n is small.
- **At tier C, both gaps are smaller and not significant:** -0.025 [-0.054, +0.005] for women and -0.013 [-0.058, +0.031] for Black workers.

The model still works well for every group, but it is less sharp for women. A fairness review should look at this before any deployment, including calibration and error rates by group.

**Calibration.**
- Brier scores fall from about 0.22 (tier A) to 0.141 (tier E ensemble).
- Logistic regression and gradient boosting are trained without class weights, so their probabilities stay on the natural scale.
- The Keras network is trained with balanced class weights, so its probabilities run lower (its F1 threshold is about 0.3). The threshold is chosen on training predictions, so this does not affect F1.

## Limitations

1. **Same-survey bias, strongest at tiers D–E.**
   - Appraisals and satisfaction come from the same person in the same questionnaire, so mood colours both.
   - "I feel supported" and "I'm satisfied with my job" are different questions but not independent measurements.
   - The tier E number is best read as "how well job experience predicts satisfaction", not as a forecast made before the job starts.
2. **Prediction, not causation.** This is one survey at one point in time. Satisfied workers may *report* more belonging, rather than belonging causing satisfaction.
3. **The pre-hire tier is still self-reported.** In this survey, the "have" side (remote days, advancement offered, voice) comes from the worker, not the employer. In a real matching setting, the employer's description of the job may be less accurate.
4. **O\*NET and job averages are coarse.** Gallup has 25 occupation groups, so any job-level information is a group average.
5. **One main train/test split.**
   - The bootstrap covers test-sample noise, and the training out-of-fold AUCs agree within 0.01.
   - Five random splits (logistic regression) vary by about ±0.01. The main split is the most favourable one for tier C.
   - The out-of-fold predictions use hyperparameters tuned on the whole training set, so they are slightly optimistic. They are used only to choose the F1 threshold and as a consistency check.
6. **Importance is about the model, not the world.** Correlated items share credit (see supported vs belonging), and grouped importance grows with group size.
7. **Survey weights are used only for evaluation, not training.** Weighted AUCs are at most 0.011 below the unweighted ones.

## How to reproduce

Run these from the repository root.

1. Make sure these two files are in place:
   - `notebooks/gallup_eda/output/gallup_fsu_fit_model.csv`: run `git lfs pull`, or build it with `python notebooks/gallup_eda/engineer_fit_features.py`;
   - `data/raw/JFF_formatted_public_release_values.csv`: from the team Drive.
2. Run the tests:
   ```bash
   cd modeling && python -m unittest test_gallup_full_models && cd ..
   ```
3. Train everything. This took about 20 minutes on an 8-core laptop, mostly for the random forest and Keras runs:
   ```bash
   python modeling/gallup_full_models.py
   ```
4. Check the saved results independently and run the robustness checks (each takes under 2 minutes):
   ```bash
   python modeling/verify_full_results.py
   python modeling/gallup_full_robustness.py
   ```
