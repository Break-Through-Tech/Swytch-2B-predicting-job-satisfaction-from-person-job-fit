"""How high can job-satisfaction prediction go, and which information gets it there?

The fit models (gallup_fit_models.py) use 8 background answers plus 7
person-job fit scores and reach a test ROC-AUC of about 0.73. Gallup asked
about 150 questions, so this script adds the rest of the survey in tiers.
Each tier adds information that is one step further from what a job-matching
platform could know before someone takes a job:

  A  Team background: the 8 answers every earlier model used.
  B  + full profile: demographics, occupation, industry, contract type,
       pay type, hours, benefits, union.
  C  + want/have pairs and the 7 fit scores. A-C is "pre-hire" information:
       a platform could ask the person and the employer for it.
  D  + job conditions the worker reports: schedule control, overtime,
       work-family interference, training, discrimination, harassment,
       monitoring, injuries.
  E  + how the worker appraises the job: autonomy, variety, learning, pace,
       time, support, belonging, fair pay, safety, respect, job security.
  F  + general well-being (life satisfaction, happiness, worthwhile, anxiety,
       health, finances). NOT a usable model: these are outcomes that move
       with job satisfaction, not information about the job. It is shown
       only as a ceiling / leakage check.

Never used: Q4 (the target), Q47 (satisfaction with a second job), IDs,
dates, free text, survey design fields and the survey weight.

Method. Same respondents, target (satisfied = Q4 >= 7) and 80/20 stratified
split (random_state=42) as the earlier scripts. All tuning uses 5-fold
cross-validation on the training set only. Survey answers are treated as
categories, so codes such as 6 = "not sure", 7 = "not applicable" and
-98 = "no response" are never read as points on a scale. The F1 threshold
is chosen on out-of-fold training predictions, never on the test set.
Confidence intervals come from a paired bootstrap of the test set.

Usage (from the repository root, after engineer_fit_features.py):
    python modeling/gallup_full_models.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gallup_fit_models import ALL_FIT, KerasModel, bootstrap_indices, paired_delta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIT_DATA_PATH = PROJECT_ROOT / "notebooks" / "gallup_eda" / "output" / "gallup_fsu_fit_model.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "JFF_formatted_public_release_values.csv"
OUTPUT_DIR = PROJECT_ROOT / "modeling" / "output" / "full"

SEED = 42
FOLDS = 5
PERMUTATION_REPEATS = 20
NO_RESPONSE = -98
NOT_ASKED = "not asked"

# tier -> (label, numeric columns added, categorical columns added)
TIERS = {
    "A": ("Team background", ["Q20", "Q51"], ["Q8", "Q9", "Q14", "Q16", "Q17", "Q40"]),
    "B": (
        "+ full profile and job facts",
        ["Q5", "Q21", "Q22", "Q53"],
        # person
        ["MARITAL_STATUS", "EDUCATION", "Q65", "Q60", "Q61_1", "Q61_2", "Q61_3", "Q61_4", "Q61_5", "Q63"]
        + ["Q54A", "Q54B", "Q54C", "Q58", "DIVISION"]
        # job facts a posting or employer could state
        + ["Q6", "Q7", "Q10", "Q12", "Q15", "WORKER_TYPE", "Q30", "Q55", "Q56_1", "Q56_2", "Q56_3", "Q56_4"]
        + [f"Q26_{i}" for i in range(1, 8)],
    ),
    "C": (
        "+ want/have pairs and fit scores",
        ALL_FIT,
        ["Q19", "Q33H", "Q32A", "Q32B", "Q32C", "Q28", "Q18", "Q34", "Q31A", "Q31B", "Q31C"],
    ),
    "D": (
        "+ job conditions",
        [],
        ["Q23", "Q24", "Q25A", "Q25B", "Q25C", "Q25D", "Q27", "Q29", "Q35A", "Q35B", "Q35C", "Q35D"]
        + [f"Q37{c}" for c in "ABCDEFGHI"]
        + ["Q38A", "Q38B", "Q38C", "Q39"],
    ),
    "E": (
        "+ job appraisals",
        [],
        ["Q33A", "Q33B", "Q33C", "Q33D", "Q33E", "Q33F", "Q33G", "Q33I", "Q33J", "Q36", "Q41"],
    ),
    "F": ("+ general well-being (leakage ceiling, not usable)", [], ["Q1", "Q2", "Q3", "Q42", "Q43", "Q59"]),
}
USABLE_TIERS = "ABCDE"
HEADLINE_TIERS = ("C", "E")  # best pre-hire model, best model overall
ENSEMBLE_MEMBERS = ("Logistic regression", "Gradient boosting", "Keras neural network")
NEVER_USED = ["Q4", "Q47", "ENTITY_ID", "WEIGHT"]

# Groups for permutation importance: each groups the survey items about one
# work-value factor, labelled with the closest O*NET work value.
FACTORS = {
    "Feeling supported (Support)": ["Q33F"],
    "Belonging & respect (Relationships)": ["Q33G", "Q36"],
    "Discrimination & harassment (Relationships)": [f"Q37{c}" for c in "ABCDEFGHI"] + ["Q38A", "Q38B", "Q38C"],
    "Fair pay (Working Conditions)": ["Q33I"],
    "Job security (Working Conditions)": ["Q41"],
    "Learning, variety & training (Achievement)": ["Q33B", "Q33C", "Q35A", "Q35C", "Q35D"],
    "Advancement (Recognition)": ["Q33H", "Q34", "Q35B", "advancement_fit", "advancement_onet_fit"],
    "Autonomy & schedule control (Independence)": ["Q33A", "Q25A", "Q25B", "Q25C", "Q25D", "Q39"],
    "Voice over pay, conditions, technology (Independence)": [
        "Q31A", "Q31B", "Q31C", "Q32A", "Q32B", "Q32C", "voice_fit_pay", "voice_fit_conditions", "voice_fit_tech",
    ],
    "Workload & pace (Working Conditions)": ["Q33D", "Q33E", "Q23", "Q24"],
    "Work-life balance & hours fit (Working Conditions)": ["Q27", "Q28", "hours_fit"],
    "Remote-work fit (Working Conditions)": ["Q18", "Q19", "remote_fit"],
    "Physical safety (Working Conditions)": ["Q33J", "Q29"],
}
PROFILE_GROUP = "Background, profile & job facts"

# Appraisal items averaged over other workers in the same job (experiment 2).
APPRAISAL_ITEMS = TIERS["E"][2]
JOB_CELL = ["Q6", "Q7"]  # occupation group x industry
SMOOTHING = 20  # a cell's mean counts like this many respondents at the overall mean


def tier_columns(tier: str) -> tuple[list[str], list[str]]:
    """Numeric and categorical columns of a tier, including every earlier tier."""
    numeric, categorical = [], []
    for name, (_, num, cat) in TIERS.items():
        numeric += num
        categorical += cat
        if name == tier:
            return numeric, categorical
    raise KeyError(tier)


def factor_groups(columns: list[str]) -> dict[str, list[str]]:
    """FACTORS restricted to the given columns; everything else is the profile group."""
    groups = {name: [c for c in cols if c in columns] for name, cols in FACTORS.items()}
    groups = {name: cols for name, cols in groups.items() if cols}
    in_factor = {c for cols in groups.values() for c in cols}
    groups[PROFILE_GROUP] = [c for c in columns if c not in in_factor]
    return groups


def numeric_answers(values: pd.Series) -> pd.Series:
    """Numbers, with blanks (question not asked) and -98 (no response) as NaN."""
    numbers = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return numbers.mask(numbers == NO_RESPONSE)


def coded_answers(values: pd.Series) -> pd.Series:
    """Answer codes as unordered categories. "-98" (no response), "-99" (not
    selected), "6" (not sure) etc. each stay their own category; a blank
    (question not asked) becomes NOT_ASKED."""
    codes = pd.to_numeric(values.astype(str).str.strip(), errors="coerce").astype("Int64")
    return codes.astype(str).where(codes.notna(), NOT_ASKED).astype("category")


def load_data(fit_path: Path = FIT_DATA_PATH, raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """The 18,381 modelling respondents with every raw Gallup answer and the fit scores."""
    fit = pd.read_csv(fit_path, usecols=["ENTITY_ID", "satisfied", *ALL_FIT])
    raw = pd.read_csv(raw_path, encoding="cp1252", low_memory=False)
    if not fit["ENTITY_ID"].isin(raw["ENTITY_ID"]).all():
        raise ValueError("every modelling respondent must match a raw Gallup row")
    return fit.merge(raw, on="ENTITY_ID", how="left", validate="one_to_one")


def prepare(data: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {**{c: numeric_answers(data[c]) for c in numeric}, **{c: coded_answers(data[c]) for c in categorical}}
    )


def likert(values: pd.Series) -> pd.Series:
    """1-5 agree/disagree scale; 6 (not sure), 7 (not applicable) and -98 become NaN."""
    numbers = numeric_answers(values)
    return numbers.where(numbers.between(1, 5))


def cell_means(source: pd.DataFrame, target: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """For each target row, the smoothed mean answer of source respondents in the same job cell."""
    def key(frame):
        return frame[JOB_CELL].astype(str).agg("|".join, axis=1)

    source_key, target_key = key(source), key(target)
    out = {}
    for item in items:
        answers = likert(source[item])
        overall = answers.mean()
        by_cell = answers.groupby(source_key)
        smoothed = (by_cell.sum() + SMOOTHING * overall) / (by_cell.count() + SMOOTHING)
        out[f"job_avg_{item}"] = target_key.map(smoothed).fillna(overall).to_numpy(dtype=float)
    return pd.DataFrame(out, index=target.index)


def job_level_features(data: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, items: list[str]) -> pd.DataFrame:
    """Average appraisals of *other* workers in the same occupation x industry.

    A matching platform cannot ask a candidate how supported they will feel,
    but it could learn how supported current workers in that kind of job
    feel. Training rows get out-of-fold means, so a person's own answer never
    feeds their own feature; test rows get means over the whole training set.
    """
    features = pd.DataFrame(np.nan, index=data.index, columns=[f"job_avg_{i}" for i in items])
    train = data.iloc[train_idx]
    for fit_part, held_out in KFold(FOLDS, shuffle=True, random_state=SEED).split(train):
        features.iloc[train_idx[held_out]] = cell_means(train.iloc[fit_part], train.iloc[held_out], items).to_numpy()
    features.iloc[test_idx] = cell_means(train, data.iloc[test_idx], items).to_numpy()
    return features


# --------------------------------------------------------------------------- models


def one_hot_preprocessor(numeric: list[str], categorical: list[str], dense: bool = False) -> ColumnTransformer:
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=not dense)
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric), ("categorical", encoder, categorical)],
        sparse_threshold=0 if dense else 0.3,
    )


class Model:
    """Uniform wrapper: fit(X, y) and predict_proba(X) -> P(satisfied)."""

    def __init__(self, name, estimator, grid=None, n_iter=None):
        self.name, self.estimator, self.grid, self.n_iter = name, estimator, grid, n_iter
        self.params = {}

    def tune_and_fit(self, X, y):
        """Pick hyperparameters by 5-fold CV ROC-AUC on (X, y), then refit on all of it."""
        if self.grid:
            cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
            if self.n_iter:
                search = RandomizedSearchCV(self.estimator, self.grid, n_iter=self.n_iter, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1)
            else:
                search = GridSearchCV(self.estimator, self.grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X, y)
            self.params = search.best_params_
            self.estimator = self.estimator.set_params(**self.params)
        return self.fit(X, y)

    def fit(self, X, y):
        self.fitted = clone(self.estimator).fit(X, y)
        return self

    def fresh(self):
        """Unfitted copy with the tuned hyperparameters (for out-of-fold predictions)."""
        return Model(self.name, clone(self.estimator))

    def predict_proba(self, X):
        return self.fitted.predict_proba(X)[:, 1]


class NeuralNet(KerasModel):
    """The Track A Keras network, on one-hot encoded inputs."""

    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = numeric, categorical
        self.preprocessor = one_hot_preprocessor(numeric, categorical, dense=True)
        self.params = {}

    def tune_and_fit(self, X, y):
        return self.fit(X, y)

    def fresh(self):
        return NeuralNet(self.numeric, self.categorical)


class Ensemble:
    """Mean predicted probability of several fitted models."""

    name = "Ensemble (mean of LR, boosting, neural net)"

    def __init__(self, members):
        self.members = members

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def logistic(numeric, categorical) -> Model:
    pipeline = Pipeline([("prep", one_hot_preprocessor(numeric, categorical)), ("clf", LogisticRegression(max_iter=5000))])
    return Model("Logistic regression", pipeline, {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]})


def boosting(numeric, categorical) -> Model:
    """Gradient boosting with native categorical splits and missing-value handling."""
    estimator = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=SEED
    )
    grid = {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 1.0, 10.0],
        "max_features": [0.3, 0.6, 1.0],
    }
    return Model("Gradient boosting", estimator, grid, n_iter=20)


def forest(numeric, categorical) -> Model:
    pipeline = Pipeline(
        [("prep", one_hot_preprocessor(numeric, categorical)), ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))]
    )
    grid = {"clf__max_depth": [10, 20, None], "clf__min_samples_leaf": [1, 5, 10, 20], "clf__max_features": ["sqrt", 0.2]}
    return Model("Random forest", pipeline, grid, n_iter=8)


# --------------------------------------------------------------------------- evaluation


def out_of_fold(model, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions on the training set with the model's tuned hyperparameters."""
    oof = np.zeros(len(y))
    for fit_part, held_out in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y):
        oof[held_out] = model.fresh().fit(X.iloc[fit_part], y[fit_part]).predict_proba(X.iloc[held_out])
    return oof


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def auc_ci(y: np.ndarray, prob: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    aucs = [roc_auc_score(y[s], prob[s]) for s in draws]
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def evaluate(y, prob, draws, weights, oof_y, oof_prob) -> dict:
    """Test-set metrics. The F1 threshold is chosen on out-of-fold training
    predictions (oof_y, oof_prob), so the test set never influences it."""
    threshold = best_f1_threshold(oof_y, oof_prob)
    pred = (prob >= threshold).astype(int)
    low, high = auc_ci(y, prob, draws)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "roc_auc_ci_low": low,
        "roc_auc_ci_high": high,
        "roc_auc_survey_weighted": roc_auc_score(y, prob, sample_weight=weights),
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "f1_at_0.5": f1_score(y, (prob >= 0.5).astype(int)),
        "threshold": threshold,
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "train_oof_roc_auc": roc_auc_score(oof_y, oof_prob),
    }


def grouped_permutation_importance(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Mean test ROC-AUC drop when every column in a group is shuffled together.
    X must have a RangeIndex. Shuffling keeps each column's dtype."""
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y, model.predict_proba(X))
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            order = rng.permutation(len(X))
            shuffled = X.copy()
            for c in cols:
                shuffled[c] = X[c].iloc[order].set_axis(X.index)
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)))
        rows.append({"factor": name, "n_items": len(cols), "auc_drop_mean": np.mean(drops), "auc_drop_sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


def univariate_screen(data, y, train_idx, test_idx, columns) -> pd.DataFrame:
    """Test ROC-AUC of each answer alone (its satisfaction rate by answer code
    in the training set). Flags items that are near-copies of the target.
    Numeric answers are cut into deciles first."""
    numeric = {c for t in TIERS for c in TIERS[t][1]}
    rows = []
    for c in columns:
        if c in numeric:
            codes = pd.qcut(numeric_answers(data[c]), 10, duplicates="drop").astype(str)
        else:
            codes = coded_answers(data[c]).astype(str)
        rate = pd.Series(y[train_idx]).groupby(codes.iloc[train_idx].to_numpy()).mean()
        score = codes.iloc[test_idx].map(rate).fillna(y[train_idx].mean()).to_numpy(dtype=float)
        tier = next(t for t in TIERS if c in tier_columns(t)[0] + tier_columns(t)[1])
        rows.append({"column": c, "tier": tier, "test_roc_auc_alone": roc_auc_score(y[test_idx], score)})
    return pd.DataFrame(rows).sort_values("test_roc_auc_alone", ascending=False)


def subgroup_auc(y, prob, data_test: pd.DataFrame) -> pd.DataFrame:
    """Test ROC-AUC within demographic and job subgroups (does the model work for everyone?)."""
    age = numeric_answers(data_test["Q51"])
    subgroups = {
        "Gender": numeric_answers(data_test["Q65"]).map({1: "Female", 2: "Male", 3: "Nonbinary"}),
        "Age": pd.cut(age, [0, 29, 44, 59, 120], labels=["18-29", "30-44", "45-59", "60+"]).astype(str),
        "Education": numeric_answers(data_test["EDUCATION"]).map(lambda v: "Bachelor's or higher" if v in (5, 6) else ("Less than bachelor's" if v in (1, 2, 3, 4) else np.nan)),
        "Employer": numeric_answers(data_test["Q9"]).map({1: "Government", 2: "Private company", 3: "Private household", 4: "Self-employed", 5: "Family business"}),
        "Hispanic": numeric_answers(data_test["Q60"]).map({1: "Hispanic", 2: "Not Hispanic"}),
        "Black": numeric_answers(data_test["Q61_3"]).map({1: "Black", -99: "Not Black"}),
    }
    rows = []
    for variable, labels in subgroups.items():
        labels = pd.Series(np.asarray(labels, dtype=object))
        for label in labels.dropna().unique():
            if label == "nan":
                continue
            mask = (labels == label).to_numpy()
            if mask.sum() >= 50 and 0 < y[mask].mean() < 1:
                rows.append({"variable": variable, "group": label, "n": int(mask.sum()), "share_satisfied": y[mask].mean(), "roc_auc": roc_auc_score(y[mask], prob[mask])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    y = data["satisfied"].to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=SEED)
    y_train, y_test = y[train_idx], y[test_idx]
    w_test = numeric_answers(data["WEIGHT"]).to_numpy()[test_idx]
    draws = bootstrap_indices(len(test_idx))
    print(f"Respondents {len(data):,}  train {len(train_idx):,}  test {len(test_idx):,}  satisfied {y.mean():.3f}")
    print(f"No-skill reference: ROC-AUC 0.5, PR-AUC {y_test.mean():.3f}, F1 of 'always satisfied' {2 * y_test.mean() / (1 + y_test.mean()):.3f}\n")

    def split(numeric, categorical, extra=None):
        X = prepare(data, numeric, categorical)
        if extra is not None:
            X = pd.concat([X, extra], axis=1)
        return X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)

    metrics, gains, probs, fitted, oofs, test_frames = [], [], {}, {}, {}, {}

    def run(tier, label, model, X_train, X_test, tune=True):
        """Tune, fit, predict the test set and record metrics (F1 threshold from out-of-fold training predictions)."""
        if tune:
            model.tune_and_fit(X_train, y_train)
        key = (tier, model.name)
        oofs[key] = out_of_fold(model, X_train, y_train) if tune else np.mean([oofs[(tier, n)] for n in ENSEMBLE_MEMBERS], axis=0)
        fitted[key], probs[key] = model, model.predict_proba(X_test)
        row = {"tier": tier, "feature_set": label, "model": model.name, "n_answers": X_train.shape[1]}
        row.update(evaluate(y_test, probs[key], draws, w_test, oof_y=y_train, oof_prob=oofs[key]))
        if getattr(model, "params", None):
            row["hyperparameters"] = str(model.params)
        metrics.append(row)
        print(f"  {tier:9s} {model.name:45s} ROC-AUC {row['roc_auc']:.4f} [{row['roc_auc_ci_low']:.3f}, {row['roc_auc_ci_high']:.3f}]  F1 {row['f1']:.3f}  ({time.time() - start:.0f}s)", flush=True)

    # 1. Every tier with logistic regression and gradient boosting; the headline
    #    tiers also with random forest, the neural net and the ensemble.
    for tier, (label, _, _) in TIERS.items():
        numeric, categorical = tier_columns(tier)
        X_train, X_test = split(numeric, categorical)
        test_frames[tier] = X_test
        print(f"Tier {tier}: {label} ({len(numeric) + len(categorical)} answers)")
        makers = (logistic, boosting, forest, NeuralNet) if tier in HEADLINE_TIERS else (logistic, boosting)
        for make in makers:
            run(tier, label, make(numeric, categorical), X_train, X_test)
        if tier in HEADLINE_TIERS:
            run(tier, label, Ensemble([fitted[(tier, n)] for n in ENSEMBLE_MEMBERS]), X_train, X_test, tune=False)

    # 2. Experiment: can job-level averages stand in for the person's own appraisals?
    print("Experiment: tier C + average appraisals of other workers in the same occupation x industry")
    job_avg = job_level_features(data, train_idx, test_idx, APPRAISAL_ITEMS)
    numeric, categorical = tier_columns("C")
    X_train, X_test = split(numeric, categorical, extra=job_avg)
    for make in (logistic, boosting):
        run("C+job_avg", "C + job-level average appraisals", make(numeric + list(job_avg.columns), categorical), X_train, X_test)

    # Paired ROC-AUC gains with bootstrap CIs.
    def gain(model_name, comparison, base_key, new_key):
        gains.append({"model": model_name, "comparison": comparison, **paired_delta(y_test, probs[base_key], probs[new_key], draws)})

    tiers = list(TIERS)
    for name in ("Logistic regression", "Gradient boosting"):
        for prev, tier in zip(tiers, tiers[1:]):
            gain(name, f"{tier} vs {prev}", (prev, name), (tier, name))
        for tier in ("C", "E"):
            gain(name, f"{tier} vs A", ("A", name), (tier, name))
        gain(name, "C + job-level averages vs C", ("C", name), ("C+job_avg", name))
        gain(name, "E vs C + job-level averages", ("C+job_avg", name), ("E", name))
    for tier in HEADLINE_TIERS:
        for name in ("Logistic regression", "Gradient boosting", "Random forest", "Keras neural network"):
            gain(Ensemble.name, f"{tier}: ensemble vs {name}", (tier, name), (tier, Ensemble.name))

    # 3. Which work-value factors drive the predictions? (test set)
    print("\nGrouped permutation importance (ensemble, test set) ...")
    importance = []
    for tier in ("E", "C"):
        X_test = test_frames[tier]
        table = grouped_permutation_importance(fitted[(tier, Ensemble.name)], X_test, y_test, factor_groups(list(X_test.columns)))
        importance.append(table.assign(tier=tier, model=Ensemble.name))
        print(f"\nTier {tier}:\n{table.round(4).to_string(index=False)}")
    single = grouped_permutation_importance(fitted[("E", "Gradient boosting")], test_frames["E"], y_test, {c: [c] for c in test_frames["E"].columns})
    importance.append(single.assign(tier="E", model="Gradient boosting (single answers)"))

    # 4. Leakage screen and subgroup checks.
    screen = univariate_screen(data, y, train_idx, test_idx, [c for t in TIERS for c in TIERS[t][1] + TIERS[t][2]])
    data_test = data.iloc[test_idx].reset_index(drop=True)
    subgroups = pd.concat([subgroup_auc(y_test, probs[(t, Ensemble.name)], data_test).assign(tier=t) for t in HEADLINE_TIERS])

    metrics = pd.DataFrame(metrics)
    gains = pd.DataFrame(gains)
    metrics.to_csv(OUTPUT_DIR / "full_model_metrics.csv", index=False)
    gains.to_csv(OUTPUT_DIR / "full_tier_gains.csv", index=False)
    pd.concat(importance).to_csv(OUTPUT_DIR / "full_factor_importance.csv", index=False)
    screen.to_csv(OUTPUT_DIR / "full_univariate_screen.csv", index=False)
    subgroups.to_csv(OUTPUT_DIR / "full_subgroup_auc.csv", index=False)
    predictions = {"ENTITY_ID": data["ENTITY_ID"].to_numpy()[test_idx], "satisfied": y_test}
    predictions.update({f"{tier} | {name}": p for (tier, name), p in probs.items()})
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "full_test_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nTest-set metrics (F1 threshold chosen on out-of-fold training predictions):")
    print(metrics.drop(columns=["hyperparameters", "feature_set"], errors="ignore").round(4).to_string(index=False))
    print(f"\nPaired ROC-AUC gains, 95% bootstrap CI:\n{gains.round(4).to_string(index=False)}")
    print(f"\nTop single answers (gradient boosting, tier E):\n{single.head(15).round(4).to_string(index=False)}")
    print(f"\nSubgroup ROC-AUC:\n{subgroups.round(3).to_string(index=False)}")
    print(f"\nSaved results to {OUTPUT_DIR} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
