"""Demographic baselines for job satisfaction.

Compares logistic regression, a shallow decision tree, and a random forest
on the same background features, split, and survey weights. The background
set adds hours, weeks worked, and self-employment to demographics and
earnings. O*NET scores stay in the modeling table for a separate comparison.
Category encoding and imputation are fit on the training split. Logistic regression
also standardizes numeric columns. WTSSNRPS is a survey weight, not a
predictor. Test metrics are survey-weighted. Precision, recall, F1, and the
confusion matrix use a threshold chosen on the training split. A fixed
cutoff of 0.5 labels everyone satisfied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "merged_onet_gss_dataset.csv"
TARGET = "SATJOB_BINARY"
WEIGHT = "WTSSNRPS"
CATEGORICAL = ["SEX", "RACE", "MARITAL", "DEGREE", "REGION"]
NUMERIC = ["AGE"]
INCOME = "REALRINC"
HOURS = "HOURS"
WEEKS_WORKED = "WEEKSWRK"
SELF_EMPLOYED = "WRKSLF"
RANDOM_STATE = 42
TEST_SIZE = 0.2
BOOTSTRAP_DRAWS = 1000
# A probability of 0.5 is reported only as a reference. The model scores sit
# above that line because most respondents are satisfied.
REFERENCE_THRESHOLD = 0.5

REFERENCE_LEVELS = {
    "SEX": "1 (male)",
    "RACE": "1 (white)",
    "MARITAL": "1 (married)",
    "DEGREE": "0 (less than high school)",
    "REGION": "1",
    "WRKSLF": "1 (self-employed)",
}


def make_pipeline(
    numeric: list[str],
    categorical: list[str],
    model: BaseEstimator,
    scale: bool,
) -> Pipeline:
    """Impute and one-hot encode, then fit a clone of the chosen model.

    Numeric columns are standardized only for logistic regression. Trees do
    not use coefficient scale, so they keep the imputed values as they are.
    """
    transformers = []
    if numeric:
        numeric_steps: list[tuple[str, BaseEstimator]] = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    transformers.append(
        (
            "categorical",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical,
        )
    )
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("model", clone(model)),
        ]
    )


def weighted_auc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores, sample_weight=weights))


def predicted_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def select_threshold(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Training threshold that maximizes survey-weighted Youden's J.

    Youden's J is the true positive rate minus the false positive rate. The
    search uses the training scores only, so the test set does not choose it.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    positive = y_true == 1
    negative = ~positive
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    best_threshold = float(np.median(scores))
    best_j = -np.inf
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive_rate = float(weights[positive & predicted].sum()) / positive_weight
        false_positive_rate = float(weights[negative & predicted].sum()) / negative_weight
        separation = true_positive_rate - false_positive_rate
        if separation > best_j:
            best_j = separation
            best_threshold = float(threshold)
    return best_threshold


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Survey-weighted precision, recall, and F1 at the decision threshold."""
    predicted = predicted_labels(scores, threshold)
    shared = {"sample_weight": weights, "zero_division": np.nan}
    precision = float(precision_score(y_true, predicted, **shared))
    recall = float(recall_score(y_true, predicted, **shared))
    f1 = float(f1_score(y_true, predicted, **shared))
    return precision, recall, f1


def confusion_table(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Unweighted test counts. Rows are actual labels and columns are predictions."""
    predicted = predicted_labels(scores, threshold)
    counts = confusion_matrix(y_true, predicted, labels=[0, 1])
    return pd.DataFrame(
        counts,
        index=["actual dissatisfied (0)", "actual satisfied (1)"],
        columns=["predicted dissatisfied", "predicted satisfied"],
    )


def metric_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    threshold: float = REFERENCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile intervals for weighted AUC, precision, recall, and F1.

    Each draw resamples the test rows with replacement. The returned arrays
    are ordered AUC, precision, recall, F1.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    estimates = np.empty((draws, 4))
    for i in range(draws):
        sample = rng.integers(0, n, n)
        y_sample = y_true[sample]
        if np.unique(y_sample).size < 2:
            estimates[i] = np.nan
            continue
        estimates[i, 0] = weighted_auc(y_sample, scores[sample], weights[sample])
        estimates[i, 1:] = threshold_metrics(y_sample, scores[sample], weights[sample], threshold)
    low, high = np.nanpercentile(estimates, [2.5, 97.5], axis=0)
    return low, high


def feature_names(pipeline: Pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    names = list(numeric)
    encoder = pipeline.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
    names.extend(encoder.get_feature_names_out(categorical).tolist())
    return names


def fit_and_report(
    frame: pd.DataFrame,
    label: str,
    numeric: list[str],
    model: BaseEstimator,
    scale: bool,
    categorical: list[str] | None = None,
) -> None:
    categories = CATEGORICAL if categorical is None else categorical
    predictors = numeric + categories
    x = frame[predictors]
    y = frame[TARGET].to_numpy()
    weights = frame[WEIGHT].to_numpy()
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x,
        y,
        weights,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = make_pipeline(numeric, categories, model, scale)
    pipeline.fit(x_train, y_train, model__sample_weight=w_train)
    train_scores = pipeline.predict_proba(x_train)[:, 1]
    scores = pipeline.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_train, train_scores, w_train)
    auc = weighted_auc(y_test, scores, w_test)
    precision, recall, f1 = threshold_metrics(y_test, scores, w_test, threshold)
    low, high = metric_intervals(y_test, scores, w_test, threshold=threshold)
    matrix = confusion_table(y_test, scores, threshold)
    always_satisfied = np.unique(predicted_labels(scores, REFERENCE_THRESHOLD)).size == 1
    estimator = pipeline.named_steps["model"]

    print(f"\n{label}")
    print(f"Train rows: {len(x_train):,}   Test rows: {len(x_test):,}")
    print(f"Test share satisfied: {y_test.mean():.3f}")
    print(f"Survey-weighted test AUC: {auc:.3f}  (95% interval {low[0]:.3f} to {high[0]:.3f})")
    if always_satisfied:
        print("At a threshold of 0.50, every test row is predicted satisfied.")
    print(
        "Decision threshold, chosen on the training split to maximize "
        f"Youden's J: {threshold:.3f}"
    )
    print(
        f"Survey-weighted precision: {precision:.3f}  "
        f"(95% interval {low[1]:.3f} to {high[1]:.3f})"
    )
    print(
        f"Survey-weighted recall: {recall:.3f}  "
        f"(95% interval {low[2]:.3f} to {high[2]:.3f})"
    )
    print(f"Survey-weighted F1: {f1:.3f}  (95% interval {low[3]:.3f} to {high[3]:.3f})")
    print("Confusion matrix on the test set (row counts):")
    print(matrix.to_string())
    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_.ravel()
        names = feature_names(pipeline, numeric, categories)
        effects = (
            pd.DataFrame({"feature": names, "log_odds": coefficients, "odds_ratio": np.exp(coefficients)})
            .sort_values("log_odds", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )
        print("Largest coefficients, versus the dropped reference level:")
        print(effects.head(8).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def baseline_models() -> list[tuple[str, BaseEstimator, bool]]:
    """Model name, estimator, and whether numeric columns are standardized."""
    return [
        ("Logistic regression", LogisticRegression(max_iter=1000, solver="lbfgs"), True),
        (
            "Decision tree",
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=RANDOM_STATE),
            False,
        ),
        (
            "Random forest",
            RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=RANDOM_STATE),
            False,
        ),
    ]


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    print("Reference levels dropped by one-hot encoding:")
    for column, level in REFERENCE_LEVELS.items():
        print(f"  {column}: {level}")
    print("Decision tree: max depth 4, at least 50 training rows per leaf.")
    print("Random forest: 200 trees, at least 20 training rows per leaf.")
    work_situation = NUMERIC + [INCOME, HOURS, WEEKS_WORKED]
    work_categories = CATEGORICAL + [SELF_EMPLOYED]
    feature_sets = [
        ("Demographics", NUMERIC, CATEGORICAL),
        ("Demographics plus own earnings (REALRINC)", NUMERIC + [INCOME], CATEGORICAL),
        (
            "Demographics, earnings, hours, weeks worked, and self-employment",
            work_situation,
            work_categories,
        ),
    ]
    for model_name, model, scale in baseline_models():
        for feature_label, numeric, categorical in feature_sets:
            fit_and_report(
                frame,
                f"{model_name}: {feature_label}",
                numeric,
                model,
                scale,
                categorical,
            )


if __name__ == "__main__":
    main()
