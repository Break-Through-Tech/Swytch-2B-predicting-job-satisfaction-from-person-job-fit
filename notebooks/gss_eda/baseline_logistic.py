"""Demographic baseline for job satisfaction.

Predicts SATJOB_BINARY from background characteristics only. Job conditions
and O*NET scores stay in the modeling table for later models. Category
encoding and imputation are fit on the training split. WTSSNRPS is a
survey weight, not a predictor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "merged_onet_gss_dataset.csv"
TARGET = "SATJOB_BINARY"
WEIGHT = "WTSSNRPS"
CATEGORICAL = ["SEX", "RACE", "MARITAL", "DEGREE", "REGION"]
NUMERIC = ["AGE"]
INCOME = "REALRINC"
RANDOM_STATE = 42
TEST_SIZE = 0.2
BOOTSTRAP_DRAWS = 1000

REFERENCE_LEVELS = {
    "SEX": "1 (male)",
    "RACE": "1 (white)",
    "MARITAL": "1 (married)",
    "DEGREE": "0 (less than high school)",
    "REGION": "1",
}


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
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
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("model", model),
        ]
    )


def weighted_auc(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores, sample_weight=weights))


def auc_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    """Percentile interval for the survey-weighted test AUC."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    estimates = np.empty(draws)
    for i in range(draws):
        sample = rng.integers(0, n, n)
        y_sample = y_true[sample]
        if np.unique(y_sample).size < 2:
            estimates[i] = np.nan
            continue
        estimates[i] = weighted_auc(y_sample, scores[sample], weights[sample])
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def feature_names(pipeline: Pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    names = list(numeric)
    encoder = pipeline.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
    names.extend(encoder.get_feature_names_out(categorical).tolist())
    return names


def fit_and_report(frame: pd.DataFrame, label: str, numeric: list[str]) -> None:
    predictors = numeric + CATEGORICAL
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

    pipeline = make_pipeline(numeric, CATEGORICAL)
    pipeline.fit(x_train, y_train, model__sample_weight=w_train)
    scores = pipeline.predict_proba(x_test)[:, 1]
    auc = weighted_auc(y_test, scores, w_test)
    low, high = auc_interval(y_test, scores, w_test)

    coefficients = pipeline.named_steps["model"].coef_.ravel()
    names = feature_names(pipeline, numeric, CATEGORICAL)
    effects = (
        pd.DataFrame({"feature": names, "log_odds": coefficients, "odds_ratio": np.exp(coefficients)})
        .sort_values("log_odds", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n{label}")
    print(f"Train rows: {len(x_train):,}   Test rows: {len(x_test):,}")
    print(f"Test share satisfied: {y_test.mean():.3f}")
    print(f"Survey-weighted test AUC: {auc:.3f}  (95% interval {low:.3f} to {high:.3f})")
    print("Largest coefficients, versus the dropped reference level:")
    print(effects.head(8).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    print("Reference levels dropped from the model:")
    for column, level in REFERENCE_LEVELS.items():
        print(f"  {column}: {level}")
    fit_and_report(frame, "Demographics", NUMERIC)
    fit_and_report(frame, "Demographics plus own earnings (REALRINC)", NUMERIC + [INCOME])


if __name__ == "__main__":
    main()
