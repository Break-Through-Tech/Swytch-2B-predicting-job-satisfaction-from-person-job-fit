"""Group-held-out validation for Gallup person-job fit models.

Compares background-only and background + fit logistic models when the test
fold contains occupations, industries, or occupation-industry cells that are
not represented in the training fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "notebooks" / "gallup_eda" / "output" / "gallup_fsu_fit_model.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "JFF_formatted_public_release_values.csv"
OUTPUT_DIR = PROJECT_ROOT / "modeling" / "output" / "heldout"
SEED = 42
FOLDS = 5

BACKGROUND_NUMERIC = ["Q20", "Q51"]
BACKGROUND_CATEGORICAL = ["Q8", "Q9", "Q14", "Q16", "Q17", "Q40"]
FIT_FEATURES = [
    "remote_fit",
    "advancement_fit",
    "voice_fit_pay",
    "voice_fit_conditions",
    "voice_fit_tech",
    "hours_fit",
    "advancement_onet_fit",
]


def group_values(data: pd.DataFrame, strategy: str) -> np.ndarray:
    """Return a one-dimensional group key for one held-out strategy."""
    if strategy == "occupation":
        columns = ["Q6"]
    elif strategy == "industry":
        columns = ["Q7"]
    elif strategy == "occupation_industry":
        columns = ["Q6", "Q7"]
    else:
        raise ValueError(f"Unknown held-out strategy: {strategy}")

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Missing group columns: {', '.join(missing)}")

    if len(columns) == 1:
        return data[columns[0]].astype("string").fillna("missing").to_numpy()
    return (
        data[columns]
        .astype("string")
        .fillna("missing")
        .agg("|".join, axis=1)
        .to_numpy()
    )


def split_groups(data: pd.DataFrame, strategy: str, n_splits: int = FOLDS):
    """Create GroupKFold splits, retaining only folds with both classes."""
    y = data["satisfied"].to_numpy()
    groups = group_values(data, strategy)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(f"Need at least two groups for {strategy}")
    splits = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
        data, y, groups
    ):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("Training and test groups overlap")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            continue
        splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError(f"No valid splits for {strategy}")
    return splits


def make_model(include_fit: bool) -> Pipeline:
    numeric = BACKGROUND_NUMERIC + (FIT_FEATURES if include_fit else [])
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=include_fit)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, BACKGROUND_CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate_fold(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    y_train = data["satisfied"].to_numpy()[train_idx]
    y_test = data["satisfied"].to_numpy()[test_idx]
    rows = []

    for feature_set, include_fit in [
        ("background_only", False),
        ("background_plus_fit", True),
    ]:
        model = make_model(include_fit)
        numeric = BACKGROUND_NUMERIC + (FIT_FEATURES if include_fit else [])
        columns = numeric + BACKGROUND_CATEGORICAL
        model.fit(data.iloc[train_idx][columns], y_train)
        probability = model.predict_proba(data.iloc[test_idx][columns])[:, 1]
        pred = model.predict(data.iloc[test_idx][columns])
        rows.append(
            {
                "feature_set": feature_set,
                "roc_auc": roc_auc_score(y_test, probability),
                "pr_auc": average_precision_score(y_test, probability),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            }
        )

    return {
        "background_roc_auc": rows[0]["roc_auc"],
        "fit_roc_auc": rows[1]["roc_auc"],
        "roc_auc_improvement": rows[1]["roc_auc"] - rows[0]["roc_auc"],
        "background_pr_auc": rows[0]["pr_auc"],
        "fit_pr_auc": rows[1]["pr_auc"],
        "background_balanced_accuracy": rows[0]["balanced_accuracy"],
        "fit_balanced_accuracy": rows[1]["balanced_accuracy"],
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    raw = pd.read_csv(RAW_DATA_PATH, encoding="cp1252", low_memory=False)
    if not data["ENTITY_ID"].isin(raw["ENTITY_ID"]).all():
        raise ValueError("Every modeling respondent must match a raw Gallup row")
    data = data.merge(
        raw[["ENTITY_ID", "Q6", "Q7"]],
        on="ENTITY_ID",
        how="left",
        validate="one_to_one",
    )
    if "satisfied" not in data.columns:
        raise KeyError("Dataset is missing satisfied")

    rows = []
    for strategy in ("occupation", "industry", "occupation_industry"):
        splits = split_groups(data, strategy)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            metrics = evaluate_fold(data, train_idx, test_idx)
            rows.append(
                {
                    "strategy": strategy,
                    "fold": fold,
                    "train_groups": len(set(group_values(data, strategy)[train_idx])),
                    "test_groups": len(set(group_values(data, strategy)[test_idx])),
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    **metrics,
                }
            )

    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby("strategy", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_roc_auc_improvement=("roc_auc_improvement", "mean"),
            sd_roc_auc_improvement=("roc_auc_improvement", "std"),
            mean_background_roc_auc=("background_roc_auc", "mean"),
            mean_fit_roc_auc=("fit_roc_auc", "mean"),
        )
    )
    pr_improvement = (
        fold_results.assign(
            pr_auc_improvement=lambda frame: frame["fit_pr_auc"] - frame["background_pr_auc"]
        )
        .groupby("strategy", as_index=False)["pr_auc_improvement"]
        .mean()
    )
    summary = summary.merge(pr_improvement, on="strategy", validate="one_to_one")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "heldout_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "heldout_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
