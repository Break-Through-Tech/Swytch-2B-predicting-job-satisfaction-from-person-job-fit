"""Track A: does person-job fit improve job satisfaction prediction?

Compares background-only models with the same models plus the fit features
from notebooks/gallup_eda/engineer_fit_features.py, for logistic regression,
random forest and a Keras neural network.

Fair comparison: every model uses the same 18,381 respondents, the same
target (satisfied = Q4 >= 7), the same 80/20 stratified split
(random_state=42) and class-balanced training. Imputation, scaling and
encoding are fit on training data only. Background preprocessing is the same
as modeling/gallup_baseline.py, so the logistic regression baseline reproduces
its ROC-AUC (0.6271).

The 95% confidence interval for each improvement comes from a paired
bootstrap of the test set: each draw resamples respondents and scores the
baseline and fit predictions for the same people.

Usage (from the repository root, after engineer_fit_features.py):
    python modeling/gallup_fit_models.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "notebooks" / "gallup_eda" / "output" / "gallup_fsu_fit_model.csv"
OUTPUT_DIR = PROJECT_ROOT / "modeling" / "output" / "fit"

SEED = 42
BOOTSTRAP_DRAWS = 2000
PERMUTATION_REPEATS = 20

# Same background features as modeling/gallup_baseline.py.
BACKGROUND_NUMERIC = ["Q20", "Q51"]
BACKGROUND_CATEGORICAL = ["Q8", "Q9", "Q14", "Q16", "Q17", "Q40"]

OUTLINE_FIT = ["remote_fit", "advancement_fit"]
GALLUP_FIT = OUTLINE_FIT + ["voice_fit_pay", "voice_fit_conditions", "voice_fit_tech", "hours_fit"]
ALL_FIT = GALLUP_FIT + ["advancement_onet_fit"]

# The raw answers each fit feature is built from. Used only for the check
# "does the fit formula add anything beyond the answers it is made of?".
# One-hot encoded because codes such as 8 (WFH not possible), 6-7 (not sure /
# not applicable) and 5 (not applicable) are not points on a scale.
FIT_INPUTS = ["Q18", "Q19", "Q33H", "Q34", "Q31A", "Q31B", "Q31C", "Q32A", "Q32B", "Q32C", "Q28"]
# Q28 is left out of both halves: it already compares preferred and actual hours.
WANT_ANSWERS = ["Q19", "Q33H", "Q32A", "Q32B", "Q32C"]
HAVE_ANSWERS = ["Q18", "Q34", "Q31A", "Q31B", "Q31C"]

BASELINE = "Background only"


def feature_sets(onet_columns: list[str]) -> dict[str, dict[str, list[str]]]:
    return {
        BASELINE: {},
        "Background + remote & advancement fit": {"fit": OUTLINE_FIT},
        "Background + all Gallup fit": {"fit": GALLUP_FIT},
        "Background + all fit (incl. O*NET advancement)": {"fit": ALL_FIT},
        "Background + 407 O*NET features (reference)": {"onet": onet_columns},
    }


def make_preprocessor(extra: dict[str, list[str]], scale: bool, dense: bool = False) -> ColumnTransformer:
    def numeric(indicator: bool = False) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=indicator))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    def categorical() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
            ]
        )

    parts = [
        ("numeric", numeric(), BACKGROUND_NUMERIC + extra.get("onet", [])),
        ("categorical", categorical(), BACKGROUND_CATEGORICAL),
    ]
    if extra.get("fit"):
        # The missing-value indicator lets the model tell "could not be
        # scored" (e.g. WFH not possible, no opinion on promotion) apart from
        # a typical score.
        parts.append(("fit", numeric(indicator=True), extra["fit"]))
    if extra.get("raw"):
        parts.append(("raw", categorical(), extra["raw"]))
    return ColumnTransformer(parts, sparse_threshold=0 if dense else 0.3)


def columns_of(extra: dict[str, list[str]]) -> list[str]:
    return BACKGROUND_NUMERIC + BACKGROUND_CATEGORICAL + [c for cols in extra.values() for c in cols]


class LogisticModel:
    name = "Logistic regression"

    def __init__(self, extra):
        self.model = Pipeline(
            [
                ("preprocessor", make_preprocessor(extra, scale=True)),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class ForestModel:
    """Random forest tuned with the same search as modeling/gallup_random_forest.py."""

    name = "Random forest"
    grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_leaf": [1, 5, 10],
    }

    def __init__(self, extra):
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(extra, scale=False)),
                ("classifier", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)),
            ]
        )
        self.search = RandomizedSearchCV(
            pipeline,
            self.grid,
            n_iter=8,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.search.fit(X, y)
        self.best_params = self.search.best_params_
        return self

    def predict_proba(self, X):
        return self.search.best_estimator_.predict_proba(X)[:, 1]


class KerasModel:
    """Small feed-forward network. 15% of the training rows are held out for early stopping."""

    name = "Keras neural network"

    def __init__(self, extra):
        self.preprocessor = make_preprocessor(extra, scale=True, dense=True)

    def fit(self, X, y):
        import keras
        import tensorflow as tf

        keras.utils.set_random_seed(SEED)
        tf.config.experimental.enable_op_determinism()
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
        self.prep = clone(self.preprocessor).fit(X_fit)
        A_fit = self.prep.transform(X_fit).astype("float32")
        A_val = self.prep.transform(X_val).astype("float32")

        self.model = keras.Sequential(
            [
                keras.Input(shape=(A_fit.shape[1],)),
                keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fit)
        history = self.model.fit(
            A_fit,
            np.asarray(y_fit),
            validation_data=(A_val, np.asarray(y_val)),
            epochs=200,
            batch_size=256,
            class_weight={0: weights[0], 1: weights[1]},
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)],
            verbose=0,
        )
        self.epochs = int(np.argmax(history.history["val_auc"])) + 1
        return self

    def predict_proba(self, X):
        return self.model.predict(self.prep.transform(X).astype("float32"), verbose=0, batch_size=4096).ravel()


ALGORITHMS = [LogisticModel, ForestModel, KerasModel]


def evaluate(y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))


def paired_delta(y, base_prob, new_prob, draws) -> dict:
    """AUC(new) - AUC(base) with a 95% paired bootstrap CI over test respondents."""
    deltas = []
    for sample in draws:
        if y[sample].min() == y[sample].max():
            continue
        deltas.append(roc_auc_score(y[sample], new_prob[sample]) - roc_auc_score(y[sample], base_prob[sample]))
    deltas = np.asarray(deltas)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "auc_improvement": roc_auc_score(y, new_prob) - roc_auc_score(y, base_prob),
        "ci_low": low,
        "ci_high": high,
        "share_of_draws_above_0": (deltas > 0).mean(),
    }


def permutation_importance(model, X, y, columns, groups=None) -> pd.DataFrame:
    """Mean ROC-AUC drop when a column (or group of columns) is shuffled on the test set."""
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y, model.predict_proba(X))
    groups = groups or {c: [c] for c in columns}
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)))
        rows.append({"feature": name, "auc_drop_mean": np.mean(drops), "auc_drop_sd": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    onet_columns = [c for c in data.columns if c.startswith("onet_")]
    y = data["satisfied"].to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=SEED)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Respondents: {len(data):,}  train {len(train_idx):,}  test {len(test_idx):,}  share satisfied {y.mean():.3f}")

    sets = feature_sets(onet_columns)
    draws = bootstrap_indices(len(test_idx))
    metrics, deltas, predictions = [], [], {"ENTITY_ID": data["ENTITY_ID"].to_numpy()[test_idx], "satisfied": y_test}
    fitted = {}

    for algorithm in ALGORITHMS:
        probs = {}
        for set_name, extra in sets.items():
            cols = columns_of(extra)
            model = algorithm(extra).fit(data.iloc[train_idx][cols], y_train)
            prob = model.predict_proba(data.iloc[test_idx][cols])
            probs[set_name] = prob
            predictions[f"{algorithm.name} | {set_name}"] = prob
            row = {"algorithm": algorithm.name, "feature_set": set_name, "n_features_in": len(cols), **evaluate(y_test, prob)}
            if isinstance(model, ForestModel):
                row["notes"] = str({k.split("__")[1]: v for k, v in model.best_params.items()})
            if isinstance(model, KerasModel):
                row["notes"] = f"best epoch {model.epochs}"
            metrics.append(row)
            fitted[(algorithm.name, set_name)] = model
            print(f"{algorithm.name:22s} {set_name:48s} ROC-AUC {row['roc_auc']:.4f}  PR-AUC {row['pr_auc']:.4f}")
        for set_name, prob in probs.items():
            if set_name != BASELINE:
                deltas.append({"algorithm": algorithm.name, "comparison": f"{set_name} vs {BASELINE}", **paired_delta(y_test, probs[BASELINE], prob, draws)})

    # Does the fit formula add anything beyond the raw answers it is built from?
    # And which half carries the signal: what people want, or what the job gives?
    # Logistic regression only.
    halves_probs = {}
    for label, extra in [
        ("Background + 'want' answers only", {"raw": WANT_ANSWERS}),
        ("Background + 'have' answers only", {"raw": HAVE_ANSWERS}),
        ("Background + raw want/have answers", {"raw": FIT_INPUTS}),
        ("Background + raw answers + all fit", {"raw": FIT_INPUTS, "fit": ALL_FIT}),
    ]:
        cols = columns_of(extra)
        model = LogisticModel(extra).fit(data.iloc[train_idx][cols], y_train)
        halves_probs[label] = model.predict_proba(data.iloc[test_idx][cols])
        metrics.append({"algorithm": LogisticModel.name, "feature_set": label, "n_features_in": len(cols), **evaluate(y_test, halves_probs[label])})
        print(f"{LogisticModel.name:22s} {label:48s} ROC-AUC {metrics[-1]['roc_auc']:.4f}")
    for label in ["Background + 'want' answers only", "Background + 'have' answers only", "Background + raw want/have answers"]:
        deltas.append(
            {
                "algorithm": LogisticModel.name,
                "comparison": f"{label} vs {BASELINE}",
                **paired_delta(y_test, predictions[f"{LogisticModel.name} | {BASELINE}"], halves_probs[label], draws),
            }
        )
    deltas.append(
        {
            "algorithm": LogisticModel.name,
            "comparison": "+ all fit vs raw want/have answers alone",
            **paired_delta(y_test, halves_probs["Background + raw want/have answers"], halves_probs["Background + raw answers + all fit"], draws),
        }
    )

    # Robustness: 5-fold cross-validated ROC-AUC for logistic regression on all respondents.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for set_name, extra in sets.items():
        cols = columns_of(extra)
        scores = cross_val_score(LogisticModel(extra).model, data[cols], y, cv=cv, scoring="roc_auc")
        for row in metrics:
            if row["algorithm"] == LogisticModel.name and row["feature_set"] == set_name:
                row["cv5_roc_auc_mean"], row["cv5_roc_auc_sd"] = scores.mean(), scores.std()

    # Permutation importance for the full fit model of each algorithm (test set).
    full = "Background + all fit (incl. O*NET advancement)"
    cols = columns_of(sets[full])
    importance = []
    for algorithm in ALGORITHMS:
        table = permutation_importance(fitted[(algorithm.name, full)], data.iloc[test_idx][cols].reset_index(drop=True), y_test, cols)
        importance.append(table.assign(algorithm=algorithm.name))
        print(f"\nPermutation importance, {algorithm.name}:")
        print(table.round(4).to_string(index=False))
    groups = {"all 8 background features": BACKGROUND_NUMERIC + BACKGROUND_CATEGORICAL, "all 7 fit features": ALL_FIT}
    for algorithm in ALGORITHMS:
        table = permutation_importance(fitted[(algorithm.name, full)], data.iloc[test_idx][cols].reset_index(drop=True), y_test, cols, groups)
        importance.append(table.assign(algorithm=algorithm.name))
        print(f"\nGrouped permutation importance, {algorithm.name}:")
        print(table.round(4).to_string(index=False))

    metrics = pd.DataFrame(metrics)
    deltas = pd.DataFrame(deltas)
    metrics.to_csv(OUTPUT_DIR / "fit_model_metrics.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "fit_auc_improvement.csv", index=False)
    pd.concat(importance).to_csv(OUTPUT_DIR / "fit_permutation_importance.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "fit_test_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nTest-set metrics (threshold 0.5):")
    print(metrics.drop(columns=["notes"], errors="ignore").round(4).to_string(index=False))
    print(f"\nROC-AUC improvement over {BASELINE}, 95% paired bootstrap CI ({BOOTSTRAP_DRAWS} draws):")
    print(deltas.round(4).to_string(index=False))
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
