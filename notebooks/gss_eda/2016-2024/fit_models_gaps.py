"""Baseline plus O*NET means, plus pay, hours, and education gaps.

One specification. Demographics, REALRINC, HOURS, and weeks worked stay in
the model. The seven O*NET means are added, then the gaps beside them.

income_gap is log(REALRINC) minus the training-set occupation median of
log(REALRINC). hours_gap is HOURS minus the training-set occupation mean.
income_percentile is the person's place in the training-set earnings
distribution of the same occupation. educ_gap maps DEGREE onto the O*NET job
zone scale and subtracts jz_mean: less than high school is 1, high school is
2, junior college is 3, bachelor is 4, and graduate is 5. A positive educ_gap
is more schooling than that occupation's job zone.

income_gap_x_achievement and income_gap_x_recognition multiply the pay gap
by Achievement and Recognition after those two scores are centered on the
training mean. The income_gap coefficient is then the slope at the average
score. Occupation medians, means, percentiles, and those centers use the
training split only. Job zone is an O*NET occupation score, so it is joined
for every row.

tests if in jobs that value achievement or recognition there is more significance regarding the pay gap when it comes to job satisfaction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_models import (
    CATEGORICAL,
    DATA_PATH,
    HOURS,
    INCOME,
    NUMERIC,
    RANDOM_STATE,
    REFERENCE_THRESHOLD,
    SELF_EMPLOYED,
    TARGET,
    TEST_SIZE,
    WEIGHT,
    WEEKS_WORKED,
    baseline_models,
    confusion_table,
    feature_names,
    lightgbm_model,
    make_pipeline,
    metric_intervals,
    predicted_labels,
    print_coefficient_ends,
    select_threshold,
    threshold_metrics,
    top_drivers,
    weighted_auc,
    weighted_log_loss,
)
from onet_gss_merge_pipeline import census_code


ONET_PATH = Path(__file__).resolve().parent / "onet_data" / "FSU_data.csv"
ONET_FEATURES = [
    "wv1val_mean",
    "wv2val_mean",
    "wv3val_mean",
    "wv4val_mean",
    "wv5val_mean",
    "wv6val_mean",
    "si1val_mean",
]
ACHIEVEMENT = "wv1val_mean"
RECOGNITION = "wv3val_mean"
JOB_ZONE = "jz_mean"
INCOME_GAP = "income_gap"
HOURS_GAP = "hours_gap"
EDUC_GAP = "educ_gap"
INCOME_PERCENTILE = "income_percentile"
INCOME_GAP_X_ACHIEVEMENT = "income_gap_x_achievement"
INCOME_GAP_X_RECOGNITION = "income_gap_x_recognition"
LOG_INCOME = "log_realrinc"
OCC_MEDIAN_LOG_INCOME = "occ_median_log_realrinc"
OCC_MEAN_HOURS = "occ_mean_hours"
# GSS DEGREE 0 through 4 lined up with O*NET job zones 1 through 5.
DEGREE_TO_ZONE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
GAP_FEATURES = [
    INCOME_GAP,
    HOURS_GAP,
    EDUC_GAP,
    INCOME_PERCENTILE,
    INCOME_GAP_X_ACHIEVEMENT,
    INCOME_GAP_X_RECOGNITION,
]
# Smaller occupation cells make the median, mean, and percentile jump around.
MIN_OCC_COUNT = 8
WORK_CATEGORIES = CATEGORICAL + [SELF_EMPLOYED]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def load_job_zone() -> pd.DataFrame:
    """O*NET job zone by Census 2010 occupation code."""
    onet = pd.read_csv(ONET_PATH, usecols=["census_num_2010", JOB_ZONE])
    onet["OCC10"] = census_code(onet["census_num_2010"])
    onet[JOB_ZONE] = pd.to_numeric(onet[JOB_ZONE], errors="coerce")
    onet = onet.dropna(subset=["OCC10"])
    return onet.groupby("OCC10", as_index=False)[JOB_ZONE].mean()


def occupation_norms(train: pd.DataFrame) -> pd.DataFrame:
    """Training-set pay median and hours mean for each occupation."""
    rows = []
    usable = train.dropna(subset=["OCC10"])
    for occupation, group in usable.groupby("OCC10"):
        income = group.loc[group[INCOME].gt(0) & group[WEIGHT].gt(0), [INCOME, WEIGHT]]
        hours = group.loc[group[HOURS].notna() & group[WEIGHT].gt(0), [HOURS, WEIGHT]]
        income_norm = np.nan
        hours_norm = np.nan
        if len(income) >= MIN_OCC_COUNT:
            income_norm = weighted_median(np.log(income[INCOME].to_numpy()), income[WEIGHT].to_numpy())
        if len(hours) >= MIN_OCC_COUNT:
            hours_norm = float(np.average(hours[HOURS].to_numpy(), weights=hours[WEIGHT].to_numpy()))
        rows.append(
            {
                "OCC10": occupation,
                OCC_MEDIAN_LOG_INCOME: income_norm,
                OCC_MEAN_HOURS: hours_norm,
            }
        )
    return pd.DataFrame(rows)


def income_reference(train: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sorted training log earnings and survey weights, by occupation."""
    references = {}
    usable = train.loc[train["OCC10"].notna() & train[INCOME].gt(0) & train[WEIGHT].gt(0), ["OCC10", INCOME, WEIGHT]]
    for occupation, group in usable.groupby("OCC10"):
        if len(group) < MIN_OCC_COUNT:
            continue
        log_income = np.log(group[INCOME].to_numpy(dtype=float))
        weights = group[WEIGHT].to_numpy(dtype=float)
        order = np.argsort(log_income)
        references[str(occupation)] = (log_income[order], np.cumsum(weights[order]))
    return references


def income_percentile(frame: pd.DataFrame, references: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.Series:
    """Share of training weight in the occupation with log earnings at or below this row."""
    percentile = np.full(len(frame), np.nan)
    occupations = frame["OCC10"].astype("string")
    income = frame[INCOME].to_numpy(dtype=float)
    for occupation, (log_income, cumulative) in references.items():
        mask = occupations.eq(occupation).to_numpy() & np.isfinite(income) & (income > 0)
        if not mask.any():
            continue
        positions = np.searchsorted(log_income, np.log(income[mask]), side="right") - 1
        share = np.zeros(int(mask.sum()))
        known = positions >= 0
        share[known] = cumulative[positions[known]] / cumulative[-1]
        percentile[mask] = share
    return pd.Series(percentile, index=frame.index)


def add_gaps(
    frame: pd.DataFrame,
    norms: pd.DataFrame,
    references: dict[str, tuple[np.ndarray, np.ndarray]],
    achievement_center: float,
    recognition_center: float,
) -> pd.DataFrame:
    """Attach training occupation norms and the person-minus-occupation gaps."""
    out = frame.merge(norms, on="OCC10", how="left")
    log_income = pd.Series(np.nan, index=out.index, dtype="float64")
    positive_income = out[INCOME].gt(0)
    log_income.loc[positive_income] = np.log(out.loc[positive_income, INCOME].to_numpy())
    out[LOG_INCOME] = log_income
    out[INCOME_GAP] = out[LOG_INCOME] - out[OCC_MEDIAN_LOG_INCOME]
    out[HOURS_GAP] = out[HOURS] - out[OCC_MEAN_HOURS]
    degree_zone = out["DEGREE"].map(DEGREE_TO_ZONE)
    out[EDUC_GAP] = degree_zone - out[JOB_ZONE]
    out[INCOME_PERCENTILE] = income_percentile(out, references)
    out[INCOME_GAP_X_ACHIEVEMENT] = out[INCOME_GAP] * (out[ACHIEVEMENT] - achievement_center)
    out[INCOME_GAP_X_RECOGNITION] = out[INCOME_GAP] * (out[RECOGNITION] - recognition_center)
    return out


def report_fit(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    label: str,
    numeric: list[str],
    model,
    scale: bool,
) -> None:
    pipeline = make_pipeline(numeric, WORK_CATEGORIES, model, scale)
    pipeline.fit(x_train[numeric + WORK_CATEGORIES], y_train, model__sample_weight=w_train)
    train_scores = pipeline.predict_proba(x_train[numeric + WORK_CATEGORIES])[:, 1]
    scores = pipeline.predict_proba(x_test[numeric + WORK_CATEGORIES])[:, 1]
    threshold = select_threshold(y_train, train_scores, w_train)
    auc = weighted_auc(y_test, scores, w_test)
    precision, recall, f1 = threshold_metrics(y_test, scores, w_test, threshold)
    low, high = metric_intervals(y_test, scores, w_test, threshold=threshold)
    matrix = confusion_table(y_test, scores, threshold)
    always_satisfied = np.unique(predicted_labels(scores, REFERENCE_THRESHOLD)).size == 1

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
    print(f"Survey-weighted precision: {precision:.3f}  (95% interval {low[1]:.3f} to {high[1]:.3f})")
    print(f"Survey-weighted recall: {recall:.3f}  (95% interval {low[2]:.3f} to {high[2]:.3f})")
    print(f"Survey-weighted F1: {f1:.3f}  (95% interval {low[3]:.3f} to {high[3]:.3f})")
    print(f"Survey-weighted test log loss: {weighted_log_loss(y_test, scores, w_test):.3f}")
    print("Confusion matrix on the test set (row counts):")
    print(matrix.to_string())
    estimator = pipeline.named_steps["model"]
    names = feature_names(pipeline, numeric, WORK_CATEGORIES)
    if hasattr(estimator, "coef_"):
        print_coefficient_ends(estimator, names)
        effects = pd.DataFrame({"feature": names, "log_odds": estimator.coef_.ravel()})
        gap_effects = effects[effects["feature"].isin([INCOME, HOURS, *GAP_FEATURES])]
        print("Pay, hours, and education-gap coefficients:")
        print(gap_effects.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return
    heading, drivers = top_drivers(estimator, names)
    print(heading)
    print(drivers.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    frame["OCC10"] = census_code(frame["OCC10"])
    frame = frame.merge(load_job_zone(), on="OCC10", how="left")
    matched = frame.dropna(subset=ONET_FEATURES).copy()
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    print(f"O*NET match: {len(matched):,} rows. Dropped {len(frame) - len(matched):,} rows.")
    missing_zone = int(matched[JOB_ZONE].isna().sum())
    print(f"Rows missing an O*NET job zone: {missing_zone:,}.")
    print("One model: demographics, earnings, hours, O*NET means, and the gaps.")
    print("LightGBM: 200 trees, learning rate 0.05, at least 20 training rows per leaf.")

    train, test = train_test_split(
        matched,
        test_size=TEST_SIZE,
        stratify=matched[TARGET],
        random_state=RANDOM_STATE,
    )
    norms = occupation_norms(train)
    references = income_reference(train)
    achievement_center = float(np.average(train[ACHIEVEMENT], weights=train[WEIGHT]))
    recognition_center = float(np.average(train[RECOGNITION], weights=train[WEIGHT]))
    train = add_gaps(train, norms, references, achievement_center, recognition_center)
    test = add_gaps(test, norms, references, achievement_center, recognition_center)
    usable_income = int(norms[OCC_MEDIAN_LOG_INCOME].notna().sum())
    usable_hours = int(norms[OCC_MEAN_HOURS].notna().sum())
    print(
        f"Training occupations with at least {MIN_OCC_COUNT} people: "
        f"{usable_income:,} for pay, {usable_hours:,} for hours, {len(references):,} for pay rank."
    )
    print(
        "Training educ_gap mean "
        f"{train[EDUC_GAP].mean():.3f}. "
        f"Share above the occupation job zone: {train[EDUC_GAP].gt(0).mean():.3f}."
    )

    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()
    w_train = train[WEIGHT].to_numpy()
    w_test = test[WEIGHT].to_numpy()
    numeric = NUMERIC + [INCOME, HOURS, WEEKS_WORKED] + ONET_FEATURES + GAP_FEATURES
    models = baseline_models() + [("LightGBM", lightgbm_model(), False)]
    for model_name, model, scale in models:
        report_fit(
            train,
            test,
            y_train,
            y_test,
            w_train,
            w_test,
            f"{model_name}: Baseline plus O*NET means and gaps",
            numeric,
            model,
            scale,
        )


if __name__ == "__main__":
    main()
