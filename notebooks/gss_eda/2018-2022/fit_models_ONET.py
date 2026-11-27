"""Background models plus O*NET occupation scores, GSS 2018 and 2022.

Uses the same split, weights, and metrics as baseline_models.py, plus LightGBM.
Rows missing an O*NET occupation score are dropped. The earnings model is
fit again on that smaller sample so a change in AUC comes from the scores.
"""

from __future__ import annotations

import pandas as pd

from baseline_models import INCOME, NUMERIC, DATA_PATH, baseline_models, fit_and_report, lightgbm_model


# Occupation scores already joined on OCC10. wv1 through wv6 are work values.
# si1 is Reading Comprehension importance.
ONET_FEATURES = [
    "wv1val_mean",
    "wv2val_mean",
    "wv3val_mean",
    "wv4val_mean",
    "wv5val_mean",
    "wv6val_mean",
    "si1val_mean",
]


def matched_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep respondents whose occupation joined to every O*NET score."""
    return frame.dropna(subset=ONET_FEATURES).copy()


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    matched = matched_sample(frame)
    dropped = len(frame) - len(matched)
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    print(f"O*NET match: {len(matched):,} rows. Dropped {dropped:,} rows missing a score.")
    print("Same logistic, tree, and forest settings as baseline_models.py.")
    print("LightGBM: 200 trees, learning rate 0.05, at least 20 training rows per leaf.")
    background = NUMERIC + [INCOME]
    feature_sets = [
        ("Demographics + own earnings", background),
        ("Demographics + own earnings + O*NET", background + ONET_FEATURES),
    ]
    models = baseline_models() + [
        (
            "LightGBM",
            lightgbm_model(),
            False,
        )
    ]
    for model_name, model, scale in models:
        for feature_label, numeric in feature_sets:
            fit_and_report(
                frame=matched,
                label=f"{model_name}: {feature_label}",
                numeric=numeric,
                model=model,
                scale=scale,
            )


if __name__ == "__main__":
    main()
