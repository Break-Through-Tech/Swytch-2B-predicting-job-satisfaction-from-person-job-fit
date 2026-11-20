"""Track B: which way of scoring fit works best?

The Outline's fit scores are two-sided: getting more than you want counts as
the same mismatch as getting less. In the data only one side hurts. People
who get more work-from-home days or more say at work than they want are about
as satisfied as people with a perfect match; people who get less are not.
This script compares, with logistic regression on the same split as
gallup_fit_models.py:

    two-sided fit   the Outline formulas (engineer_fit_features.py)
    unmet need      one-sided: how far the job falls short of what the
                    person wants (0 = need met, 1 = largest shortfall)
    raw answers     the want and have answers themselves, one-hot encoded

Usage (from the repository root, after engineer_fit_features.py):
    python modeling/gallup_fit_scoring_comparison.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from gallup_fit_models import (
    ALL_FIT,
    BASELINE,
    DATA_PATH,
    FIT_INPUTS,
    OUTPUT_DIR,
    SEED,
    LogisticModel,
    bootstrap_indices,
    columns_of,
    evaluate,
    paired_delta,
)

UNMET = [
    "remote_unmet",
    "advancement_unmet",
    "voice_unmet_pay",
    "voice_unmet_conditions",
    "voice_unmet_tech",
    "hours_want_more",
    "hours_want_fewer",
    "advancement_onet_unmet",
]


def in_range(values: pd.Series, low: float, high: float) -> pd.Series:
    return values.where(values.between(low, high))


def add_unmet_need(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    wanted, actual = in_range(out["Q19"], 0, 7), in_range(out["Q18"], 0, 7)
    out["remote_unmet"] = (wanted - actual).clip(lower=0) / 7

    # Unmet only when the person wants advancement and it is not offered.
    q33h, q34 = out["Q33H"], out["Q34"]
    scored = q33h.isin([1, 2, 3, 4, 5]) & q34.isin([1, 2])
    out["advancement_unmet"] = (q33h.isin([1, 2]) & q34.eq(2)).astype(float).where(scored)

    # Q31 = influence they have, Q32 = influence they should have; 1 a lot .. 4 none.
    for letter, name in [("A", "pay"), ("B", "conditions"), ("C", "tech")]:
        have, should = in_range(out[f"Q31{letter}"], 1, 4), in_range(out[f"Q32{letter}"], 1, 4)
        out[f"voice_unmet_{name}"] = (have - should).clip(lower=0) / 3

    q28 = in_range(out["Q28"], 1, 3)
    out["hours_want_more"] = q28.eq(1).astype(float).where(q28.notna())
    out["hours_want_fewer"] = q28.eq(3).astype(float).where(q28.notna())

    interest = 1 - (in_range(q33h, 1, 5) - 1) / 4
    offered = (in_range(out["onet_wv3val_mean"], 1, 7) - 1) / 6
    out["advancement_onet_unmet"] = (interest - offered).clip(lower=0)
    return out


def main() -> None:
    data = add_unmet_need(pd.read_csv(DATA_PATH, low_memory=False))
    y = data["satisfied"].to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=SEED)
    draws = bootstrap_indices(len(test_idx))

    sets = {
        BASELINE: {},
        "Two-sided fit (Outline formulas)": {"fit": ALL_FIT},
        "Unmet need (one-sided)": {"fit": UNMET},
        "Two-sided fit + unmet need": {"fit": ALL_FIT + UNMET},
        "Raw want/have answers": {"raw": FIT_INPUTS},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, probs = [], {}
    for name, extra in sets.items():
        cols = columns_of(extra)
        model = LogisticModel(extra).fit(data.iloc[train_idx][cols], y[train_idx])
        probs[name] = model.predict_proba(data.iloc[test_idx][cols])
        cv_auc = cross_val_score(LogisticModel(extra).model, data[cols], y, cv=cv, scoring="roc_auc")
        rows.append({"feature_set": name, **evaluate(y[test_idx], probs[name]), "cv5_roc_auc_mean": cv_auc.mean(), "cv5_roc_auc_sd": cv_auc.std()})

    comparisons = [
        ("Two-sided fit (Outline formulas)", BASELINE),
        ("Unmet need (one-sided)", BASELINE),
        ("Unmet need (one-sided)", "Two-sided fit (Outline formulas)"),
        ("Two-sided fit + unmet need", "Unmet need (one-sided)"),
        ("Raw want/have answers", "Unmet need (one-sided)"),
    ]
    deltas = pd.DataFrame(
        [{"comparison": f"{new} vs {base}", **paired_delta(y[test_idx], probs[base], probs[new], draws)} for new, base in comparisons]
    )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "fit_scoring_comparison.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "fit_scoring_comparison_deltas.csv", index=False)

    pd.set_option("display.width", 250)
    print("Logistic regression, same 80/20 split; test-set metrics at threshold 0.5:")
    print(results[["feature_set", "roc_auc", "pr_auc", "f1", "balanced_accuracy", "cv5_roc_auc_mean", "cv5_roc_auc_sd"]].round(4).to_string(index=False))
    print("\nROC-AUC differences, 95% paired bootstrap CI:")
    print(deltas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
