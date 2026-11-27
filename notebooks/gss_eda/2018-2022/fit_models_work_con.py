"""O*NET occupation scores plus four quality-of-work-life items.

Uses the same split, weights, and metrics as baseline_models.py. Rows missing
an O*NET score are dropped. The O*NET model is fit again on that sample, then
the four work items are added.

WKVSFAM and OVERWORK stay on the GSS scale, so 4 means the job rarely
interferes with family life and the person is not overworked. WKDECIDE and
JOBSECOK are reversed with 5 minus the code, so 4 means the person often
takes part in decisions and job security is very good.
"""

from __future__ import annotations

import pandas as pd

from baseline_models import INCOME, NUMERIC, DATA_PATH, baseline_models, fit_and_report, lightgbm_model
from fit_models_ONET import ONET_FEATURES, matched_sample


WORK_ITEMS = ["WKVSFAM", "OVERWORK", "WKDECIDE", "JOBSECOK"]
# Higher raw codes on these two mean less of the thing the question names.
FLIP_ITEMS = ["WKDECIDE", "JOBSECOK"]


def orient_work_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep codes 1 through 4 and point every item toward a better situation."""
    out = frame.copy()
    for column in WORK_ITEMS:
        values = pd.to_numeric(out[column], errors="coerce")
        kept = values.where(values.between(1, 4))
        if column in FLIP_ITEMS:
            kept = 5 - kept
        out[column] = kept
    return out


def main() -> None:
    frame = orient_work_items(pd.read_csv(DATA_PATH))
    matched = matched_sample(frame)
    dropped = len(frame) - len(matched)
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    print(f"O*NET match: {len(matched):,} rows. Dropped {dropped:,} rows missing a score.")
    print("Work items, share answered after orienting codes 1-4:")
    for column in WORK_ITEMS:
        print(f"  {column}: {matched[column].notna().mean():.1%}")
    print("Same logistic, tree, and forest settings as baseline_models.py.")
    print("LightGBM: 200 trees, learning rate 0.05, at least 20 training rows per leaf.")
    onet = NUMERIC + [INCOME] + ONET_FEATURES
    feature_sets = [
        ("Demographics + own earnings + O*NET", onet),
        ("Demographics + own earnings + O*NET + work conditions", onet + WORK_ITEMS),
    ]
    models = baseline_models() + [("LightGBM", lightgbm_model(), False)]
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
