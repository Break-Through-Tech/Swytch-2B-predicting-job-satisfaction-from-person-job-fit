"""Add person-job fit features to the Gallup + O*NET modeling dataset.

A fit feature compares what a respondent wants with what their job gives them.
Only Gallup questions that ask about a preference can be used, and there are
four of them: Q19 (work-from-home days wanted), Q33H (interest in
advancement), Q32A-C (influence the worker believes they should have) and
Q28 (hours/pay preference).

Features (all 0-1, higher = closer match, NaN when it cannot be scored):

    remote_fit            1 - |Q19 - Q18| / 7. Wanted vs actual days working
                          exclusively from home. Q19 = 8 ("not possible for
                          your job") is NaN.
    advancement_fit       1 if wants advancement (Q33H 1-2) and the employer
                          offers it (Q34 = 1), or does not want it (Q33H 4-5)
                          and it is not offered (Q34 = 2); 0 for the two
                          mismatches; NaN for neutral, not sure, not
                          applicable or missing.
    voice_fit_pay         1 - |Q32 - Q31| / 3, for compensation (A),
    voice_fit_conditions  working conditions (B) and new technologies (C).
    voice_fit_tech        Q31 = influence the worker has, Q32 = influence they
                          believe they should have (1 a lot .. 4 none; 5 "not
                          applicable" is NaN). This is the "voice gap" in the
                          American Job Quality Study report (pages 30 and 45).
    hours_fit             1 if the worker prefers the same hours and pay
                          (Q28 = 2), 0 if they want more or fewer.
    advancement_onet_fit  1 - |interest in advancement - O*NET Recognition|,
                          both scaled 0-1. O*NET's Recognition work value
                          (wv3) covers advancement, leadership and prestige,
                          on a 1-7 extent scale. It is an average for the
                          respondent's broad occupation group.

No other O*NET fit is built. The other Gallup job items (Q33A-J, Q25,
Q36) describe the job as experienced, not what the person wants, so comparing
them with O*NET would compare two job descriptions, not a person with a job.

Usage (from the repository root):
    python notebooks/gallup_eda/engineer_fit_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODEL_DATA = HERE / "output" / "gallup_fsu_pre_model.csv"
GALLUP_RAW = PROJECT_ROOT / "data" / "raw" / "JFF_formatted_public_release_values.csv"
OUTPUT = HERE / "output" / "gallup_fsu_fit_model.csv"

# Raw Gallup questions not already in the modeling dataset.
EXTRA_QUESTIONS = ["Q18", "Q19", "Q33H", "Q34"]

FIT_FEATURES = [
    "remote_fit",
    "advancement_fit",
    "voice_fit_pay",
    "voice_fit_conditions",
    "voice_fit_tech",
    "hours_fit",
    "advancement_onet_fit",
]

ONET_RECOGNITION = "onet_wv3val_mean"


def valid(values: pd.Series, low: float, high: float) -> pd.Series:
    """Keep answers inside the documented range; everything else becomes NaN."""
    values = pd.to_numeric(values, errors="coerce")
    return values.where(values.between(low, high))


def scale_01(values: pd.Series, scale_min: float, scale_max: float) -> pd.Series:
    return (values - scale_min) / (scale_max - scale_min)


def remote_fit(q18: pd.Series, q19: pd.Series) -> pd.Series:
    actual = valid(q18, 0, 7)
    wanted = valid(q19, 0, 7)  # 8 = not possible for your job
    return 1 - (wanted - actual).abs() / 7


def advancement_fit(q33h: pd.Series, q34: pd.Series) -> pd.Series:
    q33h = pd.to_numeric(q33h, errors="coerce")
    q34 = pd.to_numeric(q34, errors="coerce")
    wants = q33h.isin([1, 2])
    does_not_want = q33h.isin([4, 5])
    offered = q34.eq(1)
    not_offered = q34.eq(2)
    fit = pd.Series(np.nan, index=q33h.index)
    fit[(wants & offered) | (does_not_want & not_offered)] = 1.0
    fit[(wants & not_offered) | (does_not_want & offered)] = 0.0
    return fit


def voice_fit(have: pd.Series, should_have: pd.Series) -> pd.Series:
    have = valid(have, 1, 4)  # 5 = not applicable
    should_have = valid(should_have, 1, 4)
    return 1 - (should_have - have).abs() / 3


def hours_fit(q28: pd.Series) -> pd.Series:
    q28 = valid(q28, 1, 3)
    return q28.eq(2).astype(float).where(q28.notna())


def advancement_onet_fit(q33h: pd.Series, recognition: pd.Series) -> pd.Series:
    # Q33H: 1 strongly agree (interested) .. 5 strongly disagree; 6-7 are not a scale point.
    interest = 1 - scale_01(valid(q33h, 1, 5), 1, 5)
    offered = scale_01(valid(recognition, 1, 7), 1, 7)
    return 1 - (interest - offered).abs()


def add_fit_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["remote_fit"] = remote_fit(out["Q18"], out["Q19"])
    out["advancement_fit"] = advancement_fit(out["Q33H"], out["Q34"])
    out["voice_fit_pay"] = voice_fit(out["Q31A"], out["Q32A"])
    out["voice_fit_conditions"] = voice_fit(out["Q31B"], out["Q32B"])
    out["voice_fit_tech"] = voice_fit(out["Q31C"], out["Q32C"])
    out["hours_fit"] = hours_fit(out["Q28"])
    out["advancement_onet_fit"] = advancement_onet_fit(out["Q33H"], out[ONET_RECOGNITION])
    return out


def merge_questions(model: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    for name, frame in [("modeling dataset", model), ("Gallup file", raw)]:
        if not frame["ENTITY_ID"].is_unique:
            raise ValueError(f"ENTITY_ID is not unique in the {name}")
    extra = raw[["ENTITY_ID", *EXTRA_QUESTIONS]].copy()
    extra[EXTRA_QUESTIONS] = extra[EXTRA_QUESTIONS].apply(pd.to_numeric, errors="coerce").replace(-98, np.nan)
    merged = model.merge(extra, on="ENTITY_ID", how="left", validate="one_to_one")
    if len(merged) != len(model):
        raise ValueError(f"Merge changed the row count: {len(model)} -> {len(merged)}")
    missing_ids = merged[EXTRA_QUESTIONS].isna().all(axis=1).sum()
    if missing_ids:
        print(f"Warning: {missing_ids} respondents have none of {EXTRA_QUESTIONS}")
    return merged


def summarize(data: pd.DataFrame) -> None:
    print(f"\nRows: {len(data):,}  (share satisfied {data['satisfied'].mean():.3f})")
    print("\nFit feature coverage and share satisfied by value:")
    for feature in FIT_FEATURES:
        values = data[feature]
        print(
            f"\n{feature}: scored for {values.notna().mean():.1%} "
            f"(min {values.min():.2f}, max {values.max():.2f}, mean {values.mean():.3f})"
        )
        groups = pd.cut(values, [-0.01, 0.5, 0.75, 0.99, 1.0], labels=["0-0.5", "0.5-0.75", "0.75-0.99", "1 (match)"])
        groups = groups.astype("string").fillna("not scored")
        table = data.groupby(groups)["satisfied"].agg(respondents="size", share_satisfied="mean")
        print(table.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-data", type=Path, default=MODEL_DATA)
    parser.add_argument("--gallup", type=Path, default=GALLUP_RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    model = pd.read_csv(args.model_data, low_memory=False)
    raw = pd.read_csv(args.gallup, encoding="cp1252", low_memory=False)
    print(f"Modeling dataset: {model.shape}, Gallup file: {raw.shape}")

    data = add_fit_features(merge_questions(model, raw))
    for feature in FIT_FEATURES:
        assert data[feature].dropna().between(0, 1).all(), feature
    assert len(data) == len(model) and data["ENTITY_ID"].is_unique
    assert data[model.columns].equals(model), "original columns changed"

    summarize(data)
    data.to_csv(args.output, index=False)
    print(f"\nSaved {data.shape} to {args.output}")


if __name__ == "__main__":
    main()
