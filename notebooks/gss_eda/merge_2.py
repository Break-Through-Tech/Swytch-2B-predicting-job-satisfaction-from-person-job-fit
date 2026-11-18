"""Build the GSS 2006/2012/2014 modeling table and link it to O*NET.

Reads the planned columns from the 2006, 2012, and 2014 extract. Keeps people
who are currently working, gave a valid job-satisfaction answer, have a civilian
occupation, answered all five job-preference ranks, and have a positive survey
weight. Then joins the FSU Census 2010-O*NET 26.1 scores on the Census 2010
occupation code and drops rows with no O*NET match, so every model sees the
same rows. Predictor gaps stay missing. Categorical codes stay numeric so
encoding can happen inside the training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from select_job_features import SOURCE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSS_PATH = PROJECT_ROOT / "data" / "interim" / "gss_2006_2012_2014_all_cols.parquet"
ONET_PATH = Path(__file__).resolve().parent / "onet_data" / "FSU_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "2merged_onet_gss_dataset.csv"

TARGET_YEARS = [2006, 2012, 2014]
WEIGHT = "WTSSNRPS"
# Worker work values: each respondent ranks five job traits from 1 (most
# important) to 5 (fifth). These are the worker side of person-job fit.
PREFERENCE_RANKS = ["JOBINC", "JOBSEC", "JOBHOUR", "JOBPROMO", "JOBMEANS"]
# Rows with no O*NET match have no fit score. Dropping them keeps the baseline
# and fit models on identical rows. Set to False to keep them for inspection.
DROP_UNMATCHED_ONET = True

# Occupation scores for the later fit models. The baseline model does not use them.
# wv1 Achievement, wv2 Working Conditions, wv3 Recognition, wv4 Relationships,
# wv5 Support, wv6 Independence. si1 is Reading Comprehension importance.
ONET_FEATURES = [
    "wv1val_mean",
    "wv2val_mean",
    "wv3val_mean",
    "wv4val_mean",
    "wv5val_mean",
    "wv6val_mean",
    "si1val_mean",
]
EMPLOYED_WRKSTAT = {1, 2, 3}
VALID_SATJOB = {1, 2, 3, 4}

OUTPUT_COLUMNS = [
    "YEAR",
    "ID",
    "BALLOT",
    "WTSSNRPS",
    "SATJOB",
    "SATJOB_BINARY",
    "AGE",
    "SEX",
    "RACE",
    "MARITAL",
    "DEGREE",
    "REGION",
    "REALINC",
    "REALRINC",
    "WRKSTAT",
    "WRKSLF",
    "WEEKSWRK",
    "HOURS",
    "HRS1",
    "HRS2",
    "JOBLOSE",
    "JOBFIND",
    "WRKSCHED",
    "WRKHOME",
    "YEARSJOB",
    "WRKTYPE",
    "OCC10",
    *PREFERENCE_RANKS,
    *ONET_FEATURES,
]

INTEGER_COLUMNS = [
    "YEAR",
    "ID",
    "BALLOT",
    "SATJOB",
    "SATJOB_BINARY",
    "AGE",
    "SEX",
    "RACE",
    "MARITAL",
    "DEGREE",
    "REGION",
    "WRKSTAT",
    "WRKSLF",
    "WEEKSWRK",
    "HOURS",
    "HRS1",
    "HRS2",
    "JOBLOSE",
    "JOBFIND",
    "WRKSCHED",
    "WRKHOME",
    "YEARSJOB",
    "WRKTYPE",
    *PREFERENCE_RANKS,
]


def census_code(series: pd.Series) -> pd.Series:
    """Turn 1020.0 and '1020.0' into the same '1020' key the O*NET file uses."""
    numeric = pd.to_numeric(series, errors="coerce").round()
    return numeric.astype("Int64").astype("string")


def as_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def load_gss() -> pd.DataFrame:
    columns = [name for name in SOURCE_COLUMNS if name != "ID_"]
    columns += [name for name in PREFERENCE_RANKS if name not in columns]
    gss = pd.read_parquet(GSS_PATH, columns=columns)
    gss = gss[gss["YEAR"].isin(TARGET_YEARS)].copy()
    return gss


def modeling_sample(gss: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep current workers with a real satisfaction answer, a civilian
    occupation, all five preference ranks, and a usable survey weight."""
    employed = gss["WRKSTAT"].isin(EMPLOYED_WRKSTAT)
    valid_satjob = gss["SATJOB"].isin(VALID_SATJOB)
    occ = pd.to_numeric(gss["OCC10"], errors="coerce")
    valid_occ = occ.notna() & (occ > 0) & (occ < 9800)
    ranks = gss[PREFERENCE_RANKS].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    ranks = ranks.where((ranks >= 1) & (ranks <= 5))
    has_ranks = ranks.notna().all(axis=1)
    valid_weight = pd.to_numeric(gss[WEIGHT], errors="coerce").gt(0)

    steps = [
        ("respondents in target years", pd.Series(True, index=gss.index)),
        ("currently working", employed),
        ("valid SATJOB", valid_satjob),
        ("civilian OCC10", valid_occ),
        ("all five preference ranks", has_ranks),
        ("positive WTSSNRPS", valid_weight),
    ]
    year = gss["YEAR"]
    running = pd.Series(True, index=gss.index)
    counts = {}
    for label, mask in steps:
        running &= mask
        counts[label] = running.groupby(year).sum()
    funnel = pd.DataFrame(counts).T.reindex(columns=TARGET_YEARS).fillna(0).astype(int)

    # Each requirement on its own, among respondents who have all five ranks.
    # This shows which filter removes rank holders (not cumulative).
    holders = pd.DataFrame(
        {
            "have all five ranks": has_ranks,
            "rank holders who are working": has_ranks & employed,
            "rank holders with valid SATJOB": has_ranks & valid_satjob,
            "rank holders with civilian OCC10": has_ranks & valid_occ,
        }
    )
    holders = holders.groupby(year).sum().T.reindex(columns=TARGET_YEARS).fillna(0).astype(int)
    out = gss.loc[running].copy()
    out[PREFERENCE_RANKS] = ranks.loc[out.index]

    out["SEX"] = out["SEX"].where(out["SEX"].isin([1, 2]))
    out["RACE"] = out["RACE"].where(out["RACE"].isin([1, 2, 3]))
    out["MARITAL"] = out["MARITAL"].where(out["MARITAL"].isin([1, 2, 3, 4, 5]))
    out["DEGREE"] = out["DEGREE"].where(out["DEGREE"].isin([0, 1, 2, 3, 4]))
    out["REGION"] = out["REGION"].where(out["REGION"].between(1, 9))
    out["WEEKSWRK"] = out["WEEKSWRK"].where(out["WEEKSWRK"].between(0, 52))

    hours = out["HRS1"].where(out["WRKSTAT"].isin([1, 2]))
    hours = hours.fillna(out["HRS2"].where(out["WRKSTAT"].eq(3)))
    out["HOURS"] = hours.where(hours.between(0, 168))

    # 1 and 2 are very or moderately satisfied. 3 and 4 are dissatisfied.
    out["SATJOB_BINARY"] = out["SATJOB"].le(2).astype(int)
    out["OCC10"] = census_code(out["OCC10"])
    return out, funnel, holders


def load_onet() -> pd.DataFrame:
    onet = pd.read_csv(ONET_PATH, dtype={"census_num_2010": str}, usecols=["census_num_2010", *ONET_FEATURES])
    onet["census_num_2010"] = census_code(onet["census_num_2010"])
    onet = onet.dropna(subset=["census_num_2010"])
    return onet.groupby("census_num_2010", as_index=False)[ONET_FEATURES].mean()


def merge_onet(gss: pd.DataFrame, onet: pd.DataFrame) -> pd.DataFrame:
    merged = gss.merge(onet, left_on="OCC10", right_on="census_num_2010", how="left")
    merged = merged.drop(columns=["census_num_2010"])
    for column in INTEGER_COLUMNS:
        merged[column] = as_nullable_int(merged[column])
    return merged[OUTPUT_COLUMNS]


def main() -> None:
    gss = load_gss()
    sample, funnel, holders = modeling_sample(gss)
    merged = merge_onet(sample, load_onet())

    matched = merged[ONET_FEATURES].notna().all(axis=1)
    funnel.loc["matched to O*NET (all seven scores)"] = (
        matched.groupby(merged["YEAR"]).sum().reindex(TARGET_YEARS).fillna(0).astype(int)
    )
    if DROP_UNMATCHED_ONET:
        merged = merged.loc[matched].reset_index(drop=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("GSS waves in the modeling table:")
    print(merged["YEAR"].value_counts().sort_index().to_string())
    print("\nRow funnel by year (cumulative):")
    print(funnel.assign(total=funnel.sum(axis=1)).to_string())
    print("\nRespondents with all five ranks, and which requirement removes them (not cumulative):")
    print(holders.to_string())
    ranked = gss[PREFERENCE_RANKS].apply(pd.to_numeric, errors="coerce").ge(1).all(axis=1)
    for column in ["WRKSTAT", "SATJOB"]:
        print(f"\n{column} among respondents with all five ranks, by year:")
        print(gss.loc[ranked].groupby("YEAR")[column].value_counts(dropna=False).unstack(fill_value=0).to_string())

    ranks = merged[PREFERENCE_RANKS].to_numpy(dtype=float)
    valid = (np.sort(ranks, axis=1) == np.arange(1, 6)).all(axis=1)
    print(f"\nRows whose five ranks are a valid 1-5 ranking: {valid.mean():.1%}")
    print(f"SATJOB_BINARY rate (satisfied = very or moderate): {merged['SATJOB_BINARY'].mean():.3f}")

    print("\nShare of rows with a value, by year (columns near 0 cannot be used in that year):")
    present = merged.drop(columns=["YEAR"]).notna().groupby(merged["YEAR"]).mean().round(2).T
    print(present.to_string())
    print("\nMissing percent in the modeling table:")
    missing = (merged.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(missing.to_string())
    print(f"\nWrote {merged.shape[0]:,} rows x {merged.shape[1]} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()