"""Build the GSS 2018 and 2022 modeling table and link it to O*NET.

Reads the planned columns from the 2018-2022 extract, keeps people who are
currently working and gave a valid job-satisfaction answer, and joins O*NET
on the Census 2010 occupation code. The four quality-of-work-life items
WKVSFAM, OVERWORK, WKDECIDE, and JOBSECOK are kept on their GSS codes.
Categorical codes stay numeric so
encoding can happen inside the training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

GSS_EDA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GSS_EDA))

from select_job_features import SOURCE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GSS_PATH = PROJECT_ROOT / "data" / "interim" / "gss_2018_2022_all_cols.parquet"
ONET_PATH = GSS_EDA / "onet_data" / "FSU_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "merged_onet_gss_2018_2022.csv"
TARGET_YEARS = [2018, 2022]

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
    "WKVSFAM",
    "OVERWORK",
    "WKDECIDE",
    "JOBSECOK",
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
    "WKVSFAM",
    "OVERWORK",
    "WKDECIDE",
    "JOBSECOK",
]


def census_code(series: pd.Series) -> pd.Series:
    """Turn 1020.0 and '1020.0' into the same '1020' key the O*NET file uses."""
    numeric = pd.to_numeric(series, errors="coerce").round()
    return numeric.astype("Int64").astype("string")


def as_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


# Quality-of-work-life items kept on the GSS codes. Model scripts orient them.
QWL_FEATURES = ["WKVSFAM", "OVERWORK", "WKDECIDE", "JOBSECOK"]


def load_gss() -> pd.DataFrame:
    columns = [name for name in SOURCE_COLUMNS if name != "ID_"] + QWL_FEATURES
    gss = pd.read_parquet(GSS_PATH, columns=columns)
    return gss[gss["YEAR"].isin(TARGET_YEARS)].copy()


def modeling_sample(gss: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Keep current workers with a real satisfaction answer and a civilian occupation."""
    employed = gss["WRKSTAT"].isin(EMPLOYED_WRKSTAT)
    valid_satjob = gss["SATJOB"].isin(VALID_SATJOB)
    occ = pd.to_numeric(gss["OCC10"], errors="coerce")
    valid_occ = occ.notna() & (occ > 0) & (occ < 9800)
    funnel = [
        ("2018 and 2022 respondents", len(gss)),
        ("currently working", int(employed.sum())),
        ("valid SATJOB", int((employed & valid_satjob).sum())),
        ("civilian OCC10", int((employed & valid_satjob & valid_occ).sum())),
    ]
    out = gss.loc[employed & valid_satjob & valid_occ].copy()

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
    return out, funnel


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
    if not GSS_PATH.exists():
        raise FileNotFoundError(
            f"2018-2022 extract not found at: {GSS_PATH}\n"
            "Run filter_gss20182022.py first."
        )
    gss = load_gss()
    sample, funnel = modeling_sample(gss)
    merged = merge_onet(sample, load_onet())
    merged.to_csv(OUTPUT_PATH, index=False)

    matched = merged[ONET_FEATURES[0]].notna()
    print("GSS waves in the modeling table:")
    print(merged["YEAR"].value_counts().sort_index().to_string())
    print("\nRow funnel:")
    for label, count in funnel:
        print(f"  {label}: {count:,}")
    print(
        f"\nO*NET match: {int(matched.sum()):,} of {len(merged):,} rows "
        f"({matched.mean():.1%}) on OCC10."
    )
    print(f"SATJOB_BINARY rate (satisfied = very or moderate): {merged['SATJOB_BINARY'].mean():.3f}")
    print("\nMissing percent in the modeling table:")
    missing = (merged.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(missing.to_string())
    print(f"\nWrote {merged.shape[0]:,} rows x {merged.shape[1]} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
