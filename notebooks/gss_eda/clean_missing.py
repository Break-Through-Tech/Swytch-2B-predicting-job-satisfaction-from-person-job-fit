"""
# 3
Clean the 2016-2024 GSS job-satisfaction subset for modeling.

- restrict sample to active workers (WRKSTAT 1-3), valid SATJOB (1-4) and valid civilian 2010 Census occupation codes (0 < OCC10 < 9800)
- combine HRS1 and HRS2 into a single HOURS variable
- convert negative GSS survey sentinels (< 0) across all survey items to NaN so item nonresponse is correctly preserved for pipeline imputation
- retain core demographics, socioeconomic controls, and survey weights for modeling

"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_subset.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "gss_job_satisfaction_clean.parquet"

# GSS work status: 1 full-time, 2 part-time, 3 has a job but not at work last week.
EMPLOYED_WRKSTAT = (1, 2, 3)
# 1 very satisfied, 2 moderately satisfied, 3 a little dissatisfied, 4 very dissatisfied.
VALID_SATJOB = (1, 2, 3, 4)

OUTPUT_COLUMNS = [
    "YEAR",
    "ID_", 
    "BALLOT",
    "REGION", 
    "WTSSNRPS",
    "SATJOB",
    "WRKSTAT",
    "WRKSLF",
    "OCC10",
    "OCC20",
    "HOURS",
    "REALRINC",
    "REALINC",
    "DEGREE",
    "AGE",
    "SEX",
    "RACE",
    "MARITAL",
    "WEEKSWRK",
    "JOBLOSE",
    "JOBFIND",
    "WRKSCHED",
    "WRKHOME",
    "YEARSJOB",
    "WRKTYPE",
    "HRS1", 
    "HRS2"
]

def describe_row_drops(df: pd.DataFrame) -> pd.DataFrame:
    """Count rows kept after each sample restriction, in order."""
    employed = df["WRKSTAT"].isin(EMPLOYED_WRKSTAT)
    valid_satjob = df["SATJOB"].isin(VALID_SATJOB)
    # Exclude missing, negative sentinels, and military/unclassifiable codes (>= 9800)
    valid_occ = df["OCC10"].notna() & (df["OCC10"] > 0) & (df["OCC10"] < 9800)
    steps = [
        ("Starting rows", len(df)),
        ("Currently working (WRKSTAT 1, 2, or 3)", int(employed.sum())),
        ("...and valid SATJOB (1-4)", int((employed & valid_satjob).sum())),
        ("...and valid civilian OCC10 (kept)", int((employed & valid_satjob & valid_occ).sum())),
    ]
    return pd.DataFrame(steps, columns=["Step", "Rows"])


def clean_gss(df: pd.DataFrame) -> pd.DataFrame:
    """Return the modeling sample with one hours column and structural holes removed.

    Rows kept: currently working, valid SATJOB, non-missing OCC10.
    Columns dropped: JOBLOSE (never asked on ballot 3), RINCOME (duplicate of
    REALRINC), and the separate HRS1/HRS2 items after they are combined.
    Remaining NaNs are item nonresponse and are left missing.
    """
    required = {
        "WRKSTAT",
        "SATJOB",
        "OCC10",
        "HRS1",
        "HRS2",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Subset is missing required columns: {sorted(missing)}")

    employed = df["WRKSTAT"].isin(EMPLOYED_WRKSTAT)
    valid_satjob = df["SATJOB"].isin(VALID_SATJOB)
    valid_occ = df["OCC10"].notna() & (df["OCC10"] > 0) & (df["OCC10"] < 9800)
    out = df.loc[employed & valid_satjob & valid_occ].copy()

    # HRS1 is hours last week for full-time and part-time workers.
    # HRS2 is usual weekly hours, asked only when WRKSTAT == 3.
    # create combined HOURS
    hours = out["HRS1"].copy()
    temp_not_at_work = out["WRKSTAT"] == 3
    hours.loc[temp_not_at_work] = out.loc[temp_not_at_work, "HRS2"]
    out["HOURS"] = hours

    # Clean negative sentinels (< 0 -> NaN) dynamically across numeric features
    sentinel_skip = {"YEAR", "ID_", "OCC10"}  # identifiers & codes
    for col in out.select_dtypes(include=["number"]).columns:
        if col not in sentinel_skip:
            out[col] = out[col].mask(out[col] < 0, np.nan)

    # Clean bounds
    if "AGE" in out.columns:
        out["AGE"] = out["AGE"].mask(out["AGE"] > 89, np.nan)
    if "DEGREE" in out.columns:
        out["DEGREE"] = out["DEGREE"].mask(out["DEGREE"] > 4, np.nan)
    out["OCC10"] = out["OCC10"].astype("Int64")

    out = out.drop(columns=["HRS1", "HRS2", "RINCOME"], errors="ignore")
    final_cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
    out = out.loc[:, final_cols].reset_index(drop=True)
    return out


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Percent and count of remaining NaNs, highest first."""
    summary = pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
        }
    )
    return summary.sort_values("pct_missing", ascending=False)


def write_clean(df: pd.DataFrame, output_path: Path = OUTPUT_PARQUET) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")


def main() -> pd.DataFrame:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Subset parquet not found at: {INPUT_PARQUET}")

    raw = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(raw):,} rows and {raw.shape[1]} columns from {INPUT_PARQUET.name}.")
    print("\nRow restrictions:")
    print(describe_row_drops(raw).to_string(index=False))

    cleaned = clean_gss(raw)
    write_clean(cleaned)
    print(f"\nSaved {len(cleaned):,} rows and {cleaned.shape[1]} columns to {OUTPUT_PARQUET}.")
    print("\nRemaining missing values (left as NaN, not filled):")
    print(null_summary(cleaned).to_string())
    return cleaned


if __name__ == "__main__":
    main()
