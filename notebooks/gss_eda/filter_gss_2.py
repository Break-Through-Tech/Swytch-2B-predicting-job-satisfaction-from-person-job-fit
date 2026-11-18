"""2006, 2012, 2014"""
"""
Filter the GSS cumulative file to the years that can measure person-job fit.

Reasoning: the project defines fit as how closely a person's own work values
match their occupation. The only GSS items that record a worker's values are
the job-preference ranks (JOBINC, JOBSEC, JOBHOUR, JOBPROMO, JOBMEANS), and
they were not asked in 2016-2024. For JOBHOUR the documented modern years are
2006, 2012, and 2014. The year-by-year counts printed below confirm whether the
other four ranks follow the same pattern in this file.

The years are not contiguous, so they are passed as a list, not a range.
All columns are kept here; a later step slices the features that are needed.
"""

from pathlib import Path
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gss7224_r3a.dta"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE_PATH = INTERIM_DATA_DIR / "gss_2006_2012_2014_all_cols.parquet"

TARGET_YEARS = [2006, 2012, 2014]
PREFERENCE_RANKS = ["JOBINC", "JOBSEC", "JOBHOUR", "JOBPROMO", "JOBMEANS"]


def filter_gss(years: list[int] = TARGET_YEARS) -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw GSS data not found at: {RAW_DATA_PATH}\n"
            "Please make sure the raw .dta file is placed inside data/raw/."
        )

    print(f"Reading raw GSS data from {RAW_DATA_PATH.name}...")
    # convert_categoricals=False retains numeric codes (crucial for custom
    # missing-value mappings). Stata's special missing codes (.i, .d, ...)
    # are read as NaN.
    df = pd.read_stata(RAW_DATA_PATH, convert_categoricals=False)
    print(f"Loaded raw dataset with shape: {df.shape}")

    # Standardize column names to uppercase to avoid case mismatch across
    # releases
    df.columns = df.columns.str.upper()

    if "YEAR" not in df.columns:
        raise KeyError("Could not find 'YEAR' column in the dataset.")

    # Filter for target waves
    print(f"Filtering rows for years {years}...")
    df_filtered = df[df["YEAR"].isin(years)].copy()

    absent_years = [y for y in years if y not in set(df_filtered["YEAR"])]
    if absent_years:
        raise ValueError(f"No rows found for years: {absent_years}")

    # Ensure output directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet with zstd compression for high compression ratio
    print(f"Saving filtered dataset to {OUTPUT_FILE_PATH}...")
    df_filtered.to_parquet(OUTPUT_FILE_PATH, compression="zstd", index=False)

    print("\n--- Summary ---")
    print(f"Final shape: {df_filtered.shape} (rows, columns)")
    print("Respondent breakdown by year:")
    print(df_filtered["YEAR"].value_counts().sort_index())

    print("\nNon-missing preference ranks by year:")
    present = [c for c in PREFERENCE_RANKS if c in df_filtered.columns]
    missing = [c for c in PREFERENCE_RANKS if c not in df_filtered.columns]
    if missing:
        print(f"Not in this file: {', '.join(missing)}")
    if present:
        counts = df_filtered[present].notna().groupby(df_filtered["YEAR"]).sum()
        counts["all_present"] = (
            df_filtered[present].notna().all(axis=1).groupby(df_filtered["YEAR"]).sum()
        )
        print(counts.to_string())

    print("\nDone! Interim file created successfully.")


if __name__ == "__main__":
    filter_gss()