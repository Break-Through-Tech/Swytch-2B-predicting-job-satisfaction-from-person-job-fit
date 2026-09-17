"""
filter gss for the most recent years 2016-2024: [insert reasoning here]
slices the columns for relevant features.
"""

from pathlib import Path
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gss7224_r3a.dta"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE_PATH = INTERIM_DATA_DIR / "gss_2016_2024_all_cols.parquet"


def filter_gss(start_year: int = 2016, end_year: int = 2024) -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw GSS data not found at: {RAW_DATA_PATH}\n"
            "Please make sure the raw .dta file is placed inside data/raw/."
        )

    print(f"Reading raw GSS data from {RAW_DATA_PATH.name}...")
    # convert_categoricals=False retains numeric codes (crucial for custom missing-value mappings)
    df = pd.read_stata(RAW_DATA_PATH, convert_categoricals=False)
    print(f"Loaded raw dataset with shape: {df.shape}")

    # Standardize column names to uppercase to avoid case mismatch across releases
    df.columns = df.columns.str.upper()

    if "YEAR" not in df.columns:
        raise KeyError("Could not find 'YEAR' column in the dataset.")

    # Filter for target waves
    print(f"Filtering rows for years {start_year} to {end_year}...")
    df_filtered = df[(df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)].copy()

    # Ensure output directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet with zstd compression for high compression ratio
    print(f"Saving filtered dataset to {OUTPUT_FILE_PATH}...")
    df_filtered.to_parquet(OUTPUT_FILE_PATH, compression="zstd", index=False)

    print("\n--- Summary ---")
    print(f"Final shape: {df_filtered.shape} (rows, columns)")
    print("Respondent breakdown by year:")
    print(df_filtered["YEAR"].value_counts().sort_index())
    print("\nDone! Interim file created successfully.")


if __name__ == "__main__":
    filter_gss()