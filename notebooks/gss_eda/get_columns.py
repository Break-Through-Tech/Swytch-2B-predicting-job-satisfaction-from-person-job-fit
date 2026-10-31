import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

"""
# 2
extracts only the necessary columns from our 2016-2024 gss dataset


"""

INPUT_PARQUET = Path("data/interim/gss_2016_2024_all_cols.parquet")
OUTPUT_PARQUET = Path("data/interim/gss_job_satisfaction_subset.parquet")

DESIRED_COLUMNS = [
   "YEAR", "ID_", "BALLOT", "REGION", "WTSSNRPS", "AGE", "SEX", "RACE", "MARITAL", "DEGREE", "WRKSTAT", "WRKSLF", "WEEKSWRK", "SATJOB", "REALINC", "REALRINC", "OCC10", "OCC20", "JOBLOSE", "JOBFIND", "WRKSCHED", "WRKHOME", "YEARSJOB", "WRKTYPE", "HRS1", "HRS2"
]

def extract_gss_subset(input_path: Path, output_path: Path, target_cols: list[str]) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found at: {input_path.resolve()}")

    # Read Parquet schema metadata without loading the data into RAM
    parquet_file = pq.ParquetFile(input_path)
    available_cols = parquet_file.schema.names
    available_cols_lookup = {col.upper(): col for col in available_cols}

    matched_cols = []
    missing_cols = []

    for col in target_cols:
        col_upper = col.upper()
        if col_upper in available_cols_lookup:
            matched_cols.append(available_cols_lookup[col_upper])
        else:
            missing_cols.append(col)

    print(f"Columns found in file ({len(matched_cols)}): {matched_cols}")
    if missing_cols:
        print(f"Columns not in schema (skipped): {missing_cols}")

    # Column projection directly at disk read
    print(f"\nReading selected columns from '{input_path}'...")
    df_subset = pd.read_parquet(input_path, columns=matched_cols)

    # Save the pruned dataset
    df_subset.to_parquet(output_path, index=False, compression="snappy")
    print(f"Successfully saved {len(df_subset):,} rows and {len(df_subset.columns)} columns to '{output_path}'.")
    
    return df_subset

if __name__ == "__main__":
    df = extract_gss_subset(INPUT_PARQUET, OUTPUT_PARQUET, DESIRED_COLUMNS)
    print("\nDataFrame Summary:")
    print(df.info())
    print(df.shape)