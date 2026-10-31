import pyarrow.parquet as pq
import pandas as pd

schema = pq.ParquetFile("data/interim/gss_2016_2024_all_cols.parquet").schema.names

# Search for job/work-related variables fielded in recent waves
work_candidates = [
    col for col in schema 
    if any(stem in col for stem in ["WRK", "JOB", "PROMO", "PAY", "SEC", "QUAL"])
]

# Check which ones have substantive responses (> 0 non-null)
sample_df = pd.read_parquet("data/interim/gss_2016_2024_all_cols.parquet", columns=work_candidates)
valid_counts = sample_df.notna().sum()
print("Fielded work candidates with responses:")
print(valid_counts[valid_counts > 0].sort_values(ascending=False).head(30))