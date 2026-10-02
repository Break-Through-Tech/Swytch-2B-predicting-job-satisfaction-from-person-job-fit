"""Create a searchable variable catalog for a filtered GSS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "gss_2016_2024_all_cols.parquet"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "gss7224_r3a.dta"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "gss_variable_catalog.csv"


def load_variable_labels(raw_path: Path) -> dict[str, str]:
    """Read variable labels from the source Stata file when it is available."""
    if not raw_path.exists():
        return {}

    try:
        import pyreadstat

        # The GSS release contains legacy-encoded value labels.
        _, metadata = pyreadstat.read_dta(
            raw_path, metadataonly=True, encoding="latin1"
        )
    except (ImportError, OSError, ValueError):
        return {}

    return {
        str(name).upper(): str(label)
        for name, label in metadata.column_names_to_labels.items()
        if label
    }  

def build_catalog(data: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Summarize every column without changing the survey values."""
    records = []
    for position, column in enumerate(data.columns):
        series = data[column]
        non_null = series.notna().sum()
        samples = series.dropna().drop_duplicates().head(5).tolist()
        records.append(
            {
                "position": position,
                "variable": column,
                "label": labels.get(str(column).upper(), ""),
                "dtype": str(series.dtype),
                "row_count": len(series),
                "non_null_count": int(non_null),
                "missing_count": int(len(series) - non_null),
                "missing_pct": round((1 - non_null / len(series)) * 100, 2)
                if len(series)
                else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": json.dumps(samples, default=str),
            }
        )

    return pd.DataFrame.from_records(records)


def extract_features(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    raw_path: Path = DEFAULT_RAW,
) -> pd.DataFrame:
    """Create and save the variable catalog for the filtered dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {input_path}")

    data = pd.read_parquet(input_path)
    labels = load_variable_labels(raw_path)
    catalog = build_catalog(data, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catalog every variable in a filtered GSS Parquet file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    catalog = extract_features(args.input, args.output, args.raw)
    print(f"Cataloged {len(catalog):,} variables from {args.input.name}.")
    print(f"Saved catalog to {args.output}.")