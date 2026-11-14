"""Generate an exploratory data analysis report for the selected GSS features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_features.parquet"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_feature_catalog.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "eda"


def summarize_columns(data: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in data.columns:
        series = data[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        record = {
            "variable": column,
            "label": catalog.set_index("variable").get("label", pd.Series(dtype=str)).get(column, ""),
            "dtype": str(series.dtype),
            "row_count": len(series),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if numeric:
            record.update(
                {
                    "min": series.min(),
                    "p01": series.quantile(0.01),
                    "median": series.median(),
                    "p99": series.quantile(0.99),
                    "max": series.max(),
                }
            )
        records.append(record)
    return pd.DataFrame(records).sort_values("missing_pct", ascending=False)


def build_quality_flags(data: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    flags = summary[["variable", "missing_pct", "unique_count"]].copy()
    flags["high_missingness"] = flags["missing_pct"] >= 50
    flags["near_constant"] = flags["unique_count"] <= 1
    flags["review_reason"] = ""
    flags.loc[flags["high_missingness"], "review_reason"] = "module-specific or high missingness"
    flags.loc[flags["near_constant"], "review_reason"] = "no usable variation"

    if "HRS1" in data:
        invalid_hours = data["HRS1"].notna() & ~data["HRS1"].between(0, 89)
        flags.loc[flags["variable"] == "HRS1", "invalid_range_count"] = int(invalid_hours.sum())
    if "AGE" in data:
        invalid_age = data["AGE"].notna() & ~data["AGE"].between(0, 89)
        flags.loc[flags["variable"] == "AGE", "invalid_range_count"] = int(invalid_age.sum())
    return flags


def save_plots(data: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = summary.head(25).sort_values("missing_pct")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(missing["variable"], missing["missing_pct"], color="#d97706")
    axis.set(xlabel="Missing values (%)", title="Top 25 variables by missingness")
    figure.tight_layout()
    figure.savefig(output_dir / "missingness.png", dpi=160)
    plt.close(figure)

    if {"YEAR", "SATJOB"}.issubset(data.columns):
        outcome = data.dropna(subset=["SATJOB"]).copy()
        outcome["SATJOB"] = outcome["SATJOB"].astype(int).astype(str)
        proportions = pd.crosstab(outcome["YEAR"], outcome["SATJOB"], normalize="index")
        proportions = proportions.reindex(columns=["1", "2", "3", "4"], fill_value=0)
        axis = proportions.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        axis.set(xlabel="Survey year", ylabel="Share of valid SATJOB responses", title="Job satisfaction by year")
        axis.legend(title="SATJOB code", labels=["Very satisfied", "Moderately satisfied", "A little dissatisfied", "Very dissatisfied"])
        figure = axis.get_figure()
        figure.tight_layout()
        figure.savefig(output_dir / "job_satisfaction_by_year.png", dpi=160)
        plt.close(figure)


def write_report(data: pd.DataFrame, summary: pd.DataFrame, flags: pd.DataFrame, output_dir: Path) -> None:
    valid_outcome = data["SATJOB"].notna().sum() if "SATJOB" in data else 0
    high_missing = flags.loc[flags["high_missingness"], "variable"].tolist()
    report = [
        "# GSS exploratory data analysis",
        "",
        f"Rows: {len(data):,}",
        f"Columns: {len(data.columns):,}",
        f"Rows with a valid SATJOB response: {valid_outcome:,}",
        "",
        "## Initial cleaning decisions",
        "",
        "- Keep SATJOB values 1-4 as an ordered outcome; do not treat 3 and 4 as missing.",
        "- Treat missing SATJOB as unavailable outcome data and exclude those rows only for supervised modeling.",
        "- Do not impute occupation, work-hours, or work-condition variables before checking survey skip logic.",
        "- Review variables with at least 50% missingness individually; many are asked only in selected survey modules.",
        "- Keep YEAR and occupation codes as identifiers/categories, not continuous measurements.",
        "",
        "## Variables requiring review",
        "",
        f"{len(high_missing):,} selected variables have at least 50% missingness.",
        "The complete list is in `quality_flags.csv`; the full descriptive table is in `column_summary.csv`.",
    ]
    (output_dir / "eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_eda(data_path: Path, catalog_path: Path, output_dir: Path) -> None:
    data = pd.read_parquet(data_path)
    catalog = pd.read_csv(catalog_path)
    summary = summarize_columns(data, catalog)
    flags = build_quality_flags(data, summary)

    if {"YEAR", "ID"}.issubset(data.columns):
        duplicate_count = int(data.duplicated(["YEAR", "ID"]).sum())
        flags.attrs["duplicate_year_id_count"] = duplicate_count

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "column_summary.csv", index=False)
    flags.to_csv(output_dir / "quality_flags.csv", index=False)
    if "YEAR" in data:
        data["YEAR"].value_counts().sort_index().rename("row_count").to_csv(output_dir / "rows_by_year.csv")
    if {"YEAR", "SATJOB"}.issubset(data.columns):
        data.groupby(["YEAR", "SATJOB"], dropna=False).size().rename("row_count").to_csv(output_dir / "satjob_by_year.csv")
    save_plots(data, summary, output_dir)
    write_report(data, summary, flags, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GSS EDA tables and plots.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.data, args.catalog, args.output)
    print(f"EDA outputs written to {args.output}")