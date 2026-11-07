"""Select the GSS columns for the job-satisfaction models.

The source file is the 2016-2024 GSS extract. Selection is the planned
variable list, split into a demographic baseline and later job features.
Values are copied unchanged. Row filters and the O*NET join happen in
onet_gss_merge_pipeline.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "gss_2016_2024_all_cols.parquet"
DEFAULT_FEATURE_CATALOG = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_feature_catalog.csv"
DEFAULT_FEATURE_DATA = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_features.parquet"

# ID_ is the name used in some extracts; this release stores the respondent id as ID.
# OCC20 is not on the 2016-2024 file. OCC10 is the Census 2010 occupation code.
SOURCE_COLUMNS = [
    "YEAR",
    "ID",
    "BALLOT",
    "WTSSNRPS",
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
    "SATJOB",
    "OCC10",
    "JOBLOSE",
    "JOBFIND",
    "WRKSCHED",
    "WRKHOME",
    "YEARSJOB",
    "WRKTYPE",
    "HRS1",
    "HRS2",
]

# variable, group, baseline model, later job model, note
FEATURE_ROLES = [
    ("YEAR", "design", False, False, "Survey year. Optional control, not a person-level trait."),
    ("ID", "design", False, False, "Respondent id. ID_ is not in this file. Do not use as a feature."),
    ("BALLOT", "design", False, False, "Questionnaire form. Explains which items were asked. Not a predictor."),
    ("WTSSNRPS", "design", False, False, "Survey weight. Pass as sample_weight, not as a feature."),
    ("AGE", "baseline", True, False, "Demographic baseline."),
    ("SEX", "baseline", True, False, "1 male, 2 female."),
    ("RACE", "baseline", True, False, "1 white, 2 black, 3 other."),
    ("MARITAL", "baseline", True, False, "1 married, 2 widowed, 3 divorced, 4 separated, 5 never married."),
    ("DEGREE", "baseline", True, False, "Highest degree. Use this rather than years of school."),
    ("REGION", "baseline", True, False, "Census division. This extract only contains codes 1-4."),
    ("REALINC", "baseline", True, False, "Family income in constant dollars. Use this or REALRINC, not both."),
    ("REALRINC", "baseline", True, False, "Respondent earnings in constant dollars. Use this or REALINC, not both."),
    ("SATJOB", "outcome", False, False, "1 very satisfied through 4 very dissatisfied. Drop when fitting."),
    ("WRKSTAT", "job", False, True, "Kept rows are 1 full-time, 2 part-time, 3 has a job but not at work last week."),
    ("WRKSLF", "job", False, True, "1 self-employed, 2 works for someone else."),
    ("WEEKSWRK", "job", False, True, "Weeks worked in the past year."),
    ("HRS1", "job", False, True, "Hours worked last week. Prefer HOURS, which folds HRS2 in for WRKSTAT 3."),
    ("HRS2", "job", False, True, "Usual weekly hours. Asked only when WRKSTAT is 3, so it is mostly empty."),
    ("JOBLOSE", "job", False, True, "Chance of losing the job. Asked on ballots 1 and 2 only."),
    ("JOBFIND", "job", False, True, "Chance of finding another job. Asked on ballots 1 and 2 only."),
    ("WRKSCHED", "job", False, True, "Work schedule. Asked in 2018 and 2022 only."),
    ("WRKHOME", "job", False, True, "Work from home. Asked in 2018 and 2022 only."),
    ("YEARSJOB", "job", False, True, "Years with the current employer. Asked in 2018 and 2022 only."),
    ("WRKTYPE", "job", False, True, "Type of employer. Asked in 2018 and 2022 only."),
    ("OCC10", "linkage", False, False, "Census 2010 occupation code. Join key for O*NET, not a predictor."),
    ("ID_", "unavailable", False, False, "Not in this GSS release. The id column is ID."),
    ("OCC20", "unavailable", False, False, "Not in this GSS release. Link occupations with OCC10."),
]


def select_columns(available: set[str]) -> list[str]:
    """Return planned columns that exist on this extract, in list order."""
    missing = [name for name in SOURCE_COLUMNS if name not in available]
    if missing:
        raise KeyError(f"Planned columns are missing from the GSS extract: {missing}")
    return list(SOURCE_COLUMNS)


def build_feature_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Describe each planned variable, including names that are not in the file."""
    rows = []
    for variable, group, baseline, job_model, note in FEATURE_ROLES:
        in_source = variable in data.columns
        missing_pct = round(float(data[variable].isna().mean() * 100), 2) if in_source else None
        rows.append(
            {
                "variable": variable,
                "feature_group": group,
                "in_source_file": in_source,
                "baseline_model": baseline,
                "job_model": job_model,
                "missing_pct_all_respondents": missing_pct,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def extract_job_features(
    data_path: Path = DEFAULT_DATA,
    feature_catalog_path: Path = DEFAULT_FEATURE_CATALOG,
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy the planned columns from the 2016-2024 GSS extract."""
    data = pd.read_parquet(data_path)
    years = sorted(int(year) for year in data["YEAR"].dropna().unique())
    if years[0] < 2016 or years[-1] > 2024:
        raise ValueError(f"Expected GSS years 2016-2024, found {years}.")

    columns = select_columns(set(data.columns))
    feature_data = data[columns].copy()
    feature_catalog = build_feature_catalog(feature_data)

    feature_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    feature_data_path.parent.mkdir(parents=True, exist_ok=True)
    feature_catalog.to_csv(feature_catalog_path, index=False)
    feature_data.to_parquet(feature_data_path, compression="zstd", index=False)
    return feature_catalog, feature_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the planned GSS baseline and job-satisfaction columns."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--feature-catalog", type=Path, default=DEFAULT_FEATURE_CATALOG)
    parser.add_argument("--feature-data", type=Path, default=DEFAULT_FEATURE_DATA)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_catalog, selected_data = extract_job_features(
        args.data,
        args.feature_catalog,
        args.feature_data,
    )
    years = selected_data["YEAR"].value_counts().sort_index()
    print(f"GSS years: {', '.join(str(int(year)) for year in years.index)}")
    print(years.to_string())
    print(
        f"Selected {selected_data.shape[1]} columns for "
        f"{len(selected_data):,} respondents."
    )
    print(selected_catalog.groupby("feature_group").size().to_string())
    print(f"Saved catalog: {args.feature_catalog}")
    print(f"Saved data: {args.feature_data}")
