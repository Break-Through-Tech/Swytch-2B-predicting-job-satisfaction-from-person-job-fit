"""Select GSS variables useful for studying job satisfaction.

The input catalog is used for label-aware selection, so this script works across
GSS releases where variable names or survey sections differ. The source values
are copied unchanged; GSS special missing-value codes should be recoded in a
separate modeling step after reviewing the selected variables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "interim" / "gss_2016_2024_all_cols.parquet"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "interim" / "gss_variable_catalog.csv"
DEFAULT_FEATURE_CATALOG = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_feature_catalog.csv"
DEFAULT_FEATURE_DATA = PROJECT_ROOT / "data" / "interim" / "gss_job_satisfaction_features.parquet"

# Exact names protect the core analysis variables even if their labels change.
CORE_VARIABLES = {
    "outcome": {"SATJOB", "SATJOB1", "SATJOBHV", "JOBSAT", "EXJOBSAT"},
    "work": {"HRS1", "HRS2", "WRKSTAT", "EVWORK", "JOBLOSE", "JOBSECOK", "TRYNEWJB"},
    "demographic": {
        "YEAR", "AGE", "SEX", "RACE", "EDUC", "DEGREE", "MARITAL", "CHILDS",
        "INCOME", "RINCOME", "REGION", "RELIG", "ATTEND", "HEALTH",
    },
}

CATEGORY_PATTERNS = {
    "outcome": re.compile(
        r"satisf|happy|job satisfaction|satisfaction with (?:their|the) job",
        re.IGNORECASE,
    ),
    "work": re.compile(
        r"hours? worked|hours? (?:usually )?work|labor force|employment status|"
        r"job security|lose job|laid off|find (?:a )?job|new job|occupation|industry|"
        r"supervisor|boss|co-?worker|coworker|union|benefit|promotion|work stress|"
        r"autonomy|meaning(?:ful)? work",
        re.IGNORECASE,
    ),
    "demographic": re.compile(
        r"^age$|sex|gender|race|highest year of school|education|degree|marital|"
        r"number of children|family income|respondent.?s income|region of residence|"
        r"religious preference|religious services|health|citizenship|veteran|urban",
        re.IGNORECASE,
    ),
}

# These labels refer to someone other than the respondent and can create a noisy
# work-feature shortlist when broad label matching is used.
NON_RESPONDENT_WORK = re.compile(r"mother|father|parent|spouse|partner|child", re.IGNORECASE)


def classify_row(row: pd.Series) -> tuple[str, str] | None:
    variable = str(row["variable"]).upper()
    label = str(row.get("label", ""))
    text = f"{variable} {label}"

    for category, variables in CORE_VARIABLES.items():
        if variable in variables:
            return category, "core variable"

    for category in ("outcome", "work", "demographic"):
        if category == "work" and NON_RESPONDENT_WORK.search(label):
            continue
        if CATEGORY_PATTERNS[category].search(text):
            return category, "label/variable keyword match"
    return None


def select_feature_catalog(
    catalog: pd.DataFrame,
    max_missing_pct: float = 90.0,
) -> pd.DataFrame:
    """Return usable, categorized variables relevant to job satisfaction."""
    required = {"variable", "label", "missing_pct"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")

    selected = []
    for _, row in catalog.iterrows():
        if float(row["missing_pct"]) > max_missing_pct:
            continue
        classification = classify_row(row)
        if classification is None:
            continue
        category, reason = classification
        record = row.to_dict()
        record["feature_category"] = category
        record["selection_reason"] = reason
        selected.append(record)

    if not selected:
        raise ValueError("No features matched the selection rules.")

    result = pd.DataFrame(selected)
    category_order = {"outcome": 0, "work": 1, "demographic": 2}
    result["_category_order"] = result["feature_category"].map(category_order)
    result = result.sort_values(["_category_order", "missing_pct", "variable"])
    return result.drop(columns="_category_order").reset_index(drop=True)


def extract_job_features(
    data_path: Path = DEFAULT_DATA,
    catalog_path: Path = DEFAULT_CATALOG,
    feature_catalog_path: Path = DEFAULT_FEATURE_CATALOG,
    feature_data_path: Path = DEFAULT_FEATURE_DATA,
    max_missing_pct: float = 90.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select feature metadata and write the corresponding respondent data."""
    catalog = pd.read_csv(catalog_path)
    feature_catalog = select_feature_catalog(catalog, max_missing_pct)

    data = pd.read_parquet(data_path)
    available = set(data.columns)
    variables = [name for name in feature_catalog["variable"] if name in available]
    if not variables:
        raise ValueError("None of the selected catalog variables are in the data file.")

    feature_data = data[variables].copy()
    feature_catalog = feature_catalog[feature_catalog["variable"].isin(variables)].copy()

    feature_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    feature_data_path.parent.mkdir(parents=True, exist_ok=True)
    feature_catalog.to_csv(feature_catalog_path, index=False)
    feature_data.to_parquet(feature_data_path, compression="zstd", index=False)
    return feature_catalog, feature_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select GSS work, demographic, and job-satisfaction variables."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--feature-catalog", type=Path, default=DEFAULT_FEATURE_CATALOG)
    parser.add_argument("--feature-data", type=Path, default=DEFAULT_FEATURE_DATA)
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=90.0,
        help="Exclude variables with more than this percentage of missing values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_catalog, selected_data = extract_job_features(
        args.data,
        args.catalog,
        args.feature_catalog,
        args.feature_data,
        args.max_missing_pct,
    )
    print(f"Selected {len(selected_catalog):,} variables for {len(selected_data):,} respondents.")
    print(selected_catalog.groupby("feature_category").size().to_string())
    print(f"Saved catalog: {args.feature_catalog}")
    print(f"Saved data: {args.feature_data}")
