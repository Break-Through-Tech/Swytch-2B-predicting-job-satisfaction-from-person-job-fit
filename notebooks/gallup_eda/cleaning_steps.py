"""
Gallup American Job Quality Study — data cleaning Steps 1-3 (pre-O*NET merge).

Updates Steps 1-3 from notebooks/gallup_eda/eda.ipynb so the resulting frame
is ready for the O*NET SOC crosswalk join (flagged as a TODO in the notebook).

Steps:
    1. Reconcile missing-value representations (-98 sentinel -> NaN).
    2. Clean target Q4 and build df_model.
    3. Clean occupation (Q6 / Q6_TEXT): keep missing/No-response Q6 (flag,
       don't drop) and produce normalized Q6_TEXT_norm audit metadata.

Output:
    data/interim/gallup_steps1to3.parquet  (df_model, ready for O*NET merge)

Usage:
    python notebooks/gallup_eda/cleaning_steps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

LABELED_CSV = RAW_DATA_DIR / "JFF_formatted_public_release_labels.csv"
UNLABELED_CSV = RAW_DATA_DIR / "JFF_formatted_public_release_values.csv"
OUTPUT_FILE = INTERIM_DATA_DIR / "gallup_steps1to3.parquet"

MISSING_SENTINEL = -98
SATISFACTION_THRESHOLD = 7
MIXED_TYPE_COLS = ["S1", "REGION", "DIVISION"]


def load_data(
    labeled_path: Path = LABELED_CSV, unlabeled_path: Path = UNLABELED_CSV
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labeled and unlabeled Gallup CSVs."""
    if not unlabeled_path.exists():
        raise FileNotFoundError(
            f"Unlabeled CSV not found at: {unlabeled_path}\n"
            "Please place JFF_formatted_public_release_values.csv inside data/raw/."
        )

    # Labeled CSV is optional and only retained for notebook comparisons.
    df_labeled = pd.read_csv(labeled_path, encoding="cp1252") if labeled_path.exists() else None

    print(f"Reading unlabeled data from {unlabeled_path.name} ...")
    df_unlabeled = pd.read_csv(unlabeled_path, encoding="cp1252")
    print(f"  loaded unlabeled shape: {df_unlabeled.shape}")

    return df_labeled, df_unlabeled


def step1_reconcile_missing(df_unlabeled: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Reconcile missing-value representations.

    The values CSV uses -98 as a sentinel for "No response"; convert to NaN so
    missingness is represented consistently across both frames.
    """
    print("\n=== Step 1: Reconcile missing-value representations (-98 -> NaN) ===")
    print(f"Shape before: {df_unlabeled.shape}")

    df_unlabeled_clean = reconcile_missing_values(df_unlabeled)

    print(f"Shape after : {df_unlabeled_clean.shape}")
    print(
        "Total missing in unlabeled (after -98 -> NaN): "
        f"{df_unlabeled_clean.isna().sum().sum()}"
    )

    missing = df_unlabeled_clean.isna().sum().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(10)
    if not top_missing.empty:
        print("Top columns with missing values (unlabeled):")
        print(top_missing)

    return df_unlabeled_clean


def reconcile_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the Gallup no-response sentinel without changing raw columns."""
    return df.replace(MISSING_SENTINEL, np.nan)


def step2_clean_target(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Clean numeric Q4 and build df_model.

    - Drop rows with missing Q4.
    - Coerce Q4 to numeric (dropping any remaining non-numeric values).
    - Derive the binary target `satisfied = (Q4 >= 7)`.
    """
    print("\n=== Step 2: Clean target variable Q4 (job satisfaction) ===")
    print("Q4 value counts (labeled, incl. NaN):")
    print(df_labeled["Q4"].value_counts(dropna=False).sort_index())

    df_model = df_labeled[df_labeled["Q4"].notna()].copy()
    print(f"Rows before dropping missing Q4: {len(df_labeled)}")
    print(f"Rows after  dropping missing Q4: {len(df_model)}")

    df_model["Q4"] = pd.to_numeric(df_model["Q4"], errors="coerce")
    df_model = df_model[df_model["Q4"].notna()].copy()
    print(f"Rows after coercing Q4 to numeric: {len(df_model)}")

    df_model["satisfied"] = (df_model["Q4"] >= SATISFACTION_THRESHOLD).astype(int)
    print("Satisfied class distribution:")
    print(df_model["satisfied"].value_counts())

    return df_model


def step3_clean_occupation(df_model: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Clean occupation (Q6 / Q6_TEXT) in preparation for the O*NET merge.

    - Keep rows with missing/No-response Q6 (flag them instead of dropping).
    - Normalize Q6_TEXT write-ins into Q6_TEXT_norm (lowercased, stripped) for
      audit metadata. Exact code matching is the only initial linkage method.
    - Cast mixed-type columns to str for consistency.
    """
    print("\n=== Step 3: Clean occupation (Q6 / Q6_TEXT) ===")
    print("Q6 value counts (top 20):")
    print(df_model["Q6"].value_counts(dropna=False).head(20))

    # Flag rows with missing / "No response" occupation; keep them (do not drop).
    no_occ = (
        df_model["Q6"].isna()
        | (df_model["Q6"].astype(str).str.strip() == "No response")
    )
    print(f"Rows with missing/No-response Q6 (kept): {int(no_occ.sum())}")

    # Treat explicit "No response" strings in Q6 as NaN for downstream consistency.
    df_model.loc[
        df_model["Q6"].astype(str).str.strip() == "No response", "Q6"
    ] = np.nan

    # Normalize write-in text for audit and later review, not as a join key.
    df_model["Q6_TEXT_norm"] = (
        df_model["Q6_TEXT"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"-98": "", "no response": ""})
    )

    print("Sample Q6_TEXT write-ins:")
    print(
        df_model["Q6_TEXT"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .head(20)
    )
    print("Normalized write-ins (sample):")
    print(
        df_model.loc[df_model["Q6_TEXT_norm"] != "", "Q6_TEXT_norm"]
        .value_counts()
        .head(20)
    )

    # Cast mixed-type columns to str (notebook does this in Step 4; done here
    # so the frame is tidy before the merge).
    for col in MIXED_TYPE_COLS:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str).str.strip()

    # Gallup Q6 is a broad category, not a Census/SOC code. Q6=24 means
    # "Something else"; Q6_CODED can supply a coded category for that case.
    q6 = pd.to_numeric(df_model["Q6"], errors="coerce")
    coded = pd.to_numeric(df_model["Q6_CODED"], errors="coerce")
    df_model["gallup_occupation_group"] = q6.where(q6.ne(24), coded)
    df_model["gallup_occupation_group"] = df_model["gallup_occupation_group"].where(
        df_model["gallup_occupation_group"].between(1, 25) & df_model["gallup_occupation_group"].ne(24)
    ).astype("Int64")
    df_model["occupation_source"] = np.select(
        [q6.eq(24) & coded.notna(), q6.notna() & q6.ne(24)],
        ["Q6_CODED", "Q6"], default="missing_or_uncoded")
    df_model["occupation_missing"] = df_model["gallup_occupation_group"].isna().astype("int8")

    print("\n# Exact-code linkage is implemented in occupation_linkage.py")
    return df_model


def run(
    labeled_path: Path = LABELED_CSV,
    unlabeled_path: Path = UNLABELED_CSV,
    output_path: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Execute Steps 1-3 and save the cleaned modeling frame."""
    df_labeled, df_unlabeled = load_data(labeled_path, unlabeled_path)

    
    df_unlabeled = step1_reconcile_missing(df_unlabeled)
    if df_labeled is not None:
        df_labeled = reconcile_missing_values(df_labeled)

    df_model = step2_clean_target(df_unlabeled)

    
    df_model = step3_clean_occupation(df_model)

    # Persist the cleaned frame for the O*NET merge step.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned modeling frame to {output_path}  shape={df_model.shape}")

    print("\n--- Summary ---")
    print(f"Final shape: {df_model.shape} (rows, columns)")
    print(f"Columns: {list(df_model.columns)}")
    print("df_model is ready for the O*NET SOC crosswalk merge.")
    return df_model


if __name__ == "__main__":
    run()


