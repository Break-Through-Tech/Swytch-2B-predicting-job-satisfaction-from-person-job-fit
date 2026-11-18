"""Pre-modeling data checks for the GSS + O*NET job satisfaction project.

Check 1: which GSS years have the five job-preference ranks (JOBINC, JOBSEC,
         JOBHOUR, JOBPROMO, JOBMEANS), and whether the ranks look valid.
Check 2: how many rows survive in the candidate years once the outcome,
         occupation code, O*NET values, and survey weight are required, and
         which occupation codes fail to match O*NET.

Run it on the re-pulled file that includes the five rank columns. It only
reads the data and prints tables; it changes nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "merged_onet_gss_dataset.csv"
CANDIDATE_YEARS = [2006, 2012, 2014]
YEAR = "YEAR"
TARGET = "SATJOB_BINARY"
WEIGHT = "WTSSNRPS"
OCCUPATION = "OCC10"
PREFS = ["JOBINC", "JOBSEC", "JOBHOUR", "JOBPROMO", "JOBMEANS"]
ONET_COLUMNS = [f"wv{i}val_mean" for i in range(1, 7)] + ["si1val_mean"]
# GSS stores these ranks either as numbers 1-5 or as text labels, depending
# on how the file was exported.
RANK_LABELS = {"most impt": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def to_rank(series: pd.Series) -> pd.Series:
    """Convert a rank column to floats 1-5. Anything else becomes NaN."""
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype(float)
    else:
        text = series.astype("string").str.strip().str.lower()
        out = pd.to_numeric(text.map(RANK_LABELS), errors="coerce")
        out = out.fillna(pd.to_numeric(text, errors="coerce")).astype(float)
    return out.where(out.between(1, 5))


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    section("Columns")
    needed = [YEAR, TARGET, WEIGHT, OCCUPATION] + PREFS + ONET_COLUMNS
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        print("Missing from the file:", ", ".join(missing))
        if any(c in missing for c in PREFS):
            print("The preference ranks are not in this file. Re-pull the GSS with them.")
    else:
        print("All required columns are present.")
    prefs = [c for c in PREFS if c in frame.columns]
    onet = [c for c in ONET_COLUMNS if c in frame.columns]
    return prefs, onet


def check_ranks(frame: pd.DataFrame, prefs: list[str]) -> pd.DataFrame:
    section("Check 1: preference ranks by year")
    if not prefs:
        print("No rank columns to check.")
        return pd.DataFrame(index=frame.index)
    ranks = frame[prefs].apply(to_rank)
    present = ranks.notna()
    all_five = present.all(axis=1)
    by_year = present.groupby(frame[YEAR]).sum().astype(int)
    by_year.insert(0, "rows", frame.groupby(YEAR).size())
    by_year["all_five"] = all_five.groupby(frame[YEAR]).sum().astype(int)
    if TARGET in frame.columns:
        both = all_five & frame[TARGET].notna()
        by_year["all_five_and_outcome"] = both.groupby(frame[YEAR]).sum().astype(int)
    has_any = by_year[prefs].sum(axis=1) > 0
    print("Years with at least one rank value (counts are non-missing rows):")
    print(by_year[has_any].to_string())

    print("\nCandidate years:")
    for year in CANDIDATE_YEARS:
        if year not in by_year.index:
            print(f"  {year}: not in the file")
        elif by_year.loc[year, "all_five"] == 0:
            print(f"  {year}: in the file, but no row has all five ranks")
        else:
            print(f"  {year}: {by_year.loc[year, 'all_five']:,} rows with all five ranks")

    complete = ranks[all_five]
    if len(complete):
        ordered = np.sort(complete.to_numpy(), axis=1)
        valid = (ordered == np.arange(1, 6)).all(axis=1)
        share = pd.Series(valid, index=complete.index).groupby(frame.loc[complete.index, YEAR]).mean()
        print("\nShare of fully ranked rows that are a valid 1-5 ranking (each rank used once):")
        print(share.round(3).to_string())
        print("A value well below 1.0 means the coding is not what I assumed. Check the codebook.")
    return ranks


def check_funnel(frame: pd.DataFrame, ranks: pd.DataFrame, onet: list[str]) -> None:
    section("Check 2: usable rows in the candidate years")
    sub = frame[frame[YEAR].isin(CANDIDATE_YEARS)]
    if sub.empty:
        print("No rows in the candidate years.")
        return
    occupation = pd.to_numeric(sub[OCCUPATION], errors="coerce") if OCCUPATION in sub else None
    steps = {"rows in year": pd.Series(True, index=sub.index)}
    if len(ranks.columns):
        steps["+ all five ranks"] = ranks.loc[sub.index].notna().all(axis=1)
    if TARGET in sub:
        steps["+ outcome (SATJOB_BINARY)"] = sub[TARGET].notna()
    if occupation is not None:
        steps["+ valid occupation code"] = occupation.notna() & (occupation > 0)
    if onet:
        steps["+ all O*NET values"] = sub[onet].notna().all(axis=1)
    if WEIGHT in sub:
        steps["+ positive survey weight"] = pd.to_numeric(sub[WEIGHT], errors="coerce").gt(0)

    running = pd.Series(True, index=sub.index)
    rows = {}
    for name, mask in steps.items():
        running &= mask
        rows[name] = running.groupby(sub[YEAR]).sum()
    funnel = pd.DataFrame(rows).T.reindex(columns=CANDIDATE_YEARS).fillna(0).astype(int)
    funnel["total"] = funnel.sum(axis=1)
    print("Rows remaining after each requirement is added (cumulative):")
    print(funnel.to_string())
    final = funnel.iloc[-1]
    print(f"\nUsable rows: {final['total']:,} total")
    print("By year: " + ", ".join(f"{y}: {final[y]:,}" for y in CANDIDATE_YEARS))

    if onet:
        print("\nShare of rows missing each O*NET column, by year:")
        missing = sub[onet].isna().groupby(sub[YEAR]).mean().round(3)
        print(missing.T.to_string())
        if occupation is not None:
            coded = occupation.notna() & (occupation > 0)
            unmatched = sub[coded & sub[onet].isna().any(axis=1)]
            print(f"\nRows with an occupation code but no O*NET match: {len(unmatched):,} "
                  f"of {int(coded.sum()):,}")
            if len(unmatched):
                print("Most common unmatched OCC10 codes:")
                print(unmatched[OCCUPATION].value_counts().head(15).to_string())
    if OCCUPATION in sub:
        print("\nShare of rows with no valid occupation code, by year:")
        bad = ~(occupation.notna() & (occupation > 0))
        print(bad.groupby(sub[YEAR]).mean().round(3).to_string())


def main() -> None:
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(frame):,} rows from {DATA_PATH.name}")
    if YEAR not in frame.columns:
        print(f"No {YEAR} column, so nothing more can be checked.")
        return
    prefs, onet = check_columns(frame)
    ranks = check_ranks(frame, prefs)
    check_funnel(frame, ranks, onet)


if __name__ == "__main__":
    main()