"""Link Gallup occupation codes to an O*NET-SOC crosswalk.

The module keeps source occupation fields, performs exact code matching, and
produces respondent-level audit fields without duplicating respondents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_SENTINELS = frozenset(
    {"", "na", "nan", "none", "null", "no response", "missing", "-98", "-99"}
)


def normalize_occupation_code(
    value: object,
    *,
    width: int | None = None,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
) -> str | None:
    """Return a canonical string code, or ``None`` for missing/invalid input.

    Decimal strings are accepted only when their fractional part is zero.
    Codes are never converted through floating point, which preserves leading
    zeros before optional explicit padding.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    if text.lower() in {str(item).strip().lower() for item in sentinels}:
        return None
    if not text.isdigit():
        return None
    if width is not None:
        if width < 1:
            raise ValueError("width must be positive")
        text = text.zfill(width)
    return text


def normalize_code_column(
    series: pd.Series,
    *,
    width: int | None = None,
) -> pd.Series:
    """Normalize a pandas code column while retaining pandas' nullable string type."""
    return series.map(
        lambda value: normalize_occupation_code(value, width=width),
        na_action=None,
    ).astype("string")


def validate_code_column(
    series: pd.Series,
    *,
    width: int | None = None,
) -> pd.DataFrame:
    """Summarize valid, missing, and invalid occupation code values."""
    normalized = normalize_code_column(series, width=width)
    raw = series.astype("string").str.strip()
    missing = raw.isna() | raw.str.lower().isin(DEFAULT_SENTINELS)
    return pd.DataFrame(
        {
            "raw_value": raw,
            "normalized_code": normalized,
            "status": np.select(
                [missing, normalized.notna()],
                ["missing", "valid"],
                default="invalid",
            ),
        }
    )


def _read_table(path: Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as error:
            raise ImportError(
                "Parquet output requires pyarrow or fastparquet; use a .csv "
                "output path when those optional dependencies are unavailable."
            ) from error
        return
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError("output must end in .parquet or .csv")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {', '.join(missing)}")


def prepare_crosswalk(
    crosswalk: pd.DataFrame,
    *,
    source_code_col: str,
    onet_code_col: str,
    source_code_width: int | None = None,
    numeric_attributes: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalize and aggregate a crosswalk to one row per source code."""
    attributes = list(numeric_attributes)
    _require_columns(
        crosswalk,
        [source_code_col, onet_code_col, *attributes],
        "Crosswalk",
    )
    prepared = crosswalk.copy()
    prepared["source_code"] = normalize_code_column(
        prepared[source_code_col], width=source_code_width
    )
    prepared["onet_code"] = prepared[onet_code_col].astype("string").str.strip()
    prepared = prepared[
        prepared["source_code"].notna() & prepared["onet_code"].notna()
    ].copy()
    for column in attributes:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    aggregation: dict[str, str] = {
        "onet_code": lambda values: "|".join(sorted(set(values.dropna().astype(str))))
    }
    aggregation.update({column: "mean" for column in attributes})
    grouped = prepared.groupby("source_code", as_index=False).agg(aggregation)
    counts = (
        prepared.groupby("source_code")["onet_code"]
        .nunique()
        .rename("candidate_count")
    )
    grouped = grouped.merge(counts, on="source_code", how="left", validate="one_to_one")
    return grouped


def link_respondents(
    respondents: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    respondent_code_col: str,
    source_code_col: str,
    onet_code_col: str,
    source_code_width: int | None = None,
    numeric_attributes: Iterable[str] = (),
) -> pd.DataFrame:
    """Left-join respondents to an aggregated exact-code crosswalk."""
    _require_columns(respondents, [respondent_code_col], "Respondents")
    prepared = prepare_crosswalk(
        crosswalk,
        source_code_col=source_code_col,
        onet_code_col=onet_code_col,
        source_code_width=source_code_width,
        numeric_attributes=numeric_attributes,
    )
    result = respondents.copy()
    result["occupation_code_normalized"] = normalize_code_column(
        result[respondent_code_col], width=source_code_width
    )
    result = result.merge(
        prepared,
        left_on="occupation_code_normalized",
        right_on="source_code",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    result["match_status"] = np.where(
        result["_merge"].eq("both"), "matched", "unmatched"
    )
    result["match_method"] = np.where(
        result["match_status"].eq("matched"), "exact_code", "none"
    )
    result["candidate_count"] = result["candidate_count"].fillna(0).astype("int64")
    result["match_confidence"] = np.where(
        result["match_status"].eq("matched"), 1.0, 0.0
    )
    return result.drop(columns=["source_code", "_merge"])


def linkage_diagnostics(
    linked: pd.DataFrame,
    *,
    respondent_code_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and source-code-level linkage coverage diagnostics."""
    _require_columns(
        linked,
        [
            respondent_code_col,
            "occupation_code_normalized",
            "match_status",
            "candidate_count",
        ],
        "Linked respondents",
    )
    overall = pd.DataFrame(
        [
            {
                "respondent_count": len(linked),
                "matched_count": int(linked["match_status"].eq("matched").sum()),
                "unmatched_count": int(linked["match_status"].eq("unmatched").sum()),
                "match_rate": float(linked["match_status"].eq("matched").mean())
                if len(linked)
                else 0.0,
            }
        ]
    )
    by_code = (
        linked.groupby("occupation_code_normalized", dropna=False)
        .agg(
            respondent_count=(respondent_code_col, "size"),
            matched_count=("match_status", lambda values: int(values.eq("matched").sum())),
            candidate_count=("candidate_count", "max"),
        )
        .reset_index()
    )
    by_code["match_rate"] = (
        by_code["matched_count"] / by_code["respondent_count"]
    )
    return overall, by_code


def write_provenance(
    output_path: Path,
    *,
    input_paths: Iterable[Path],
    crosswalk_path: Path,
    source_code_col: str,
    onet_code_col: str,
    source_code_width: int | None,
    sheet_name: str | int,
) -> None:
    """Write reproducibility metadata beside a linkage output."""
    def file_metadata(path: Path) -> dict[str, object]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"path": str(path), "sha256": digest}

    metadata = {
        "input_files": [file_metadata(path) for path in input_paths],
        "crosswalk": file_metadata(crosswalk_path),
        "crosswalk_sheet": sheet_name,
        "source_code_column": source_code_col,
        "onet_code_column": onet_code_col,
        "source_code_width": source_code_width,
        "match_scope": "exact_code_only",
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_audit_outputs(
    output_path: Path,
    overall: pd.DataFrame,
    by_code: pd.DataFrame,
) -> None:
    """Write audit tables as Excel or as two CSV files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        overall.to_csv(
            output_path.with_name(f"{output_path.stem}_overall.csv"), index=False
        )
        by_code.to_csv(
            output_path.with_name(f"{output_path.stem}_by_source_code.csv"),
            index=False,
        )
        return
    if output_path.suffix.lower() != ".xlsx":
        raise ValueError("audit-output must end in .xlsx or .csv")
    try:
        with pd.ExcelWriter(output_path) as writer:
            overall.to_excel(writer, sheet_name="overall", index=False)
            by_code.to_excel(writer, sheet_name="by_source_code", index=False)
    except ImportError as error:
        raise ImportError(
            "Excel audit output requires an installed pandas Excel writer; "
            "use a .csv audit-output path to avoid that optional dependency."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--respondent-code-col", required=True)
    parser.add_argument("--source-code-col", required=True)
    parser.add_argument("--onet-code-col", required=True)
    parser.add_argument("--attribute", action="append", default=[])
    parser.add_argument("--source-code-width", type=int)
    parser.add_argument("--sheet-name", default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheet_name: str | int = (
        int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    )
    respondents = _read_table(args.respondents)
    crosswalk = _read_table(args.crosswalk, sheet_name=sheet_name)
    linked = link_respondents(
        respondents,
        crosswalk,
        respondent_code_col=args.respondent_code_col,
        source_code_col=args.source_code_col,
        onet_code_col=args.onet_code_col,
        source_code_width=args.source_code_width,
        numeric_attributes=args.attribute,
    )
    overall, by_code = linkage_diagnostics(
        linked, respondent_code_col=args.respondent_code_col
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_table(linked, args.output)
    write_audit_outputs(args.audit_output, overall, by_code)
    write_provenance(
        args.output.with_suffix(".metadata.json"),
        input_paths=[args.respondents],
        crosswalk_path=args.crosswalk,
        source_code_col=args.source_code_col,
        onet_code_col=args.onet_code_col,
        source_code_width=args.source_code_width,
        sheet_name=sheet_name,
    )


if __name__ == "__main__":
    main()
