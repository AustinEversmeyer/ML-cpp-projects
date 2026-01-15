#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from preprocessing_utilities import combine_inputs  # noqa: E402
from preprocess import DEFAULT_TIME_FIELDS, transform_columns  # noqa: E402
from tests.test_preprocess_and_analysis import HEADERLESS_SCHEMA  # noqa: E402


def _paths() -> list[Path]:
    base = TOOLS_DIR / "examples"
    return [
        base / "sample1.csv",
        base / "sample2.txt",
        base / "headerless_schema_sample.csv",
    ]


def _assert_vertical(df: pd.DataFrame) -> None:
    expected_rows = 3 + 5 + 3
    if len(df) != expected_rows:
        raise AssertionError(f"Vertical combine should have {expected_rows} rows, found {len(df)}")
    if "truth" not in df.columns:
        raise AssertionError("Vertical combine should retain 'truth' column.")


def _assert_horizontal(df: pd.DataFrame) -> None:
    expected_times = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0}
    times = set(pd.to_numeric(df["time"], errors="coerce").dropna().tolist())
    if times != expected_times:
        raise AssertionError(f"Horizontal combine time grid mismatch. Expected {expected_times}, got {times}")
    if "truth" in df.columns:
        raise AssertionError("Horizontal combine should drop truth column prefixes.")

    stem_prefixes = ["sample1:", "sample2:", "headerless_schema_sample:"]
    for prefix in stem_prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            raise AssertionError(f"Missing columns for prefix {prefix} after horizontal combine.")
        if not df[cols].notna().any().any():
            raise AssertionError(f"All values are NaN for columns under prefix {prefix}.")


def _assert_interpolated(df: pd.DataFrame) -> None:
    # For headerless file, times 0.0, 0.5, 1.0 were present. After interpolation+ffill,
    # we expect non-null values across the full grid 0.0..4.0 for its columns.
    cols = [c for c in df.columns if c.startswith("headerless_schema_sample:temperature")]
    if not cols:
        raise AssertionError("Missing interpolated temperature column for headerless file.")
    temp_col = df[cols[0]]
    if temp_col.isna().any():
        raise AssertionError("Interpolated temperature column contains NaNs after interpolate/ffill.")


def main() -> None:
    paths = _paths()

    df_vert, summary_vert = combine_inputs(
        paths,
        None,
        None,
        cap_to_min=False,
        time_align=False,
        time_fields=DEFAULT_TIME_FIELDS,
        transform_fn=transform_columns,
        per_file_schema=HEADERLESS_SCHEMA,
        combine_mode="vertical",
        align_strategy="none",
    )
    print("Vertical combine: OK")
    _assert_vertical(df_vert)
    if summary_vert.get("combine_mode") != "vertical":
        raise AssertionError("Vertical summary combine_mode mismatch.")
    print(f"Vertical shape: {df_vert.shape}")
    print("Vertical preview:")
    print(df_vert.head())

    df_horiz, summary_horiz = combine_inputs(
        paths,
        None,
        None,
        cap_to_min=False,
        time_align=False,
        time_fields=DEFAULT_TIME_FIELDS,
        transform_fn=transform_columns,
        per_file_schema=HEADERLESS_SCHEMA,
        combine_mode="horizontal",
        align_strategy="ffill",
    )
    print("Horizontal combine (ffill): OK")
    _assert_horizontal(df_horiz)
    if summary_horiz.get("combine_mode") != "horizontal":
        raise AssertionError("Horizontal summary combine_mode mismatch.")
    if summary_horiz.get("align_strategy") != "ffill":
        raise AssertionError("Horizontal summary align_strategy mismatch.")
    preview_cols = [c for c in df_horiz.columns if c.endswith(":temperature")][:3]
    print("Horizontal (ffill) preview:")
    print(df_horiz[["time"] + preview_cols].head())

    df_interp, summary_interp = combine_inputs(
        paths,
        None,
        None,
        cap_to_min=False,
        time_align=False,
        time_fields=DEFAULT_TIME_FIELDS,
        transform_fn=transform_columns,
        per_file_schema=HEADERLESS_SCHEMA,
        combine_mode="horizontal",
        align_strategy="interpolate",
    )
    print("Horizontal combine (interpolate): OK")
    _assert_horizontal(df_interp)
    _assert_interpolated(df_interp)
    if summary_interp.get("align_strategy") != "interpolate":
        raise AssertionError("Interpolate summary align_strategy mismatch.")
    preview_cols_interp = [c for c in df_interp.columns if c.endswith(":temperature")][:3]
    print("Horizontal (interpolate) preview:")
    print(df_interp[["time"] + preview_cols_interp].head())

    print("✅ combine_modes tests passed.")


if __name__ == "__main__":
    main()
