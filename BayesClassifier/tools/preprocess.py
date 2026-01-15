#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from preprocessing_utilities import (
    combine_inputs,
    normalize_features,
    print_summary,
    write_csv,
)

# ---------- Editable defaults (override by passing args to run_preprocess) ----------
DEFAULT_INPUTS: List[str] = [
    str(Path(__file__).parent / "examples" / "sample1.csv"),
    str(Path(__file__).parent / "examples" / "sample2.txt"),
]
DEFAULT_FEATURES: Optional[List[str]] = None
DEFAULT_OUTPUT: str = str(Path(__file__).parent / "examples" / "combined_output.csv")
DEFAULT_DELIMITER: Optional[str] = None
DEFAULT_CAP_TO_MIN: bool = False
DEFAULT_TIME_ALIGN: bool = False
DEFAULT_TIME_FIELDS: List[str] = ["time", "timestep", "timestamp", "time_val", "t"]
DEFAULT_COMBINE_MODE: str = "vertical"  # "vertical" stacks rows; "horizontal" joins on time
DEFAULT_ALIGN_STRATEGY: str = "none"  # "none", "ffill", "interpolate"
DEFAULT_PER_FILE_SCHEMA: Dict[str, Dict[str, Any]] = {}
# ----------------------------------------------------------------

DF_INFO: Dict[str, Dict[str, Any]] = {
    # "sample1": {"length_val": 123, "constant_val": 42},
    # "sample2": {"status": "ok"},
}


def transform_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Hook to edit entire columns at once for a single input file. Return the modified DataFrame.

    Example tweaks you can add here:
        if source_name == "sensorA":
            df["temperature"] = df["temperature"] - 273.15
        df["ratio"] = df["f1"] / (df["f2"] + 1e-9)
        return df[["temperature", "ratio"]]  # re-order / drop columns if desired
    """
    return df  # default: pass-through


def run_preprocess(
    inputs: Optional[List[str]] = None,
    features: Optional[List[str] | str] = None,
    output_path: Optional[str | Path] = None,
    delimiter: Optional[str] = None,
    cap_to_min: bool = DEFAULT_CAP_TO_MIN,
    time_align: bool = DEFAULT_TIME_ALIGN,
    time_fields: Optional[List[str]] = None,
    per_file_schema: Optional[Dict[str, Dict[str, Any]]] = None,
    combine_mode: str = DEFAULT_COMBINE_MODE,
    align_strategy: str = DEFAULT_ALIGN_STRATEGY,
) -> pd.DataFrame:
    paths = [Path(p) for p in (inputs if inputs is not None else DEFAULT_INPUTS)]
    feats = normalize_features(features if features is not None else DEFAULT_FEATURES)
    out_path = Path(output_path) if output_path is not None else Path(DEFAULT_OUTPUT)
    delim = delimiter if delimiter is not None else DEFAULT_DELIMITER
    tf = time_fields if time_fields is not None else DEFAULT_TIME_FIELDS
    schema = per_file_schema if per_file_schema is not None else DEFAULT_PER_FILE_SCHEMA

    combined_df, summary = combine_inputs(
        paths,
        feats,
        delim,
        cap_to_min,
        time_align,
        tf,
        transform_columns,
        schema,
        combine_mode,
        align_strategy,
    )
    write_csv(combined_df, out_path)
    print_summary(summary, paths)
    return combined_df


if __name__ == "__main__":
    run_preprocess()
