#!/usr/bin/env python3
"""Collect classification CSVs from a results directory tree, remap integer
truth labels (and predicted classes / probability columns) to human-readable
strings using per-group mappings, and produce merged CSVs ready for analysis.

Expected directory layout (group name can appear at any depth)::

    results_root/
        .../group1/.../Run1/.../classifications.csv
        .../group1/.../Run2/.../classifications.csv
        .../group2/.../Run1/.../classifications.csv
        ...

A JSON mapping file supplies per-group label maps::

    {
        "group1": {"0": "Cat", "1": "Dog"},
        "group2": {"0": "Healthy", "1": "Faulty", "2": "Degraded"}
    }

Outputs (written to a single output directory):
    - <group>_merged.csv   per-group merged file
    - all_merged.csv       collective file across all groups
    - A "source_file" column is added so each row is traceable.
    - A "group" column is added to identify the group.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------- Editable defaults ----------
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_MAPPING_FILE = Path("config/label_mappings.json")
DEFAULT_OUTPUT_DIR = Path("tools/outputs/collected_results")
DEFAULT_CSV_FILENAME = "classifications.csv"
DEFAULT_TRUTH_COLUMN = "truth_label"
DEFAULT_PRED_COLUMN = "predicted_class"
DEFAULT_PROB_PREFIX = "prob_"
# ----------------------------------------


def log(msg: str) -> None:
    print(f"[collect] {msg}")


def load_mappings(mapping_path: Path) -> Dict[str, Dict[str, str]]:
    """Load the JSON mapping file. Returns {group_name: {old_label: new_label}}."""
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Mapping file must be a JSON object, got {type(raw).__name__}")
    for group, mapping in raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"Mapping for group '{group}' must be a JSON object, got {type(mapping).__name__}"
            )
        raw[group] = {str(k): str(v) for k, v in mapping.items()}
    return raw


def detect_group(
    csv_path: Path,
    results_root: Path,
    known_groups: List[str],
) -> Optional[str]:
    """Determine the group name from the path.

    Scans every directory component of *csv_path* (relative to *results_root*)
    for a match against *known_groups* (the keys from the mapping file).
    Returns the first match, or ``None`` if no component matches.

    For example, with ``known_groups=["group1"]``::

        results/some/nested/group1/Run1/classifications.csv  ->  "group1"
    """
    try:
        relative = csv_path.resolve().relative_to(results_root.resolve())
    except ValueError:
        raise ValueError(
            f"CSV path {csv_path} is not under results root {results_root}"
        )
    for part in relative.parts:
        if part in known_groups:
            return part
    return None


def find_csv_files(
    results_root: Path,
    csv_filename: str = DEFAULT_CSV_FILENAME,
) -> List[Path]:
    """Recursively find all matching CSV files under *results_root*."""
    found = sorted(results_root.rglob(csv_filename))
    return found


def remap_dataframe(
    df: pd.DataFrame,
    label_map: Dict[str, str],
    truth_col: str = DEFAULT_TRUTH_COLUMN,
    pred_col: str = DEFAULT_PRED_COLUMN,
    prob_prefix: str = DEFAULT_PROB_PREFIX,
) -> pd.DataFrame:
    """Apply label remapping to a single DataFrame.

    Remaps values in *truth_col* and *pred_col*, and renames any
    probability columns whose suffix matches a key in *label_map*.
    """
    df = df.copy()

    if truth_col in df.columns:
        df[truth_col] = df[truth_col].astype(str).map(
            lambda v: label_map.get(v, v)
        )

    if pred_col in df.columns:
        df[pred_col] = df[pred_col].astype(str).map(
            lambda v: label_map.get(v, v)
        )

    rename_cols = {}
    for col in df.columns:
        if col.startswith(prob_prefix):
            suffix = col[len(prob_prefix):]
            if suffix in label_map:
                rename_cols[col] = f"{prob_prefix}{label_map[suffix]}"
    if rename_cols:
        df.rename(columns=rename_cols, inplace=True)

    return df


def run_collect_and_remap(
    results_root: Optional[Path] = None,
    mapping_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    csv_filename: Optional[str] = None,
    truth_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    prob_prefix: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Main entry point.

    Returns a dict of ``{group_name: merged_dataframe, "all": collective_df}``.
    """
    results_root = Path(results_root or DEFAULT_RESULTS_ROOT)
    mapping_file = Path(mapping_file or DEFAULT_MAPPING_FILE)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    csv_filename = csv_filename or DEFAULT_CSV_FILENAME
    truth_col = truth_col or DEFAULT_TRUTH_COLUMN
    pred_col = pred_col or DEFAULT_PRED_COLUMN
    prob_prefix = prob_prefix or DEFAULT_PROB_PREFIX

    log(f"Results root : {results_root}")
    log(f"Mapping file : {mapping_file}")
    log(f"Output dir   : {output_dir}")
    log(f"CSV filename : {csv_filename}")

    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    mappings = load_mappings(mapping_file)
    known_groups = list(mappings.keys())
    log(f"Loaded mappings for groups: {known_groups}")

    csv_files = find_csv_files(results_root, csv_filename)
    log(f"Found {len(csv_files)} CSV file(s)")
    if not csv_files:
        raise FileNotFoundError(
            f"No '{csv_filename}' files found under {results_root}"
        )

    group_frames: Dict[str, List[pd.DataFrame]] = {}

    for csv_path in csv_files:
        group = detect_group(csv_path, results_root, known_groups)
        if group is None:
            log(f"  {csv_path}  ->  SKIPPED (no known group in path)")
            continue
        log(f"  {csv_path}  ->  group '{group}'")

        df = pd.read_csv(csv_path)
        df["source_file"] = str(csv_path)
        df["group"] = group

        df = remap_dataframe(
            df, mappings[group],
            truth_col=truth_col,
            pred_col=pred_col,
            prob_prefix=prob_prefix,
        )

        group_frames.setdefault(group, []).append(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}

    all_dfs = []
    for group, frames in sorted(group_frames.items()):
        merged = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"{group}_merged.csv"
        merged.to_csv(out_path, index=False)
        log(f"Wrote {len(merged)} rows -> {out_path}")
        result[group] = merged
        all_dfs.append(merged)

    collective = pd.concat(all_dfs, ignore_index=True)
    collective_path = output_dir / "all_merged.csv"
    collective.to_csv(collective_path, index=False)
    log(f"Wrote {len(collective)} rows -> {collective_path}")
    result["all"] = collective

    log("Done.")
    return result


if __name__ == "__main__":
    run_collect_and_remap()
