import csv
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd


def detect_delimiter(sample_lines: List[str], fallback: str = ",") -> str:
    cleaned = [line.lstrip() for line in sample_lines if line.strip()]
    if not cleaned:
        return fallback

    sample = "".join(cleaned)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", " ", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        if " " in sample:
            return " "
        return fallback


def normalize_features(features: Optional[List[str] | str]) -> Optional[List[str]]:
    if features is None:
        return None
    if isinstance(features, str):
        return [features]
    return list(features)


def _schema_for_path(
    path: Path, per_file_schema: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    if not per_file_schema:
        return {}

    keys = [str(path), path.name, path.stem]
    for key in keys:
        if key in per_file_schema:
            value = per_file_schema[key]
            return dict(value) if isinstance(value, dict) else {}
    return {}


def _fill_header_names(
    count: int, provided: Optional[List[Any]] = None, prefix: str = "column"
) -> List[str]:
    """
    Create column names for headerless files.

    Rules:
    - If provided is None: all columns become f"{prefix}{i}".
    - If provided has entries:
        - None / "" -> becomes f"{prefix}{i}"
        - otherwise str(value) is used.
    - If provided is shorter than count: remaining are filled with f"{prefix}{i}".
    - If provided is longer than count: extra names are ignored.
    """
    result: List[str] = []
    provided_list: List[Any] = list(provided) if provided is not None else []
    for i in range(count):
        name = provided_list[i] if i < len(provided_list) else None
        if name is None:
            result.append(f"{prefix}{i}")
        else:
            s = str(name).strip()
            result.append(s if s else f"{prefix}{i}")
    return result


def _coerce_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().all():
            # nothing convertible; keep original
            continue
        # keep originals where conversion failed (including real NaNs), otherwise use numeric
        df[col] = df[col].where(converted.isna(), converted)
    return df


def _combine_horizontally(
    tables: List[pd.DataFrame],
    time_columns: List[Optional[str]],
    paths: List[Path],
    align_strategy: str,
) -> pd.DataFrame:
    if any(col is None for col in time_columns):
        missing = [str(paths[i]) for i, col in enumerate(time_columns) if col is None]
        raise ValueError(f"Horizontal combine requires a time column in all inputs. Missing: {', '.join(missing)}")

    time_union: set[float] = set()
    numeric_time_columns: List[pd.Series] = []
    for df, tcol in zip(tables, time_columns):
        ts = pd.to_numeric(df[tcol], errors="coerce")
        time_union.update(ts.dropna().tolist())
        numeric_time_columns.append(ts)

    if not time_union:
        raise ValueError("No valid time values found for horizontal combination.")

    union_index = pd.Index(sorted(time_union), name="time")
    merged = pd.DataFrame(index=union_index)
    align_strategy = align_strategy.lower()

    for path, df, ts, tcol in zip(paths, tables, numeric_time_columns, time_columns):
        working = df.copy()
        working["__time__"] = ts
        working = working.loc[~working["__time__"].isna()].copy()
        working = working.set_index("__time__")

        drop_cols = [c for c in ["truth", tcol] if c in working.columns]
        working = working.drop(columns=drop_cols, errors="ignore")

        working = working.rename(columns=lambda c: f"{path.stem}:{c}")
        working = working.reindex(union_index)

        if align_strategy == "ffill":
            working = working.ffill()
        elif align_strategy == "interpolate":
            numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
            if numeric_cols:
                working[numeric_cols] = working[numeric_cols].interpolate(method="linear", limit_direction="both")
            working = working.ffill()

        merged = merged.join(working, how="outer")

    combined = merged.reset_index().rename(columns={"time": "time"})
    return combined


def read_table(
    path: Path,
    override_delimiter: Optional[str],
    per_file_schema: Optional[Dict[str, Dict[str, Any]]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with path.open() as f:
        sample_lines: List[str] = []
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            if line.strip():
                sample_lines.append(line)

    schema = _schema_for_path(path, per_file_schema)
    delimiter = schema.get("delimiter") or override_delimiter or detect_delimiter(sample_lines)

    read_kwargs: Dict[str, Any] = {"skipinitialspace": True}
    if delimiter == " ":
        read_kwargs.update({"sep": r"\s+", "engine": "python"})
    else:
        read_kwargs["sep"] = delimiter

    decision: Dict[str, Any] = {
        "delimiter": delimiter,
        "schema_applied": bool(schema),
        "has_header": True,
        "decision_source": "default",
        "assigned_columns": None,
    }

    explicit_has_header = schema.get("has_header")
    explicit_columns = schema.get("columns")

    if explicit_has_header is not None:
        decision["has_header"] = bool(explicit_has_header)
        decision["decision_source"] = "schema.has_header"

    if decision["has_header"]:
        if explicit_columns is not None:
            raise ValueError(
                f"{path}: per_file_schema specifies 'columns' but has_header=True. "
                "Either remove 'columns' or set has_header=False."
            )
        df = pd.read_csv(path, header=0, **read_kwargs)
    else:
        df = pd.read_csv(path, header=None, **read_kwargs)
        names = _fill_header_names(len(df.columns), explicit_columns, prefix="column")
        df.columns = names
        decision["assigned_columns"] = names
        if explicit_columns is None:
            decision["decision_source"] = "schema.has_header (no columns)"
        else:
            decision["decision_source"] = "schema.has_header + columns"

    df = df.dropna(how="all")
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.map(lambda v: v.strip() if isinstance(v, str) else v)
    df = _coerce_cells(df)
    return df.reset_index(drop=True), decision


def resolve_features(features: Optional[List[str]], available: List[str]) -> List[str]:
    if features is None:
        return list(available)

    resolved: List[str] = []
    seen: set[str] = set()

    for feat in features:
        if feat not in seen:
            resolved.append(feat)
            seen.add(feat)

    if "truth" in available and "truth" not in resolved:
        resolved.insert(0, "truth")

    return resolved


def fuzzy_match_features(
    df: pd.DataFrame, targets: List[str], cutoff: float = 0.5
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Map target names onto existing columns using substring and fuzzy matching.
    Returns the possibly-augmented DataFrame and a list of (target, matched_column) pairs applied.
    """
    available = list(df.columns)
    lower_available = {col.lower(): col for col in available if isinstance(col, str)}
    applied: List[Tuple[str, str]] = []

    for feat in targets:
        if feat in df.columns:
            continue
        key = feat.lower() if isinstance(feat, str) else str(feat)

        substr_match = None
        for lower_name, orig_name in lower_available.items():
            if key in lower_name or lower_name in key:
                substr_match = orig_name
                break

        if substr_match:
            df[feat] = df[substr_match]
            applied.append((feat, substr_match))
            continue

        match = difflib.get_close_matches(key, list(lower_available.keys()), n=1, cutoff=cutoff)
        if match:
            source = lower_available[match[0]]
            df[feat] = df[source]
            applied.append((feat, source))
    return df, applied


def find_time_column(df: pd.DataFrame, candidates: List[str], cutoff: float = 0.5) -> Optional[str]:
    """
    Attempt to locate a time/timestep column using exact, substring, then fuzzy matching.
    Returns the column name if found, else None.
    """
    available = list(df.columns)
    lower_available = {col.lower(): col for col in available if isinstance(col, str)}

    for cand in candidates:
        if cand in df.columns:
            return cand
        cand_lower = cand.lower()
        if cand_lower in lower_available:
            return lower_available[cand_lower]

    for cand in candidates:
        key = cand.lower()
        for lower_name, orig_name in lower_available.items():
            if key in lower_name or lower_name in key:
                return orig_name

    for cand in candidates:
        key = cand.lower()
        match = difflib.get_close_matches(key, list(lower_available.keys()), n=1, cutoff=cutoff)
        if match:
            return lower_available[match[0]]
    return None


def time_align_tables(
    tables: List[pd.DataFrame],
    time_columns: List[Optional[str]],
    table_info: List[Dict[str, Any]],
    enabled: bool,
    paths: List[Path],
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    status: Dict[str, Any] = {"enabled": enabled, "window": None, "skipped_reason": None}
    if not enabled or not tables:
        return tables, status

    if any(col is None for col in time_columns):
        missing = [str(paths[i]) for i, col in enumerate(time_columns) if col is None]
        status["skipped_reason"] = f"No time column found in: {', '.join(missing)}"
        return tables, status

    mins: List[float] = []
    maxs: List[float] = []
    for df, tcol in zip(tables, time_columns):
        ts = pd.to_numeric(df[tcol], errors="coerce")
        mins.append(ts.min())
        maxs.append(ts.max())
    start = max(mins)
    end = min(maxs)
    if pd.isna(start) or pd.isna(end) or start > end:
        status["skipped_reason"] = "No overlapping time window."
        return tables, status

    aligned_tables = []
    for info, df, tcol in zip(table_info, tables, time_columns):
        ts = pd.to_numeric(df[tcol], errors="coerce")
        mask = (ts >= start) & (ts <= end)
        aligned_df = df.loc[mask].reset_index(drop=True)
        info["rows_after_time_align"] = len(aligned_df)
        aligned_tables.append(aligned_df)
    status["window"] = (float(start), float(end))
    return aligned_tables, status


def cap_tables_to_min(tables: List[pd.DataFrame], enabled: bool) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    status: Dict[str, Any] = {"enabled": enabled, "applied": False, "min_len": None}
    if not enabled or not tables:
        return tables, status

    min_len = min(len(df) for df in tables)
    tables = [df.iloc[:min_len].reset_index(drop=True) for df in tables]
    status["applied"] = True
    status["min_len"] = int(min_len)
    return tables, status


def combine_inputs(
    paths: List[Path],
    features: Optional[List[str]],
    delimiter: Optional[str],
    cap_to_min: bool,
    time_align: bool,
    time_fields: List[str],
    transform_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    per_file_schema: Optional[Dict[str, Dict[str, Any]]],
    combine_mode: str = "vertical",
    align_strategy: str = "none",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tables: List[pd.DataFrame] = []
    time_columns: List[Optional[str]] = []
    table_info: List[Dict[str, Any]] = []

    for path in paths:
        df, header_decision = read_table(path, delimiter, per_file_schema)
        df["truth"] = path.stem
        df = transform_fn(df, path.stem)
        df["truth"] = path.stem  # ensure truth survives user transforms
        time_col = find_time_column(df, time_fields)
        tables.append(df)
        time_columns.append(time_col)
        table_info.append(
            {
                "path": str(path),
                "rows_initial": len(df),
                "time_column": time_col,
                "delimiter": header_decision.get("delimiter"),
                "has_header": header_decision.get("has_header"),
                "header_decision_source": header_decision.get("decision_source"),
                "assigned_columns": header_decision.get("assigned_columns"),
                "schema_applied": header_decision.get("schema_applied"),
            }
        )

    summary: Dict[str, Any] = {
        "time_align": {"enabled": time_align, "window": None, "skipped_reason": None},
        "cap_to_min": {"enabled": cap_to_min, "applied": False, "min_len": None},
        "fuzzy_matches": [],
        "selected_features": None,
        "combine_mode": combine_mode,
        "align_strategy": align_strategy,
        "tables": table_info,
    }

    if not tables:
        return pd.DataFrame(), summary

    tables, summary["time_align"] = time_align_tables(tables, time_columns, table_info, time_align, paths)

    if combine_mode not in ("vertical", "horizontal"):
        raise ValueError(f"Unknown combine_mode={combine_mode}; expected 'vertical' or 'horizontal'.")

    if combine_mode == "horizontal":
        combined_df = _combine_horizontally(tables, time_columns, paths, align_strategy)
    else:
        tables, cap_status = cap_tables_to_min(tables, cap_to_min)
        summary["cap_to_min"].update(cap_status)
        if cap_status.get("min_len") is not None:
            for info in table_info:
                info["rows_after_cap"] = cap_status["min_len"]
        combined_df = pd.concat(tables, axis=0, ignore_index=True, sort=False)

    selected_features = resolve_features(features, list(combined_df.columns))
    combined_df, matches = fuzzy_match_features(combined_df, selected_features)
    summary["fuzzy_matches"] = matches
    for feature in selected_features:
        if feature not in combined_df.columns:
            combined_df[feature] = pd.NA
    combined_df = combined_df[selected_features]
    summary["selected_features"] = selected_features

    return combined_df, summary


def write_csv(df: pd.DataFrame, output_path: Path):
    if df.empty:
        print("No rows to write; exiting.", file=sys.stderr)
        return

    df.to_csv(output_path, index=False, na_rep="")


def print_summary(summary: Dict[str, Any], input_paths: List[Path]):
    print("\nPreprocess summary:")
    print(f"  inputs: {', '.join(str(p) for p in input_paths)}")
    print(f"  combine_mode: {summary.get('combine_mode')}")
    print(f"  align_strategy: {summary.get('align_strategy')}")
    print(f"  selected_features: {summary.get('selected_features')}")
    if summary["fuzzy_matches"]:
        print("  fuzzy matches:")
        for target, source in summary["fuzzy_matches"]:
            print(f"    {target} <- {source}")
    else:
        print("  fuzzy matches: none")

    if summary["time_align"]["enabled"]:
        if summary["time_align"]["window"]:
            start, end = summary["time_align"]["window"]
            print(f"  time_align: applied window [{start}, {end}]")
        else:
            print(f"  time_align: skipped ({summary['time_align']['skipped_reason']})")
    else:
        print("  time_align: disabled")

    if summary["cap_to_min"]["enabled"]:
        if summary["cap_to_min"]["applied"]:
            print(f"  cap_to_min: applied length {summary['cap_to_min']['min_len']}")
        else:
            print("  cap_to_min: enabled but not applied")
    else:
        print("  cap_to_min: disabled")

    for info in summary["tables"]:
        msg = f"  table {info['path']}: rows={info.get('rows_initial')}"
        if "rows_after_time_align" in info:
            msg += f" -> after time_align {info['rows_after_time_align']}"
        if "rows_after_cap" in info:
            msg += f" -> after cap {info['rows_after_cap']}"
        msg += f", time_col={info.get('time_column')}"
        msg += f", has_header={info.get('has_header')} ({info.get('header_decision_source')})"
        msg += f", delimiter={repr(info.get('delimiter'))}"
        if info.get("assigned_columns"):
            msg += f", assigned_columns={info.get('assigned_columns')}"
        if info.get("schema_applied"):
            msg += ", schema_applied=True"
        print(msg)
