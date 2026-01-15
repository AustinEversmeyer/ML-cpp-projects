#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from preprocessing_utilities import combine_inputs

from preprocess import DEFAULT_TIME_FIELDS, transform_columns
from tests.test_preprocess_and_analysis import HEADERLESS_SCHEMA  # noqa: E402


def main() -> None:
    """
    Confirm schema logic against a headered file and the headerless sample together.
    """
    header_file = TOOLS_DIR / "examples" / "sample1.csv"
    headerless_file = TOOLS_DIR / "examples" / "headerless_schema_sample.csv"
    schema = HEADERLESS_SCHEMA.get(headerless_file.name, {})

    inputs = [header_file, headerless_file]
    df, summary = combine_inputs(
        inputs,
        None,
        None,
        False,
        False,
        DEFAULT_TIME_FIELDS,
        transform_columns,
        HEADERLESS_SCHEMA,
    )

    print("DEFAULT_PER_FILE_SCHEMA test (mixed header usage)")
    for idx, (path, info) in enumerate(zip(inputs, summary["tables"])):
        label = "headerless" if idx == 1 else "headered"
        print(f"  {label} input: {path.name}")
        print(f"    schema_applied={info['schema_applied']}  has_header={info['has_header']}")
        if path == headerless_file and info.get("assigned_columns"):
            print(f"    assigned_columns={info['assigned_columns']}")
        if path == headerless_file and not info["schema_applied"]:
            raise ValueError("Headerless input did not pick up the schema.")
        if path == header_file and info["schema_applied"]:
            raise ValueError("Headered input unexpectedly applied the schema.")

    print(f"  schema for headerless file: {schema}")
    print(f"  data columns: {list(df.columns)}")
    print("  sample rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
