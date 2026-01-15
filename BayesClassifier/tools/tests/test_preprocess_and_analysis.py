#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")

import analyze_predictions as analysis
from preprocess import DEFAULT_TIME_FIELDS, run_preprocess

HEADERLESS_SCHEMA: dict[str, dict[str, Any]] = {
    "headerless_schema_sample.csv": {
        "has_header": False,
        "columns": ["timestep", "temperature", "pressure"],
    }
}


def _build_sample_predictions() -> pd.DataFrame:
    """
    Create a tiny prediction CSV that exercises accuracy, confusion matrices,
    log-loss / ROC AUC, and the binary good/bad collapse.
    """
    rows = [
        ("ClassA", "ClassA", [0.9, 0.05, 0.05]),
        ("ClassA", "ClassB", [0.4, 0.5, 0.1]),
        ("ClassB", "ClassB", [0.1, 0.8, 0.1]),
        ("ClassB", "ClassC", [0.2, 0.3, 0.5]),
        ("ClassC", "ClassB", [0.1, 0.6, 0.3]),
    ]
    labels = ["ClassA", "ClassB", "ClassC"]
    records: list[dict[str, float | str]] = []

    for truth, pred, probs in rows:
        record = {
            "truth_label": truth,
            "predicted_class": pred,
            "predicted_prob": probs[labels.index(pred)],
        }
        for label, prob in zip(labels, probs):
            record[f"prob_{label}"] = prob
        records.append(record)

    return pd.DataFrame.from_records(records)


def _assert_files_exist(paths: Iterable[Path], base_dir: Path) -> None:
    missing = [str(p.relative_to(base_dir)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Analysis outputs missing: {', '.join(missing)}")


def main() -> None:
    base = TOOLS_DIR
    temp_root = Path(tempfile.mkdtemp(prefix="preprocess-analysis-"))
    print(f"Running preprocess test; outputs stored under {temp_root}")

    combined_path = temp_root / "combined_output.csv"
    inputs = [
        str(base / "examples" / "sample1.csv"),
        str(base / "examples" / "sample2.txt"),
        str(base / "examples" / "headerless_schema_sample.csv"),
    ]

    combined_df = run_preprocess(
        inputs=inputs,
        output_path=combined_path,
        delimiter=None,
        cap_to_min=False,
        time_align=False,
        time_fields=DEFAULT_TIME_FIELDS,
        per_file_schema=HEADERLESS_SCHEMA,
    )
    if combined_df.empty:
        raise ValueError("Preprocessed DataFrame is empty; expected combined data.")
    if "truth" not in combined_df.columns:
        raise ValueError("Preprocessed output missing the expected 'truth' column.")
    print(f"  - combined {len(combined_df)} rows from {len(inputs)} files")
    print(f"  - saved combined CSV to {combined_path}")

    predictions = _build_sample_predictions()
    predictions_path = temp_root / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    analysis_output = temp_root / "analysis_outputs"
    analysis.DEFAULT_CSV_PATH = str(predictions_path)
    analysis.DEFAULT_OUTPUT_DIR = str(analysis_output)
    analysis.DEFAULT_SHOW = False
    analysis.DEFAULT_GOOD_CLASSES = ["ClassA"]
    analysis.DEFAULT_CAST_LABELS_TO_CATEGORY = True

    print("Running analyze_predictions on the synthetic CSV ...")
    analysis.main()

    required_files = [
        analysis_output / "classification_report.csv",
        analysis_output / "metrics_summary.json",
        analysis_output / "confusion_matrix_raw.png",
        analysis_output / "confusion_matrix_normalized.png",
        analysis_output / "confusion_counts.png",
        analysis_output / "confusion_counts_binary_good_bad.png",
        analysis_output / "top_misclassifications.csv",
    ]
    _assert_files_exist(required_files, analysis_output)

    metrics_path = analysis_output / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("samples") != len(predictions):
        raise ValueError("Metrics summary reports unexpected sample count.")

    report_path = analysis_output / "classification_report.csv"
    report = pd.read_csv(report_path)
    if "precision" not in report.columns:
        raise ValueError("Classification report missing precision column.")

    print(f"  - analysis outputs written to {analysis_output}")
    print("✅ preprocess + analyze_predictions scripts executed successfully.")


if __name__ == "__main__":
    main()
