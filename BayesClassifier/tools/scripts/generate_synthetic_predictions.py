#!/usr/bin/env python3
"""Generate prediction CSVs for testing the analysis tool."""

from __future__ import annotations

from pathlib import Path
import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Editable defaults ----------
RUN_WINE = True  # set False to skip generating the wine predictions
RUN_SYNTHETIC = True  # set False to skip the synthetic expansion
RUN_FEATURE_DATA = True  # set False to skip generating feature inputs
RUN_WINE_INPUT = True  # set False to skip generating wine inputs and PDFs

OUTPUT = Path("tools/outputs/synthetic_prediction_temp.csv")
TEST_SIZE = 0.3
RANDOM_STATE = 0
MAX_ITER = 5000

SYNTHETIC_SOURCE = OUTPUT
SYNTHETIC_OUTPUT = Path("tools/outputs/synthetic_predictions.csv")
SYNTHETIC_ROWS = 500_000
SYNTHETIC_NOISE_STD = 0.02
SYNTHETIC_SEED = 0

MODEL_CONFIG = Path("config/model/model.configuration.example.json")
FEATURE_OUTPUT = Path("tools/outputs/synthetic_feature_input.txt")
FEATURE_ROWS_PER_CLASS = 20000
FEATURE_SEED = 0

WINE_INPUT_OUTPUT = Path("tools/outputs/synthetic_wine_input.txt")
WINE_MODEL_CONFIG = Path("tools/outputs/wine_model.configuration.json")
WINE_INFERENCE_CONFIG = Path("tools/outputs/wine_inference_config.json")
WINE_PDF_OUTPUT_DIR = Path("tools/outputs/wine_pdfs")
# --------------------------------------


def generate_wine_predictions(output_path: Path = OUTPUT) -> None:
    print(f"[training] Loading wine dataset ...")
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    labels = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[training] Training logistic regression ...")
    model = LogisticRegression(max_iter=MAX_ITER).fit(X_train_scaled, y_train)
    pred_idx = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)

    df = pd.DataFrame(
        {
            "truth_label": [labels[i] for i in y_test],
            "predicted_class": [labels[i] for i in pred_idx],
            "predicted_prob": proba.max(axis=1),
        }
    )
    for i, label in enumerate(labels):
        df[f"prob_{label}"] = proba[:, i]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[training] Saved predictions to {output_path.resolve()} with shape {df.shape}")


def renormalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Ensure each row sums to 1.0; if a row collapses to 0, use uniform."""
    probs = np.clip(probs, 1e-9, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    probs[~zero_rows] = probs[~zero_rows] / row_sums[~zero_rows]
    if zero_rows.any():
        probs[zero_rows] = 1.0 / probs.shape[1]
    return probs


def generate_synthetic_predictions(
    source_path: Path,
    output_path: Path,
    rows: int,
    noise_std: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if not source_path.exists():
        raise FileNotFoundError(f"Source predictions not found: {source_path}")

    print(f"[synthetic] Loading source predictions from {source_path}")
    base = pd.read_csv(source_path)
    prob_cols: List[str] = [c for c in base.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("No probability columns found (expected columns prefixed with 'prob_').")

    labels = [c[len("prob_") :] for c in prob_cols]

    sampled = base.sample(rows, replace=True, random_state=seed).reset_index(drop=True)
    probs = sampled[prob_cols].to_numpy(dtype=float)

    noise = rng.normal(0.0, noise_std, size=probs.shape)
    probs_noisy = renormalize_probabilities(probs + noise)

    argmax_idx = np.argmax(probs_noisy, axis=1)
    sampled["predicted_class"] = [labels[i] for i in argmax_idx]
    sampled["predicted_prob"] = probs_noisy[np.arange(rows), argmax_idx]
    for i, col in enumerate(prob_cols):
        sampled[col] = probs_noisy[:, i]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)
    print(f"[synthetic] Wrote {len(sampled)} rows to {output_path.resolve()}")


def load_model_config(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")
    with model_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    classes = data.get("classes", [])
    if not classes:
        raise ValueError("Model configuration contains no classes.")

    feature_names = [feat["name"] for feat in classes[0].get("features", [])]
    if not feature_names:
        raise ValueError("First class has no features defined.")

    class_models = []
    for cls in classes:
        class_name = cls.get("name", "Unknown")
        feature_map = {}
        for feat in cls.get("features", []):
            feature_map[feat["name"]] = {
                "type": feat.get("type", "").lower(),
                "params": feat.get("params", {}),
            }
        aligned = [feature_map.get(name) for name in feature_names]
        class_models.append(
            {
                "name": class_name,
                "feature_models": aligned,
            }
        )

    return {"feature_names": feature_names, "classes": class_models}


def sample_feature_value(model_def: dict | None, rng: np.random.Generator) -> float:
    if not model_def:
        return float("nan")
    dist_type = model_def["type"]
    params = model_def["params"]
    if dist_type == "gaussian":
        mean = float(params.get("mean", 0.0))
        sigma = float(params.get("sigma", 1.0))
        return float(rng.normal(mean, sigma))
    if dist_type == "rayleigh":
        sigma = float(params.get("sigma", 1.0))
        return float(rng.rayleigh(sigma))
    return float("nan")


def generate_feature_input(
    model_path: Path,
    output_path: Path,
    rows_per_class: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    model = load_model_config(model_path)
    feature_names = model["feature_names"]
    classes = model["classes"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        header = "timestep truth " + " ".join(feature_names)
        handle.write(header + "\n")
        timestep = 0
        for cls in classes:
            for _ in range(rows_per_class):
                values = [
                    sample_feature_value(model_def, rng)
                    for model_def in cls["feature_models"]
                ]
                feature_text = " ".join(str(value) for value in values)
                line = f"{timestep} {cls['name']} {feature_text}"
                handle.write(line + "\n")
                timestep += 1

    print(
        f"[features] Wrote {rows_per_class * len(classes)} rows to {output_path.resolve()}"
    )


def generate_wine_feature_input(output_path: Path) -> dict:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    labels = list(data.target_names)
    feature_names = list(data.feature_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        header = "timestep truth " + " ".join(feature_names)
        handle.write(header + "\n")
        timestep = 0
        for row_idx in range(len(X)):
            class_name = labels[int(y.iloc[row_idx])]
            values = X.iloc[row_idx].to_list()
            feature_text = " ".join(str(value) for value in values)
            handle.write(f"{timestep} {class_name} {feature_text}\n")
            timestep += 1

    print(f"[features] Wrote wine input to {output_path.resolve()}")
    return {"labels": labels, "feature_names": feature_names}


def generate_wine_model_config(output_path: Path) -> dict:
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    labels = list(data.target_names)
    feature_names = list(data.feature_names)

    classes = []
    for class_index, class_name in enumerate(labels):
        class_rows = X[y == class_index]
        features = []
        for feature_name in feature_names:
            values = class_rows[feature_name].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sigma = float(np.std(values, ddof=1))
            if sigma <= 0.0:
                sigma = 1.0
            features.append(
                {
                    "name": feature_name,
                    "type": "gaussian",
                    "params": {"mean": mean, "sigma": sigma},
                }
            )
        classes.append(
            {
                "name": class_name,
                "prior": 1.0 / len(labels),
                "features": features,
            }
        )

    payload = {"computation_mode": "log", "classes": classes}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"[model] Wrote wine model config to {output_path.resolve()}")
    return {"labels": labels, "feature_names": feature_names}


def generate_wine_inference_config(
    output_path: Path, input_path: Path, model_path: Path, feature_names: list[str]
) -> None:
    payload = {
        "input_file": str(input_path.resolve()),
        "input_format": "text",
        "model_config": str(model_path.resolve()),
        "layout": {
            "truth_field": "truth",
            "feature_fields": feature_names,
            "delimiter": "SPACE",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"[config] Wrote wine inference config to {output_path.resolve()}")


def main() -> None:
    if RUN_WINE:
        generate_wine_predictions(OUTPUT)
    else:
        print("[training] Skipped (RUN_WINE=False)")

    if RUN_SYNTHETIC:
        generate_synthetic_predictions(
            SYNTHETIC_SOURCE,
            SYNTHETIC_OUTPUT,
            rows=SYNTHETIC_ROWS,
            noise_std=SYNTHETIC_NOISE_STD,
            seed=SYNTHETIC_SEED,
        )
    else:
        print("[synthetic] Skipped (RUN_SYNTHETIC=False)")

    if RUN_FEATURE_DATA:
        generate_feature_input(
            MODEL_CONFIG,
            FEATURE_OUTPUT,
            rows_per_class=FEATURE_ROWS_PER_CLASS,
            seed=FEATURE_SEED,
        )
    else:
        print("[features] Skipped (RUN_FEATURE_DATA=False)")

    if RUN_WINE_INPUT:
        wine_meta = generate_wine_feature_input(WINE_INPUT_OUTPUT)
        generate_wine_model_config(WINE_MODEL_CONFIG)
        generate_wine_inference_config(
            WINE_INFERENCE_CONFIG,
            WINE_INPUT_OUTPUT,
            WINE_MODEL_CONFIG,
            wine_meta["feature_names"],
        )
        print(f"[features] Wine assets ready in {WINE_INPUT_OUTPUT.parent.resolve()}")
    else:
        print("[features] Skipped (RUN_WINE_INPUT=False)")


if __name__ == "__main__":
    main()
