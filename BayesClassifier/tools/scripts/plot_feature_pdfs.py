#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_CONFIG = ""
INFERENCE_CONFIG = "config/classifier/inference.batch_text.example.json"
OUTPUT_DIR = "output/pdfs"
BINS = 30

def resolve_path(base_dir: Path, path_value: str, root_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        if root_dir is not None:
            root_candidate = root_dir / path_value
            if root_candidate.exists():
                path = root_candidate
            else:
                path = base_dir / path
        else:
            path = base_dir / path
    return path.resolve()


def parse_delimiter(token: str) -> str:
    if token == "SPACE":
        return " "
    if token == "TAB":
        return "\t"
    if len(token) == 1:
        return token
    raise ValueError(f"Invalid delimiter specification: {token}")


def tokenize(line: str, delimiter: str) -> list[str]:
    if delimiter in (" ", "\t"):
        return line.split()
    return [token.strip() for token in line.split(delimiter)]


def load_inference_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    input_format = data.get("input_format", "text")
    if input_format != "text":
        raise ValueError("This tool currently supports text input only.")

    layout = data.get("layout", {})
    feature_fields = layout.get("feature_fields")
    if not feature_fields:
        raise ValueError("layout.feature_fields must be provided for text input.")

    delimiter_token = layout.get("delimiter", "SPACE")
    config_dir = config_path.parent
    input_file = data.get("input_file")
    if not input_file:
        raise ValueError("input_file must be set in the inference config.")

    return {
        "input_file": resolve_path(config_dir, input_file),
        "truth_field": layout.get("truth_field", "truth"),
        "feature_fields": feature_fields,
        "delimiter": parse_delimiter(delimiter_token),
        "model_config": data.get("model_config"),
    }


def load_model_config(model_path: Path) -> dict:
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
                "feature_names": feature_names,
                "feature_models": aligned,
            }
        )

    return {"feature_names": feature_names, "classes": class_models}


def read_text_input(input_path: Path,
                    truth_field: str,
                    feature_fields: list[str],
                    delimiter: str) -> dict:
    data_by_class: dict[str, list[list[float]]] = {}

    with input_path.open("r", encoding="utf-8") as handle:
        header_tokens = None
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            header_tokens = tokenize(stripped, delimiter)
            break

        if header_tokens is None:
            raise ValueError("Input file has no header row.")

        header_index = {name: idx for idx, name in enumerate(header_tokens)}
        if truth_field not in header_index:
            raise ValueError(f"Missing truth field '{truth_field}' in header.")

        feature_indices = []
        for name in feature_fields:
            if name not in header_index:
                raise ValueError(f"Missing feature '{name}' in header.")
            feature_indices.append(header_index[name])

        truth_index = header_index[truth_field]

        for line in handle:
            content = line.split("#", 1)[0].strip()
            if not content:
                continue
            tokens = tokenize(content, delimiter)
            if truth_index >= len(tokens):
                continue
            class_name = tokens[truth_index].strip()
            if not class_name:
                continue

            if class_name not in data_by_class:
                data_by_class[class_name] = [[] for _ in feature_fields]

            for idx, token_index in enumerate(feature_indices):
                if token_index >= len(tokens):
                    continue
                token = tokens[token_index].strip()
                if not token:
                    continue
                try:
                    value = float(token)
                except ValueError:
                    continue
                if math.isfinite(value):
                    data_by_class[class_name][idx].append(value)

    return data_by_class


def gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coef * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def rayleigh_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    pdf = (x / (sigma * sigma)) * np.exp(-0.5 * (x * x) / (sigma * sigma))
    pdf[x < 0.0] = 0.0
    return pdf


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def plot_feature_pdfs(model_config: str,
                      inference_config: str,
                      output_dir: str = "output/pdfs",
                      bins: int = 30) -> None:
    root_dir = Path(__file__).resolve().parents[2]
    inference_path = Path(inference_config).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inference = load_inference_config(inference_path)
    model_config_value = model_config or inference.get("model_config")
    if not model_config_value:
        raise ValueError("model_config must be provided.")
    model_path = resolve_path(inference_path.parent, model_config_value, root_dir=root_dir)
    model = load_model_config(model_path)
    data_by_class = read_text_input(
        inference["input_file"],
        inference["truth_field"],
        inference["feature_fields"],
        inference["delimiter"],
    )

    feature_names = model["feature_names"]
    classes = model["classes"]

    for feature_index, feature_name in enumerate(feature_names):
        data_values = []
        for class_name, per_feature in data_by_class.items():
            if feature_index < len(per_feature):
                data_values.extend(per_feature[feature_index])

        data_min = min(data_values) if data_values else None
        data_max = max(data_values) if data_values else None

        model_min = None
        model_max = None
        for cls in classes:
            model_def = cls["feature_models"][feature_index]
            if not model_def:
                continue
            dist_type = model_def["type"]
            params = model_def["params"]
            if dist_type == "gaussian":
                mean = float(params.get("mean", 0.0))
                sigma = float(params.get("sigma", 1.0))
                model_min = mean - 4.0 * sigma if model_min is None else min(model_min, mean - 4.0 * sigma)
                model_max = mean + 4.0 * sigma if model_max is None else max(model_max, mean + 4.0 * sigma)
            elif dist_type == "rayleigh":
                sigma = float(params.get("sigma", 1.0))
                model_min = 0.0 if model_min is None else min(model_min, 0.0)
                model_max = 4.0 * sigma if model_max is None else max(model_max, 4.0 * sigma)

        x_min_candidates = [v for v in (data_min, model_min) if v is not None]
        x_max_candidates = [v for v in (data_max, model_max) if v is not None]
        if not x_min_candidates or not x_max_candidates:
            x_min, x_max = 0.0, 1.0
        else:
            x_min = min(x_min_candidates)
            x_max = max(x_max_candidates)
            if x_min == x_max:
                x_min -= 1.0
                x_max += 1.0

        padding = 0.05 * (x_max - x_min)
        x_min -= padding
        x_max += padding

        x_values = np.linspace(x_min, x_max, 400)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for class_index, cls in enumerate(classes):
            model_def = cls["feature_models"][feature_index]
            if not model_def:
                continue
            dist_type = model_def["type"]
            params = model_def["params"]
            color = color_cycle[class_index % len(color_cycle)]
            label_prefix = cls["name"]

            if dist_type == "gaussian":
                mean = float(params.get("mean", 0.0))
                sigma = float(params.get("sigma", 1.0))
                y_values = gaussian_pdf(x_values, mean, sigma)
                ax.plot(x_values, y_values, color=color, label=f"{label_prefix} PDF")
            elif dist_type == "rayleigh":
                sigma = float(params.get("sigma", 1.0))
                y_values = rayleigh_pdf(x_values, sigma)
                ax.plot(x_values, y_values, color=color, label=f"{label_prefix} PDF")

            if cls["name"] in data_by_class:
                samples = data_by_class[cls["name"]][feature_index]
                if samples:
                    ax.hist(
                        samples,
                        bins=bins,
                        density=True,
                        alpha=0.25,
                        color=color,
                        label=f"{label_prefix} data",
                    )

        ax.set_title(f"Feature '{feature_name}': PDFs vs data")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.2)

        filename = f"pdf_{sanitize_filename(feature_name)}.png"
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    plot_feature_pdfs(
        model_config=MODEL_CONFIG,
        inference_config=INFERENCE_CONFIG,
        output_dir=OUTPUT_DIR,
        bins=BINS,
    )
