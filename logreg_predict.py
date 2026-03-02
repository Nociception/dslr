#!/usr/bin/env python3

import csv
import sys
import json
from typing import List

import numpy as np
import polars as pl
import numpy.typing as npt

from logreg_train import apply_standardization, predict_ovr


def load_model(
    path: str,
) -> tuple[
    dict[str, tuple[npt.NDArray[np.float64], np.float64]],
    np.ndarray,
    np.ndarray,
    List[str],
]:
    with open(path, "r") as f:
        data = json.load(f)

    models = {}
    for house, v in data["classes"].items():
        models[house] = (
            np.array(v["weights"], dtype=np.float64),
            float(v["intercept"]),
        )

    means = np.array(data["standardization"]["means"], dtype=np.float64)
    stds = np.array(data["standardization"]["stds"], dtype=np.float64)
    features = data["features"]

    return models, means, stds, features


def load_test_dataset(path: str, feature_names: List[str]) -> np.ndarray:
    df = pl.read_csv(path)
    df = df.select(feature_names)

    df = df.fill_null(strategy="mean")

    return df.to_numpy()


def save_predictions(preds: np.ndarray, output_path: str = "houses.csv") -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for idx, house in enumerate(preds):
            writer.writerow([idx, house])

    print(f"Predictions saved to {output_path}")


def main(test_path: str, model_path: str = "model.json") -> None:
    models, means, stds, features = load_model(model_path)
    X_test = load_test_dataset(test_path, features)
    X_test = apply_standardization(X_test, means, stds)
    preds = predict_ovr(models, X_test)
    save_predictions(preds)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./predict.py datasets/dataset_test.csv")
        exit(1)
    main(sys.argv[1])
