from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mean_absolute_percentage_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    mask = np.abs(y_true_arr) > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    precision = float(np.sum((y_true_arr == 1) & (y_pred_arr == 1)) / max(np.sum(y_pred_arr == 1), 1))
    recall = float(np.sum((y_true_arr == 1) & (y_pred_arr == 1)) / max(np.sum(y_true_arr == 1), 1))
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def evaluate_model(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    metric: str = "rmse",
) -> float:
    """Evaluate regression or classification outputs with a chosen metric."""
    if metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    if metric == "mape":
        return mean_absolute_percentage_error(y_true, y_pred)
    if metric == "rmse":
        return root_mean_squared_error(y_true, y_pred)
    if metric == "accuracy":
        return float(np.mean(np.asarray(list(y_true), dtype=int) == np.asarray(list(y_pred), dtype=int)))
    raise ValueError(f"Unsupported metric: {metric}")
