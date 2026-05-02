"""Evaluation metrics and statistical tests (Phases 4-6)."""

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and directional accuracy."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Directional accuracy: sign match (excluding exact zeros to avoid degenerate counts)
    mask = y_true != 0
    dir_acc = float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))

    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc}


def compute_per_regime_metrics(y_true, y_pred, regime_labels) -> dict:
    """Return metrics dict keyed by regime label."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    regime_labels = np.asarray(regime_labels).flatten()

    out = {}
    for r in np.unique(regime_labels):
        mask = regime_labels == r
        if mask.sum() == 0:
            continue
        out[int(r)] = compute_metrics(y_true[mask], y_pred[mask])
    return out


def diebold_mariano_test(errors_a, errors_b) -> dict:
    """Run Diebold-Mariano test, return statistic and p-value."""
    raise NotImplementedError("Phase 6")
