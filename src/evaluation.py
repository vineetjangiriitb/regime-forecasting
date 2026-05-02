"""Evaluation metrics and statistical tests (Phases 4-6). Stubs only — not implemented yet."""


def compute_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and directional accuracy."""
    raise NotImplementedError("Phase 4")


def compute_per_regime_metrics(y_true, y_pred, regime_labels) -> dict:
    """Return metrics dict keyed by regime label."""
    raise NotImplementedError("Phase 5")


def diebold_mariano_test(errors_a, errors_b) -> dict:
    """Run Diebold-Mariano test, return statistic and p-value."""
    raise NotImplementedError("Phase 6")
