"""TFT baseline and regime-conditioned models (Phases 4-5). Stubs only — not implemented yet."""


def build_global_tft(train_df, config: dict):
    """Build TFT global baseline (no regime input)."""
    raise NotImplementedError("Phase 4")


def build_separate_tft(train_df, regime_labels, config: dict):
    """Build one TFT per regime (TFT-Separate variant)."""
    raise NotImplementedError("Phase 5")


def build_conditioned_tft(train_df, regime_labels, config: dict):
    """Build single TFT with regime label as static covariate (TFT-Conditioned)."""
    raise NotImplementedError("Phase 5")


def build_ensemble_tft(train_df, regime_probs, config: dict):
    """Build soft-routing ensemble using HMM regime probabilities (TFT-Ensemble)."""
    raise NotImplementedError("Phase 5")
