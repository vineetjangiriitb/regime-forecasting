"""TFT baseline and regime-conditioned models (Phases 4-5)."""

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


# Default config — small TFT for CPU training on ~4500 train rows (Mac-friendly)
DEFAULT_CONFIG = {
    "max_encoder_length": 60,
    "max_prediction_length": 1,
    "hidden_size": 16,
    "attention_head_size": 4,
    "dropout": 0.15,
    "hidden_continuous_size": 8,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 25,
    "patience": 6,
    "gradient_clip_val": 0.1,
}


def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_idx, group_id, lagged macro, and cyclic time features required by TFT."""
    out = df.copy()
    out = out.sort_index()

    out["time_idx"] = np.arange(len(out), dtype=np.int64)
    out["group_id"] = "global"  # single series

    # Lag macro features 1 day to prevent lookahead
    out["fed_funds_lag1"]  = out["fed_funds_rate"].shift(1)
    out["yield_spread_lag1"] = out["yield_spread"].shift(1)

    # Cyclic time features
    dow   = out.index.dayofweek
    month = out.index.month
    out["dow_sin"]   = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"]   = np.cos(2 * np.pi * dow / 5)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out.dropna()


def build_tft_dataset(df: pd.DataFrame, config: dict, training_cutoff: int = None) -> TimeSeriesDataSet:
    """Build a TimeSeriesDataSet for TFT (global baseline; no regime input)."""
    if training_cutoff is None:
        training_cutoff = df["time_idx"].max()

    return TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="sp500_return",
        group_ids=["group_id"],
        min_encoder_length=config["max_encoder_length"],
        max_encoder_length=config["max_encoder_length"],
        min_prediction_length=config["max_prediction_length"],
        max_prediction_length=config["max_prediction_length"],
        static_categoricals=["group_id"],
        time_varying_known_reals=[
            "time_idx", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "fed_funds_lag1", "yield_spread_lag1",
        ],
        time_varying_unknown_reals=["sp500_return", "vix"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )


def build_global_tft(training_dataset: TimeSeriesDataSet, config: dict) -> TemporalFusionTransformer:
    """Instantiate TFT model from a training dataset (no regime input)."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["attention_head_size"],
        dropout=config["dropout"],
        hidden_continuous_size=config["hidden_continuous_size"],
        loss=MAE(),
        log_interval=0,
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )


def build_separate_tft(train_df, regime_labels, config: dict):
    """Build one TFT per regime (TFT-Separate variant)."""
    raise NotImplementedError("Phase 5")


def build_conditioned_tft(train_df, regime_labels, config: dict):
    """Build single TFT with regime label as static covariate (TFT-Conditioned)."""
    raise NotImplementedError("Phase 5")


def build_ensemble_tft(train_df, regime_probs, config: dict):
    """Build soft-routing ensemble using HMM regime probabilities (TFT-Ensemble)."""
    raise NotImplementedError("Phase 5")
