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


def prepare_tft_dataframe_with_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Same as prepare_tft_dataframe but keeps regime_label as a string categorical column."""
    out = prepare_tft_dataframe(df)
    out["regime"] = out["regime_label"].astype(int).astype(str)
    return out


def build_conditioned_tft_dataset(df: pd.DataFrame, config: dict, training_cutoff: int = None) -> TimeSeriesDataSet:
    """Build TimeSeriesDataSet with regime as time-varying known categorical (TFT-Conditioned)."""
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
        time_varying_known_categoricals=["regime"],
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


def get_regime_indices(dataset: TimeSeriesDataSet, df: pd.DataFrame, target_regime: int) -> list:
    """Return indices into a TimeSeriesDataSet where the prediction-day's regime equals target_regime.

    pytorch-forecasting's `dataset.index` exposes `time` (encoder-start time_idx) and
    `sequence_length`. The prediction day's time_idx = time + sequence_length - max_prediction_length.
    """
    regime_at_t = df.set_index("time_idx")["regime_label"].to_dict()
    idx = dataset.index
    pred_times = (idx["time"] + idx["sequence_length"] - dataset.max_prediction_length).values
    return [i for i, t in enumerate(pred_times) if regime_at_t.get(int(t)) == target_regime]


def hmm_posteriors_for_dates(hmm_model, scaler, df: pd.DataFrame, dates) -> np.ndarray:
    """Return [n_dates, K] HMM posteriors for the given dates using fitted HMM + scaler.

    Features (vol, yield_spread, vix) are computed from `df` on `dates` and standardized
    by the fitted train-only scaler. No refitting — this is anti-lookahead-safe.
    """
    feats = df.loc[dates, ["realized_vol_20d", "yield_spread", "vix"]].values
    X = scaler.transform(feats)
    return hmm_model.predict_proba(X)
