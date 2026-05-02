"""Train TFT Global baseline (Phase 4) and save predictions.

Run as a standalone script (not in Jupyter) to avoid memory issues
on systems with limited RAM. Saves:
  - results/tft_global.ckpt (model state dict)
  - results/tft_global_predictions.pkl (val_dates, y_actual, y_pred, metrics)
"""

import sys
import pickle
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet

from src.evaluation import compute_metrics
from src.tft_model import (
    prepare_tft_dataframe,
    build_tft_dataset,
    build_global_tft,
    DEFAULT_CONFIG,
)


def main():
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42, workers=True)

    DATA_RAW = ROOT / "data/raw"
    RESULTS = ROOT / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    TRAIN_END = "2017-12-31"
    VAL_END = "2020-12-31"

    print("Loading data...")
    df = pd.read_csv(DATA_RAW / "master.csv", index_col=0, parse_dates=True).sort_index()
    df_phase4 = df.loc[:VAL_END].copy()
    val_mask = df_phase4.index > TRAIN_END
    y_val_actual_full = df_phase4.loc[val_mask, "sp500_return"].values
    val_dates_full = df_phase4.index[val_mask]

    print(f"Train+Val rows: {df_phase4.shape[0]:,}")
    print(f"Val rows: {val_mask.sum():,}")

    print("Preparing TFT features...")
    tft_df = prepare_tft_dataframe(df_phase4)
    training_cutoff = tft_df.loc[tft_df.index <= TRAIN_END, "time_idx"].max()

    config = DEFAULT_CONFIG.copy()
    print("Config:", config)

    print("Building datasets...")
    training_ds = build_tft_dataset(tft_df, config, training_cutoff=training_cutoff)
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, tft_df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=training_cutoff + 1,
    )

    train_dl = training_ds.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=0)
    val_dl = validation_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    print(f"Train samples: {len(training_ds):,} | Val samples: {len(validation_ds):,}")

    print("Building TFT model...")
    model = build_global_tft(training_ds, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    early_stop = EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", verbose=True)
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="cpu",
        devices=1,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[early_stop],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )

    print("\nTraining...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"Done. Stopped at epoch {trainer.current_epoch}")

    print("\nPredicting on validation (CPU)...")
    predict_trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    raw = model.predict(val_dl, mode="prediction", return_x=False, trainer_kwargs={"accelerator": "cpu", "devices": 1, "logger": False, "enable_progress_bar": False})
    y_val_tft = raw.cpu().numpy().flatten() if hasattr(raw, "cpu") else np.asarray(raw).flatten()

    n_pred = len(y_val_tft)
    y_actual = y_val_actual_full[-n_pred:] if n_pred < len(y_val_actual_full) else y_val_actual_full
    val_dates = val_dates_full[-n_pred:] if n_pred < len(val_dates_full) else val_dates_full

    metrics = compute_metrics(y_actual, y_val_tft)
    print(f"TFT Global: MAE={metrics['mae']:.5f} RMSE={metrics['rmse']:.5f} DirAcc={metrics['dir_acc']*100:.2f}%")

    print("\nSaving artifacts...")
    torch.save(model.state_dict(), RESULTS / "tft_global.ckpt")
    with open(RESULTS / "tft_global_predictions.pkl", "wb") as f:
        pickle.dump({
            "val_dates": val_dates,
            "y_actual": y_actual,
            "y_pred":   y_val_tft,
            "metrics":  metrics,
            "config":   config,
            "stopped_epoch": trainer.current_epoch,
        }, f)
    print("Saved tft_global.ckpt + tft_global_predictions.pkl")


if __name__ == "__main__":
    main()
