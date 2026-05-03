"""Train TFT Global baseline (Phase 4) — full GPU config for Google Colab.

Run on Colab with a T4/A100 GPU. Saves artifacts with _colab suffix so both
the Mac CPU run and this run coexist in results/.

Outputs:
  results/tft_global_colab.ckpt
  results/tft_global_colab_predictions.pkl
  results/metrics/phase4_baselines_colab.csv
"""

import sys
import pickle
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet

from src.evaluation import compute_metrics
from src.tft_model import (
    prepare_tft_dataframe,
    build_tft_dataset,
    build_global_tft,
)

# Full-capacity config — no hardware compromises
COLAB_CONFIG = {
    "max_encoder_length": 60,
    "max_prediction_length": 1,
    "hidden_size": 64,          # was 16 on Mac
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 16,  # was 8 on Mac
    "learning_rate": 3e-4,
    "batch_size": 128,          # was 32 on Mac
    "max_epochs": 50,           # was 25 on Mac
    "patience": 8,              # was 6 on Mac
    "gradient_clip_val": 0.1,
}


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42, workers=True)

    DATA_RAW = ROOT / "data/raw"
    RESULTS = ROOT / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)

    TRAIN_END = "2017-12-31"
    VAL_END = "2020-12-31"

    print("Loading data...")
    df = pd.read_csv(DATA_RAW / "master.csv", index_col=0, parse_dates=True).sort_index()
    df_phase4 = df.loc[:VAL_END].copy()
    val_mask = df_phase4.index > TRAIN_END
    y_val_actual_full = df_phase4.loc[val_mask, "sp500_return"].values
    val_dates_full = df_phase4.index[val_mask]
    print(f"Train+Val rows: {df_phase4.shape[0]:,}  |  Val rows: {val_mask.sum():,}")

    print("Preparing TFT features...")
    tft_df = prepare_tft_dataframe(df_phase4)
    training_cutoff = tft_df.loc[tft_df.index <= TRAIN_END, "time_idx"].max()

    config = COLAB_CONFIG.copy()
    print("Config:", config)

    print("Building datasets...")
    training_ds = build_tft_dataset(tft_df, config, training_cutoff=training_cutoff)
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, tft_df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=training_cutoff + 1,
    )

    # num_workers=4 for fast data loading on Colab (multiple CPU cores available)
    train_dl = training_ds.to_dataloader(
        train=True, batch_size=config["batch_size"], num_workers=4, persistent_workers=True
    )
    val_dl = validation_ds.to_dataloader(
        train=False, batch_size=256, num_workers=4, persistent_workers=True
    )
    print(f"Train sequences: {len(training_ds):,}  |  Val sequences: {len(validation_ds):,}")

    print("Building TFT model...")
    model = build_global_tft(training_ds, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=config["patience"], mode="min", verbose=True
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(RESULTS),
        filename="tft_global_colab_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # accelerator="gpu" — uses Colab T4/A100
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices=1,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,
        num_sanity_val_steps=0,
    )

    print("\nTraining (GPU)...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    stopped_epoch = trainer.current_epoch
    print(f"Done. Stopped at epoch {stopped_epoch}")

    print("\nPredicting on validation...")
    raw = model.predict(
        val_dl, mode="prediction", return_x=False,
        trainer_kwargs={"accelerator": "gpu", "devices": 1, "logger": False, "enable_progress_bar": False}
    )
    y_val_tft = raw.cpu().numpy().flatten() if hasattr(raw, "cpu") else np.asarray(raw).flatten()

    n_pred = len(y_val_tft)
    y_actual = y_val_actual_full[-n_pred:] if n_pred < len(y_val_actual_full) else y_val_actual_full
    val_dates = val_dates_full[-n_pred:] if n_pred < len(val_dates_full) else val_dates_full

    metrics = compute_metrics(y_actual, y_val_tft)
    print(
        f"\nTFT Global (Colab): "
        f"MAE={metrics['mae']:.5f}  "
        f"RMSE={metrics['rmse']:.5f}  "
        f"DirAcc={metrics['dir_acc']*100:.2f}%"
    )

    print("\nSaving artifacts...")
    torch.save(model.state_dict(), RESULTS / "tft_global_colab.ckpt")
    with open(RESULTS / "tft_global_colab_predictions.pkl", "wb") as f:
        pickle.dump({
            "val_dates": val_dates,
            "y_actual": y_actual,
            "y_pred": y_val_tft,
            "metrics": metrics,
            "config": config,
            "stopped_epoch": stopped_epoch,
        }, f)

    # Save comparison CSV with both runs if the Mac run exists
    rows = [{"run": "Colab GPU", "hidden_size": config["hidden_size"],
              "batch_size": config["batch_size"], "epochs": stopped_epoch, **metrics}]
    mac_pkl = RESULTS / "tft_global_predictions.pkl"
    if mac_pkl.exists():
        with open(mac_pkl, "rb") as f:
            mac = pickle.load(f)
        rows.append({
            "run": "Mac CPU",
            "hidden_size": mac["config"]["hidden_size"],
            "batch_size": mac["config"]["batch_size"],
            "epochs": mac["stopped_epoch"],
            **mac["metrics"],
        })

    import pandas as pd_inner
    pd_inner.DataFrame(rows).to_csv(RESULTS / "metrics/phase4_tft_comparison.csv", index=False)
    print("Saved: tft_global_colab.ckpt + tft_global_colab_predictions.pkl + phase4_tft_comparison.csv")


if __name__ == "__main__":
    main()
