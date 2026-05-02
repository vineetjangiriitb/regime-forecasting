"""Data download and feature engineering for regime-aware forecasting (Phase 1)."""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web


# Tickers and series to download
EQUITY_TICKERS = ["^GSPC", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP"]
FRED_SERIES = {
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "FEDFUNDS": "fed_funds_rate",
}
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"


def download_yahoo_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance, return adjusted close prices."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    df.columns = [t.replace("^", "").lower() for t in df.columns]
    return df


def download_fred_data(series: dict, start: str, end: str) -> pd.DataFrame:
    """Download macro series from FRED, return daily-reindexed DataFrame."""
    frames = []
    for series_id, col_name in series.items():
        s = web.DataReader(series_id, "fred", start, end)
        s.columns = [col_name]
        frames.append(s)
    return pd.concat(frames, axis=1)


def merge_data(yahoo_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Yahoo and FRED DataFrames on trading days, forward-fill FRED gaps."""
    fred_aligned = fred_df.reindex(yahoo_df.index).ffill()
    df = pd.concat([yahoo_df, fred_aligned], axis=1)
    return df.dropna(subset=["gspc"])


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: daily returns, 20-day realized vol, yield spread, growth/value spread."""
    df = df.copy()
    df["sp500_return"] = np.log(df["gspc"] / df["gspc"].shift(1))
    df["realized_vol_20d"] = df["sp500_return"].rolling(20).std() * np.sqrt(252)
    df["yield_spread"] = df["treasury_10y"] - df["treasury_2y"]
    df["growth_value_spread"] = np.log(df["xlk"] / df["xlp"])
    return df.dropna()


def load_clean_data(save_path: str = None) -> pd.DataFrame:
    """End-to-end: download, merge, compute features, optionally save to CSV."""
    yahoo_df = download_yahoo_data(EQUITY_TICKERS, START_DATE, END_DATE)
    fred_df = download_fred_data(FRED_SERIES, START_DATE, END_DATE)
    df = merge_data(yahoo_df, fred_df)
    df = compute_features(df)
    if save_path:
        df.to_csv(save_path)
    return df
