# Regime-Aware Forecasting

**Hypothesis:** A two-stage pipeline that first detects macro regimes (HMM) and then trains regime-specific forecasters (TFT) outperforms a single global model on equity return prediction.

## Project Structure

```
regime-forecasting/
├── data/raw/              # Downloaded CSVs — never modify
├── notebooks/
│   └── 01_data_foundation.ipynb   # Phase 1 (current)
├── src/
│   ├── data_loader.py     # Download + feature engineering
│   ├── hmm_model.py       # Regime detection (Phase 2)
│   ├── tft_model.py       # TFT variants (Phases 4-5)
│   └── evaluation.py      # Metrics + stat tests (Phases 4-6)
├── results/
│   ├── figures/           # All visualizations
│   └── metrics/           # Saved metric tables
├── requirements.txt
└── README.md
```

## Data

| Source | Series | Description |
|--------|--------|-------------|
| Yahoo Finance | `^GSPC` | S&P 500 price |
| Yahoo Finance | `^VIX` | CBOE Volatility Index |
| Yahoo Finance | `XLK, XLF, XLE, XLV, XLY, XLP` | Sector ETFs |
| FRED | `DGS10`, `DGS2` | 10Y and 2Y Treasury yields |
| FRED | `FEDFUNDS` | Federal Funds Rate |

**Date range:** Jan 2000 – Dec 2023  
**Train:** 2000–2017 | **Validation:** 2018–2020 | **Test (locked):** 2021–2023

## Phases

| Phase | Description | Weeks |
|-------|-------------|-------|
| 1 | Data Foundation | 1 |
| 2 | Regime Detection (HMM) | 1.5 |
| 3 | Regime Validation | 0.5 |
| 4 | Forecasting Baseline (ARIMA + TFT Global) | 2 |
| 5 | Regime-Conditioned Models | 2.5 |
| 6 | Rigorous Evaluation | 1.5 |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_foundation.ipynb
```

## Key Rules

- No lookahead bias — macro data must be lagged
- Test set (2021–2023) is sacred — do not touch before Phase 6
- Report per-regime metrics, not just overall
- HMM transition matrix diagonal must be > 0.90
