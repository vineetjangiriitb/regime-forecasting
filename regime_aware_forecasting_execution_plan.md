# Regime-Aware Forecasting: Execution Plan

## Project Overview

**Problem Statement:**
Build a two-stage pipeline:
1. An unsupervised regime-detection model (HMM) that segments historical market data into macro-regimes based on interest rate environment, volatility, and sector rotation signals
2. A regime-conditioned forecasting model (TFT) trained separately per regime

**Core Research Question:** Does regime-aware forecasting significantly outperform a single global model — and by how much, and in which regimes?

**Total Duration:** ~9–10 weeks at 3–5 hrs/week on this project

---

## Project Structure

```
regime-forecasting/
├── data/
│   └── raw/                  # Downloaded CSVs (never modify)
├── notebooks/
│   ├── 01_data_foundation.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_regime_validation.ipynb
│   ├── 04_forecasting_baseline.ipynb
│   ├── 05_regime_conditioned.ipynb
│   └── 06_final_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── hmm_model.py
│   ├── tft_model.py
│   └── evaluation.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

---

## Phase 1: Data Foundation

**Goal:** Build a single clean, unified DataFrame covering Jan 2000 – Dec 2023 with no missing values.

### Data to Collect

| Series | Source | Library |
|---|---|---|
| S&P 500 daily price + returns | Yahoo Finance | `yfinance` |
| Nasdaq 100 daily price | Yahoo Finance | `yfinance` |
| S&P 500 Value ETF (IVE) | Yahoo Finance | `yfinance` |
| S&P 500 Growth ETF (IVW) | Yahoo Finance | `yfinance` |
| Utilities ETF (XLU) | Yahoo Finance | `yfinance` |
| VIX index | Yahoo Finance | `yfinance` |
| Federal Funds Rate | FRED | `pandas_datareader` |
| 2-Year Treasury Yield | FRED | `pandas_datareader` |
| 10-Year Treasury Yield | FRED | `pandas_datareader` |

### Derived Features to Compute

- **Daily returns** — `pct_change()` on S&P 500 closing price
- **Realized volatility** — rolling 20-day standard deviation of daily returns
- **Yield curve spread** — 10-year yield minus 2-year yield
- **Growth vs Value spread** — IVW daily return minus IVE daily return

### Key Implementation Notes

- Macro series (FRED) update at different frequencies than daily price data — forward-fill macro series to align to trading days
- Drop any dates where the market was closed (holidays) — use S&P 500 trading days as the master index
- Standardize all features to zero mean, unit variance before any model training

### Visualizations — Phase 1

**Tool:** `matplotlib` with `seaborn` styling (`sns.set_theme(style="darkgrid")`)
**Save all figures to:** `results/figures/phase1/`

---

**VIZ 1.1 — Feature Dashboard (4-panel figure)**
A 2×2 grid of subplots sharing the same x-axis (date), all spanning 2000–2023.

```
Panel 1 (top-left):   S&P 500 closing price — line chart
Panel 2 (top-right):  VIX index — line chart, color the area under curve red
Panel 3 (bottom-left): Yield curve spread (10yr - 2yr) — line chart,
                        shade area below zero in red (inversion periods)
Panel 4 (bottom-right): Realized volatility (20-day) vs VIX — two lines
                         on same axes, different colors, with legend
```

What to look for manually:
- Panel 3 must go below zero in 2006–2007 (before 2008 crash) and briefly in 2019
- Panel 2 must spike sharply in 2008, 2020, and moderately in 2022
- Panel 4 lines must broadly track each other — if they diverge wildly, something is wrong in your realized vol calculation

Save as: `results/figures/phase1/viz1_feature_dashboard.png`

---

**VIZ 1.2 — Correlation Heatmap**
Seaborn heatmap of Pearson correlations between all features in your master DataFrame.

```python
import seaborn as sns
corr = df[feature_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
```

What to look for manually:
- VIX and realized volatility should be strongly positively correlated (>0.6)
- Yield curve spread and VIX should be moderately negatively correlated (high fear = inverted curve)
- S&P 500 returns and VIX should be negatively correlated (market up = fear down)
- Any correlation above 0.95 between two features = potential redundancy, investigate

Save as: `results/figures/phase1/viz2_correlation_heatmap.png`

---

**VIZ 1.3 — Missing Data Audit Plot**
Before you claim your DataFrame is clean, visualize it.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.imshow(df[feature_cols].isna().T, aspect='auto', cmap='Reds', interpolation='none')
plt.yticks(range(len(feature_cols)), feature_cols)
plt.title("Missing Data Map — Red = NaN")
```

What to look for manually:
- Should be entirely white (no red) after your cleaning pipeline
- If you see red stripes at the start (pre-2000 for some series) or at weekends/holidays, your date alignment has a bug

Save as: `results/figures/phase1/viz3_missing_data_audit.png`

---

### Validation Gate ✓

You pass Phase 1 when you can answer **yes** to all four:

- [ ] Single clean DataFrame from Jan 2000 – Dec 2023 with zero NaN values in any column
- [ ] VIZ 1.1 Panel 3 shows yield curve spread going **negative before 2008** (the inversion signal)
- [ ] VIZ 1.1 Panel 2 shows VIX with three clear spikes: **2008, March 2020, and 2022**
- [ ] VIZ 1.1 Panel 4 shows realized volatility broadly tracking VIX
- [ ] VIZ 1.2 shows VIX–realized vol correlation above 0.6
- [ ] VIZ 1.3 is entirely white — zero missing values confirmed visually

**Do not proceed to Phase 2 until all boxes are checked.**

---

## Phase 2: Regime Detection

**Goal:** Train an HMM that outputs a regime label (0, 1, 2...) for every trading day from 2000–2023.

### HMM Input Features

Start with exactly these three — they are the cleanest regime signals:
1. Realized volatility (20-day rolling)
2. Yield curve spread (10yr minus 2yr)
3. VIX index

### Implementation Steps

```
1. Standardize all 3 features (zero mean, unit variance)
2. Train GaussianHMM from hmmlearn with K=2
3. Repeat for K=3, K=4, K=5
4. Compute BIC for each K — select K with lowest BIC (elbow point)
5. Extract Viterbi path for optimal K — this is your regime label series
6. Save regime labels as a new column in your master DataFrame
```

### Key Concepts

- **Baum-Welch algorithm** — how the HMM learns its parameters (expectation-maximization)
- **Viterbi path** — the single most likely sequence of hidden states given your observations
- **BIC (Bayesian Information Criterion)** — penalizes model complexity; use to select K without overfitting

### Library

```python
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=K, covariance_type="full", n_iter=1000)
model.fit(features_standardized)
regime_labels = model.predict(features_standardized)  # Viterbi path
```

### Visualizations — Phase 2

**Save all figures to:** `results/figures/phase2/`

---

**VIZ 2.1 — BIC Curve**
Line plot of BIC score vs K (number of regimes), for K = 2, 3, 4, 5.

```python
plt.plot(k_values, bic_scores, marker='o', linewidth=2)
plt.xlabel("Number of Regimes (K)")
plt.ylabel("BIC Score")
plt.title("HMM Model Selection — BIC vs K")
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Selected K={optimal_k}')
```

What to look for manually:
- BIC should decrease from K=2 onward and flatten or increase after the optimal K
- The elbow (point of diminishing returns) is your K — typically 3 or 4 for market data
- If BIC keeps strictly decreasing through K=5, try K=6 before deciding

Save as: `results/figures/phase2/viz1_bic_curve.png`

---

**VIZ 2.2 — Regime Timeline (the most important visualization in the project)**
S&P 500 price chart (2000–2023) with regime labels as colored background bands.

```python
colors = {0: '#90EE90', 1: '#FF6B6B', 2: '#FFD700', 3: '#87CEEB'}
# (green=calm, red=crisis, yellow=tightening, blue=other)

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(dates, sp500_price, color='black', linewidth=0.8, zorder=3)

for regime_id, color in colors.items():
    mask = (regime_labels == regime_id)
    ax.fill_between(dates, sp500_price.min(), sp500_price.max(),
                    where=mask, alpha=0.3, color=color, label=f'Regime {regime_id}')

ax.set_title("S&P 500 with HMM Regime Labels (2000–2023)")
ax.legend()
```

What to look for manually:
- Red (crisis) bands must cover: 2008–2009, March 2020, and significantly overlap with 2022
- Green (calm) bands must cover: 2003–2007 bull run, 2013–2019 expansion
- Bands should be **continuous stretches of weeks/months** — not flickering every few days
- If you see rapid color switching, increase HMM smoothing or recheck your features

Save as: `results/figures/phase2/viz2_regime_timeline.png`
**This is your single most important diagnostic plot — if it looks wrong, stop here.**

---

**VIZ 2.3 — Feature Distributions Per Regime**
A 3-panel violin plot — one panel per HMM input feature — showing distribution of each feature broken down by regime.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
features = ['realized_vol', 'yield_spread', 'vix']
titles = ['Realized Volatility', 'Yield Curve Spread', 'VIX']

for ax, feat, title in zip(axes, features, titles):
    data_by_regime = [df[df['regime'] == k][feat].values for k in range(optimal_k)]
    ax.violinplot(data_by_regime, positions=range(optimal_k))
    ax.set_title(title)
    ax.set_xlabel("Regime")
```

What to look for manually:
- Each feature's violin should look **clearly different across regimes**
- If two regimes have nearly identical distributions across all three features, reduce K
- The crisis regime violin for VIX should be centered much higher than the calm regime

Save as: `results/figures/phase2/viz3_feature_distributions_per_regime.png`

---

**VIZ 2.4 — Regime Transition Matrix Heatmap**
Heatmap of the HMM's learned transition probabilities.

```python
sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f'To R{i}' for i in range(optimal_k)],
            yticklabels=[f'From R{i}' for i in range(optimal_k)])
plt.title("HMM Regime Transition Probabilities")
```

What to look for manually:
- Diagonal values should be **above 0.90** — regime stays stable once entered
- If diagonal is below 0.85, the HMM is too unstable — consider reducing K

Save as: `results/figures/phase2/viz4_transition_matrix.png`

---

### Validation Gate ✓

- [ ] VIZ 2.1 BIC curve shows a clear elbow — optimal K identified
- [ ] VIZ 2.2 regime timeline: crisis bands visually align with **2008–09, March 2020, and 2022**
- [ ] VIZ 2.2 shows continuous regime bands (weeks/months), not rapid flickering
- [ ] VIZ 2.3 violin plots show clearly different distributions per regime for all 3 features
- [ ] VIZ 2.4 transition matrix diagonal values all above 0.90

**This is the most important gate in the entire project. If regimes don't make economic sense, nothing downstream is valid.**

---

## Phase 3: Regime Validation (Economic Sanity Check)

**Goal:** Confirm that the HMM discovered economically meaningful structure — not just statistical clusters.

### What to Compute Per Regime

For each regime label (0, 1, 2...):
- Mean daily return of S&P 500
- Mean daily return: Nasdaq vs S&P Value ETF (who outperforms?)
- Mean VIX level
- Mean yield curve spread
- Average regime duration (trading days)
- % of total history spent in this regime
- Which real-world events fall in this regime?

### Target Output — Regime Characterization Table

| Regime | Name I'd Give It | Mean Daily Return | Mean VIX | Yield Spread | Real Events |
|---|---|---|---|---|---|
| 0 | Bull/Calm | ~+0.07% | ~14 | ~+1.5% | 2003–07, 2013–19 |
| 1 | Crisis | ~-0.15% | ~32 | ~-0.2% | 2008–09, Mar 2020 |
| 2 | Tightening/Uncertain | ~-0.02% | ~22 | ~+0.3% | 2022, 2015–16 |

*(Exact numbers will come from your data — the above are indicative targets)*

### Visualizations — Phase 3

**Save all figures to:** `results/figures/phase3/`

---

**VIZ 3.1 — Regime Statistics Bar Chart (4-panel)**
A 2×2 grid showing the mean value of 4 key variables per regime, as grouped bar charts.

```
Panel 1: Mean daily S&P 500 return per regime (with error bars = std dev)
Panel 2: Mean VIX per regime
Panel 3: Mean yield curve spread per regime
Panel 4: Mean Nasdaq return minus mean Value ETF return per regime
         (positive = growth outperforms, negative = value outperforms)
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
regime_names = ['Bull/Calm', 'Crisis', 'Tightening']  # replace with your names
# For each panel, compute per-regime means and plot as bar chart
# Add a horizontal dashed line at zero for return panels
```

What to look for manually:
- Panel 1: crisis regime bar should be the only clearly negative bar
- Panel 2: crisis regime VIX bar should be 2–3x taller than calm regime
- Panel 4: growth outperforms in calm regime (positive), value outperforms or is flat in tightening regime

Save as: `results/figures/phase3/viz1_regime_statistics.png`

---

**VIZ 3.2 — Regime Duration Histogram**
For each regime, show a histogram of how long individual regime episodes last (in trading days).

```python
fig, axes = plt.subplots(1, optimal_k, figsize=(5*optimal_k, 4))

for k, ax in enumerate(axes):
    # Compute run-length encoding of regime_labels to get episode durations
    durations = compute_regime_durations(regime_labels, k)  # implement with itertools.groupby
    ax.hist(durations, bins=20, color=colors[k], edgecolor='black')
    ax.axvline(np.mean(durations), color='red', linestyle='--',
               label=f'Mean: {np.mean(durations):.0f} days')
    ax.set_title(f'Regime {k} — Episode Duration')
    ax.set_xlabel("Duration (trading days)")
    ax.legend()
```

What to look for manually:
- Calm regime: most episodes should be long (50–200 days), right-skewed distribution
- Crisis regime: shorter but intense (20–80 days typically)
- No regime should have a mean duration below 10 trading days — if it does, HMM is too noisy

Save as: `results/figures/phase3/viz2_regime_duration_histogram.png`

---

**VIZ 3.3 — Regime Time Allocation Pie Chart**
Simple pie chart showing what % of the full 2000–2023 history each regime occupies.

```python
regime_counts = pd.Series(regime_labels).value_counts().sort_index()
plt.pie(regime_counts, labels=regime_names, autopct='%1.1f%%',
        colors=[colors[k] for k in regime_counts.index])
plt.title("% of History Spent in Each Regime (2000–2023)")
```

What to look for manually:
- Calm regime should be the largest slice (~55–65% of history)
- Crisis regime should be the smallest (~10–20%)
- If crisis regime is >30%, your K may be too low and it's absorbing moderate-stress periods

Save as: `results/figures/phase3/viz3_regime_time_allocation.png`

---

### Validation Gate ✓

- [ ] Characterization table completed with real numbers from your data
- [ ] Each regime has an **economically interpretable name** you can defend
- [ ] VIZ 3.1 Panel 1 shows crisis regime as the only clearly negative return bar
- [ ] VIZ 3.2 shows mean regime duration above 10 trading days for all regimes
- [ ] VIZ 3.3 shows calm regime occupying the majority of history (>50%)
- [ ] The "crisis" regime contains 2008–09 and March 2020 — if it doesn't, your HMM needs retuning
- [ ] You can write a 2–3 sentence plain-English description of each regime

---

## Phase 4: Forecasting Baseline

**Goal:** Build two baseline models trained on the full dataset with **no regime conditioning**. These are the benchmarks to beat.

### Train / Validation / Test Split

```
Train:      Jan 2000 – Dec 2017
Validation: Jan 2018 – Dec 2020
Test:       Jan 2021 – Dec 2023   ← HOLD OUT. Do not touch until Phase 6.
```

**Critical:** Never use test set data for any tuning decisions. It is only opened once in Phase 6.

### Baseline 1 — ARIMA

```python
from pmdarima import auto_arima

model_arima = auto_arima(train_returns, seasonal=False, information_criterion='bic')
forecasts_arima = model_arima.predict(n_periods=len(val_returns))
```

Target: S&P 500 **daily returns** (not price — returns are stationary)

### Baseline 2 — Temporal Fusion Transformer (TFT)

Library: `pytorch-forecasting`

**Inputs to TFT:**
- Past returns (lookback window: 60 trading days)
- Past VIX
- Past yield curve spread
- Federal Funds Rate
- Day of week, month (time features — encode cyclically)

**No regime label passed as input — this is the global baseline.**

### Metrics to Record (on Validation Set)

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error on daily returns |
| RMSE | Root Mean Squared Error on daily returns |
| Directional Accuracy | % of days where predicted direction (up/down) matches actual |

**Expected ranges (validation set):**
- MAE: 0.008 – 0.012 (0.8% – 1.2% daily return error)
- Directional accuracy: 52% – 54%

> ⚠️ If directional accuracy exceeds 60%, you almost certainly have **lookahead bias** somewhere. Stop and audit your data pipeline.

### Visualizations — Phase 4

**Save all figures to:** `results/figures/phase4/`

---

**VIZ 4.1 — Train/Val Split Timeline**
A simple horizontal bar chart showing which portion of the data is train vs validation vs test (held out).

```python
fig, ax = plt.subplots(figsize=(14, 2))
ax.barh(0, width=train_days, left=0, color='#4CAF50', label='Train (2000–2017)')
ax.barh(0, width=val_days, left=train_days, color='#FF9800', label='Validation (2018–2020)')
ax.barh(0, width=test_days, left=train_days+val_days, color='#F44336',
        label='Test — HELD OUT (2021–2023)', alpha=0.4, hatch='//')
ax.set_yticks([])
ax.set_xlabel("Trading Days")
ax.legend(loc='upper left')
ax.set_title("Data Split — Test Set Locked Until Phase 6")
```

What to look for manually:
- Test set (red hatched) should be visually separate and clearly labeled as locked
- Approximate proportions: train ~72%, val ~13%, test ~13%

Save as: `results/figures/phase4/viz1_data_split.png`

---

**VIZ 4.2 — ARIMA Forecast vs Actual (Validation Set)**
Line chart of actual daily returns vs ARIMA predictions on the validation set (2018–2020).

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top panel: full validation period
axes[0].plot(val_dates, actual_returns, color='black', linewidth=0.6, label='Actual', alpha=0.7)
axes[0].plot(val_dates, arima_preds, color='blue', linewidth=0.6, label='ARIMA', alpha=0.7)
axes[0].set_title("ARIMA: Forecast vs Actual — Full Validation Period (2018–2020)")
axes[0].legend()

# Bottom panel: zoom in to one specific quarter (e.g. Q1 2020 — COVID crash)
axes[1].plot(q1_2020_dates, q1_2020_actual, color='black', linewidth=1, label='Actual')
axes[1].plot(q1_2020_dates, q1_2020_arima, color='blue', linewidth=1, label='ARIMA')
axes[1].set_title("ARIMA: Zoom — Q1 2020 (COVID Crash Period)")
axes[1].legend()
```

What to look for manually:
- ARIMA should roughly follow the magnitude of returns but will miss sharp spikes
- During COVID crash (Feb–March 2020), ARIMA should lag badly — this is expected and fine
- If ARIMA perfectly tracks every move, you have lookahead bias

Save as: `results/figures/phase4/viz2_arima_forecast.png`

---

**VIZ 4.3 — TFT Forecast vs Actual (Validation Set)**
Same structure as VIZ 4.2 but for TFT, plus a residual plot.

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Full validation period
axes[0].plot(val_dates, actual_returns, color='black', linewidth=0.6, label='Actual')
axes[0].plot(val_dates, tft_preds, color='darkorange', linewidth=0.6, label='TFT Global')
axes[0].set_title("TFT Global: Forecast vs Actual — Validation Period")

# Panel 2: Zoom into 2022 rate-hike period
axes[1].plot(y2022_dates, y2022_actual, color='black', linewidth=1, label='Actual')
axes[1].plot(y2022_dates, y2022_tft, color='darkorange', linewidth=1, label='TFT Global')
axes[1].set_title("TFT Global: Zoom — 2022 Rate-Hike Period")

# Panel 3: Residuals over time (actual minus predicted)
residuals = actual_returns - tft_preds
axes[2].bar(val_dates, residuals, color=np.where(residuals > 0, 'green', 'red'),
            width=1, alpha=0.6)
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_title("TFT Global: Residuals Over Time")
```

What to look for manually:
- Residuals should be roughly centered around zero with no obvious trend
- Large residual clusters (model consistently wrong in one direction) in a specific period = regime mismatch
- TFT should noticeably outperform ARIMA in volatile periods (wider return swings)

Save as: `results/figures/phase4/viz3_tft_baseline_forecast.png`

---

**VIZ 4.4 — Baseline Metrics Comparison Bar Chart**
Side-by-side bar chart comparing ARIMA vs TFT Global on MAE, RMSE, and directional accuracy.

```python
metrics = ['MAE', 'RMSE', 'Directional Accuracy']
arima_vals = [arima_mae, arima_rmse, arima_dir_acc]
tft_vals = [tft_mae, tft_rmse, tft_dir_acc]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, arima_vals, width, label='ARIMA', color='steelblue')
ax.bar(x + width/2, tft_vals, width, label='TFT Global', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Baseline Model Comparison — Validation Set")
ax.legend()
```

What to look for manually:
- TFT bars should be shorter (lower error) than ARIMA for MAE and RMSE
- TFT directional accuracy bar should be taller
- Write these exact numbers into your metrics table — they are the benchmark for Phase 5

Save as: `results/figures/phase4/viz4_baseline_comparison.png`

---

### Validation Gate ✓

- [ ] ARIMA trained and producing forecasts — MAE and RMSE computed and written down
- [ ] TFT trained and producing forecasts — MAE, RMSE, directional accuracy computed and written down
- [ ] VIZ 4.3 residuals look roughly centered around zero with no systematic drift
- [ ] VIZ 4.4 confirms TFT outperforms ARIMA on at least 2 of 3 metrics (if not, TFT setup has a bug)
- [ ] Directional accuracy is between 51–56% — if above 60%, audit for lookahead bias immediately

---

## Phase 5: Regime-Conditioned Forecasting

**Goal:** Build three variants of regime-aware TFT and compare against Phase 4 baselines.

### Variant A — TFT-Separate

Train a **completely separate TFT model for each regime**.

```
During training: filter data to only rows where regime_label == k, train TFT_k
During inference: get current regime from HMM, route to TFT_k
```

### Variant B — TFT-Conditioned

Train a **single TFT** but pass regime label as a **static categorical input**.

```
TFT input = [past returns, VIX, yield spread, Fed rate, time features, regime_label]
The model learns internally how to condition on regime
```

### Variant C — TFT-Ensemble (Soft Routing)

Train separate models per regime (like Variant A) but at inference time use the **HMM's probability distribution** over regimes — not the hard label — as blending weights.

```
If HMM says: P(regime=0)=0.7, P(regime=1)=0.3
Forecast = 0.7 * TFT_0_forecast + 0.3 * TFT_1_forecast
```

This is theoretically cleaner than hard routing because it handles regime transitions gracefully.

### Comparison Table to Fill (Validation Set)

| Model | Overall MAE | Crisis Regime MAE | Calm Regime MAE | Directional Acc |
|---|---|---|---|---|
| ARIMA (baseline) | | | | |
| TFT Global | | | | |
| TFT-Separate | | | | |
| TFT-Conditioned | | | | |
| TFT-Ensemble | | | | |

**The most interesting finding will be per-regime metrics** — where exactly does regime-awareness help? Crisis periods? Rate-tightening periods?

### Visualizations — Phase 5

**Save all figures to:** `results/figures/phase5/`

---

**VIZ 5.1 — Per-Regime Forecast Comparison (the key result plot)**
For each regime separately, a side-by-side bar chart comparing MAE across all 5 models.

```python
regimes = ['Calm', 'Crisis', 'Tightening']
models = ['ARIMA', 'TFT Global', 'TFT-Separate', 'TFT-Conditioned', 'TFT-Ensemble']
colors_model = ['steelblue', 'darkorange', 'green', 'purple', 'crimson']

fig, axes = plt.subplots(1, len(regimes), figsize=(16, 5), sharey=True)

for ax, regime_name in zip(axes, regimes):
    mae_vals = [get_mae_for_regime(model, regime_name) for model in models]
    ax.bar(models, mae_vals, color=colors_model)
    ax.set_title(f'{regime_name} Regime — MAE')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(tft_global_mae[regime_name], color='darkorange',
               linestyle='--', alpha=0.5, label='TFT Global baseline')

fig.suptitle("Per-Regime MAE: All Models vs Baseline", fontsize=14)
```

What to look for manually:
- In the Crisis regime panel: regime-aware models (Separate/Conditioned/Ensemble) bars should be shorter than TFT Global
- In the Calm regime panel: all bars should be roughly equal — if regime models perform much worse in calm periods, something is wrong
- The regime where improvement is largest tells you the core finding

Save as: `results/figures/phase5/viz1_per_regime_mae_comparison.png`
**This plot is the centerpiece of your final GitHub README.**

---

**VIZ 5.2 — Forecast Comparison During a Regime Transition**
Zoom into a specific period containing a regime switch — e.g., January–June 2022 (calm → tightening transition) — and overlay all model forecasts.

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top panel: actual returns + all model forecasts
axes[0].plot(transition_dates, actual, color='black', linewidth=1.2, label='Actual', zorder=5)
axes[0].plot(transition_dates, tft_global, color='darkorange', linewidth=0.8,
             linestyle='--', label='TFT Global')
axes[0].plot(transition_dates, tft_ensemble, color='crimson', linewidth=0.8,
             label='TFT-Ensemble')
axes[0].set_title("Forecast Comparison During Regime Transition (Jan–Jun 2022)")
axes[0].legend()

# Bottom panel: regime probability over time (soft HMM output)
axes[1].stackplot(transition_dates, hmm_probs.T,
                  labels=regime_names, colors=['#90EE90', '#FF6B6B', '#FFD700'])
axes[1].set_title("HMM Regime Probabilities During Same Period")
axes[1].set_ylabel("Probability")
axes[1].legend(loc='upper left')
```

What to look for manually:
- The bottom panel should show regime probabilities shifting (green decreasing, yellow/red increasing) during the transition
- TFT-Ensemble should respond to this shift faster than TFT Global
- If both models look identical during the transition, soft routing isn't working — debug the ensemble blending

Save as: `results/figures/phase5/viz2_regime_transition_forecast.png`

---

**VIZ 5.3 — Full Comparison Table Heatmap**
Visualize the entire metrics table as a color-coded heatmap — green = better than TFT Global, red = worse.

```python
# Build metrics DataFrame: rows=models, columns=metrics
metrics_df = pd.DataFrame({
    'Overall MAE': [...],
    'Crisis MAE': [...],
    'Calm MAE': [...],
    'Tightening MAE': [...],
    'Dir. Accuracy': [...]
}, index=models)

# Normalize relative to TFT Global baseline
normalized = (metrics_df - metrics_df.loc['TFT Global']) / metrics_df.loc['TFT Global']

# For MAE: negative normalized value = improvement (green)
# For Dir. Accuracy: positive normalized value = improvement (green)
sns.heatmap(normalized, annot=True, fmt=".1%", cmap="RdYlGn_r",
            center=0, linewidths=0.5)
plt.title("Model Performance Relative to TFT Global Baseline\n(Green = Better)")
```

What to look for manually:
- The column "Crisis MAE" should have the most green for regime-aware models
- No model should be dramatically red in the "Calm MAE" column — that would mean regime conditioning is hurting on easy periods
- The "Overall MAE" column tells a mixed story — the per-regime columns are the honest result

Save as: `results/figures/phase5/viz3_full_comparison_heatmap.png`

---

### Validation Gate ✓

- [ ] All three variants trained and producing validation set metrics
- [ ] VIZ 5.1 shows at least one regime-aware model outperforming TFT Global in at least one regime
- [ ] Per-regime metrics computed — you can identify **which regime drives the improvement**
- [ ] VIZ 5.2 shows regime probability shift during the 2022 transition period
- [ ] Full comparison table (Phase 5 table) is completely filled in with real numbers

---

## Phase 6: Rigorous Evaluation & Story

**Goal:** Final evaluation on held-out test set (2021–2023). Build the complete project narrative.

### Test Period Context (2021–2023)

This period contains a genuine regime transition:
- 2021: Post-COVID recovery bull run
- 2022: Aggressive Fed rate hikes, bear market — clear regime switch
- 2023: Stabilization, recovery

This is ideal for testing regime-awareness because the model must detect and adapt to a real transition it hasn't been trained on.

### Statistical Significance Testing

Use the **Diebold-Mariano (DM) test** — the standard test for comparing forecast accuracy.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
# DM test implementation — compare forecast errors of two models
# Null hypothesis: both models have equal forecast accuracy
# p < 0.05 means the difference is statistically significant
```

### Final Deliverables

**1. Key Metrics Table (Test Set)**

| Model | MAE | RMSE | Directional Acc | DM Test p-value vs TFT Global |
|---|---|---|---|---|
| ARIMA | | | | — |
| TFT Global | | | | — |
| Best Regime Model | | | | |

**2. Final Visualizations**
**Save all to:** `results/figures/phase6/`

---

**VIZ 6.1 — The Hero Plot (README cover figure)**
S&P 500 price chart (full 2000–2023) with HMM regime background + annotated key events.

```python
fig, ax = plt.subplots(figsize=(20, 7))

# Price line
ax.plot(dates, sp500_price, color='black', linewidth=0.8, zorder=4)

# Regime background
for regime_id, color in colors.items():
    mask = (regime_labels == regime_id)
    ax.fill_between(dates, 0, sp500_price.max()*1.05,
                    where=mask, alpha=0.25, color=color, label=regime_names[regime_id])

# Annotate key events
annotations = {
    '2008\nCrisis': '2008-09-15',
    'COVID\nCrash': '2020-03-23',
    'Fed Hikes\nBegin': '2022-03-16',
}
for label, date in annotations.items():
    price_at_date = sp500_price.loc[date]
    ax.annotate(label, xy=(pd.Timestamp(date), price_at_date),
                xytext=(pd.Timestamp(date), price_at_date * 1.15),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center')

ax.set_title("S&P 500 (2000–2023) — HMM Macro Regime Detection", fontsize=14)
ax.set_ylabel("S&P 500 Price")
ax.legend(loc='upper left')
ax.set_yscale('log')  # log scale makes long-term trends clearer
```

Save as: `results/figures/phase6/viz1_hero_regime_chart.png`
**This is the figure that goes at the top of your GitHub README.**

---

**VIZ 6.2 — Test Set: Global vs Best Regime Model**
Direct forecast comparison on the held-out 2021–2023 test set.

```python
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Panel 1: 2021 (COVID recovery bull run)
axes[0].plot(y2021_dates, y2021_actual, 'k-', linewidth=0.8, label='Actual')
axes[0].plot(y2021_dates, y2021_global, color='darkorange', linewidth=0.8,
             linestyle='--', label='TFT Global')
axes[0].plot(y2021_dates, y2021_best, color='crimson', linewidth=0.8,
             label='Best Regime Model')
axes[0].set_title("2021 — Post-COVID Bull Run (Calm Regime)")

# Panel 2: 2022 (rate hike bear market)
axes[1].plot(y2022_dates, y2022_actual, 'k-', linewidth=0.8, label='Actual')
axes[1].plot(y2022_dates, y2022_global, color='darkorange', linewidth=0.8,
             linestyle='--', label='TFT Global')
axes[1].plot(y2022_dates, y2022_best, color='crimson', linewidth=0.8,
             label='Best Regime Model')
axes[1].set_title("2022 — Rate Hike Bear Market (Tightening Regime)")

# Panel 3: 2023 (stabilization)
axes[2].plot(y2023_dates, y2023_actual, 'k-', linewidth=0.8, label='Actual')
axes[2].plot(y2023_dates, y2023_global, color='darkorange', linewidth=0.8,
             linestyle='--', label='TFT Global')
axes[2].plot(y2023_dates, y2023_best, color='crimson', linewidth=0.8,
             label='Best Regime Model')
axes[2].set_title("2023 — Stabilization/Recovery")

for ax in axes:
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.5)
```

Save as: `results/figures/phase6/viz2_test_set_comparison.png`

---

**VIZ 6.3 — Final Per-Regime MAE Bar Chart (Test Set)**
The same structure as VIZ 5.1 but now on the test set — the definitive result.

Same code as VIZ 5.1 but using test set metrics.

Save as: `results/figures/phase6/viz3_final_per_regime_results.png`

---

**3. One-Sentence Finding**

Template:
> *"Regime-conditioned TFT outperforms a global TFT by X% on MAE during [regime name] regimes (p=Y, Diebold-Mariano test), while performing comparably during calm regimes — suggesting that explicit macro-regime supervision is a useful inductive bias in data-scarce forecasting settings."*

Fill in X and Y from your actual results.

---

**4. README for GitHub**

Structure:
```
## What this project does (2 sentences)
## Key finding (1 sentence + VIZ 6.1 as the cover figure)
## Pipeline (Phase 1 → 2 → 3 → 4 → 5 flow diagram)
## How to reproduce (setup + run instructions)
## Results table (test set metrics)
## Limitations (honest, 3 bullet points)
```

### Validation Gate ✓ (Project Complete)

- [ ] Test set metrics computed for all models — test set was never touched before this phase
- [ ] Diebold-Mariano test run — result is statistically significant (p < 0.05) in at least one regime
- [ ] VIZ 6.1 hero plot is clean, annotated, and export-ready (300 DPI PNG)
- [ ] VIZ 6.2 test set comparison shows visible difference between TFT Global and best regime model in 2022 panel
- [ ] VIZ 6.3 final per-regime bar chart matches or improves on validation set results from Phase 5
- [ ] One-sentence finding written with real numbers filled in
- [ ] GitHub repo live with clean README and VIZ 6.1 as cover figure
- [ ] You can explain the entire project — methodology, findings, and limitations — in 5 minutes without referring to notes

---

## Visualization Standards (Apply Across All Phases)

These standards ensure all plots are consistent, readable, and export-ready for GitHub.

### Setup (put this at the top of every notebook)

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Global style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 150,           # screen display
    'savefig.dpi': 300,          # export quality
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
})

# Consistent regime color palette (use across ALL regime plots)
REGIME_COLORS = {
    0: '#4CAF50',   # green  — calm/bull
    1: '#F44336',   # red    — crisis
    2: '#FF9800',   # orange — tightening
    3: '#2196F3',   # blue   — if 4th regime exists
}

# Consistent model color palette (use across ALL forecast comparison plots)
MODEL_COLORS = {
    'ARIMA':           'steelblue',
    'TFT Global':      'darkorange',
    'TFT-Separate':    '#4CAF50',
    'TFT-Conditioned': 'purple',
    'TFT-Ensemble':    'crimson',
    'Actual':          'black',
}
```

### Saving Convention

```python
import os

def save_fig(fig, phase, viz_name):
    """Save figure to correct directory with consistent naming."""
    path = f"results/figures/phase{phase}/{viz_name}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")

# Usage: save_fig(fig, phase=1, viz_name="viz1_feature_dashboard")
```

### Complete Figure Inventory

| Figure | Phase | Filename | Purpose |
|---|---|---|---|
| VIZ 1.1 | 1 | viz1_feature_dashboard.png | 4-panel raw data overview |
| VIZ 1.2 | 1 | viz2_correlation_heatmap.png | Feature redundancy check |
| VIZ 1.3 | 1 | viz3_missing_data_audit.png | Data cleanliness proof |
| VIZ 2.1 | 2 | viz1_bic_curve.png | K selection for HMM |
| VIZ 2.2 | 2 | viz2_regime_timeline.png | **Core diagnostic — most important** |
| VIZ 2.3 | 2 | viz3_feature_distributions_per_regime.png | Regime separability |
| VIZ 2.4 | 2 | viz4_transition_matrix.png | Regime stability check |
| VIZ 3.1 | 3 | viz1_regime_statistics.png | Economic characterization |
| VIZ 3.2 | 3 | viz2_regime_duration_histogram.png | Regime persistence check |
| VIZ 3.3 | 3 | viz3_regime_time_allocation.png | Regime prevalence pie |
| VIZ 4.1 | 4 | viz1_data_split.png | Train/val/test split diagram |
| VIZ 4.2 | 4 | viz2_arima_forecast.png | ARIMA baseline visual |
| VIZ 4.3 | 4 | viz3_tft_baseline_forecast.png | TFT Global baseline visual |
| VIZ 4.4 | 4 | viz4_baseline_comparison.png | ARIMA vs TFT side-by-side |
| VIZ 5.1 | 5 | viz1_per_regime_mae_comparison.png | **Key result — centerpiece** |
| VIZ 5.2 | 5 | viz2_regime_transition_forecast.png | Transition period behavior |
| VIZ 5.3 | 5 | viz3_full_comparison_heatmap.png | All models all metrics |
| VIZ 6.1 | 6 | viz1_hero_regime_chart.png | **README cover figure** |
| VIZ 6.2 | 6 | viz2_test_set_comparison.png | Final test set evaluation |
| VIZ 6.3 | 6 | viz3_final_per_regime_results.png | Definitive result chart |

**Total: 19 figures across 6 phases.**

---

| Phase | Content | Estimated Duration |
|---|---|---|
| 1 | Data Foundation | 1 week |
| 2 | Regime Detection | 1.5 weeks |
| 3 | Regime Validation | 0.5 weeks |
| 4 | Forecasting Baseline | 2 weeks |
| 5 | Regime-Conditioned Models | 2.5 weeks |
| 6 | Rigorous Evaluation | 1.5 weeks |
| **Total** | | **~9–10 weeks** |

---

## Library Reference

```
pip install yfinance pandas_datareader hmmlearn pytorch-forecasting pytorch-lightning pmdarima statsmodels scikit-learn matplotlib seaborn
```

| Library | Used In | Purpose |
|---|---|---|
| `yfinance` | Phase 1 | Price data download |
| `pandas_datareader` | Phase 1 | FRED macro data |
| `hmmlearn` | Phase 2 | HMM training and Viterbi path |
| `pmdarima` | Phase 4 | Auto ARIMA model selection |
| `pytorch-forecasting` | Phase 4, 5 | TFT implementation |
| `pytorch-lightning` | Phase 4, 5 | TFT training loop |
| `statsmodels` | Phase 6 | Diebold-Mariano test |
| `scikit-learn` | Throughout | Preprocessing, metrics |
| `matplotlib` | Throughout | All visualizations (primary) |
| `seaborn` | Throughout | Heatmaps, violin plots, styling |

---

## Key Rules to Never Break

1. **No lookahead bias** — macro data (Fed rate decisions) must be lagged by 1 day before use as features. The Fed announces at 2pm ET — you cannot use that day's rate as a feature for that same day's forecast.
2. **Test set is sacred** — never open 2021–2023 data until Phase 6. Any tuning decision made using test data invalidates the entire project.
3. **Regime labels for inference use only HMM trained on train set** — do not refit the HMM on the full dataset before generating labels for the validation/test periods.
4. **Report per-regime metrics, not just overall** — an overall improvement can hide the fact that the model only helps in one regime. Per-regime breakdown is the honest and interesting result.
5. **State limitations honestly** — this project does not demonstrate trading profitability. The claim is strictly: *regime-aware models produce more accurate return forecasts in specific macro environments*. Nothing more.

---

*Last updated: May 2026*
*Project: Macro-Regime Aware Deep Learning for Equity Return Forecasting*
*Author: Vineet Jangir — IIT Bombay*
