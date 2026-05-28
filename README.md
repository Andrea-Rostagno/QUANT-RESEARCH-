# DAX Quant Research Lab

Professional intraday quantitative research framework for DAX using MetaTrader 5 data.

## Goal

Build a systematic pipeline that **discovers, validates, and ranks** statistically robust intraday strategy candidates on the DAX — with realistic transaction costs, walk-forward validation, and no look-ahead bias.

## Architecture

```
dax-quant-lab/
├── configs/
│   └── dax_m1.yaml          # All parameters: symbol, costs, features, labeling
├── data/
│   ├── raw/                 # MT5 OHLCV downloads (Parquet)
│   ├── interim/             # Resampled timeframes
│   ├── processed/           # Cleaned & merged data
│   └── features/            # Engineered features + labels
├── notebooks/
│   ├── 01_mt5_data_ingestion.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_labeling_triple_barrier.ipynb
│   ├── 04_regime_detection.ipynb
│   ├── 05_supervised_models.ipynb
│   ├── 06_pattern_discovery.ipynb
│   ├── 07_backtest_walk_forward.ipynb
│   └── 08_strategy_report.ipynb
├── src/
│   ├── data/                # MT5 loader
│   ├── features/            # Technical, volatility, time, cross-asset
│   ├── labels/              # Triple-barrier labeling
│   ├── models/              # ML training, regime, explainability
│   ├── backtest/            # Event-driven backtester
│   └── utils/               # Config loader
├── models/                  # Saved model artifacts
├── reports/                 # Strategy candidate tables (CSV/HTML)
└── requirements.txt
```

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. MetaTrader 5

- Install MT5 from your broker (Key To Markets)
- Log in to a live or demo account
- Keep MT5 running while executing notebooks

### 3. Symbol configuration

Edit `configs/dax_m1.yaml` and set `mt5.symbol` to your broker's DAX symbol.
Run notebook `01_mt5_data_ingestion.ipynb` → cell **"Find DAX symbol"** to auto-detect candidates.

### 4. Run notebooks in order

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Each notebook saves its outputs to Parquet so later notebooks can load them independently.

## Key design principles

| Principle | Implementation |
|-----------|----------------|
| No look-ahead bias | All features use only past data; labels computed from future window |
| No data leakage | Strict chronological split; purge+embargo between train/test |
| Realistic costs | Spread + slippage + commission configurable in YAML |
| Walk-forward validation | 5-fold expanding-window splits |
| Financial metrics | Profit factor, expectancy, Sharpe, Sortino, max drawdown |
| Strategy classification | REJECTED / NEEDS_MORE_DATA / PAPER_TEST / EA_CANDIDATE |

## Strategy output

The final report (`reports/strategy_candidates.csv`) ranks all tested strategy candidates with:

- Entry conditions, side, timeframe
- TP/SL/holding-time rules
- Net performance after costs
- Walk-forward stability
- Recommendation status

## Notes

- The MQL5 Expert Advisor is a future step — not part of this repo yet.
- All TODO comments in code require broker confirmation before production use.
