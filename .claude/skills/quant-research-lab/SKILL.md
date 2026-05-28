\---

name: quant-research-lab

description: Use this skill when working on the DAX intraday quantitative research project. It enforces a professional quant workflow: MT5 data ingestion, feature engineering, triple-barrier labeling, regime detection, ML modeling, pattern discovery, walk-forward validation, and realistic backtesting with costs.

\---



\# Quant Research Lab Skill



You are working on a professional intraday quantitative research project for DAX trading using MT5 data.



\## Core objective



Build a research infrastructure that discovers, validates, ranks, and reports statistically robust intraday trading strategy candidates.



Do not create toy examples unless explicitly requested. Prefer modular, production-quality Python code and Jupyter notebooks.



\## Required project structure



Use this structure:



\- data/raw

\- data/interim

\- data/processed

\- data/features

\- notebooks

\- src/data

\- src/features

\- src/labels

\- src/models

\- src/backtest

\- src/utils

\- configs

\- reports

\- models



\## Research principles



Always avoid:



\- look-ahead bias

\- data leakage

\- random train/test split on time series

\- overfitting

\- using future data in features

\- ignoring spread, slippage, commissions

\- reporting only accuracy

\- optimizing on the test set



Always prefer:



\- chronological train/test split

\- walk-forward validation

\- purged/embargoed validation when appropriate

\- realistic transaction costs

\- robust metrics

\- feature importance and stability checks

\- clear strategy candidate reports



\## Data sources



Primary source:



\- MetaTrader 5 via Python package MetaTrader5



Main asset:



\- DAX, symbol to be configured by the user in configs/dax\_m1.yaml



Auxiliary assets may include:



\- NASDAQ / US100

\- S\&P500 / US500

\- Dow Jones / US30

\- Euro Stoxx 50

\- EURUSD

\- XAUUSD

\- VIX or volatility proxy, if available

\- Bund/bonds proxy, if available



\## Feature groups



Create features for:



1\. Returns and candle structure

2\. Volatility and ATR

3\. Momentum and trend

4\. Mean reversion

5\. Volume and tick-volume behavior

6\. Spread and execution quality

7\. Session/time-of-day behavior

8\. Cross-asset lead-lag relationships

9\. Regime detection

10\. News/sentiment placeholders for future extension



\## Labeling



Implement triple-barrier labeling:



\- Long label

\- Short label

\- No-trade label

\- TP hit

\- SL hit

\- Timeout

\- Time to barrier

\- Maximum favorable excursion

\- Maximum adverse excursion



Labels must be calculated strictly using future windows after the decision timestamp.



\## Models



Implement baseline models first:



\- Logistic Regression

\- Random Forest

\- XGBoost

\- LightGBM

\- CatBoost



Then add:



\- clustering/regime detection

\- association rules

\- SHAP analysis

\- optional neural networks only after the tabular baseline works



\## Backtesting



The backtest must include:



\- spread

\- slippage

\- commissions

\- trade direction

\- TP

\- SL

\- max holding time

\- time filters

\- one-position-at-a-time option

\- realistic execution assumptions



\## Output



The final output must rank strategy candidates with:



\- strategy\_id

\- entry conditions

\- side: long/short

\- timeframe

\- TP rule

\- SL rule

\- holding-time rule

\- number of trades

\- net profit

\- profit factor

\- expectancy

\- Sharpe/Sortino

\- max drawdown

\- win rate

\- average R

\- stability by month

\- walk-forward performance

\- recommendation: REJECTED, NEEDS\_MORE\_DATA, PAPER\_TEST, EA\_CANDIDATE



\## Coding rules



Use clean Python modules.

Keep notebooks readable.

Every notebook must explain what it does.

Every function must have docstrings.

Save intermediate datasets as Parquet.

Use config files instead of hardcoded symbols where possible.

