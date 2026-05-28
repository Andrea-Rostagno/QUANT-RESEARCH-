# CLAUDE.md — DAX Quant Research Lab

## Project overview

This is a professional intraday quantitative research lab for DAX trading using MetaTrader 5.
The goal is to discover, validate, and rank strategy candidates — not to implement a pre-defined strategy.

## Critical rules (never violate)

- **No look-ahead bias**: features must use only data available at decision time.
- **No data leakage**: train/test splits are strictly chronological; use purge + embargo.
- **No random splits**: never use `train_test_split(shuffle=True)` on time series.
- **Costs always included**: every backtest includes spread + slippage + commission.
- **No invented strategies**: the framework generates and filters candidates; do not hardcode signals.
- **Financial metrics first**: accuracy is insufficient; use profit factor, expectancy, Sharpe, max drawdown.

## Architecture

- `configs/dax_m1.yaml` — all parameters (symbol, features, labeling, backtest costs)
- `src/` — pure Python modules, importable from notebooks
- `notebooks/` — progressive pipeline: 01 → 02 → ... → 08
- `data/` — Parquet files at each stage (raw → interim → processed → features)
- `models/` — saved model artifacts
- `reports/` — CSV/HTML strategy candidate reports

## Coding standards

- Every public function must have a docstring.
- No hardcoded symbols or parameters — always read from config.
- Save intermediate results as Parquet files to `data/` subdirectories.
- Use `loguru` for logging in modules.
- Use `tqdm` for progress bars in long loops.
- Import `src` modules using `sys.path.insert(0, "..")` at the top of notebooks.

## Broker context

- Broker: Key To Markets
- Platform: MetaTrader 5 (Windows)
- Main asset: DAX (symbol TBD — see TODO in configs/dax_m1.yaml)
- Commission: TODO confirm
- Spread: TODO confirm
- Contract spec: TODO confirm

## Important TODOs

Search for `# TODO` in the codebase to find all items requiring broker confirmation.

## MT5 symbol discovery

If the DAX symbol is unknown, call `mt5_loader.find_dax_symbol()` which searches for:
`DAX`, `GER`, `DE40`, `GER40`, `XGER`

## Notebook execution order

01 → 02 → 03 → 04 → 05 → 06 → 07 → 08

Each notebook is self-contained but expects outputs from previous notebooks in `data/`.
