"""
cross_asset.py — Cross-asset features for DAX intraday research.

Features:
  - Log returns of auxiliary symbols (aligned to DAX bars)
  - Rolling correlation between DAX and each auxiliary symbol
  - Lead-lag cross-correlations (auxiliary leads DAX)
  - Beta of DAX vs S&P / US30
  - Relative strength: DAX vs US500 / NAS100

All look-back windows use only past data — no look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_aux(dax: pd.DataFrame, aux: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Align auxiliary OHLCV to DAX index via forward-fill.

    Parameters
    ----------
    dax:
        DAX OHLCV DataFrame (target index).
    aux:
        Auxiliary symbol OHLCV DataFrame.

    Returns
    -------
    tuple of (dax_ret, aux_ret)
        Log return series aligned to dax.index.
    """
    aux_close = aux["close"].reindex(dax.index, method="ffill")
    dax_ret   = np.log(dax["close"] / dax["close"].shift(1))
    aux_ret   = np.log(aux_close / aux_close.shift(1))
    return dax_ret, aux_ret


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_cross_returns(
    dax: pd.DataFrame,
    aux_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute log returns for all auxiliary symbols, aligned to DAX bars.

    Parameters
    ----------
    dax:
        DAX M1 OHLCV.
    aux_dict:
        Dict of symbol → OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns ``ret_{symbol}`` for each auxiliary symbol.
    """
    frames: dict[str, pd.Series] = {}
    for sym, aux_df in aux_dict.items():
        try:
            _, aux_ret = _align_aux(dax, aux_df)
            frames[f"ret_{sym}"] = aux_ret
        except Exception as exc:
            logger.warning(f"Could not compute returns for {sym}: {exc}")
    return pd.DataFrame(frames, index=dax.index)


# ---------------------------------------------------------------------------
# Rolling correlations
# ---------------------------------------------------------------------------

def compute_rolling_correlation(
    dax: pd.DataFrame,
    aux_dict: dict[str, pd.DataFrame],
    windows: list[int] = (20, 60),
) -> pd.DataFrame:
    """Compute rolling Pearson correlation between DAX and each auxiliary symbol.

    Parameters
    ----------
    dax:
        DAX M1 OHLCV.
    aux_dict:
        Dict of symbol → OHLCV DataFrame.
    windows:
        Rolling window sizes in bars.

    Returns
    -------
    pd.DataFrame
        Columns ``corr_{symbol}_{window}`` for each combination.
    """
    dax_ret = np.log(dax["close"] / dax["close"].shift(1))
    frames: dict[str, pd.Series] = {}

    for sym, aux_df in aux_dict.items():
        try:
            aux_close = aux_df["close"].reindex(dax.index, method="ffill")
            aux_ret   = np.log(aux_close / aux_close.shift(1))
            for w in windows:
                corr = dax_ret.rolling(w).corr(aux_ret)
                frames[f"corr_{sym}_{w}"] = corr
        except Exception as exc:
            logger.warning(f"Correlation failed for {sym}: {exc}")

    return pd.DataFrame(frames, index=dax.index)


# ---------------------------------------------------------------------------
# Lead-lag
# ---------------------------------------------------------------------------

def compute_lead_lag(
    dax: pd.DataFrame,
    aux_dict: dict[str, pd.DataFrame],
    max_lag: int = 5,
    window: int = 60,
) -> pd.DataFrame:
    """Compute lagged correlation between auxiliary returns and DAX returns.

    A positive lag means the auxiliary symbol leads DAX by ``lag`` bars.

    Parameters
    ----------
    dax:
        DAX M1 OHLCV.
    aux_dict:
        Dict of symbol → OHLCV DataFrame.
    max_lag:
        Maximum number of bars to look back.
    window:
        Rolling window for correlation.

    Returns
    -------
    pd.DataFrame
        Columns ``leadlag_{symbol}_lag{k}`` for each symbol and lag k=1..max_lag.
    """
    dax_ret = np.log(dax["close"] / dax["close"].shift(1))
    frames: dict[str, pd.Series] = {}

    for sym, aux_df in aux_dict.items():
        try:
            aux_close = aux_df["close"].reindex(dax.index, method="ffill")
            aux_ret   = np.log(aux_close / aux_close.shift(1))
            for lag in range(1, max_lag + 1):
                # aux_ret shifted back by lag: correlation(dax_ret_t, aux_ret_{t-lag})
                corr = dax_ret.rolling(window).corr(aux_ret.shift(lag))
                frames[f"leadlag_{sym}_lag{lag}"] = corr
        except Exception as exc:
            logger.warning(f"Lead-lag failed for {sym}: {exc}")

    return pd.DataFrame(frames, index=dax.index)


# ---------------------------------------------------------------------------
# Beta and relative strength
# ---------------------------------------------------------------------------

def compute_rolling_beta(
    dax: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 60,
    label: str = "benchmark",
) -> pd.Series:
    """Compute rolling beta of DAX relative to a benchmark.

    beta = cov(r_dax, r_bench) / var(r_bench)

    Parameters
    ----------
    dax:
        DAX M1 OHLCV.
    benchmark:
        Benchmark OHLCV (e.g. US500).
    window:
        Rolling window.
    label:
        Name suffix for the output column.

    Returns
    -------
    pd.Series
        Rolling beta series.
    """
    dax_ret   = np.log(dax["close"] / dax["close"].shift(1))
    bench_cls = benchmark["close"].reindex(dax.index, method="ffill")
    bench_ret = np.log(bench_cls / bench_cls.shift(1))

    cov = dax_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var(ddof=1)
    beta = cov / var.replace(0, np.nan)
    return beta.rename(f"beta_{label}_{window}")


def compute_relative_strength(
    dax: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 20,
    label: str = "bench",
) -> pd.Series:
    """Compute rolling relative performance: DAX cumret / benchmark cumret.

    Values > 1 → DAX outperforming; < 1 → underperforming.

    Returns
    -------
    pd.Series
        Relative strength ratio.
    """
    dax_ret   = np.log(dax["close"] / dax["close"].shift(1))
    bench_cls = benchmark["close"].reindex(dax.index, method="ffill")
    bench_ret = np.log(bench_cls / bench_cls.shift(1))

    dax_cum   = dax_ret.rolling(window).sum()
    bench_cum = bench_ret.rolling(window).sum()
    rs = dax_cum - bench_cum
    return rs.rename(f"rel_strength_{label}_{window}")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all_cross_asset(
    dax: pd.DataFrame,
    aux_dict: dict[str, pd.DataFrame],
    cfg: dict,
) -> pd.DataFrame:
    """Compute all cross-asset features in one call.

    Parameters
    ----------
    dax:
        DAX M1 OHLCV.
    aux_dict:
        Dict of enabled auxiliary symbol DataFrames.
    cfg:
        Feature config section (``cfg["features"]``).

    Returns
    -------
    pd.DataFrame
        All cross-asset feature columns.
    """
    if not aux_dict:
        logger.warning("No auxiliary symbols provided — cross-asset features will be empty.")
        return pd.DataFrame(index=dax.index)

    max_lag  = cfg.get("lead_lag_max_lag", 5)
    corr_wins = [20, 60]

    parts: list[pd.DataFrame | pd.Series] = [
        compute_cross_returns(dax, aux_dict),
        compute_rolling_correlation(dax, aux_dict, corr_wins),
        compute_lead_lag(dax, aux_dict, max_lag),
    ]

    # Beta and relative strength vs US500 if available
    for bench_sym in ("US500", "US30"):
        if bench_sym in aux_dict:
            parts.append(compute_rolling_beta(dax, aux_dict[bench_sym], 60, bench_sym))
            parts.append(compute_relative_strength(dax, aux_dict[bench_sym], 20, bench_sym))

    return pd.concat(parts, axis=1)
