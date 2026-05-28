"""
volatility.py — Volatility estimators for OHLCV data.

All estimators are computed from past bars only (no look-ahead bias).

Features:
  - ATR (Average True Range, Wilder's smoothing)
  - Realised volatility (annualised, multiple windows)
  - Rolling close-to-close volatility
  - Parkinson volatility (high-low estimator)
  - Garman-Klass volatility estimator
  - Volatility regime ratio (short / long window)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Trading minutes per year for the DAX (approximately)
# DAX: ~21.5 h/day × 252 days ≈ 5418 M1 bars/year
_BARS_PER_YEAR_M1 = 5418


# ---------------------------------------------------------------------------
# True Range & ATR
# ---------------------------------------------------------------------------

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute the True Range bar-by-bar.

    TR = max(H-L, |H-Cprev|, |L-Cprev|)

    Returns
    -------
    pd.Series
        True Range with the same index as ``df``.
    """
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        h - l,
        (h - c_prev).abs(),
        (l - c_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rename("true_range")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR using Wilder's smoothing.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    period:
        Smoothing period.

    Returns
    -------
    pd.Series
        ATR series.
    """
    tr = compute_true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr.rename(f"atr_{period}")


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as percentage of close price."""
    atr = compute_atr(df, period)
    return (atr / df["close"]).rename(f"atr_pct_{period}")


# ---------------------------------------------------------------------------
# Realised / rolling volatility
# ---------------------------------------------------------------------------

def compute_realized_volatility(
    df: pd.DataFrame,
    windows: list[int] = (5, 15, 30, 60),
    annualise: bool = True,
) -> pd.DataFrame:
    """Compute realised volatility over multiple rolling windows.

    Uses sum of squared log returns (Rogers-Satchell style, simplified).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    windows:
        Rolling window sizes in bars (M1 = minutes).
    annualise:
        If True, convert to annualised volatility.

    Returns
    -------
    pd.DataFrame
        Columns ``rvol_{window}``.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    frames: dict[str, pd.Series] = {}
    for w in windows:
        rv = np.sqrt(log_ret.pow(2).rolling(w).sum())
        if annualise:
            rv = rv * np.sqrt(_BARS_PER_YEAR_M1 / w)
        frames[f"rvol_{w}"] = rv
    return pd.DataFrame(frames, index=df.index)


def compute_rolling_volatility(
    df: pd.DataFrame,
    windows: list[int] = (20, 60, 120, 240),
    annualise: bool = True,
) -> pd.DataFrame:
    """Compute rolling close-to-close standard deviation volatility.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    windows:
        Rolling window sizes.
    annualise:
        If True, annualise the vol.

    Returns
    -------
    pd.DataFrame
        Columns ``vol_{window}``.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    frames: dict[str, pd.Series] = {}
    for w in windows:
        vol = log_ret.rolling(w).std(ddof=1)
        if annualise:
            vol = vol * np.sqrt(_BARS_PER_YEAR_M1)
        frames[f"vol_{w}"] = vol
    return pd.DataFrame(frames, index=df.index)


# ---------------------------------------------------------------------------
# Advanced volatility estimators
# ---------------------------------------------------------------------------

def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualise: bool = True,
) -> pd.Series:
    """Parkinson (high-low) volatility estimator.

    More efficient than close-to-close when intrabar extremes are informative.

    Formula: sqrt( (1 / (4*n*ln2)) * sum(ln(H/L)^2) )

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    window:
        Rolling window.
    annualise:
        If True, annualise by ``sqrt(_BARS_PER_YEAR_M1 / window)``.

    Returns
    -------
    pd.Series
        Parkinson volatility.
    """
    ln_hl = np.log(df["high"] / df["low"]).pow(2)
    park = np.sqrt(ln_hl.rolling(window).sum() / (4 * window * np.log(2)))
    if annualise:
        park = park * np.sqrt(_BARS_PER_YEAR_M1 / window)
    return park.rename(f"parkinson_vol_{window}")


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualise: bool = True,
) -> pd.Series:
    """Garman-Klass OHLC volatility estimator.

    More efficient than Parkinson as it uses open and close as well.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    window:
        Rolling window.
    annualise:
        If True, annualise.

    Returns
    -------
    pd.Series
        Garman-Klass volatility.
    """
    log_hl  = np.log(df["high"] / df["low"]).pow(2)
    log_co  = np.log(df["close"] / df["open"]).pow(2)
    gk_inst = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    gk = np.sqrt(gk_inst.rolling(window).mean())
    if annualise:
        gk = gk * np.sqrt(_BARS_PER_YEAR_M1)
    return gk.rename(f"gk_vol_{window}")


# ---------------------------------------------------------------------------
# Regime features
# ---------------------------------------------------------------------------

def compute_volatility_ratio(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 120,
) -> pd.Series:
    """Ratio of short-window to long-window rolling volatility.

    Values > 1 indicate an expanding volatility regime.
    Values < 1 indicate a quiet / compressed regime.

    Returns
    -------
    pd.Series
        ``vol_ratio_{short}_{long}``.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_s = log_ret.rolling(short).std(ddof=1)
    vol_l = log_ret.rolling(long).std(ddof=1)
    ratio = vol_s / vol_l.replace(0, np.nan)
    return ratio.rename(f"vol_ratio_{short}_{long}")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all_volatility(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all volatility features in one call.

    Parameters
    ----------
    df:
        M1 OHLCV DataFrame.
    cfg:
        Feature config section (``cfg["features"]``).

    Returns
    -------
    pd.DataFrame
        All volatility columns.
    """
    parts: list[pd.DataFrame | pd.Series] = []

    atr_period = cfg.get("atr_period", 14)
    parts.append(compute_atr(df, atr_period))
    parts.append(compute_atr_pct(df, atr_period))
    parts.append(compute_true_range(df))
    parts.append(compute_realized_volatility(df, cfg.get("realized_vol_windows", [5, 15, 30, 60])))
    parts.append(compute_rolling_volatility(df, cfg.get("rolling_vol_windows", [20, 60, 120, 240])))
    parts.append(compute_parkinson_volatility(df, 20))
    parts.append(compute_garman_klass_volatility(df, 20))
    parts.append(compute_volatility_ratio(df, 20, 120))

    return pd.concat(parts, axis=1)
