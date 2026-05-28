"""
technical.py — Technical indicator features for OHLCV data.

All functions are look-ahead-bias-free: they operate on closed bars only
and use pandas rolling/shift operations that reference past data exclusively.

Features computed:
  - Log returns
  - Candle body, wicks, range
  - EMA (multiple periods)
  - RSI
  - MACD
  - ADX
  - Bollinger Bands
  - Z-score from EMA / rolling mean / VWAP
  - Volume z-score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_log_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Compute log returns: log(close_t / close_{t-1}).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    col:
        Column to compute returns on (default: "close").

    Returns
    -------
    pd.Series
        Log return series with the same index as ``df``.
    """
    return np.log(df[col] / df[col].shift(1)).rename(f"log_ret_{col}")


def compute_returns_multi(df: pd.DataFrame, lags: list[int] = (1, 3, 5, 10, 15, 30)) -> pd.DataFrame:
    """Compute log returns at multiple lags.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    lags:
        Lag values in bars.

    Returns
    -------
    pd.DataFrame
        One column per lag: ``log_ret_{lag}``.
    """
    frames = {}
    for lag in lags:
        frames[f"log_ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
    return pd.DataFrame(frames, index=df.index)


# ---------------------------------------------------------------------------
# Candle structure
# ---------------------------------------------------------------------------

def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle body, wicks, and range in price and normalised forms.

    Features (all normalised by ``close`` to be scale-invariant):
      - ``body``       : abs(close - open)
      - ``body_pct``   : body / close
      - ``upper_wick`` : high - max(open, close)
      - ``lower_wick`` : min(open, close) - low
      - ``range``      : high - low
      - ``range_pct``  : range / close
      - ``body_ratio`` : body / range  (0 if range == 0)
      - ``direction``  : +1 if bullish, -1 if bearish, 0 if doji

    Returns
    -------
    pd.DataFrame
        Candle feature columns.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    rng = h - l

    result = pd.DataFrame({
        "body":        body,
        "body_pct":    body / c,
        "upper_wick":  upper_wick,
        "lower_wick":  lower_wick,
        "range":       rng,
        "range_pct":   rng / c,
        "body_ratio":  np.where(rng > 0, body / rng, 0.0),
        "direction":   np.sign(c - o).astype(int),
    }, index=df.index)
    return result


# ---------------------------------------------------------------------------
# Trend / momentum
# ---------------------------------------------------------------------------

def compute_ema(df: pd.DataFrame, periods: list[int] = (8, 16, 24, 64, 200)) -> pd.DataFrame:
    """Compute Exponential Moving Averages for multiple periods.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    periods:
        EMA window sizes.

    Returns
    -------
    pd.DataFrame
        Columns ``ema_{period}`` and price-relative distances
        ``ema_{period}_dist`` (close - ema) / close.
    """
    frames: dict[str, pd.Series] = {}
    for p in periods:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        frames[f"ema_{p}"]      = ema
        frames[f"ema_{p}_dist"] = (df["close"] - ema) / df["close"]
    return pd.DataFrame(frames, index=df.index)


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    period:
        Look-back period.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{period}")


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Returns
    -------
    pd.DataFrame
        Columns: ``macd``, ``macd_signal``, ``macd_hist``.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd":       macd,
        "macd_signal": macd_sig,
        "macd_hist":  macd - macd_sig,
    }, index=df.index)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, and -DI (Wilder's smoothing).

    Returns
    -------
    pd.DataFrame
        Columns: ``adx``, ``plus_di``, ``minus_di``.
    """
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    prev_h = h.shift(1)
    prev_l = l.shift(1)

    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)

    plus_dm  = np.where((h - prev_h) > (prev_l - l), np.maximum(h - prev_h, 0), 0.0)
    minus_dm = np.where((prev_l - l) > (h - prev_h), np.maximum(prev_l - l, 0), 0.0)

    atr_w    = _wilder_smooth(pd.Series(tr, index=df.index), period)
    plus_dm_s  = _wilder_smooth(pd.Series(plus_dm,  index=df.index), period)
    minus_dm_s = _wilder_smooth(pd.Series(minus_dm, index=df.index), period)

    plus_di  = 100 * plus_dm_s  / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_smooth(dx.fillna(0), period)

    return pd.DataFrame({
        "adx":      adx,
        "plus_di":  plus_di,
        "minus_di": minus_di,
    }, index=df.index)


def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands and %B.

    Returns
    -------
    pd.DataFrame
        Columns: ``bb_mid``, ``bb_upper``, ``bb_lower``, ``bb_width``, ``bb_pct_b``.
    """
    mid   = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std(ddof=1)
    upper = mid + n_std * sigma
    lower = mid - n_std * sigma
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (df["close"] - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "bb_mid":   mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Z-scores and mean reversion
# ---------------------------------------------------------------------------

def compute_zscore(
    series: pd.Series,
    window: int,
    name: str | None = None,
) -> pd.Series:
    """Rolling z-score: (x - mean) / std over the past ``window`` bars.

    Parameters
    ----------
    series:
        Input series.
    window:
        Rolling window size.
    name:
        Optional name for the output series.

    Returns
    -------
    pd.Series
        Z-score series.
    """
    mu  = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    z   = (series - mu) / std.replace(0, np.nan)
    if name:
        z = z.rename(name)
    return z


def compute_price_zscores(df: pd.DataFrame, windows: list[int] = (20, 60, 120)) -> pd.DataFrame:
    """Compute z-score of close price relative to rolling window.

    Returns
    -------
    pd.DataFrame
        Columns ``zscore_close_{w}`` for each window.
    """
    frames: dict[str, pd.Series] = {}
    for w in windows:
        frames[f"zscore_close_{w}"] = compute_zscore(df["close"], w)
    return pd.DataFrame(frames, index=df.index)


def compute_vwap(df: pd.DataFrame, session_col: str | None = None) -> pd.Series:
    """Compute session VWAP.

    If ``session_col`` is None, computes a cumulative daily VWAP
    using UTC date as session boundary.

    Parameters
    ----------
    df:
        OHLCV DataFrame (requires ``tick_volume``).
    session_col:
        Optional column name identifying session group.

    Returns
    -------
    pd.Series
        VWAP series.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"]

    if session_col and session_col in df.columns:
        group = df[session_col]
    else:
        group = df.index.normalize()

    cum_tp_vol = (typical * vol).groupby(group).cumsum()
    cum_vol    = vol.groupby(group).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.rename("vwap")


def compute_vwap_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Z-score of close relative to rolling VWAP deviation.

    Returns
    -------
    pd.Series
        ``vwap_zscore_{window}``.
    """
    vwap  = compute_vwap(df)
    diff  = df["close"] - vwap
    z     = compute_zscore(diff, window)
    return z.rename(f"vwap_zscore_{window}")


def compute_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling z-score of tick volume.

    Returns
    -------
    pd.Series
        ``vol_zscore``.
    """
    return compute_zscore(df["tick_volume"].astype(float), window, name=f"vol_zscore_{window}")


# ---------------------------------------------------------------------------
# Spread features
# ---------------------------------------------------------------------------

def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spread-derived features (requires ``spread`` column in df).

    Returns
    -------
    pd.DataFrame
        Columns: ``spread_pct``, ``spread_zscore_20``.
    """
    if "spread" not in df.columns:
        logger.warning("No 'spread' column found — spread features will be NaN.")
        idx = df.index
        return pd.DataFrame({"spread_pct": np.nan, "spread_zscore_20": np.nan}, index=idx)

    spread = df["spread"].astype(float)
    spread_pct = spread / df["close"]
    spread_z   = compute_zscore(spread, 20, "spread_zscore_20")
    return pd.DataFrame({"spread_pct": spread_pct, "spread_zscore_20": spread_z}, index=df.index)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all_technical(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all technical features in one call and return a single DataFrame.

    Parameters
    ----------
    df:
        M1 OHLCV DataFrame.
    cfg:
        Feature config section from ``configs/dax_m1.yaml``
        (i.e., ``cfg["features"]``).

    Returns
    -------
    pd.DataFrame
        All feature columns aligned to ``df.index``.
    """
    parts: list[pd.DataFrame | pd.Series] = []

    # Returns
    parts.append(compute_log_returns(df))
    parts.append(compute_returns_multi(df, lags=[1, 3, 5, 10, 15, 30]))

    # Candle structure
    parts.append(compute_candle_features(df))

    # EMA
    parts.append(compute_ema(df, cfg.get("ema_periods", [8, 16, 24, 64, 200])))

    # RSI
    parts.append(compute_rsi(df, cfg.get("rsi_period", 14)))

    # MACD
    parts.append(compute_macd(df, cfg.get("macd_fast", 12), cfg.get("macd_slow", 26), cfg.get("macd_signal", 9)))

    # ADX
    parts.append(compute_adx(df, cfg.get("adx_period", 14)))

    # Bollinger Bands
    parts.append(compute_bollinger_bands(df, cfg.get("bbands_period", 20), cfg.get("bbands_std", 2.0)))

    # Z-scores
    parts.append(compute_price_zscores(df, cfg.get("zscore_windows", [20, 60, 120])))
    for w in cfg.get("zscore_windows", [20, 60, 120]):
        parts.append(compute_vwap_zscore(df, w))

    # Volume z-score
    parts.append(compute_volume_zscore(df, cfg.get("volume_zscore_window", 20)))

    # Spread
    parts.append(compute_spread_features(df))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): equivalent to EMA with alpha=1/period."""
    return series.ewm(alpha=1 / period, adjust=False).mean()
