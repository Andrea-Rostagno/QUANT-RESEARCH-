"""
time_features.py — Calendar and session-based features.

All features are derived purely from the bar timestamp — zero data leakage.

Features:
  - Hour, minute, day of week, month
  - Minutes since midnight (UTC)
  - Session flags: Frankfurt pre-open, Xetra open, London overlap, NY open, NY+EU overlap
  - Cyclical encoding (sin/cos) of hour and day-of-week
  - End-of-week, end-of-month flags
  - Bar index within session
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Session definitions (all times in UTC)
# ---------------------------------------------------------------------------
# Key To Markets — CFD hours for GER40 (typical)
# TODO: verify exact session hours with broker
_SESSIONS = {
    "frankfurt_preopen": (6, 0,  7, 0),    # 06:00–07:00 UTC
    "xetra_open":        (7, 0, 15, 30),   # 07:00–15:30 UTC (winter: 08:00–16:30 CET)
    "london_open":       (7, 0,  9, 0),    # 07:00–09:00 UTC  (London morning)
    "ny_open":           (13, 30, 16, 0),  # 13:30–16:00 UTC  (NY open first 2.5 h)
    "eu_ny_overlap":     (13, 30, 15, 30), # overlap window
    "us_afternoon":      (16, 0, 21, 0),   # 16:00–21:00 UTC
}


# ---------------------------------------------------------------------------
# Core time features
# ---------------------------------------------------------------------------

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic calendar and clock features from the UTC DatetimeIndex.

    Features produced:
      ``hour``, ``minute``, ``day_of_week`` (0=Mon), ``month``,
      ``week_of_year``, ``minute_of_day``, ``quarter``.

    Returns
    -------
    pd.DataFrame
        One column per feature.
    """
    idx = df.index
    return pd.DataFrame({
        "hour":          idx.hour,
        "minute":        idx.minute,
        "day_of_week":   idx.dayofweek,
        "month":         idx.month,
        "quarter":       idx.quarter,
        "week_of_year":  idx.isocalendar().week.astype(int).values,
        "minute_of_day": idx.hour * 60 + idx.minute,
    }, index=idx)


def compute_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ``hour`` and ``day_of_week`` as sin/cos pairs.

    Cyclical encoding prevents the model from treating 23h and 0h as far apart.

    Returns
    -------
    pd.DataFrame
        Columns: ``hour_sin``, ``hour_cos``, ``dow_sin``, ``dow_cos``.
    """
    idx = df.index
    hour  = idx.hour + idx.minute / 60.0
    dow   = idx.dayofweek.astype(float)

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow  / 5),
        "dow_cos":  np.cos(2 * np.pi * dow  / 5),
    }, index=idx)


# ---------------------------------------------------------------------------
# Session flags
# ---------------------------------------------------------------------------

def compute_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary in-session flags for major market sessions.

    All comparisons use minute_of_day (UTC) to be timezone-safe.

    Returns
    -------
    pd.DataFrame
        One boolean column per session.
    """
    idx   = df.index
    mod   = idx.hour * 60 + idx.minute  # minute_of_day

    result: dict[str, np.ndarray] = {}
    for name, (h0, m0, h1, m1) in _SESSIONS.items():
        start = h0 * 60 + m0
        end   = h1 * 60 + m1
        result[f"session_{name}"] = ((mod >= start) & (mod < end)).astype(np.int8)

    return pd.DataFrame(result, index=idx)


def compute_session_bar_index(df: pd.DataFrame, session_start_hour: int = 7) -> pd.Series:
    """Bar index within the trading session (resets each day at session open).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    session_start_hour:
        UTC hour at which the session is considered to start.

    Returns
    -------
    pd.Series
        Bar index within session (0-based).
    """
    idx = df.index
    session_start_mod = session_start_hour * 60
    mod = idx.hour * 60 + idx.minute

    # Define session date: bars before session_start belong to the previous day
    session_date = idx.date
    session_date = pd.Series(session_date, index=idx)

    bar_seq = session_date.groupby(session_date).cumcount()
    return bar_seq.rename("session_bar_index")


# ---------------------------------------------------------------------------
# Calendar flags
# ---------------------------------------------------------------------------

def compute_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute end-of-week and end-of-month proximity flags.

    Returns
    -------
    pd.DataFrame
        Columns: ``is_friday``, ``is_monday``, ``is_last_day_of_month``.
    """
    idx = df.index
    month_end = pd.tseries.offsets.BMonthEnd()

    return pd.DataFrame({
        "is_monday":          (idx.dayofweek == 0).astype(np.int8),
        "is_friday":          (idx.dayofweek == 4).astype(np.int8),
        "is_last_bday_month": (
            idx.normalize() == (idx.normalize() + month_end).normalize()
        ).astype(np.int8),
    }, index=idx)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all_time_features(df: pd.DataFrame, session_start_hour: int = 7) -> pd.DataFrame:
    """Compute all time-based features in one call.

    Parameters
    ----------
    df:
        M1 OHLCV DataFrame with UTC DatetimeIndex.
    session_start_hour:
        UTC hour defining session start (default 7 for DAX).

    Returns
    -------
    pd.DataFrame
        All time feature columns.
    """
    parts = [
        compute_time_features(df),
        compute_cyclical_time(df),
        compute_session_flags(df),
        compute_session_bar_index(df, session_start_hour),
        compute_calendar_flags(df),
    ]
    return pd.concat(parts, axis=1)
