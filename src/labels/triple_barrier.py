"""
triple_barrier.py - Triple-barrier labeling for intraday OHLC data.

This module implements ATR-based triple-barrier labeling for long and short
events using future HIGH/LOW bars, not only future close prices.

For each timestamp t:
- entry price = close[t]
- ATR = ATR[t], computed only from past and current closed bars
- future path = bars from t+1 to t+horizon
- long TP/SL are checked using future high/low
- short TP/SL are checked using future high/low

If TP and SL are touched inside the same future candle, the default policy is
"sl_first", which is conservative because the intrabar order is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute ATR using Wilder-style exponential smoothing.

    Parameters
    ----------
    high, low, close:
        OHLC series aligned on the same DatetimeIndex.
    period:
        ATR smoothing period.

    Returns
    -------
    pd.Series
        ATR series aligned to the input index.
    """
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.rename(f"atr_{period}")


def apply_triple_barrier_ohlc(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
    horizon: int = 30,
    side: int = 1,
    ambiguous_policy: str = "sl_first",
) -> pd.DataFrame:
    """Apply ATR triple-barrier labels using future high/low path.

    Parameters
    ----------
    close:
        Close price series. Entry is close[t].
    high:
        High price series.
    low:
        Low price series.
    atr:
        ATR known at time t.
    tp_atr:
        Take-profit distance as ATR multiple.
    sl_atr:
        Stop-loss distance as ATR multiple.
    horizon:
        Max holding time in bars.
    side:
        +1 for long labels, -1 for short labels.
    ambiguous_policy:
        How to handle candles where both TP and SL are touched.
        Supported:
        - "sl_first": conservative, assume SL first
        - "tp_first": optimistic, assume TP first
        - "neutral": label as 0 with barrier_side="AMBIGUOUS"

    Returns
    -------
    pd.DataFrame
        Columns:
        label, barrier_side, time_to_exit, mfe, mae, tp_distance, sl_distance.
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 for long or -1 for short.")

    if ambiguous_policy not in {"sl_first", "tp_first", "neutral"}:
        raise ValueError("ambiguous_policy must be 'sl_first', 'tp_first', or 'neutral'.")

    close = close.astype(float)
    high = high.reindex(close.index).astype(float)
    low = low.reindex(close.index).astype(float)
    atr = atr.reindex(close.index).astype(float)

    n = len(close)

    labels = np.zeros(n, dtype=np.int8)
    barrier_hit = np.full(n, "TIMEOUT", dtype=object)
    time_exit = np.full(n, horizon, dtype=np.int32)
    mfe_arr = np.full(n, np.nan, dtype=np.float64)
    mae_arr = np.full(n, np.nan, dtype=np.float64)
    tp_dist_arr = np.full(n, np.nan, dtype=np.float64)
    sl_dist_arr = np.full(n, np.nan, dtype=np.float64)

    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    a = atr.to_numpy()

    for i in range(n):
        p0 = c[i]
        atr_i = a[i]

        if not np.isfinite(p0) or not np.isfinite(atr_i) or atr_i <= 0:
            labels[i] = 0
            barrier_hit[i] = "INVALID_ATR"
            continue

        end_idx = min(i + horizon + 1, n)

        future_high = h[i + 1 : end_idx]
        future_low = l[i + 1 : end_idx]

        if len(future_high) == 0:
            labels[i] = 0
            barrier_hit[i] = "NO_FUTURE"
            time_exit[i] = 0
            continue

        tp_distance = tp_atr * atr_i
        sl_distance = sl_atr * atr_i

        tp_dist_arr[i] = tp_distance
        sl_dist_arr[i] = sl_distance

        if side == 1:
            tp_level = p0 + tp_distance
            sl_level = p0 - sl_distance

            # MFE/MAE in points for long.
            mfe_arr[i] = np.nanmax(future_high - p0)
            mae_arr[i] = np.nanmin(future_low - p0)

            for j, (fh, fl) in enumerate(zip(future_high, future_low), start=1):
                hit_tp = fh >= tp_level
                hit_sl = fl <= sl_level

                if hit_tp and hit_sl:
                    time_exit[i] = j
                    if ambiguous_policy == "sl_first":
                        labels[i] = -1
                        barrier_hit[i] = "AMBIGUOUS_SL_FIRST"
                    elif ambiguous_policy == "tp_first":
                        labels[i] = 1
                        barrier_hit[i] = "AMBIGUOUS_TP_FIRST"
                    else:
                        labels[i] = 0
                        barrier_hit[i] = "AMBIGUOUS"
                    break

                if hit_tp:
                    labels[i] = 1
                    barrier_hit[i] = "TP"
                    time_exit[i] = j
                    break

                if hit_sl:
                    labels[i] = -1
                    barrier_hit[i] = "SL"
                    time_exit[i] = j
                    break

        else:
            tp_level = p0 - tp_distance
            sl_level = p0 + sl_distance

            # MFE/MAE in points for short.
            # Favourable = price moves down.
            mfe_arr[i] = np.nanmax(p0 - future_low)
            mae_arr[i] = np.nanmin(p0 - future_high)

            for j, (fh, fl) in enumerate(zip(future_high, future_low), start=1):
                hit_tp = fl <= tp_level
                hit_sl = fh >= sl_level

                if hit_tp and hit_sl:
                    time_exit[i] = j
                    if ambiguous_policy == "sl_first":
                        labels[i] = -1
                        barrier_hit[i] = "AMBIGUOUS_SL_FIRST"
                    elif ambiguous_policy == "tp_first":
                        labels[i] = 1
                        barrier_hit[i] = "AMBIGUOUS_TP_FIRST"
                    else:
                        labels[i] = 0
                        barrier_hit[i] = "AMBIGUOUS"
                    break

                if hit_tp:
                    labels[i] = 1
                    barrier_hit[i] = "TP"
                    time_exit[i] = j
                    break

                if hit_sl:
                    labels[i] = -1
                    barrier_hit[i] = "SL"
                    time_exit[i] = j
                    break

    return pd.DataFrame(
        {
            "label": labels,
            "barrier_side": barrier_hit,
            "time_to_exit": time_exit,
            "mfe": mfe_arr,
            "mae": mae_arr,
            "tp_distance": tp_dist_arr,
            "sl_distance": sl_dist_arr,
        },
        index=close.index,
    )


def apply_triple_barrier(
    prices: pd.Series,
    atr: pd.Series,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
    horizon: int = 30,
    side: int = 1,
) -> pd.DataFrame:
    """Backward-compatible close-only triple barrier.

    This wrapper keeps compatibility with older notebooks, but for serious
    labeling prefer apply_triple_barrier_ohlc().
    """
    logger.warning(
        "apply_triple_barrier() uses close-only path. "
        "Prefer apply_triple_barrier_ohlc() with high/low data."
    )
    return apply_triple_barrier_ohlc(
        close=prices,
        high=prices,
        low=prices,
        atr=atr,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        horizon=horizon,
        side=side,
    )


def label_all_combinations(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    tp_multiples: list[float] = (1.5, 2.0, 3.0),
    sl_multiples: list[float] = (1.0, 1.5, 2.0),
    horizons: list[int] = (5, 15, 30, 60),
    sides: list[int] = (1, -1),
    ambiguous_policy: str = "sl_first",
) -> dict[str, pd.DataFrame]:
    """Run OHLC triple-barrier labeling for multiple parameter combinations."""
    results: dict[str, pd.DataFrame] = {}

    combos = [
        (tp, sl, h, s)
        for tp in tp_multiples
        for sl in sl_multiples
        for h in horizons
        for s in sides
        if tp > sl
    ]

    logger.info(f"Labeling {len(combos)} TP/SL/horizon/side combinations.")

    for tp, sl, h, s in tqdm(combos, desc="Labeling"):
        side_str = "long" if s == 1 else "short"
        key = f"{side_str}_tp{tp}_sl{sl}_h{h}"

        results[key] = apply_triple_barrier_ohlc(
            close=close,
            high=high,
            low=low,
            atr=atr,
            tp_atr=tp,
            sl_atr=sl,
            horizon=h,
            side=s,
            ambiguous_policy=ambiguous_policy,
        )

    return results


def build_labeled_dataset_ohlc(
    features: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
    horizon: int = 30,
    ambiguous_policy: str = "sl_first",
) -> pd.DataFrame:
    """Combine features with long and short OHLC triple-barrier labels."""
    idx = features.index

    close = close.reindex(idx)
    high = high.reindex(idx)
    low = low.reindex(idx)
    atr = atr.reindex(idx)

    long_lbl = apply_triple_barrier_ohlc(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        horizon=horizon,
        side=1,
        ambiguous_policy=ambiguous_policy,
    ).add_suffix("_long")

    short_lbl = apply_triple_barrier_ohlc(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        horizon=horizon,
        side=-1,
        ambiguous_policy=ambiguous_policy,
    ).add_suffix("_short")

    out = pd.concat([features, long_lbl, short_lbl], axis=1)
    out[f"atr_{14}"] = atr
    return out


def build_labeled_dataset(
    features: pd.DataFrame,
    prices: pd.Series,
    atr: pd.Series,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
    horizon: int = 30,
) -> pd.DataFrame:
    """Backward-compatible close-only labeled dataset builder."""
    logger.warning(
        "build_labeled_dataset() uses close-only labels. "
        "Prefer build_labeled_dataset_ohlc()."
    )

    long_lbl = apply_triple_barrier(
        prices=prices,
        atr=atr,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        horizon=horizon,
        side=1,
    ).add_suffix("_long")

    short_lbl = apply_triple_barrier(
        prices=prices,
        atr=atr,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        horizon=horizon,
        side=-1,
    ).add_suffix("_short")

    return pd.concat([features, long_lbl, short_lbl], axis=1)


def label_statistics(labels: pd.DataFrame, label_col: str = "label") -> dict:
    """Compute label distribution statistics."""
    col = labels[label_col]
    n = len(col.dropna())

    n_tp = int((col == 1).sum())
    n_sl = int((col == -1).sum())
    n_neutral = int((col == 0).sum())

    return {
        "n_total": int(n),
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_neutral": n_neutral,
        "pct_tp": round(n_tp / n * 100, 2) if n else 0,
        "pct_sl": round(n_sl / n * 100, 2) if n else 0,
        "pct_neutral": round(n_neutral / n * 100, 2) if n else 0,
    }
