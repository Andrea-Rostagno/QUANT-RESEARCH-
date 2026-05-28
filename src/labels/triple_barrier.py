"""
triple_barrier.py — Triple-barrier labeling for intraday price series.

Based on the methodology described in "Advances in Financial Machine Learning"
(de Prado, 2018), adapted for M1 CFD data with ATR-based barriers.

Labels:
  +1  = Long: TP hit before SL and before horizon
  -1  = Short: SL hit before TP and before horizon (i.e., price fell enough)
   0  = Neutral: neither barrier hit within horizon (timeout / no-trade)

For each event bar the following are also recorded:
  - ``barrier_side``  : which barrier was hit first (TP / SL / TIMEOUT)
  - ``time_to_exit``  : bars until barrier hit (or horizon)
  - ``mfe``           : max favourable excursion (best price move in our direction)
  - ``mae``           : max adverse excursion (worst price move against us)

IMPORTANT: Labels are computed using FUTURE bars starting at t+1.
           The labeling function never reads data at or before the event timestamp.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    prices: pd.Series,
    atr: pd.Series,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
    horizon: int = 30,
    side: int = 1,
) -> pd.DataFrame:
    """Apply triple-barrier labeling for a single TP/SL/horizon combination.

    For each bar, we project:
      - Upper barrier : price_t + tp_atr * ATR_t  (take profit for long)
      - Lower barrier : price_t - sl_atr * ATR_t  (stop loss for long)
      - Horizon       : t + horizon bars

    For short side, barriers are mirrored:
      - Upper barrier : price_t + sl_atr * ATR_t  (stop loss for short)
      - Lower barrier : price_t - tp_atr * ATR_t  (take profit for short)

    Parameters
    ----------
    prices:
        Close price series with DatetimeIndex.
    atr:
        ATR series aligned to ``prices``.
    tp_atr:
        Take-profit multiple of ATR.
    sl_atr:
        Stop-loss multiple of ATR.
    horizon:
        Maximum bars to hold (timeout).
    side:
        +1 for long labels, -1 for short labels.

    Returns
    -------
    pd.DataFrame
        Index = same as ``prices``.
        Columns: ``label``, ``barrier_side``, ``time_to_exit``, ``mfe``, ``mae``.
    """
    n = len(prices)
    labels      = np.zeros(n, dtype=np.int8)
    barrier_hit = np.full(n, "TIMEOUT", dtype=object)
    time_exit   = np.full(n, horizon, dtype=np.int32)
    mfe_arr     = np.zeros(n, dtype=np.float32)
    mae_arr     = np.zeros(n, dtype=np.float32)

    price_vals = prices.values.astype(np.float64)
    atr_vals   = atr.reindex(prices.index).values.astype(np.float64)

    for i in range(n):
        p0  = price_vals[i]
        atr_i = atr_vals[i]

        if np.isnan(atr_i) or atr_i <= 0:
            labels[i] = 0
            continue

        # Future window (strictly after bar i)
        end_idx = min(i + horizon + 1, n)
        future  = price_vals[i + 1 : end_idx]

        if len(future) == 0:
            labels[i] = 0
            continue

        if side == 1:
            tp_level = p0 + tp_atr * atr_i
            sl_level = p0 - sl_atr * atr_i
        else:
            tp_level = p0 - tp_atr * atr_i
            sl_level = p0 + sl_atr * atr_i

        # Check each future bar
        label    = 0
        hit_side = "TIMEOUT"
        t_exit   = len(future)

        for j, fp in enumerate(future):
            if side == 1:
                if fp >= tp_level:
                    label    = 1
                    hit_side = "TP"
                    t_exit   = j + 1
                    break
                elif fp <= sl_level:
                    label    = -1
                    hit_side = "SL"
                    t_exit   = j + 1
                    break
            else:
                if fp <= tp_level:
                    label    = 1      # short TP = price went down
                    hit_side = "TP"
                    t_exit   = j + 1
                    break
                elif fp >= sl_level:
                    label    = -1     # short SL = price went up
                    hit_side = "SL"
                    t_exit   = j + 1
                    break

        # MFE / MAE
        if side == 1:
            fav_exc  = future - p0
        else:
            fav_exc  = p0 - future

        mfe_arr[i] = float(np.max(fav_exc)) if len(fav_exc) > 0 else 0.0
        mae_arr[i] = float(np.min(fav_exc)) if len(fav_exc) > 0 else 0.0

        labels[i]      = label
        barrier_hit[i] = hit_side
        time_exit[i]   = t_exit

    return pd.DataFrame({
        "label":        labels,
        "barrier_side": barrier_hit,
        "time_to_exit": time_exit,
        "mfe":          mfe_arr,
        "mae":          mae_arr,
    }, index=prices.index)


# ---------------------------------------------------------------------------
# Multi-parameter sweep
# ---------------------------------------------------------------------------

def label_all_combinations(
    prices: pd.Series,
    atr: pd.Series,
    tp_multiples: list[float] = (1.5, 2.0, 3.0),
    sl_multiples: list[float] = (1.0, 1.5),
    horizons: list[int] = (5, 15, 30, 60),
    sides: list[int] = (1, -1),
) -> dict[str, pd.DataFrame]:
    """Run triple-barrier labeling for all (tp, sl, horizon, side) combinations.

    Parameters
    ----------
    prices:
        Close price series.
    atr:
        ATR series aligned to ``prices``.
    tp_multiples:
        List of TP ATR multiples.
    sl_multiples:
        List of SL ATR multiples.
    horizons:
        List of horizon values (bars).
    sides:
        [+1] for long only, [-1] for short only, [+1, -1] for both.

    Returns
    -------
    dict
        Keys like ``"long_tp2.0_sl1.0_h30"``, values = DataFrame from
        :func:`apply_triple_barrier`.
    """
    results: dict[str, pd.DataFrame] = {}
    combos = [
        (tp, sl, h, s)
        for tp in tp_multiples
        for sl in sl_multiples
        for h  in horizons
        for s  in sides
        if tp > sl  # ensure TP > SL multiple (asymmetric)
    ]

    logger.info(f"Labeling {len(combos)} TP/SL/horizon/side combinations …")

    for tp, sl, h, s in tqdm(combos, desc="Labeling"):
        side_str = "long" if s == 1 else "short"
        key = f"{side_str}_tp{tp}_sl{sl}_h{h}"
        results[key] = apply_triple_barrier(prices, atr, tp, sl, h, s)

    return results


# ---------------------------------------------------------------------------
# Convenience: binary long / short frame
# ---------------------------------------------------------------------------

def build_labeled_dataset(
    features: pd.DataFrame,
    prices: pd.Series,
    atr: pd.Series,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
    horizon: int = 30,
) -> pd.DataFrame:
    """Combine features with triple-barrier labels (long and short) into one DataFrame.

    Rows where the label is 0 (timeout) are kept but flagged.

    Parameters
    ----------
    features:
        Feature DataFrame (no future data).
    prices:
        Close price series aligned to ``features``.
    atr:
        ATR series.
    tp_atr:
        TP multiple.
    sl_atr:
        SL multiple.
    horizon:
        Horizon bars.

    Returns
    -------
    pd.DataFrame
        ``features`` columns plus ``label_long``, ``label_short``,
        ``barrier_side_long``, ``barrier_side_short``,
        ``time_to_exit_long``, ``time_to_exit_short``,
        ``mfe_long``, ``mae_long``, ``mfe_short``, ``mae_short``.
    """
    long_lbl  = apply_triple_barrier(prices, atr, tp_atr, sl_atr, horizon,  1)
    short_lbl = apply_triple_barrier(prices, atr, tp_atr, sl_atr, horizon, -1)

    long_lbl  = long_lbl.add_suffix("_long")
    short_lbl = short_lbl.add_suffix("_short")

    return pd.concat([features, long_lbl, short_lbl], axis=1)


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def label_statistics(labels: pd.DataFrame, label_col: str = "label") -> dict:
    """Compute label distribution statistics.

    Parameters
    ----------
    labels:
        DataFrame containing a label column.
    label_col:
        Name of the label column.

    Returns
    -------
    dict
        Keys: ``n_total``, ``n_long``, ``n_short``, ``n_neutral``,
        ``pct_long``, ``pct_short``, ``pct_neutral``.
    """
    col = labels[label_col]
    n   = len(col.dropna())
    n_long    = (col ==  1).sum()
    n_short   = (col == -1).sum()
    n_neutral = (col ==  0).sum()
    return {
        "n_total":    n,
        "n_long":     int(n_long),
        "n_short":    int(n_short),
        "n_neutral":  int(n_neutral),
        "pct_long":   round(n_long    / n * 100, 2) if n else 0,
        "pct_short":  round(n_short   / n * 100, 2) if n else 0,
        "pct_neutral": round(n_neutral / n * 100, 2) if n else 0,
    }
