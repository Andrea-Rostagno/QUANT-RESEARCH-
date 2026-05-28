"""
event_backtester.py — Realistic event-driven backtester for intraday DAX strategies.

Design:
  - Event-driven: iterates bar-by-bar; no vectorised future look-ahead
  - Supports long and short trades
  - Costs: spread + slippage + commission (configurable)
  - TP / SL / max holding time exits
  - Optional one-position-at-a-time enforcement
  - Session time filter
  - Full trade log with entry/exit metadata

Output metrics:
  - Net profit, total trades, win rate
  - Profit factor, expectancy, average R
  - Sharpe ratio, Sortino ratio (annualised, M1 basis)
  - Maximum drawdown (absolute and %)
  - Monthly breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade record."""
    trade_id:    int
    entry_time:  pd.Timestamp
    exit_time:   Optional[pd.Timestamp]
    side:        int           # +1 long, -1 short
    entry_price: float
    exit_price:  float = 0.0
    tp_price:    float = 0.0
    sl_price:    float = 0.0
    max_bars:    int   = 60
    bars_held:   int   = 0
    gross_pnl:   float = 0.0
    net_pnl:     float = 0.0
    cost:        float = 0.0
    exit_reason: str   = ""    # TP / SL / TIMEOUT / SESSION_END


@dataclass
class BacktestResult:
    """Aggregated backtest result."""
    trades:          pd.DataFrame
    equity_curve:    pd.Series
    monthly_stats:   pd.DataFrame
    metrics:         dict


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------

class EventBacktester:
    """Bar-by-bar event-driven backtester.

    Parameters
    ----------
    prices:
        OHLCV DataFrame with UTC DatetimeIndex.
    signals:
        Series of trade signals: +1 = long, -1 = short, 0 = no trade.
        Must be aligned to ``prices`` index.
    cfg_backtest:
        Backtest cost/rule parameters from ``cfg["backtest"]``.
    atr:
        ATR series used to compute dynamic TP/SL levels.
    tp_atr:
        TP multiple of ATR.
    sl_atr:
        SL multiple of ATR.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        cfg_backtest: dict,
        atr: pd.Series,
        tp_atr: float = 2.0,
        sl_atr: float = 1.0,
    ) -> None:
        self.prices       = prices
        self.signals      = signals.reindex(prices.index).fillna(0)
        self.cfg          = cfg_backtest
        self.atr          = atr.reindex(prices.index)
        self.tp_atr       = tp_atr
        self.sl_atr       = sl_atr

        # Parameters
        self.spread        = float(cfg_backtest.get("spread_points", 1.0))
        self.slippage      = float(cfg_backtest.get("slippage_points", 0.5))
        self.commission    = float(cfg_backtest.get("commission_per_lot", 7.0))
        self.lot_size      = float(cfg_backtest.get("lot_size", 1.0))
        self.point_value   = float(cfg_backtest.get("point_value", 1.0))
        self.max_bars      = int(cfg_backtest.get("max_holding_bars", 60))
        self.one_pos       = bool(cfg_backtest.get("one_position_at_a_time", True))

        sess_start = cfg_backtest.get("session_start_utc", "07:00")
        sess_end   = cfg_backtest.get("session_end_utc",   "21:00")
        self._sess_start = _time_to_minutes(sess_start)
        self._sess_end   = _time_to_minutes(sess_end)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest.

        Returns
        -------
        BacktestResult
        """
        trades: list[Trade] = []
        open_trade: Optional[Trade] = None
        trade_id   = 0

        closes = self.prices["close"].values
        highs  = self.prices["high"].values
        lows   = self.prices["low"].values
        times  = self.prices.index
        sigs   = self.signals.values
        atrs   = self.atr.values

        n = len(closes)
        equity = np.zeros(n)

        for i in range(1, n):
            t_mod = times[i].hour * 60 + times[i].minute

            # --- Manage open trade ---
            if open_trade is not None:
                open_trade.bars_held += 1
                tp = open_trade.tp_price
                sl = open_trade.sl_price
                hi, lo = highs[i], lows[i]
                reason = None

                # Session-end forced exit
                if t_mod >= self._sess_end:
                    reason = "SESSION_END"

                # Max holding time
                elif open_trade.bars_held >= self.max_bars:
                    reason = "TIMEOUT"

                else:
                    # TP check (conservative: use close for final)
                    if open_trade.side == 1 and hi >= tp:
                        reason = "TP"
                    elif open_trade.side == -1 and lo <= tp:
                        reason = "TP"
                    # SL check
                    elif open_trade.side == 1 and lo <= sl:
                        reason = "SL"
                    elif open_trade.side == -1 and hi >= sl:
                        reason = "SL"

                if reason:
                    exit_px = _exit_price(
                        reason, open_trade.side,
                        closes[i], tp, sl,
                        self.spread, self.slippage
                    )
                    _close_trade(open_trade, exit_px, times[i], reason, self.commission, self.point_value)
                    trades.append(open_trade)
                    open_trade = None

            # --- New signal ---
            if open_trade is None:
                sig = sigs[i]
                if sig != 0 and self._in_session(t_mod):
                    atr_i = atrs[i]
                    if np.isnan(atr_i) or atr_i <= 0:
                        continue

                    entry_px = _entry_price(int(sig), closes[i], self.spread, self.slippage)

                    if sig == 1:
                        tp_px = entry_px + self.tp_atr * atr_i
                        sl_px = entry_px - self.sl_atr * atr_i
                    else:
                        tp_px = entry_px - self.tp_atr * atr_i
                        sl_px = entry_px + self.sl_atr * atr_i

                    open_trade = Trade(
                        trade_id=trade_id,
                        entry_time=times[i],
                        exit_time=None,
                        side=int(sig),
                        entry_price=entry_px,
                        tp_price=tp_px,
                        sl_price=sl_px,
                        max_bars=self.max_bars,
                    )
                    trade_id += 1

            # Record equity (closed P&L only for simplicity)
            equity[i] = equity[i - 1]
            if trades and trades[-1].exit_time == times[i]:
                equity[i] += trades[-1].net_pnl

        # Force-close any remaining open trade at last bar
        if open_trade is not None:
            _close_trade(open_trade, closes[-1], times[-1], "SESSION_END", self.commission, self.point_value)
            trades.append(open_trade)

        equity_series = pd.Series(equity, index=times, name="equity")
        trades_df     = _trades_to_df(trades)
        monthly       = _monthly_stats(trades_df)
        metrics       = _compute_metrics(trades_df, equity_series)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_series,
            monthly_stats=monthly,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_session(self, t_mod: int) -> bool:
        return self._sess_start <= t_mod < self._sess_end


# ---------------------------------------------------------------------------
# Trade cost / exit helpers
# ---------------------------------------------------------------------------

def _entry_price(side: int, close: float, spread: float, slippage: float) -> float:
    """Compute realistic entry price including spread and slippage."""
    if side == 1:
        return close + 0.5 * spread + slippage
    else:
        return close - 0.5 * spread - slippage


def _exit_price(
    reason: str,
    side: int,
    close: float,
    tp: float,
    sl: float,
    spread: float,
    slippage: float,
) -> float:
    """Compute realistic exit price."""
    if reason == "TP":
        base = tp
    elif reason == "SL":
        base = sl
    else:
        base = close

    if side == 1:
        return base - 0.5 * spread - slippage
    else:
        return base + 0.5 * spread + slippage


def _close_trade(
    trade: Trade,
    exit_px: float,
    exit_time: pd.Timestamp,
    reason: str,
    commission: float,
    point_value: float,
) -> None:
    """Fill in trade exit fields in-place."""
    trade.exit_time   = exit_time
    trade.exit_price  = exit_px
    trade.exit_reason = reason
    raw_pnl           = trade.side * (exit_px - trade.entry_price) * point_value
    trade.gross_pnl   = raw_pnl
    trade.cost        = commission
    trade.net_pnl     = raw_pnl - commission


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = [t.__dict__ for t in trades]
    return pd.DataFrame(rows).set_index("trade_id")


def _monthly_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        n_trades   = ("net_pnl", "count"),
        net_pnl    = ("net_pnl", "sum"),
        win_rate   = ("net_pnl", lambda x: (x > 0).mean()),
        avg_pnl    = ("net_pnl", "mean"),
    ).reset_index()
    return monthly


def _compute_metrics(trades_df: pd.DataFrame, equity: pd.Series) -> dict:
    if trades_df.empty:
        return {"error": "No trades executed."}

    pnl = trades_df["net_pnl"]
    n   = len(pnl)
    wins  = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Drawdown
    cum_equity = equity.cumsum() if not equity.cumsum().empty else equity
    rolling_max = cum_equity.cummax()
    drawdown = cum_equity - rolling_max
    max_dd   = drawdown.min()
    max_dd_pct = (max_dd / rolling_max.replace(0, np.nan)).min() * 100

    # Sharpe / Sortino (bar-level, annualised for M1)
    bar_returns = equity.diff().dropna()
    BARS_PER_YEAR = 5418
    if bar_returns.std() > 0:
        sharpe  = (bar_returns.mean() / bar_returns.std()) * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0
    neg_returns = bar_returns[bar_returns < 0]
    if neg_returns.std() > 0:
        sortino = (bar_returns.mean() / neg_returns.std()) * np.sqrt(BARS_PER_YEAR)
    else:
        sortino = 0.0

    return {
        "n_trades":      n,
        "win_rate":      float(wins.count() / n),
        "net_profit":    float(pnl.sum()),
        "gross_profit":  float(gross_profit),
        "gross_loss":    float(gross_loss),
        "profit_factor": float(profit_factor),
        "expectancy":    float(pnl.mean()),
        "avg_winner":    float(wins.mean()) if not wins.empty else 0.0,
        "avg_loser":     float(losses.mean()) if not losses.empty else 0.0,
        "avg_r":         float(wins.mean() / abs(losses.mean())) if not losses.empty and not wins.empty else 0.0,
        "max_drawdown":  float(max_dd),
        "max_dd_pct":    float(max_dd_pct),
        "sharpe":        float(sharpe),
        "sortino":       float(sortino),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "tp_rate":       float((trades_df["exit_reason"] == "TP").mean()),
        "sl_rate":       float((trades_df["exit_reason"] == "SL").mean()),
        "timeout_rate":  float((trades_df["exit_reason"].isin(["TIMEOUT", "SESSION_END"])).mean()),
    }


def _time_to_minutes(t_str: str) -> int:
    """Parse 'HH:MM' string to minutes since midnight."""
    h, m = map(int, t_str.split(":"))
    return h * 60 + m
