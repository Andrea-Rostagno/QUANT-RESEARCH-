"""
mt5_loader.py — MetaTrader 5 data ingestion module.

Responsibilities:
  - Connect to and disconnect from MT5
  - Discover the exact DAX symbol available on the broker
  - Download OHLCV bars (M1 and auxiliary timeframes)
  - Handle timezone conversion (MT5 server time → UTC)
  - Save raw data as Parquet files
  - Detect and report duplicate timestamps and time gaps

Usage (from a notebook)::

    import sys; sys.path.insert(0, "..")
    from src.data.mt5_loader import MT5Loader
    from src.utils.config import load_config

    cfg = load_config()
    loader = MT5Loader(cfg)
    loader.connect()
    df = loader.download_symbol(cfg["mt5"]["symbol"], "M1")
    loader.save_raw(df, cfg["mt5"]["symbol"], "M1")
    loader.disconnect()
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed — MT5Loader will not function.")

# ---------------------------------------------------------------------------
# Timeframe map: string → MT5 constant
# ---------------------------------------------------------------------------
_TIMEFRAME_MAP: dict[str, int] = {}
if _MT5_AVAILABLE:
    _TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,
        "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,
        "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

# DAX symbol search patterns
_DAX_PATTERNS = ["DAX", "GER", "DE40", "GER40", "XGER"]

# Minimum points gap to flag as a data gap (M1 = 60 s)
_M1_SECONDS = 60


class MT5Loader:
    """Download, validate, and persist OHLCV data from MetaTrader 5.

    Parameters
    ----------
    cfg:
        Full project configuration as returned by :func:`src.utils.config.load_config`.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._paths = self._resolve_paths()
        self._server_tz = pytz.timezone(cfg["mt5"].get("server_timezone", "Etc/UTC"))

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the MT5 connection.

        Returns
        -------
        bool
            True if connection succeeded.

        Raises
        ------
        RuntimeError
            If MT5 package is missing or initialisation fails.
        """
        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package is not installed. Run: pip install MetaTrader5")

        if not mt5.initialize():
            error = mt5.last_error()
            raise RuntimeError(f"MT5 initialization failed: {error}")

        info = mt5.terminal_info()
        logger.info(
            f"MT5 connected — build={info.build}, broker={info.company}, "
            f"connected={info.connected}"
        )
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if _MT5_AVAILABLE:
            mt5.shutdown()
            logger.info("MT5 disconnected.")

    # ------------------------------------------------------------------
    # Symbol discovery
    # ------------------------------------------------------------------

    def find_dax_symbol(self) -> list[str]:
        """Search all broker symbols for DAX candidates.

        Searches for substrings: DAX, GER, DE40, GER40, XGER (case-insensitive).

        Returns
        -------
        list[str]
            Matching symbol names sorted alphabetically.
        """
        self._require_mt5()
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            logger.warning("mt5.symbols_get() returned None")
            return []

        pattern = re.compile("|".join(_DAX_PATTERNS), re.IGNORECASE)
        candidates = sorted(s.name for s in all_symbols if pattern.search(s.name))

        if candidates:
            logger.info(f"Found {len(candidates)} DAX candidate(s): {candidates}")
        else:
            logger.warning("No DAX symbol candidates found on this broker.")
        return candidates

    def list_all_symbols(self) -> pd.DataFrame:
        """Return a DataFrame with all available broker symbols.

        Columns: name, description, digits, spread, volume_min, currency_base,
        currency_profit.
        """
        self._require_mt5()
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return pd.DataFrame()

        rows = [
            {
                "name": s.name,
                "description": s.description,
                "digits": s.digits,
                "spread": s.spread,
                "volume_min": s.volume_min,
                "currency_base": s.currency_base,
                "currency_profit": s.currency_profit,
            }
            for s in all_symbols
        ]
        return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------

    def download_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download OHLCV bars for a single symbol and timeframe.

        Parameters
        ----------
        symbol:
            MT5 symbol name (e.g. "GER40").
        timeframe:
            String timeframe code (e.g. "M1", "M5", "H1").
        start_date:
            ISO date string "YYYY-MM-DD". Falls back to config value.
        end_date:
            ISO date string "YYYY-MM-DD". Falls back to config value (None = today).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with UTC DatetimeIndex. Columns:
            open, high, low, close, tick_volume, spread, real_volume.
        """
        self._require_mt5()

        tf_const = self._resolve_timeframe(timeframe)

        start = self._parse_date(start_date or self.cfg["mt5"]["start_date"])
        end = self._parse_date(end_date or self.cfg["mt5"].get("end_date"))  # None → today

        logger.info(f"Downloading {symbol} {timeframe} from {start.date()} to {end.date()} …")

        rates = mt5.copy_rates_range(symbol, tf_const, start, end)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise ValueError(
                f"No data returned for {symbol} {timeframe}. MT5 error: {error}\n"
                "Check: symbol name, date range, market hours, MT5 connection."
            )

        df = pd.DataFrame(rates)
        df = self._convert_timestamps(df)
        df = self._clean_ohlcv(df)

        self._quality_check(df, symbol, timeframe)
        return df

    def download_all_symbols(self) -> dict[str, pd.DataFrame]:
        """Download M1 data for the primary DAX symbol and all enabled aux symbols.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping symbol → DataFrame.
        """
        from src.utils.config import get_symbol, get_aux_symbols

        primary = get_symbol(self.cfg)
        symbols = [primary] + get_aux_symbols(self.cfg, enabled_only=True)

        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.download_symbol(sym, "M1")
                results[sym] = df
                logger.info(f"  {sym}: {len(df)} bars")
            except Exception as exc:
                logger.warning(f"  {sym}: FAILED — {exc}")

        return results

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def resample_ohlcv(df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample M1 OHLCV bars to a higher timeframe.

        Resampling is done purely from M1 data — no MT5 call needed.
        This avoids introducing a separate download for each timeframe.

        Parameters
        ----------
        df_m1:
            M1 OHLCV DataFrame with UTC DatetimeIndex.
        timeframe:
            Target timeframe string: "M3", "M5", "M15", "M30", "H1", etc.

        Returns
        -------
        pd.DataFrame
            Resampled OHLCV (rows with NaN dropped).
        """
        freq_map = {
            "M2": "2min", "M3": "3min", "M4": "4min", "M5": "5min",
            "M6": "6min", "M10": "10min", "M12": "12min", "M15": "15min",
            "M20": "20min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D",
        }
        freq = freq_map.get(timeframe.upper())
        if freq is None:
            raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }
        if "real_volume" in df_m1.columns:
            agg["real_volume"] = "sum"
        if "spread" in df_m1.columns:
            agg["spread"] = "mean"

        resampled = df_m1.resample(freq).agg(agg).dropna(subset=["open"])
        return resampled

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_raw(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        """Save a raw OHLCV DataFrame to Parquet.

        File path: ``data/raw/{symbol}_{timeframe}.parquet``

        Parameters
        ----------
        df:
            OHLCV DataFrame with UTC DatetimeIndex.
        symbol:
            Symbol name (used in filename).
        timeframe:
            Timeframe code (used in filename).

        Returns
        -------
        Path
            Path to the saved Parquet file.
        """
        raw_dir = self._paths["raw"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        file_path = raw_dir / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(file_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {len(df)} rows → {file_path}")
        return file_path

    def load_raw(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load a previously saved raw Parquet file.

        Parameters
        ----------
        symbol:
            Symbol name.
        timeframe:
            Timeframe code.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with UTC DatetimeIndex.
        """
        file_path = self._paths["raw"] / f"{symbol}_{timeframe}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {file_path}\n"
                "Run notebook 01_mt5_data_ingestion first."
            )
        df = pd.read_parquet(file_path, engine="pyarrow")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        return df

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _quality_check(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Log warnings for duplicate timestamps and unexpected time gaps."""
        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            logger.warning(f"{symbol} {timeframe}: {n_dupes} duplicate timestamps detected.")

        if timeframe == "M1":
            expected_delta = pd.Timedelta(seconds=_M1_SECONDS)
            deltas = df.index.to_series().diff().dropna()
            gaps = deltas[deltas > expected_delta * 2]
            if not gaps.empty:
                logger.warning(
                    f"{symbol} {timeframe}: {len(gaps)} gaps found (>2 min).\n"
                    f"  Largest gap: {gaps.max()} starting at {gaps.idxmax()}"
                )
            else:
                logger.info(f"{symbol} {timeframe}: no significant gaps.")

        logger.info(
            f"{symbol} {timeframe}: {len(df)} bars | "
            f"{df.index[0]} → {df.index[-1]}"
        )

    @staticmethod
    def check_gaps(df: pd.DataFrame, expected_freq: str = "1min") -> pd.DataFrame:
        """Return a DataFrame of time gaps larger than the expected frequency.

        Parameters
        ----------
        df:
            OHLCV DataFrame with DatetimeIndex.
        expected_freq:
            Expected minimum step as a pandas frequency string.

        Returns
        -------
        pd.DataFrame
            Rows with columns: ``gap_start``, ``gap_end``, ``gap_duration``.
        """
        threshold = pd.tseries.frequencies.to_offset(expected_freq).nanos * 2e9 / 1e9
        deltas = df.index.to_series().diff()
        mask = deltas > pd.Timedelta(seconds=threshold)
        gaps = pd.DataFrame({
            "gap_start": df.index[mask] - deltas[mask],
            "gap_end": df.index[mask],
            "gap_duration": deltas[mask],
        })
        return gaps.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert MT5 Unix timestamps to UTC DatetimeIndex."""
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        # MT5 returns server time; if server_timezone != UTC, convert
        if self._server_tz != pytz.UTC:
            df["time"] = (
                df["time"]
                .dt.tz_localize(None)
                .dt.tz_localize(self._server_tz)
                .dt.tz_convert("UTC")
            )
        df = df.set_index("time").sort_index()
        return df

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns, drop invalid rows, remove duplicates."""
        rename_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "tick_volume",
            "spread": "spread",
            "real_volume": "real_volume",
        }
        df = df.rename(columns=rename_map)
        cols = [c for c in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
                if c in df.columns]
        df = df[cols]

        # Remove zero-price rows (market closed / bad tick)
        df = df[df["close"] > 0].copy()
        # Remove duplicates (keep first)
        df = df[~df.index.duplicated(keep="first")]
        return df

    def _resolve_timeframe(self, timeframe: str) -> int:
        """Map string timeframe to MT5 constant."""
        tf = _TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(
                f"Unknown timeframe: '{timeframe}'. "
                f"Valid options: {list(_TIMEFRAME_MAP.keys())}"
            )
        return tf

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> datetime:
        """Parse an ISO date string or return today as UTC datetime."""
        if date_str is None:
            return datetime.now(timezone.utc)
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)

    def _resolve_paths(self) -> dict[str, Path]:
        """Resolve project path dict from config."""
        from src.utils.config import get_paths
        return get_paths(self.cfg)

    @staticmethod
    def _require_mt5() -> None:
        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed.")
