"""
config.py — YAML configuration loader for the DAX Quant Research Lab.

Provides a typed interface to configs/dax_m1.yaml so that all modules
and notebooks share the same parameter values without hardcoding anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "dax_m1.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration as a nested dictionary.

    Parameters
    ----------
    path:
        Explicit path to the YAML file.  If None, falls back to the
        environment variable ``DAX_CONFIG`` and then to the default
        location ``configs/dax_m1.yaml`` relative to the project root.

    Returns
    -------
    dict
        Full configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not point to an existing file.
    """
    if path is None:
        env_path = os.environ.get("DAX_CONFIG")
        resolved = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    else:
        resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}\n"
            "Set the DAX_CONFIG environment variable or pass the path explicitly."
        )

    with open(resolved, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


def get_symbol(cfg: dict[str, Any]) -> str:
    """Return the primary DAX symbol from config."""
    return cfg["mt5"]["symbol"]


def get_aux_symbols(cfg: dict[str, Any], enabled_only: bool = True) -> list[str]:
    """Return list of auxiliary symbol names.

    Parameters
    ----------
    cfg:
        Config dict as returned by :func:`load_config`.
    enabled_only:
        If True, return only symbols where ``enabled: true`` in the YAML.
    """
    entries = cfg["mt5"].get("aux_symbols", [])
    if enabled_only:
        return [e["symbol"] for e in entries if e.get("enabled", True)]
    return [e["symbol"] for e in entries]


def get_paths(cfg: dict[str, Any], project_root: str | Path | None = None) -> dict[str, Path]:
    """Return absolute Path objects for every data directory.

    Parameters
    ----------
    cfg:
        Config dict.
    project_root:
        Root of the project.  Defaults to two levels above this file.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    return {key: root / rel for key, rel in cfg["paths"].items()}


def get_backtest_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return backtest cost parameters."""
    return cfg["backtest"]


def get_feature_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return feature engineering parameters."""
    return cfg["features"]


def get_labeling_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return triple-barrier labeling parameters."""
    return cfg["labeling"]


def get_validation_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return walk-forward validation parameters."""
    return cfg["validation"]
