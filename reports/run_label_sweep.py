from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.labels.triple_barrier import compute_atr, build_labeled_dataset_ohlc

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 200)

MODEL_DATASET_PATH = Path("data/processed/model_dataset_M1.parquet")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports/label_sweep")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COMBOS = [
    {"tp_atr": 1.5, "sl_atr": 1.0, "horizon": 30},
    {"tp_atr": 2.0, "sl_atr": 1.0, "horizon": 30},
    {"tp_atr": 2.0, "sl_atr": 1.0, "horizon": 60},
    {"tp_atr": 2.5, "sl_atr": 1.0, "horizon": 60},
    {"tp_atr": 3.0, "sl_atr": 1.5, "horizon": 60},
]


def pct(x):
    return float(x) * 100


def make_name(tp, sl, h):
    return f"labeled_dataset_M1_tp{tp}_sl{sl}_h{h}.parquet"


def get_spread_col(df):
    candidates = [
        "XGER30_spread_points",
        "XGER30_spread",
        "spread_points",
        "spread",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    spread_candidates = [
        c for c in df.columns
        if "spread" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
    ]

    return spread_candidates[0] if spread_candidates else None


def expectancy_points(df, side, spread_col=None):
    label_col = f"label_{side}"
    tp_col = f"tp_distance_{side}"
    sl_col = f"sl_distance_{side}"

    gross = np.select(
        [
            df[label_col].to_numpy() == 1,
            df[label_col].to_numpy() == -1,
            df[label_col].to_numpy() == 0,
        ],
        [
            df[tp_col].to_numpy(),
            -df[sl_col].to_numpy(),
            0.0,
        ],
        default=0.0,
    ).astype(float)

    net = gross.copy()

    if spread_col is not None:
        active = df[label_col].to_numpy() != 0
        net[active] = net[active] - df.loc[active, spread_col].to_numpy()

    return float(np.mean(gross)), float(np.mean(net))


def summarize_combo(df, tp, sl, h, out_path):
    spread_col = get_spread_col(df)

    row = {
        "tp_atr": tp,
        "sl_atr": sl,
        "horizon": h,
        "path": str(out_path),
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "nan_total": int(df.isna().sum().sum()),
        "duplicate_index": int(df.index.duplicated().sum()),
    }

    for side in ["long", "short"]:
        label_col = f"label_{side}"
        barrier_col = f"barrier_side_{side}"
        tte_col = f"time_to_exit_{side}"
        tp_col = f"tp_distance_{side}"
        sl_col = f"sl_distance_{side}"

        row[f"{side}_tp_pct"] = pct((df[label_col] == 1).mean())
        row[f"{side}_sl_pct"] = pct((df[label_col] == -1).mean())
        row[f"{side}_neutral_pct"] = pct((df[label_col] == 0).mean())

        row[f"{side}_timeout_count"] = int((df[barrier_col] == "TIMEOUT").sum())
        row[f"{side}_ambiguous_count"] = int(df[barrier_col].astype(str).str.contains("AMBIGUOUS").sum())

        row[f"{side}_tte_mean"] = float(df[tte_col].mean())
        row[f"{side}_tte_median"] = float(df[tte_col].median())
        row[f"{side}_exit_1m_pct"] = pct((df[tte_col] <= 1).mean())
        row[f"{side}_exit_3m_pct"] = pct((df[tte_col] <= 3).mean())
        row[f"{side}_exit_5m_pct"] = pct((df[tte_col] <= 5).mean())
        row[f"{side}_exit_10m_pct"] = pct((df[tte_col] <= 10).mean())

        row[f"{side}_tp_distance_mean"] = float(df[tp_col].mean())
        row[f"{side}_sl_distance_mean"] = float(df[sl_col].mean())

        gross_exp, net_exp = expectancy_points(df, side, spread_col)
        row[f"{side}_gross_expectancy_points"] = gross_exp
        row[f"{side}_net_expectancy_points_minus_1spread"] = net_exp

    row["atr_mean"] = float(df["atr_14"].mean())
    row["atr_median"] = float(df["atr_14"].median())

    if spread_col is not None:
        row["spread_col"] = spread_col
        row["spread_mean"] = float(df[spread_col].mean())
        row["spread_median"] = float(df[spread_col].median())
        row["spread_over_atr_mean"] = float((df[spread_col] / df["atr_14"]).mean())
        row["spread_over_tp_mean"] = float((df[spread_col] / df["tp_distance_long"]).mean())
        row["spread_over_sl_mean"] = float((df[spread_col] / df["sl_distance_long"]).mean())
    else:
        row["spread_col"] = None
        row["spread_mean"] = np.nan
        row["spread_median"] = np.nan
        row["spread_over_atr_mean"] = np.nan
        row["spread_over_tp_mean"] = np.nan
        row["spread_over_sl_mean"] = np.nan

    return row


def main():
    print("=" * 140)
    print("LABEL SWEEP - DAX M1")
    print("=" * 140)

    base = pd.read_parquet(MODEL_DATASET_PATH)

    print("\nLoaded model dataset:")
    print("path:", MODEL_DATASET_PATH)
    print("shape:", base.shape)
    print("from:", base.index.min())
    print("to:", base.index.max())
    print("nan total:", int(base.isna().sum().sum()))
    print("duplicate index:", int(base.index.duplicated().sum()))

    atr = compute_atr(
        high=base["XGER30_high"],
        low=base["XGER30_low"],
        close=base["XGER30_close"],
        period=14,
    )

    rows = []

    for combo in COMBOS:
        tp = combo["tp_atr"]
        sl = combo["sl_atr"]
        h = combo["horizon"]

        print("\n" + "-" * 140)
        print(f"Building labels: TP={tp} ATR | SL={sl} ATR | H={h}")
        print("-" * 140)

        labeled = build_labeled_dataset_ohlc(
            features=base,
            close=base["XGER30_close"],
            high=base["XGER30_high"],
            low=base["XGER30_low"],
            atr=atr,
            tp_atr=tp,
            sl_atr=sl,
            horizon=h,
            ambiguous_policy="sl_first",
        )

        labeled["atr_14"] = atr

        # Remove ATR warmup and rows without complete future horizon.
        labeled = labeled[labeled["atr_14"].notna()].copy()
        labeled = labeled.iloc[:-h].copy()

        out_path = OUT_DIR / make_name(tp, sl, h)
        labeled.to_parquet(out_path)

        row = summarize_combo(labeled, tp, sl, h, out_path)
        rows.append(row)

        print("saved:", out_path)
        print("shape:", labeled.shape)
        print("from:", labeled.index.min())
        print("to:", labeled.index.max())
        print("long label counts:")
        print(labeled["label_long"].value_counts(dropna=False).sort_index())
        print("short label counts:")
        print(labeled["label_short"].value_counts(dropna=False).sort_index())
        print("time_to_exit long median/mean:", round(labeled["time_to_exit_long"].median(), 3), round(labeled["time_to_exit_long"].mean(), 3))
        print("time_to_exit short median/mean:", round(labeled["time_to_exit_short"].median(), 3), round(labeled["time_to_exit_short"].mean(), 3))

    summary = pd.DataFrame(rows)

    summary_path = REPORT_DIR / "label_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected_cols = [
        "tp_atr",
        "sl_atr",
        "horizon",
        "rows",
        "long_tp_pct",
        "long_sl_pct",
        "long_neutral_pct",
        "long_tte_mean",
        "long_tte_median",
        "long_exit_3m_pct",
        "long_exit_5m_pct",
        "long_net_expectancy_points_minus_1spread",
        "short_tp_pct",
        "short_sl_pct",
        "short_neutral_pct",
        "short_tte_mean",
        "short_tte_median",
        "short_exit_3m_pct",
        "short_exit_5m_pct",
        "short_net_expectancy_points_minus_1spread",
        "atr_mean",
        "spread_mean",
        "spread_over_tp_mean",
        "spread_over_sl_mean",
    ]

    print("\n" + "=" * 140)
    print("LABEL SWEEP SUMMARY")
    print("=" * 140)
    print(summary[selected_cols].round(4))

    print("\nSaved summary:", summary_path)

    print("\nCreated labeled datasets:")
    for p in summary["path"]:
        print(p)


if __name__ == "__main__":
    main()

