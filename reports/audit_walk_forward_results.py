from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 300)

RESULTS_PATH = Path("reports/model_validation/walk_forward_results.csv")
SUMMARY_PATH = Path("reports/model_validation/walk_forward_summary.csv")

results = pd.read_csv(RESULTS_PATH)
summary = pd.read_csv(SUMMARY_PATH)

print("=" * 140)
print("WALK-FORWARD AUDIT")
print("=" * 140)

print("\nLoaded:")
print("results:", RESULTS_PATH, results.shape)
print("summary:", SUMMARY_PATH, summary.shape)

# ---------------------------------------------------------------------
# 1. Live-feasible threshold results
# ---------------------------------------------------------------------

print("\n" + "=" * 140)
print("[1] LIVE-FEASIBLE RESULTS ONLY: THRESHOLD SELECTION")
print("=" * 140)

thr = results[results["selection_type"] == "threshold"].copy()

live_rank = (
    thr.groupby(["side", "model", "threshold"], dropna=False)
    .agg(
        folds=("test_date", "nunique"),
        avg_roc_auc=("roc_auc", "mean"),
        avg_pr_auc=("pr_auc", "mean"),
        avg_test_pos_rate=("test_positive_rate", "mean"),
        total_trades_no_overlap=("n_trades_no_overlap", "sum"),
        avg_trades_no_overlap=("n_trades_no_overlap", "mean"),
        positive_folds_no_overlap=("net_total_points_no_overlap", lambda x: int((x > 0).sum())),
        negative_folds_no_overlap=("net_total_points_no_overlap", lambda x: int((x < 0).sum())),
        total_net_no_overlap=("net_total_points_no_overlap", "sum"),
        worst_fold_net_no_overlap=("net_total_points_no_overlap", "min"),
        best_fold_net_no_overlap=("net_total_points_no_overlap", "max"),
        avg_tp_no_overlap=("tp_rate_pct_no_overlap", "mean"),
    )
    .reset_index()
)

live_rank["net_per_trade_no_overlap"] = (
    live_rank["total_net_no_overlap"] / live_rank["total_trades_no_overlap"].replace(0, np.nan)
)

live_rank = live_rank.sort_values(
    ["total_net_no_overlap", "positive_folds_no_overlap", "avg_roc_auc"],
    ascending=[False, False, False],
)

print("\nBest threshold combinations ranked by total_net_no_overlap:")
print(live_rank.round(4).head(30))

# ---------------------------------------------------------------------
# 2. Diagnostic top_pct results
# ---------------------------------------------------------------------

print("\n" + "=" * 140)
print("[2] DIAGNOSTIC RESULTS ONLY: TOP_PCT SELECTION")
print("=" * 140)
print("Nota: top_pct è diagnostico/ranking, non direttamente live-feasible.")

top = results[results["selection_type"] == "top_pct"].copy()

top_rank = (
    top.groupby(["side", "model", "top_pct"], dropna=False)
    .agg(
        folds=("test_date", "nunique"),
        avg_roc_auc=("roc_auc", "mean"),
        avg_pr_auc=("pr_auc", "mean"),
        avg_test_pos_rate=("test_positive_rate", "mean"),
        total_trades_no_overlap=("n_trades_no_overlap", "sum"),
        avg_trades_no_overlap=("n_trades_no_overlap", "mean"),
        positive_folds_no_overlap=("net_total_points_no_overlap", lambda x: int((x > 0).sum())),
        negative_folds_no_overlap=("net_total_points_no_overlap", lambda x: int((x < 0).sum())),
        total_net_no_overlap=("net_total_points_no_overlap", "sum"),
        worst_fold_net_no_overlap=("net_total_points_no_overlap", "min"),
        best_fold_net_no_overlap=("net_total_points_no_overlap", "max"),
        avg_tp_no_overlap=("tp_rate_pct_no_overlap", "mean"),
    )
    .reset_index()
)

top_rank["net_per_trade_no_overlap"] = (
    top_rank["total_net_no_overlap"] / top_rank["total_trades_no_overlap"].replace(0, np.nan)
)

top_rank = top_rank.sort_values(
    ["total_net_no_overlap", "positive_folds_no_overlap", "avg_roc_auc"],
    ascending=[False, False, False],
)

print("\nBest top_pct combinations ranked by total_net_no_overlap:")
print(top_rank.round(4).head(30))

# ---------------------------------------------------------------------
# 3. Fold detail for relevant candidates
# ---------------------------------------------------------------------

candidates = [
    {
        "name": "LONG logistic threshold 0.55",
        "side": "long",
        "model": "logistic_l2_balanced",
        "selection_type": "threshold",
        "threshold": 0.55,
        "top_pct": np.nan,
    },
    {
        "name": "LONG logistic top 15%",
        "side": "long",
        "model": "logistic_l2_balanced",
        "selection_type": "top_pct",
        "threshold": np.nan,
        "top_pct": 15.0,
    },
    {
        "name": "LONG RF top 20%",
        "side": "long",
        "model": "random_forest_balanced",
        "selection_type": "top_pct",
        "threshold": np.nan,
        "top_pct": 20.0,
    },
    {
        "name": "SHORT RF threshold 0.60",
        "side": "short",
        "model": "random_forest_balanced",
        "selection_type": "threshold",
        "threshold": 0.60,
        "top_pct": np.nan,
    },
    {
        "name": "SHORT RF top 15%",
        "side": "short",
        "model": "random_forest_balanced",
        "selection_type": "top_pct",
        "threshold": np.nan,
        "top_pct": 15.0,
    },
    {
        "name": "SHORT RF top 20%",
        "side": "short",
        "model": "random_forest_balanced",
        "selection_type": "top_pct",
        "threshold": np.nan,
        "top_pct": 20.0,
    },
]

print("\n" + "=" * 140)
print("[3] FOLD-BY-FOLD DETAIL FOR CANDIDATES")
print("=" * 140)

for cand in candidates:
    sub = results[
        (results["side"] == cand["side"])
        & (results["model"] == cand["model"])
        & (results["selection_type"] == cand["selection_type"])
    ].copy()

    if cand["selection_type"] == "threshold":
        sub = sub[np.isclose(sub["threshold"], cand["threshold"], equal_nan=False)]
    else:
        sub = sub[np.isclose(sub["top_pct"], cand["top_pct"], equal_nan=False)]

    cols = [
        "test_date",
        "side",
        "model",
        "selection_type",
        "top_pct",
        "threshold",
        "test_rows",
        "test_positive_rate",
        "roc_auc",
        "pr_auc",
        "n_selected_raw",
        "tp_rate_pct_raw",
        "net_avg_points_raw",
        "net_total_points_raw",
        "n_trades_no_overlap",
        "tp_rate_pct_no_overlap",
        "net_avg_points_no_overlap",
        "net_total_points_no_overlap",
    ]

    print("\n" + "-" * 140)
    print(cand["name"])
    print("-" * 140)

    if sub.empty:
        print("No rows found.")
        continue

    print(sub[cols].round(4))

    total_net = sub["net_total_points_no_overlap"].sum()
    total_trades = sub["n_trades_no_overlap"].sum()
    pos_folds = int((sub["net_total_points_no_overlap"] > 0).sum())
    neg_folds = int((sub["net_total_points_no_overlap"] < 0).sum())
    net_per_trade = total_net / total_trades if total_trades > 0 else np.nan

    print("\nCompact:")
    print({
        "folds": int(sub["test_date"].nunique()),
        "positive_folds": pos_folds,
        "negative_folds": neg_folds,
        "total_trades_no_overlap": int(total_trades),
        "total_net_no_overlap": round(float(total_net), 4),
        "net_per_trade_no_overlap": round(float(net_per_trade), 4) if np.isfinite(net_per_trade) else np.nan,
        "worst_fold": round(float(sub["net_total_points_no_overlap"].min()), 4),
        "best_fold": round(float(sub["net_total_points_no_overlap"].max()), 4),
    })

# ---------------------------------------------------------------------
# 4. Per-day model quality only
# ---------------------------------------------------------------------

print("\n" + "=" * 140)
print("[4] MODEL QUALITY BY FOLD")
print("=" * 140)

quality = (
    results[["side", "model", "test_date", "test_positive_rate", "roc_auc", "pr_auc", "brier"]]
    .drop_duplicates()
    .sort_values(["side", "model", "test_date"])
)

print(quality.round(4))

print("\n" + "=" * 140)
print("DONE")
print("=" * 140)
