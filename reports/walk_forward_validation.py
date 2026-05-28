from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 300)

DATA_PATH = Path("data/processed/labeled_dataset_M1_tp1.5_sl1.0_h30.parquet")
OUT_DIR = Path("reports/model_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 30
RANDOM_STATE = 42


def make_stationary_features(df: pd.DataFrame) -> list[str]:
    leakage_words = [
        "label",
        "barrier",
        "time_to_exit",
        "mfe",
        "mae",
        "tp_distance",
        "sl_distance",
        "future",
        "forward",
        "target",
        "hit",
        "exit",
    ]

    raw_price_suffixes = ("_open", "_high", "_low", "_close")

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_cols = [
        c for c in numeric_cols
        if not any(w in c.lower() for w in leakage_words)
        and not c.endswith(raw_price_suffixes)
    ]

    return feature_cols


def get_net_points(df_side: pd.DataFrame, side: str) -> np.ndarray:
    label_col = f"label_{side}"
    tp_col = f"tp_distance_{side}"
    sl_col = f"sl_distance_{side}"

    gross = np.select(
        [
            df_side[label_col].to_numpy() == 1,
            df_side[label_col].to_numpy() == -1,
            df_side[label_col].to_numpy() == 0,
        ],
        [
            df_side[tp_col].to_numpy(),
            -df_side[sl_col].to_numpy(),
            0.0,
        ],
        default=0.0,
    ).astype(float)

    spread_col = "XGER30_spread_points"
    if spread_col in df_side.columns:
        active = df_side[label_col].to_numpy() != 0
        gross[active] = gross[active] - df_side.loc[active, spread_col].to_numpy()

    return gross


def get_gross_points(df_side: pd.DataFrame, side: str) -> np.ndarray:
    label_col = f"label_{side}"
    tp_col = f"tp_distance_{side}"
    sl_col = f"sl_distance_{side}"

    gross = np.select(
        [
            df_side[label_col].to_numpy() == 1,
            df_side[label_col].to_numpy() == -1,
            df_side[label_col].to_numpy() == 0,
        ],
        [
            df_side[tp_col].to_numpy(),
            -df_side[sl_col].to_numpy(),
            0.0,
        ],
        default=0.0,
    ).astype(float)

    return gross


def evaluate_bucket(df_test: pd.DataFrame, side: str, proba: np.ndarray, top_pct: float) -> dict:
    n = len(df_test)
    n_top = max(1, int(n * top_pct))

    order = np.argsort(proba)[::-1]
    selected_pos = order[:n_top]

    y_true = (df_test[f"label_{side}"].to_numpy() == 1).astype(int)
    net = get_net_points(df_test, side)
    gross = get_gross_points(df_test, side)

    return {
        "top_pct": top_pct * 100,
        "n_selected_raw": int(n_top),
        "min_proba": float(proba[selected_pos].min()),
        "avg_proba": float(proba[selected_pos].mean()),
        "tp_rate_pct_raw": float(y_true[selected_pos].mean() * 100),
        "gross_avg_points_raw": float(gross[selected_pos].mean()),
        "net_avg_points_raw": float(net[selected_pos].mean()),
        "net_total_points_raw": float(net[selected_pos].sum()),
    }


def evaluate_threshold(df_test: pd.DataFrame, side: str, proba: np.ndarray, threshold: float) -> dict:
    selected = proba >= threshold
    n_selected = int(selected.sum())

    y_true = (df_test[f"label_{side}"].to_numpy() == 1).astype(int)
    net = get_net_points(df_test, side)
    gross = get_gross_points(df_test, side)

    if n_selected == 0:
        return {
            "threshold": threshold,
            "n_selected_raw": 0,
            "tp_rate_pct_raw": np.nan,
            "gross_avg_points_raw": np.nan,
            "net_avg_points_raw": np.nan,
            "net_total_points_raw": 0.0,
        }

    return {
        "threshold": threshold,
        "n_selected_raw": n_selected,
        "tp_rate_pct_raw": float(y_true[selected].mean() * 100),
        "gross_avg_points_raw": float(gross[selected].mean()),
        "net_avg_points_raw": float(net[selected].mean()),
        "net_total_points_raw": float(net[selected].sum()),
    }


def non_overlapping_simulation(df_test: pd.DataFrame, side: str, proba: np.ndarray, mode: str, value: float) -> dict:
    tmp = df_test.copy()
    tmp["proba"] = proba

    if mode == "top_pct":
        n_top = max(1, int(len(tmp) * value))
        selected_index = tmp.sort_values("proba", ascending=False).head(n_top).index
        candidates = tmp.loc[selected_index].sort_index().copy()
    elif mode == "threshold":
        candidates = tmp[tmp["proba"] >= value].sort_index().copy()
    else:
        raise ValueError("mode must be top_pct or threshold")

    if len(candidates) == 0:
        return {
            "n_trades_no_overlap": 0,
            "tp_rate_pct_no_overlap": np.nan,
            "net_avg_points_no_overlap": np.nan,
            "net_total_points_no_overlap": 0.0,
            "avg_proba_no_overlap": np.nan,
        }

    net_all = pd.Series(get_net_points(tmp, side), index=tmp.index)
    label_col = f"label_{side}"
    tte_col = f"time_to_exit_{side}"

    kept = []
    next_available_time = pd.Timestamp.min.tz_localize("UTC")

    for ts, row in candidates.iterrows():
        if ts < next_available_time:
            continue

        kept.append(ts)

        tte = int(row[tte_col])
        tte = max(1, tte)

        next_available_time = ts + pd.Timedelta(minutes=tte)

    if len(kept) == 0:
        return {
            "n_trades_no_overlap": 0,
            "tp_rate_pct_no_overlap": np.nan,
            "net_avg_points_no_overlap": np.nan,
            "net_total_points_no_overlap": 0.0,
            "avg_proba_no_overlap": np.nan,
        }

    kept_df = tmp.loc[kept]
    kept_net = net_all.loc[kept]
    kept_y = (kept_df[label_col] == 1).astype(int)

    return {
        "n_trades_no_overlap": int(len(kept)),
        "tp_rate_pct_no_overlap": float(kept_y.mean() * 100),
        "net_avg_points_no_overlap": float(kept_net.mean()),
        "net_total_points_no_overlap": float(kept_net.sum()),
        "avg_proba_no_overlap": float(kept_df["proba"].mean()),
    }


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def main():
    print("=" * 140)
    print("WALK-FORWARD VALIDATION - DAX M1")
    print("=" * 140)

    df = pd.read_parquet(DATA_PATH)
    feature_cols = make_stationary_features(df)

    df = df.copy()
    df["date"] = pd.Series(df.index.date, index=df.index)

    dates = sorted(df["date"].unique())

    print("\nLoaded:", DATA_PATH)
    print("shape:", df.shape)
    print("from:", df.index.min())
    print("to:", df.index.max())
    print("dates:", dates)
    print("n_features stationary:", len(feature_cols))

    models = {
        "logistic_l2_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    all_rows = []
    all_predictions = []

    for side in ["long", "short"]:
        label_col = f"label_{side}"

        print("\n" + "=" * 140)
        print("SIDE:", side.upper())
        print("=" * 140)

        for test_date in dates[1:]:
            test_start = df.loc[df["date"] == test_date].index.min()

            train_mask = df.index < test_start - pd.Timedelta(minutes=HORIZON)
            test_mask = df["date"] == test_date

            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()

            if len(train) < 500 or len(test) < 100:
                print(f"Skipping {test_date}: train={len(train)}, test={len(test)}")
                continue

            X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
            X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)

            medians = X_train.median(numeric_only=True)
            X_train = X_train.fillna(medians)
            X_test = X_test.fillna(medians)

            y_train = (train[label_col] == 1).astype(int)
            y_test = (test[label_col] == 1).astype(int)

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(f"Skipping {test_date}: only one class.")
                continue

            print(f"\nFold test_date={test_date} | train={len(train)} | test={len(test)} | train_pos={y_train.mean():.3f} | test_pos={y_test.mean():.3f}")

            for model_name, model in models.items():
                proba = fit_predict(model, X_train, y_train, X_test)

                roc = roc_auc_score(y_test, proba)
                pr = average_precision_score(y_test, proba)
                brier = brier_score_loss(y_test, proba)

                base_row = {
                    "side": side,
                    "test_date": str(test_date),
                    "model": model_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_positive_rate": y_train.mean(),
                    "test_positive_rate": y_test.mean(),
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "brier": brier,
                }

                print(f"  {model_name}: roc={roc:.4f} | pr={pr:.4f} | brier={brier:.4f}")

                for top_pct in [0.05, 0.10, 0.15, 0.20]:
                    row = base_row.copy()
                    row["selection_type"] = "top_pct"
                    row.update(evaluate_bucket(test, side, proba, top_pct))
                    row.update(non_overlapping_simulation(test, side, proba, "top_pct", top_pct))
                    all_rows.append(row)

                for threshold in [0.50, 0.525, 0.55, 0.575, 0.60]:
                    row = base_row.copy()
                    row["selection_type"] = "threshold"
                    row.update(evaluate_threshold(test, side, proba, threshold))
                    row.update(non_overlapping_simulation(test, side, proba, "threshold", threshold))
                    all_rows.append(row)

                pred = pd.DataFrame({
                    "timestamp": test.index,
                    "side": side,
                    "test_date": str(test_date),
                    "model": model_name,
                    "y_true": y_test.to_numpy(),
                    "label": test[label_col].to_numpy(),
                    "proba": proba,
                    "net_points": get_net_points(test, side),
                    "time_to_exit": test[f"time_to_exit_{side}"].to_numpy(),
                    "atr_14": test["atr_14"].to_numpy(),
                    "spread": test["XGER30_spread_points"].to_numpy(),
                })

                all_predictions.append(pred)

    results = pd.DataFrame(all_rows)
    predictions = pd.concat(all_predictions, axis=0, ignore_index=True)

    results_path = OUT_DIR / "walk_forward_results.csv"
    predictions_path = OUT_DIR / "walk_forward_predictions.parquet"

    results.to_csv(results_path, index=False)
    predictions.to_parquet(predictions_path)

    print("\n" + "=" * 140)
    print("SUMMARY BY SIDE / MODEL / SELECTION")
    print("=" * 140)

    summary = (
        results
        .groupby(["side", "model", "selection_type", "top_pct", "threshold"], dropna=False)
        .agg(
            folds=("test_date", "nunique"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_pr_auc=("pr_auc", "mean"),
            avg_test_pos_rate=("test_positive_rate", "mean"),
            avg_n_raw=("n_selected_raw", "mean"),
            avg_tp_raw=("tp_rate_pct_raw", "mean"),
            avg_net_raw=("net_avg_points_raw", "mean"),
            total_net_raw=("net_total_points_raw", "sum"),
            avg_n_no_overlap=("n_trades_no_overlap", "mean"),
            avg_tp_no_overlap=("tp_rate_pct_no_overlap", "mean"),
            avg_net_no_overlap=("net_avg_points_no_overlap", "mean"),
            total_net_no_overlap=("net_total_points_no_overlap", "sum"),
        )
        .reset_index()
        .sort_values(["side", "model", "selection_type", "top_pct", "threshold"])
    )

    summary_path = OUT_DIR / "walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.round(4))

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
