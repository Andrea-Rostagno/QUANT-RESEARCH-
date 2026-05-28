from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)


DATA_PATH = Path("data/processed/labeled_dataset_M1_tp1.5_sl1.0_h30.parquet")
OUT_DIR = Path("reports/model_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 30
TRAIN_FRAC = 0.70
RANDOM_STATE = 42


def make_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
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

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    all_numeric_no_leak = [
        c for c in numeric_cols
        if not any(w in c.lower() for w in leakage_words)
    ]

    # Più prudente: rimuove prezzi assoluti OHLC, che possono essere non stazionari.
    raw_price_suffixes = ("_open", "_high", "_low", "_close")
    stationary_no_prices = [
        c for c in all_numeric_no_leak
        if not c.endswith(raw_price_suffixes)
    ]

    return {
        "all_numeric_no_leak": all_numeric_no_leak,
        "stationary_no_prices": stationary_no_prices,
    }


def temporal_split(df: pd.DataFrame, train_frac: float, embargo: int):
    n = len(df)
    split = int(n * train_frac)

    train_end = max(0, split - embargo)
    test_start = split

    train_idx = df.index[:train_end]
    embargo_idx = df.index[train_end:test_start]
    test_idx = df.index[test_start:]

    return train_idx, embargo_idx, test_idx


def get_trade_points(df_side: pd.DataFrame, side: str) -> np.ndarray:
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
    )

    return gross.astype(float)


def get_net_trade_points(df_side: pd.DataFrame, side: str) -> np.ndarray:
    gross = get_trade_points(df_side, side)
    label_col = f"label_{side}"

    spread_col = "XGER30_spread_points"
    if spread_col not in df_side.columns:
        return gross

    net = gross.copy()
    active = df_side[label_col].to_numpy() != 0
    net[active] = net[active] - df_side.loc[active, spread_col].to_numpy()
    return net


def classification_report_table(y_true, proba):
    y_pred_50 = (proba >= 0.50).astype(int)

    out = {
        "n": len(y_true),
        "positive_rate": float(np.mean(y_true)),
        "accuracy@0.50": accuracy_score(y_true, y_pred_50),
        "precision@0.50": precision_score(y_true, y_pred_50, zero_division=0),
        "recall@0.50": recall_score(y_true, y_pred_50, zero_division=0),
        "f1@0.50": f1_score(y_true, y_pred_50, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred_50, labels=[0, 1])
    return out, cm


def threshold_trade_report(df_test, side: str, proba: np.ndarray):
    y_true = (df_test[f"label_{side}"] == 1).astype(int).to_numpy()
    net_points = get_net_trade_points(df_test, side)
    gross_points = get_trade_points(df_test, side)

    rows = []

    for thr in [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70]:
        selected = proba >= thr
        n_trades = int(selected.sum())

        if n_trades == 0:
            rows.append({
                "threshold": thr,
                "n_trades": 0,
                "trade_rate_pct": 0.0,
                "tp_rate_pct": np.nan,
                "gross_avg_points": np.nan,
                "net_avg_points": np.nan,
                "net_total_points": 0.0,
            })
            continue

        rows.append({
            "threshold": thr,
            "n_trades": n_trades,
            "trade_rate_pct": n_trades / len(proba) * 100,
            "tp_rate_pct": y_true[selected].mean() * 100,
            "gross_avg_points": gross_points[selected].mean(),
            "net_avg_points": net_points[selected].mean(),
            "net_total_points": net_points[selected].sum(),
        })

    return pd.DataFrame(rows).round(4)


def top_quantile_trade_report(df_test, side: str, proba: np.ndarray):
    y_true = (df_test[f"label_{side}"] == 1).astype(int).to_numpy()
    net_points = get_net_trade_points(df_test, side)
    gross_points = get_trade_points(df_test, side)

    rows = []

    for top_pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        n_top = max(1, int(len(proba) * top_pct))
        order = np.argsort(proba)[::-1]
        selected_idx = order[:n_top]

        rows.append({
            "top_pct": top_pct * 100,
            "n_trades": n_top,
            "min_proba_in_bucket": proba[selected_idx].min(),
            "avg_proba": proba[selected_idx].mean(),
            "tp_rate_pct": y_true[selected_idx].mean() * 100,
            "gross_avg_points": gross_points[selected_idx].mean(),
            "net_avg_points": net_points[selected_idx].mean(),
            "net_total_points": net_points[selected_idx].sum(),
        })

    return pd.DataFrame(rows).round(4)


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-scores))

    return model, proba


def print_feature_importance(model_name, fitted_model, feature_cols, out_path):
    rows = []

    model_obj = fitted_model

    if isinstance(fitted_model, Pipeline):
        model_obj = fitted_model.steps[-1][1]

    if hasattr(model_obj, "feature_importances_"):
        imp = model_obj.feature_importances_
        rows = list(zip(feature_cols, imp))

    elif hasattr(model_obj, "coef_"):
        coef = model_obj.coef_.ravel()
        rows = list(zip(feature_cols, np.abs(coef)))

    if not rows:
        print(f"\nNo feature importance available for {model_name}")
        return

    fi = (
        pd.DataFrame(rows, columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    fi.to_csv(out_path, index=False)

    print(f"\nTop 25 feature importance - {model_name}")
    print(fi.head(25))


def run_one_side(df, feature_cols, feature_set_name: str, side: str):
    print("\n" + "=" * 140)
    print(f"SIDE={side.upper()} | FEATURE_SET={feature_set_name}")
    print("=" * 140)

    label_col = f"label_{side}"
    y = (df[label_col] == 1).astype(int)

    train_idx, embargo_idx, test_idx = temporal_split(df, TRAIN_FRAC, HORIZON)

    X_train = df.loc[train_idx, feature_cols].copy()
    y_train = y.loc[train_idx].copy()

    X_test = df.loc[test_idx, feature_cols].copy()
    y_test = y.loc[test_idx].copy()
    df_test = df.loc[test_idx].copy()

    # Safety cleaning.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    print("\nSplit info")
    print("train rows:", len(X_train), "| from:", train_idx.min(), "| to:", train_idx.max())
    print("embargo rows:", len(embargo_idx), "| from:", embargo_idx.min(), "| to:", embargo_idx.max())
    print("test rows:", len(X_test), "| from:", test_idx.min(), "| to:", test_idx.max())

    print("\nClass balance")
    print("train positive rate:", round(y_train.mean() * 100, 2), "%")
    print("test positive rate:", round(y_test.mean() * 100, 2), "%")
    print("n_features:", len(feature_cols))

    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
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

    all_metrics = []
    all_predictions = pd.DataFrame(index=test_idx)
    all_predictions[f"y_true_{side}"] = y_test
    all_predictions[f"label_{side}"] = df_test[f"label_{side}"]

    for model_name, model in models.items():
        print("\n" + "-" * 120)
        print(f"MODEL: {model_name}")
        print("-" * 120)

        fitted, proba = fit_predict(model, X_train, y_train, X_test)

        metrics, cm = classification_report_table(y_test.to_numpy(), proba)
        metrics["side"] = side
        metrics["feature_set"] = feature_set_name
        metrics["model"] = model_name

        all_metrics.append(metrics)

        print("\nClassification metrics")
        print(pd.DataFrame([metrics]).round(4).T)

        print("\nConfusion matrix @ 0.50")
        print(pd.DataFrame(
            cm,
            index=["true_0_not_TP", "true_1_TP"],
            columns=["pred_0", "pred_1"],
        ))

        thr_report = threshold_trade_report(df_test, side, proba)
        top_report = top_quantile_trade_report(df_test, side, proba)

        print("\nThreshold trade report")
        print(thr_report)

        print("\nTop probability bucket trade report")
        print(top_report)

        all_predictions[f"proba_{model_name}"] = proba

        safe_name = f"{side}_{feature_set_name}_{model_name}"
        thr_report.to_csv(OUT_DIR / f"threshold_report_{safe_name}.csv", index=False)
        top_report.to_csv(OUT_DIR / f"top_bucket_report_{safe_name}.csv", index=False)

        print_feature_importance(
            model_name=safe_name,
            fitted_model=fitted,
            feature_cols=feature_cols,
            out_path=OUT_DIR / f"feature_importance_{safe_name}.csv",
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / f"metrics_{side}_{feature_set_name}.csv", index=False)
    all_predictions.to_parquet(OUT_DIR / f"predictions_{side}_{feature_set_name}.parquet")

    print("\nSaved:")
    print(OUT_DIR / f"metrics_{side}_{feature_set_name}.csv")
    print(OUT_DIR / f"predictions_{side}_{feature_set_name}.parquet")

    return metrics_df


def main():
    print("=" * 140)
    print("BASELINE MODEL - DAX M1 TP=1.5ATR SL=1.0ATR H=30")
    print("=" * 140)

    df = pd.read_parquet(DATA_PATH)

    print("\nLoaded:", DATA_PATH)
    print("shape:", df.shape)
    print("from:", df.index.min())
    print("to:", df.index.max())
    print("nan total:", int(df.isna().sum().sum()))
    print("duplicate index:", int(df.index.duplicated().sum()))

    feature_sets = make_feature_sets(df)

    print("\nFeature sets")
    for name, cols in feature_sets.items():
        print(name, "->", len(cols), "features")

    all_results = []

    for feature_set_name, feature_cols in feature_sets.items():
        for side in ["long", "short"]:
            res = run_one_side(
                df=df,
                feature_cols=feature_cols,
                feature_set_name=feature_set_name,
                side=side,
            )
            all_results.append(res)

    final = pd.concat(all_results, axis=0, ignore_index=True)
    final_path = OUT_DIR / "all_baseline_metrics.csv"
    final.to_csv(final_path, index=False)

    print("\n" + "=" * 140)
    print("FINAL METRICS SUMMARY")
    print("=" * 140)

    cols = [
        "side",
        "feature_set",
        "model",
        "positive_rate",
        "accuracy@0.50",
        "precision@0.50",
        "recall@0.50",
        "f1@0.50",
        "roc_auc",
        "pr_auc",
        "brier",
    ]

    print(final[cols].round(4))
    print("\nSaved final metrics:", final_path)


if __name__ == "__main__":
    main()
