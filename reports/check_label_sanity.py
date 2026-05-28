from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)

PATH = Path("data/processed/labeled_dataset_M1_tp1.5_sl1.0_h30.parquet")

df = pd.read_parquet(PATH)

print("=" * 120)
print("LABEL SANITY CHECK - DAX M1")
print("=" * 120)

print("\n[1] BASIC INFO")
print("path:", PATH)
print("shape:", df.shape)
print("start:", df.index.min())
print("end:", df.index.max())
print("timezone:", getattr(df.index, "tz", None))
print("nan total:", int(df.isna().sum().sum()))
print("duplicate index:", int(df.index.duplicated().sum()))

required_cols = [
    "label_long",
    "label_short",
    "barrier_side_long",
    "barrier_side_short",
    "time_to_exit_long",
    "time_to_exit_short",
    "tp_distance_long",
    "sl_distance_long",
    "tp_distance_short",
    "sl_distance_short",
    "atr_14",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

spread_candidates = [
    c for c in df.columns
    if "spread" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
]

print("\n[2] SPREAD CANDIDATES")
print(spread_candidates)

spread_col = None
for c in ["XGER30_spread", "spread", "spread_points"]:
    if c in df.columns:
        spread_col = c
        break

if spread_col is None and spread_candidates:
    spread_col = spread_candidates[0]

print("selected spread_col:", spread_col)


def dist_table(s):
    counts = s.value_counts(dropna=False).sort_index()
    pct = s.value_counts(normalize=True, dropna=False).sort_index() * 100
    return pd.DataFrame({"count": counts, "pct": pct.round(2)})


print("\n[3] LABEL DISTRIBUTION LONG")
print(dist_table(df["label_long"]))

print("\n[4] LABEL DISTRIBUTION SHORT")
print(dist_table(df["label_short"]))

print("\n[5] BARRIER SIDE LONG")
print(df["barrier_side_long"].value_counts(dropna=False))

print("\n[6] BARRIER SIDE SHORT")
print(df["barrier_side_short"].value_counts(dropna=False))


print("\n[7] TIME TO EXIT LONG - OVERALL")
print(df["time_to_exit_long"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(3))

print("\n[8] TIME TO EXIT SHORT - OVERALL")
print(df["time_to_exit_short"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(3))


def exit_speed_table(label_col, tte_col, name):
    out = []
    for lab, label_name in [(-1, "SL"), (0, "TIMEOUT/NEUTRAL"), (1, "TP")]:
        sub = df[df[label_col] == lab]
        if len(sub) == 0:
            continue
        out.append({
            "side": name,
            "label": label_name,
            "n": len(sub),
            "mean_tte": sub[tte_col].mean(),
            "median_tte": sub[tte_col].median(),
            "pct_exit_1m": (sub[tte_col] <= 1).mean() * 100,
            "pct_exit_3m": (sub[tte_col] <= 3).mean() * 100,
            "pct_exit_5m": (sub[tte_col] <= 5).mean() * 100,
            "pct_exit_10m": (sub[tte_col] <= 10).mean() * 100,
        })
    return pd.DataFrame(out).round(2)


print("\n[9] EXIT SPEED BY OUTCOME - LONG")
print(exit_speed_table("label_long", "time_to_exit_long", "long"))

print("\n[10] EXIT SPEED BY OUTCOME - SHORT")
print(exit_speed_table("label_short", "time_to_exit_short", "short"))


tmp = df.copy()
tmp["date"] = tmp.index.date
tmp["hour_utc"] = tmp.index.hour

print("\n[11] LABEL LONG BY DAY - PERCENT")
long_day = (
    tmp.groupby("date")["label_long"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
    .pivot(index="date", columns="label_long", values="pct")
    .fillna(0)
    .round(2)
)
print(long_day)

print("\n[12] LABEL SHORT BY DAY - PERCENT")
short_day = (
    tmp.groupby("date")["label_short"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
    .pivot(index="date", columns="label_short", values="pct")
    .fillna(0)
    .round(2)
)
print(short_day)

print("\n[13] LABEL LONG BY HOUR UTC - PERCENT")
long_hour = (
    tmp.groupby("hour_utc")["label_long"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
    .pivot(index="hour_utc", columns="label_long", values="pct")
    .fillna(0)
    .round(2)
)
print(long_hour)

print("\n[14] LABEL SHORT BY HOUR UTC - PERCENT")
short_hour = (
    tmp.groupby("hour_utc")["label_short"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
    .pivot(index="hour_utc", columns="label_short", values="pct")
    .fillna(0)
    .round(2)
)
print(short_hour)


print("\n[15] ATR DIAGNOSTICS")
print(df["atr_14"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(4))

print("\n[16] BARRIER DISTANCE DIAGNOSTICS")
barrier_stats = pd.DataFrame({
    "tp_long": df["tp_distance_long"],
    "sl_long": df["sl_distance_long"],
    "tp_short": df["tp_distance_short"],
    "sl_short": df["sl_distance_short"],
})
print(barrier_stats.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(4))

if spread_col is not None:
    print("\n[17] SPREAD DIAGNOSTICS")
    print(df[spread_col].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(4))

    print("\n[18] SPREAD / ATR")
    spread_atr = df[spread_col] / df["atr_14"]
    print(spread_atr.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(4))

    print("\n[19] SPREAD / BARRIER DISTANCE")
    cost_ratios = pd.DataFrame({
        "spread_over_tp_long": df[spread_col] / df["tp_distance_long"],
        "spread_over_sl_long": df[spread_col] / df["sl_distance_long"],
        "spread_over_tp_short": df[spread_col] / df["tp_distance_short"],
        "spread_over_sl_short": df[spread_col] / df["sl_distance_short"],
    })
    print(cost_ratios.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(4))


def expectancy(label_col, tp_col, sl_col, side_name):
    gross = np.select(
        [
            df[label_col] == 1,
            df[label_col] == -1,
            df[label_col] == 0,
        ],
        [
            df[tp_col],
            -df[sl_col],
            0.0,
        ],
        default=0.0,
    )

    out = {
        "side": side_name,
        "n": len(df),
        "tp_pct": (df[label_col].eq(1).mean() * 100),
        "sl_pct": (df[label_col].eq(-1).mean() * 100),
        "neutral_pct": (df[label_col].eq(0).mean() * 100),
        "gross_expectancy_points": float(np.mean(gross)),
    }

    if spread_col is not None:
        active = df[label_col].ne(0).to_numpy()
        net = gross.copy()
        net[active] = net[active] - df.loc[df[label_col].ne(0), spread_col].to_numpy()
        out["rough_net_expectancy_points_minus_1spread"] = float(np.mean(net))

    return out


print("\n[20] ROUGH EXPECTANCY WITHOUT MODEL")
exp = pd.DataFrame([
    expectancy("label_long", "tp_distance_long", "sl_distance_long", "long"),
    expectancy("label_short", "tp_distance_short", "sl_distance_short", "short"),
]).round(4)
print(exp)


print("\n[21] POSSIBLE LEAKAGE / NON-FEATURE COLUMNS")
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

leakage_like = [
    c for c in df.columns
    if any(w in c.lower() for w in leakage_words)
]

for c in leakage_like:
    print(c)

print("\n[22] NUMERIC FEATURE COUNT AFTER BASIC EXCLUSION")
numeric_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c])
]

feature_cols = [
    c for c in numeric_cols
    if c not in leakage_like
]

print("numeric cols:", len(numeric_cols))
print("excluded leakage-like cols:", len([c for c in leakage_like if c in numeric_cols]))
print("candidate numeric features:", len(feature_cols))

print("\nFirst 30 candidate features:")
print(feature_cols[:30])

print("\n" + "=" * 120)
print("DONE")
print("=" * 120)
