"""Build the point-in-time stockout feature table for every scenario and print it for review."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.stockout.features import FEATURES, build_features, split_by_label_end

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 40, "display.max_colwidth", 24)
SIM = Path("data/simulated")
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
horizon = config["labels"]["stockout_horizon_days"]
daily_all = pd.read_parquet(SIM / "inventory_daily.parquet")
orders_all = pd.read_parquet(SIM / "purchase_orders.parquet")
series = pd.read_parquet(SIM / "series.parquet")
suppliers = pd.read_parquet(SIM / "suppliers.parquet")

frames = []
started = time.time()
for scenario in config["scenarios"]:
    forecasts = pd.read_parquet(f"data/processed/stockout_forecasts/{scenario}.parquet")
    origins = sorted(forecasts.origin.unique())
    mapping = split_by_label_end(pd.DatetimeIndex(origins), config, horizon)
    table = build_features(daily_all[daily_all.scenario == scenario], orders_all[orders_all.scenario == scenario], forecasts, series, suppliers,
                           pd.DatetimeIndex(origins), horizon, mapping)
    table.insert(0, "scenario", scenario)
    frames.append(table)
    print(f"{scenario}: {len(table):,} rows, {time.time() - started:.0f}s", flush=True)
data = pd.concat(frames, ignore_index=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
data.to_parquet("data/processed/stockout_features.parquet", index=False)

print("\nFEATURE SCHEMA")
print(data[FEATURES].dtypes.astype(str).to_string())
print("\nSPLITS")
print(data.groupby("split").origin.agg(["min", "max", "nunique"]).assign(rows=data.groupby("split").size()).to_string())
print("\nCLASS BALANCE (share with a stockout in the next 14 days)")
print(data.pivot_table(index="scenario", columns="split", values="label", aggfunc="mean", observed=True).mul(100).round(2).to_string())
print("labels missing:", int(data.label.isna().sum()))
print("\nMISSING VALUES (features)")
missing = data[FEATURES].isna().mean().mul(100).round(2)
print(missing[missing > 0].to_string())
print("\nFEATURE DISTRIBUTIONS (train)")
print(data[data.split == "train"][FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())
print("\n20 ROWS (baseline train, stockouts and non-stockouts)")
sample = data[(data.scenario == "baseline") & (data.split == "train")]
rows = pd.concat([sample[sample.label == 1].head(10), sample[sample.label == 0].head(10)])
print(rows[["origin", "id", "on_hand", "on_order", "fc_mean_14", "fc_q90_sum_14", "cover_days", "projected_14", "units_due_14", "units_overdue", "days_to_next_due", "supplier_late_rate", "label"]]
      .assign(id=lambda d: d.id.str.replace("_evaluation", "")).round(2).to_string(index=False))
Path("reports/step_05").mkdir(parents=True, exist_ok=True)
rows.to_csv("reports/step_05/feature_preview.csv", index=False)
print("\nLABEL RATE BY SIMPLE RISK BANDS (train, all scenarios)")
train = data[data.split == "train"]
print(train.groupby(pd.cut(train.projected_14, [-1e9, -10, -2, 0, 2, 10, 50, 1e9]), observed=True).label.agg(["size", "mean"]).round(3).to_string())
print(train.groupby(pd.cut(train.cover_days, [-1, 3, 7, 14, 21, 28, 56, 1e9]), observed=True).label.agg(["size", "mean"]).round(3).to_string())
print("\ncorrelation with label, top features:")
print(train[FEATURES].corrwith(train.label, method="spearman").abs().sort_values(ascending=False).head(10).round(3).to_string())
