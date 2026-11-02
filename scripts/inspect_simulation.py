"""Print Step 3 diagnostics: closures, disruption impact, selection bias and label boundaries."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.simulation.demand import load_m5

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 20)
SIM = Path("data/simulated")
daily = pd.read_parquet(SIM / "inventory_daily.parquet")
orders = pd.read_parquet(SIM / "purchase_orders.parquet")
series = pd.read_parquet(SIM / "series.parquet")
parameters = json.loads(Path("reports/step_03/scenario_parameters.json").read_text())
calendar, sales, _, day_columns, dates = load_m5()

totals = sales.groupby("store_id")[day_columns].sum()
christmas = [c for c, d in zip(day_columns, dates) if d.month == 12 and d.day == 25]
print("Store units on Dec 25 (all items):")
print(totals[christmas].set_axis([str(d.year) for d in dates[[day_columns.index(c) for c in christmas]]], axis=1).to_string())
print("typical store day (median units):", int(np.median(totals.to_numpy())))

disrupted = parameters["supplier_disruption"]["disrupted_suppliers"]
print("\ndisrupted suppliers:", disrupted, "| series affected:", int(series.supplier_id.isin(disrupted).sum()))
ids = series[series.supplier_id.isin(disrupted)].id
rows = []
for start, end in parameters["supplier_disruption"]["windows"]:
    tail_end = pd.Timestamp(end) + pd.Timedelta(days=30)
    for scenario in ["baseline", "supplier_disruption"]:
        part = daily[(daily.scenario == scenario) & daily.id.isin(ids) & daily.date.between(start, tail_end)]
        placed = orders[(orders.scenario == scenario) & orders.supplier_id.isin(disrupted) & orders.order_date.between(start, end)]
        rows.append({"window": start, "scenario": scenario, "fill_pct": round(100 * (1 - part.unmet.sum() / part.demand.sum()), 2),
                     "stockout_day_pct": round(100 * part.stockout_day.mean(), 2), "pos_placed": len(placed),
                     "share_delayed": round(float((placed.delay_days > 0).mean()), 3), "mean_delay": round(float(placed.delay_days.astype(float).mean()), 2)})
print(pd.DataFrame(rows).to_string(index=False))

matrix = sales[day_columns].to_numpy()
print("\nselected vs all series, units/day quartiles (all time):")
print(pd.DataFrame({"all": pd.Series(matrix.mean(axis=1)).describe(), "selected": pd.Series(sales.set_index("id").loc[series.id, day_columns].to_numpy().mean(axis=1)).describe()}).round(2).T.to_string())
print("zero-day share, all:", round(float((matrix == 0).mean()), 3), "| selected:", round(float((sales.set_index("id").loc[series.id, day_columns].to_numpy() == 0).mean()), 3))

gap = series.longest_zero_run_after_check_end >= 56
base = daily[daily.scenario == "baseline"]
late = base[base.date > "2014-12-31"].merge(series[["id", "longest_zero_run_after_check_end"]])
print("\nseries with a later 56+ day gap, 2015-16: fill", round(100 * (1 - late[late.longest_zero_run_after_check_end >= 56].unmet.sum() / late[late.longest_zero_run_after_check_end >= 56].demand.sum()), 2),
      "| mean on hand", round(float(late[late.longest_zero_run_after_check_end >= 56].on_hand_end.mean()), 1), "| other series on hand", round(float(late[late.longest_zero_run_after_check_end < 56].on_hand_end.mean()), 1))

for name, group in base.groupby("split", sort=False):
    print(f"{name}: {group.date.min().date()} to {group.date.max().date()} | label null {int(group.stockout_next_14d.isna().sum())}")
edge = base[(base.split == "train") & (base.date > pd.Timestamp("2014-12-31") - pd.Timedelta(days=14))]
print("train rows whose 14-day label looks into validation:", len(edge), "| test rows with no label:", int(base[base.split == "test"].stockout_next_14d.isna().sum()))
print("stockout episodes, baseline train:", int(((base.stockout_day) & ~(base.groupby("id").stockout_day.shift(1, fill_value=False))).where(base.split == "train", False).sum()))
crossed = orders.sort_values(["scenario", "id", "order_date"]).groupby(["scenario", "id"]).receipt_date.apply(lambda s: int((s.diff() < pd.Timedelta(0)).sum())).groupby("scenario").sum()
print("orders that arrived before an earlier order:", crossed.to_dict())
