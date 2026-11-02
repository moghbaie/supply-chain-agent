"""Run the Step 3 simulation, save outputs and print the data needed for review."""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.simulation.demand import closed_store_days, daily_prices, load_m5, select_series
from src.simulation.world import assign_suppliers, build_suppliers, run_scenario

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 30)
OUT = Path("data/simulated")
REPORT = Path("reports/step_03")
INK, MUTED, SURFACE, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3", "#2a78d6", "#eb6834"
PREVIEW = ["date", "id", "on_hand_start", "received", "demand", "sales", "unmet", "on_hand_end", "on_order_end",
           "reorder_point", "ordered", "stockout_day", "stockout_next_14d", "split"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
seed = config["seed"]
started = time.time()
calendar, sales, prices, day_columns, all_dates = load_m5()
closed = closed_store_days(sales, day_columns)
selected, eligible, funnel = select_series(sales, day_columns, all_dates, config, seed)
suppliers = build_suppliers(config, seed)
selected = assign_suppliers(selected, suppliers, seed)
demand = sales.set_index("id").loc[selected.id, day_columns].to_numpy(dtype=np.int64)
price, unlisted_weeks = daily_prices(selected, prices, calendar, all_dates)
sim_days = all_dates >= pd.Timestamp(config["dates"]["simulation_start"])

print("SELECTION funnel:", funnel, "| selected:", len(selected), "| items:", selected.item_id.nunique())
print(pd.crosstab(selected.cat_id, selected.state_id, margins=True).to_string())
print("mean daily units, all series:", round(float(sales[day_columns].to_numpy().mean()), 3),
      "| eligible:", round(float(eligible.mean_daily_units_to_check_end.mean()), 3),
      "| selected:", round(float(selected.mean_daily_units_to_check_end.mean()), 3))
later = selected[selected.longest_zero_run_after_check_end >= config["selection"]["max_zero_gap_days"]]
print(f"selected series with a 56+ day zero gap after 2014: {len(later)} of {len(selected)}")
closure_counts = pd.Series(closed.to_numpy().sum(axis=0), index=all_dates)
closure_dates = closure_counts[closure_counts > 0]
print("closure dates:", len(closure_dates), "| closed store-days:", int(closure_counts.sum()),
      "| stores closed per date:", closure_dates.value_counts().to_dict(),
      "| examples:", [str(d.date()) for d in closure_dates.index[:6]])
print("prices: unlisted weeks carried forward:", unlisted_weeks, "| missing during simulation:", int(np.isnan(price[:, sim_days]).sum()))
print("\nSUPPLIERS (synthetic)\n", suppliers.to_string(index=False))
print("series per supplier:", selected.supplier_id.value_counts().sort_index().to_dict())

dailies, order_frames, parameters = [], [], {}
for name in config["scenarios"]:
    tick = time.time()
    daily, orders, policy, disrupted = run_scenario(name, config, selected, suppliers, all_dates, demand, closed, price, seed)
    dailies.append(daily)
    order_frames.append(orders)
    parameters[name] = {"policy": asdict(policy), "disrupted_suppliers": disrupted,
                        **{k: v for k, v in config["scenarios"][name].items() if k != "policy"}}
    print(f"\nscenario {name}: {len(daily):,} daily rows, {len(orders):,} purchase orders, {time.time() - tick:.1f}s")
daily = pd.concat(dailies, ignore_index=True)
orders = pd.concat(order_frames, ignore_index=True)
print("\nSCENARIO PARAMETERS\n", json.dumps(parameters, default=str))

base = daily[daily.scenario == "baseline"]
print("\n20 INVENTORY RECORDS (baseline, one series around its first training stockout)")
first = base[base.stockout_day & (base.split == "train")].iloc[0]
window = base[(base.id == first.id) & (base.date >= first.date - pd.Timedelta(days=12))].head(20)
print(window[PREVIEW].assign(id=lambda d: d.id.str.replace("_evaluation", "")).to_string(index=False))

print("\n10 PURCHASE ORDERS (supplier_disruption)")
print(orders[orders.scenario == "supplier_disruption"].iloc[1000:1010].drop(columns=["scenario", "po_id"])
      .assign(id=lambda d: d.id.str.replace("_evaluation", "")).to_string(index=False))

print("\nLEAD TIMES by scenario (received POs)")
received = orders[orders.status == "received"]
print(received.groupby("scenario").actual_lead_days.apply(lambda s: s.astype(float).describe()).unstack().round(2).to_string())
print("share delayed:", received.assign(late=lambda d: d.delay_days > 0).groupby("scenario").late.mean().round(4).to_dict())
print("by tier, baseline:", received[received.scenario == "baseline"].merge(suppliers[["supplier_id", "tier"]])
      .groupby("tier").actual_lead_days.apply(lambda s: round(float(s.astype(float).mean()), 2)).to_dict())
print("open POs at end:", orders[orders.status == "open"].groupby("scenario").size().to_dict())

active = daily[daily.store_open & (daily.split != "burn_in")]
summary = active.groupby(["scenario", "split"], sort=False).agg(
    days=("id", "size"), demand=("demand", "sum"), unmet=("unmet", "sum"), stockout_days=("stockout_day", "sum"),
    label_rate=("stockout_next_14d", "mean"), on_hand=("on_hand_end", "mean"))
summary["fill_rate_pct"] = (100 * (1 - summary.unmet / summary.demand)).round(2)
summary["stockout_day_pct"] = (100 * summary.stockout_days / summary.days).round(2)
summary["label_rate_pct"] = (100 * summary.label_rate).round(2)
print("\nSTOCKOUTS by scenario and split")
print(summary[["days", "demand", "unmet", "fill_rate_pct", "stockout_days", "stockout_day_pct", "label_rate_pct", "on_hand"]].round(2).to_string())
base_train = base[base.split == "train"]
print("\nlabel rate by category, baseline train:", base_train.merge(selected[["id", "cat_id"]]).groupby("cat_id").stockout_next_14d.mean().round(4).to_dict())
per_series = base_train.groupby("id").stockout_day.mean()
print("per-series stockout-day rate, baseline train:", per_series.describe().round(4).to_dict(), "| series with none:", int((per_series == 0).sum()))
print("\nCOSTS (synthetic parameters)\n", active.groupby("scenario", sort=False)[["holding_cost", "lost_margin", "shipping_cost"]].sum().round(0).to_string())

residual = int((daily.on_hand_start + daily.received - daily.sales - daily.on_hand_end).abs().sum())
closed_rows = daily[~daily.store_open]
print("\nconservation residual:", residual, "| min on hand:", int(daily.on_hand_end.min()),
      "| demand != sales + unmet:", int((daily.demand != daily.sales + daily.unmet).sum()),
      "| closed-day rows:", len(closed_rows), "| closed-day demand + receipts:", int(closed_rows.demand.sum() + closed_rows.received.sum()))

OUT.mkdir(parents=True, exist_ok=True)
daily.to_parquet(OUT / "inventory_daily.parquet", index=False)
orders.to_parquet(OUT / "purchase_orders.parquet", index=False)
received[["scenario", "po_id", "id", "supplier_id", "receipt_date", "quantity"]].to_parquet(OUT / "receipts.parquet", index=False)
selected.to_parquet(OUT / "series.parquet", index=False)
suppliers.to_parquet(OUT / "suppliers.parquet", index=False)
REPORT.mkdir(parents=True, exist_ok=True)
window[PREVIEW].to_csv(REPORT / "inventory_preview.csv", index=False)
summary.round(4).to_csv(REPORT / "stockouts_by_split.csv")
(REPORT / "scenario_parameters.json").write_text(json.dumps(parameters, indent=2, default=str))
(REPORT / "selection_funnel.json").write_text(json.dumps(funnel, indent=2))

rates = per_series.sort_values()
picks = rates.index[np.linspace(0, len(rates) - 1, 6).astype(int)]
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, facecolor=SURFACE)
for ax, series_id in zip(axes.flat, picks):
    part = base[(base.id == series_id) & (base.date >= "2014-01-01") & (base.date <= "2014-12-31")]
    ax.plot(part.date, part.on_hand_end, color=BLUE, linewidth=1.5, label="On hand, end of day")
    hits = part[part.stockout_day]
    ax.scatter(hits.date, np.zeros(len(hits)), color=ORANGE, s=24, edgecolors=SURFACE, linewidths=1, zorder=3, label="Stockout day")
    ax.set_title(f"{series_id.replace('_evaluation', '')}  ({100 * rates[series_id]:.1f}% of train days)", fontsize=9, color=INK, loc="left")
    style(ax)
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=MUTED)
fig.suptitle("Synthetic stock trajectories, baseline, 2014 (lowest to highest stockout rate)", color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(REPORT / "stock_trajectories.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, sharex=True, facecolor=SURFACE)
upper = int(received.actual_lead_days.astype(float).quantile(0.995)) + 1
for ax, name in zip(axes, config["scenarios"]):
    values = received[received.scenario == name].actual_lead_days.astype(float).to_numpy()
    ax.hist(values, bins=np.arange(0, upper + 2) - 0.5, color=BLUE, rwidth=0.75, weights=np.full(len(values), 100 / len(values)))
    ax.set_title(f"{name} (median {np.median(values):.0f}, p95 {np.percentile(values, 95):.0f} days)", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Actual lead time, days", fontsize=8, color=MUTED)
    style(ax)
axes[0].set_ylabel("% of received POs", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(REPORT / "lead_times.png", dpi=130)

print(f"\nruntime {time.time() - started:.1f}s | output:", {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(OUT.glob('*.parquet'))})
