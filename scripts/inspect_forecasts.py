"""Print actual vs predicted demand for 10 SKUs, plot intervals and break down accuracy by sales volume."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 20)
INK, MUTED, SURFACE, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3", "#2a78d6", "#eb6834"
REPORT = Path("reports/step_04")
test = pd.read_parquet("data/processed/forecasts/test.parquet")
series = pd.read_parquet("data/simulated/series.parquet")
LEVELS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def pinball(frame):
    total = 0.0
    for column, level in LEVELS.items():
        error = frame.actual - frame[column]
        total += np.maximum(level * error, (level - 1) * error).sum()
    return total


observed = test[test.observed]
timestral = observed[observed.model == "timestral"]
print("saved test rows per model:", test.groupby("model", observed=True).size().to_dict())

last_origin = timestral.origin.max()
at_last = timestral[timestral.origin == last_origin]
volume = timestral.groupby("id").actual.mean().sort_values()
picks = volume.index[np.linspace(0, len(volume) - 1, 10).astype(int)]
print(f"\n10 SKUS, forecast date {last_origin.date()}, Timestral, spread from slowest to fastest seller")
rows = []
for series_id in picks:
    part = at_last[at_last.id == series_id].sort_values("horizon")
    for label, frame in (("days 1-14", part[part.horizon <= 14]), ("days 15-28", part[part.horizon > 14])):
        rows.append({"sku": series_id.replace("_evaluation", ""), "period": label, "actual": int(frame.actual.sum()), "pred_mean": round(float(frame["mean"].sum()), 1),
                     "error_pct": round(100 * float(frame["mean"].sum() - frame.actual.sum()) / max(float(frame.actual.sum()), 1), 1),
                     "day_in_80_range_pct": round(100 * float(frame.actual.between(frame.q10, frame.q90).mean()), 0)})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(REPORT / "ten_skus.csv", index=False)

sample = at_last[at_last.id == picks[6]].sort_values("horizon")
print(f"\ndaily detail, {picks[6].replace('_evaluation', '')}:")
print(sample[["horizon", "actual", "mean", "q10", "q50", "q90"]].round(2).to_string(index=False))

print("\nACCURACY BY VOLUME GROUP (test, pinball per unit; ratio below 1 favors Timestral)")
quartile = pd.qcut(volume, 4, labels=["slowest", "second", "third", "fastest"])
results = []
for group in quartile.cat.categories:
    ids = quartile.index[quartile == group]
    part = observed[observed.id.isin(ids)]
    entry = {"group": group, "series": len(ids), "units_per_day": round(float(volume[ids].mean()), 2), "share_of_units_pct": round(100 * float(part[part.model == "timestral"].actual.sum() / timestral.actual.sum()), 1)}
    for model in ["seasonal_naive", "lightgbm", "timestral"]:
        frame = part[part.model == model]
        entry[f"{model}_pinball"] = round(pinball(frame) / frame.actual.sum(), 3)
    entry["ratio_vs_lightgbm"] = round(entry["timestral_pinball"] / entry["lightgbm_pinball"], 3)
    results.append(entry)
groups = pd.DataFrame(results)
print(groups.to_string(index=False))
groups.to_csv(REPORT / "accuracy_by_volume.csv", index=False)

print("\nINTERVAL CHECK (test, share of days)")
for model in ["seasonal_naive", "lightgbm", "timestral"]:
    frame = observed[observed.model == model]
    print(f"{model:15} below q10 {100 * (frame.actual < frame.q10).mean():5.1f}% | above q90 {100 * (frame.actual > frame.q90).mean():5.1f}% | mean width {(frame.q90 - frame.q10).mean():6.2f} | actual zero {100 * (frame.actual == 0).mean():.1f}%")

print("\nACCURACY BY HORIZON DAY (Timestral, test)")
by_day = timestral.groupby("horizon").apply(lambda f: pd.Series({"wape_mean_pct": 100 * (f["mean"] - f.actual).abs().sum() / f.actual.sum(), "bias_mean_pct": 100 * (f["mean"] - f.actual).sum() / f.actual.sum()}), include_groups=False)
print(by_day.iloc[[0, 6, 13, 20, 27]].round(2).to_string())

weekly = observed.assign(target=observed.origin + pd.to_timedelta(observed.horizon, unit="D"))
fig, axes = plt.subplots(5, 2, figsize=(12, 11), sharex=True, facecolor=SURFACE)
for ax, series_id in zip(axes.flat, picks):
    part = at_last[at_last.id == series_id].sort_values("horizon")
    dates = part.origin + pd.to_timedelta(part.horizon, unit="D")
    ax.fill_between(dates, part.q10, part.q90, color=BLUE, alpha=0.18, linewidth=0, label="80% range")
    ax.plot(dates, part["mean"], color=BLUE, linewidth=2, label="Forecast mean")
    ax.scatter(dates, part.actual, color=ORANGE, s=22, edgecolors=SURFACE, linewidths=1, zorder=3, label="Actual")
    ax.set_title(series_id.replace("_evaluation", ""), fontsize=9, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
for ax in axes[-1]:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", labelcolor=MUTED, ncols=3)
fig.suptitle(f"Timestral-28M: 28-day forecast from {last_origin.date()} vs actual daily units", color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(REPORT / "forecast_vs_actual.png", dpi=125)
