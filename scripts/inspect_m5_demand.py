"""Print M5 sales coverage, zeros, launch timing, prices and zero runs that may be hidden stockouts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RAW = Path("data/raw/m5")
OUT = Path("reports/step_02/m5")
SUSPECT_PROBABILITY = 0.001

calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
evaluation = pd.read_csv(RAW / "sales_train_evaluation.csv")
validation = pd.read_csv(RAW / "sales_train_validation.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")
day_columns = [c for c in evaluation.columns if c.startswith("d_")]
validation_days = [c for c in validation.columns if c.startswith("d_")]
sales = evaluation[day_columns].to_numpy(dtype=np.int32)
dates = calendar.set_index("d").date
result = {"series": len(evaluation), "items": evaluation.item_id.nunique(), "stores": evaluation.store_id.nunique(),
          "sales_days": len(day_columns), "sales_first": str(dates[day_columns[0]].date()), "sales_last": str(dates[day_columns[-1]].date()),
          "calendar_days_without_sales": len(calendar) - len(day_columns)}
print(result)
same = (evaluation[validation_days].to_numpy() == validation[validation_days].to_numpy()).all()
result["validation_is_prefix_of_evaluation"] = bool(same and (evaluation.item_id == validation.item_id).all() and (evaluation.store_id == validation.store_id).all())
print("validation file matches evaluation on shared days:", result["validation_is_prefix_of_evaluation"], f"({len(validation_days)} days)")
print("negative sales cells:", int((sales < 0).sum()), "| consecutive calendar:", bool((calendar.date.diff().dropna() == pd.Timedelta(days=1)).all()))

print("\nseries by category/state:\n", pd.crosstab(evaluation.cat_id, evaluation.state_id, margins=True).to_string())
print("\ndaily units per series:", pd.Series(sales.mean(axis=1)).describe().round(2).to_dict())
print("all cells zero share:", round(float((sales == 0).mean()), 4))

first_sale = np.where((sales > 0).any(axis=1), (sales > 0).argmax(axis=1), -1)
post_launch = np.arange(sales.shape[1])[None, :] >= first_sale[:, None]
result["zero_share_all_pct"] = round(100 * float((sales == 0).mean()), 2)
result["zero_share_after_first_sale_pct"] = round(100 * float(((sales == 0) & post_launch).sum() / post_launch.sum()), 2)
result["series_never_sold"] = int((first_sale < 0).sum())
print(f"zero share after first sale: {result['zero_share_after_first_sale_pct']}% | leading-zero days share: {round(100 * float((~post_launch).mean()), 2)}% | never sold: {result['series_never_sold']}")

week_of_day = calendar.set_index("d").wm_yr_wk.reindex(day_columns).to_numpy()
first_price_week = prices.groupby(["store_id", "item_id"]).wm_yr_wk.min()
first_price = pd.MultiIndex.from_frame(evaluation[["store_id", "item_id"]]).map(first_price_week).to_numpy()
priced = week_of_day[None, :] >= first_price[:, None]
result["sales_before_first_price_units"] = int(sales[~priced].sum())
result["days_with_no_price_share_pct"] = round(100 * float((~priced).mean()), 2)
print(f"units sold before first listed price: {result['sales_before_first_price_units']:,} | days before first price: {result['days_with_no_price_share_pct']}%")
print("price:", prices.sell_price.describe().round(2).to_dict())
per_week = prices.groupby(["store_id", "item_id"]).size()
gaps = prices.groupby(["store_id", "item_id"]).wm_yr_wk.nunique()
print("priced weeks per series:", per_week.describe().round(1).to_dict())

records = []
for row in range(sales.shape[0]):
    if first_sale[row] < 0:
        continue
    series = sales[row, first_sale[row]:]
    positive = np.flatnonzero(series > 0)
    rate = series.mean()
    edges = np.concatenate(([-1], positive, [len(series)]))
    lengths = np.diff(edges) - 1
    starts = edges[:-1] + 1
    keep = lengths > 0
    for length, start in zip(lengths[keep], starts[keep]):
        records.append((row, int(start + first_sale[row]), int(length), float(rate), int(start + length == len(series))))
runs = pd.DataFrame(records, columns=["row", "start", "length", "rate", "at_end"])
runs["probability"] = np.exp(-runs.rate * runs.length)
suspect = runs[runs.probability < SUSPECT_PROBABILITY]
zero_days = int(((sales == 0) & post_launch).sum())
result["suspect_runs"] = len(suspect)
result["suspect_series"] = int(suspect.row.nunique())
result["suspect_days"] = int(suspect.length.sum())
result["suspect_share_of_post_launch_zero_days_pct"] = round(100 * suspect.length.sum() / zero_days, 2)
result["christmas_days_zero_share_pct"] = round(100 * float((sales[:, (dates[day_columns].dt.month == 12) & (dates[day_columns].dt.day == 25)] == 0).mean()), 2)
print(f"\nzero runs unlikely under a constant Poisson rate (p<{SUSPECT_PROBABILITY}): {len(suspect):,} runs in {result['suspect_series']:,} of {sales.shape[0]:,} series")
print(f"days in those runs: {result['suspect_days']:,} = {result['suspect_share_of_post_launch_zero_days_pct']}% of post-launch zero days")
print("run length:", suspect.length.describe().round(1).to_dict())
print("Dec 25 zero share:", result["christmas_days_zero_share_pct"], "%")
named = suspect.nlargest(10, "length").assign(id=lambda d: evaluation.id.iloc[d.row].to_numpy(), start_date=lambda d: dates[[day_columns[i] for i in d.start]].dt.date.to_numpy())
print(named[["id", "start_date", "length", "rate", "probability", "at_end"]].round(4).to_string(index=False))

total = pd.Series(sales.sum(axis=0), index=pd.to_datetime(dates[day_columns].to_numpy()))
print("\nunits by year:", total.groupby(total.index.year).sum().to_dict())
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "demand_checks.json").write_text(json.dumps(result, indent=2))
