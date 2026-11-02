"""Load M5 sales as demand, detect store closures and select series without using future data."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/m5")


def load_m5():
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    dates = pd.DatetimeIndex(calendar.set_index("d").date.reindex(day_columns))
    return calendar, sales, prices, day_columns, dates


def longest_zero_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value == 0 else 0
        longest = max(longest, current)
    return longest


def closed_store_days(sales, day_columns):
    """Return a store-by-day frame, true when none of the store's items sold anything."""
    return sales.groupby("store_id")[day_columns].sum() == 0


def select_series(sales, day_columns, dates, config, seed):
    """Sample series across category and state after dropping ones that fail the checks through gap_check_end."""
    selection = config["selection"]
    check_end = pd.Timestamp(selection["gap_check_end"])
    latest_first_sale = pd.Timestamp(selection["latest_first_sale"])
    matrix = sales[day_columns].to_numpy()
    through_check = dates <= check_end
    first_sale = dates[(matrix > 0).argmax(axis=1)]
    candidates = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    candidates["first_sale"] = first_sale
    candidates["longest_zero_run_to_check_end"] = [
        longest_zero_run(row[(dates >= start) & through_check]) for row, start in zip(matrix, first_sale)]
    candidates["longest_zero_run_after_check_end"] = [longest_zero_run(row[~through_check]) for row in matrix]
    candidates["mean_daily_units_to_check_end"] = [
        row[(dates >= start) & through_check].mean() for row, start in zip(matrix, first_sale)]

    launched = candidates[candidates.first_sale <= latest_first_sale]
    eligible = launched[launched.longest_zero_run_to_check_end < selection["max_zero_gap_days"]]
    funnel = {"all_series": len(candidates), "launched_in_time": len(launched), "eligible": len(eligible)}

    groups = eligible.groupby(["cat_id", "state_id"], sort=True)
    exact = groups.size() / len(eligible) * selection["series_count"]
    counts = np.floor(exact).astype(int)
    remainder = selection["series_count"] - int(counts.sum())
    counts.loc[(exact - counts).sort_values(ascending=False, kind="stable").index[:remainder]] += 1
    rng = np.random.default_rng(seed)
    picks = [group.iloc[rng.choice(len(group), int(counts.loc[key]), replace=False)] for key, group in groups]
    selected = pd.concat(picks).sort_values("id").reset_index(drop=True)
    return selected, eligible, funnel


def daily_prices(selected, prices, calendar, dates):
    """Return a price per series per day, carrying the last listed price across unlisted weeks."""
    week_of_day = calendar.set_index("date").wm_yr_wk.reindex(dates).to_numpy()
    listed = prices.merge(selected[["store_id", "item_id", "id"]], on=["store_id", "item_id"])
    weekly = listed.pivot(index="id", columns="wm_yr_wk", values="sell_price").reindex(selected.id)
    filled = weekly.ffill(axis=1)
    unlisted_weeks = int((filled.notna() & weekly.isna()).sum().sum())
    return filled.reindex(columns=week_of_day).to_numpy(), unlisted_weeks
