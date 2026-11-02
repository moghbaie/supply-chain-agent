"""Build escalation cases: what a planner can see at review, and a reference category from what happened next.

The reference uses the following 14 days, so it never goes into a case's visible facts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["monitor", "reorder_recommendation", "expedite_recommendation", "human_review"]
SEVERE_QUANTILE = 0.90
HORIZON_DAYS = 14
VISIBLE_COLUMNS = ["on_hand", "on_order", "fc_mean_14", "fc_q90_sum_14", "cover_days", "days_to_next_due", "units_due_14", "units_overdue",
                   "days_overdue_max", "open_pos", "ordered_today", "supplier_late_rate", "supplier_recent_late_rate", "stockout_risk",
                   "stockout_risk_lightgbm", "value_band"]


def future_lost_margin(daily, origins, horizon=HORIZON_DAYS):
    """Return lost margin in the days after each origin, per scenario and series."""
    frame = daily[["scenario", "id", "date", "lost_margin"]].sort_values(["scenario", "id", "date"])
    grouped = frame.groupby(["scenario", "id"]).lost_margin
    frame["future_lost_margin"] = grouped.transform(lambda s: s.iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1].shift(-1))
    return origins.merge(frame.rename(columns={"date": "origin"})[["scenario", "id", "origin", "future_lost_margin"]], on=["scenario", "id", "origin"], how="left")


def reference_category(frame, severe_margin):
    """Assign the category a planner would have wanted, knowing the next 14 days."""
    stockout = frame.label == 1
    severe = stockout & (frame.future_lost_margin >= severe_margin)
    expedite = stockout & ~severe & (frame.open_pos > 0)
    reorder = stockout & ~severe & (frame.open_pos == 0)
    return np.select([severe, expedite, reorder], ["human_review", "expedite_recommendation", "reorder_recommendation"], "monitor")


def value_bands(frame, edges):
    """Bucket daily sales value so requests carry no raw prices."""
    value = frame.fc_mean_14 / 14 * frame.price
    return np.select([value < edges[0], value < edges[1]], ["low", "medium"], "high")


def build_cases(features_path=Path("data/processed/stockout_features.parquet"), predictions_path=Path("data/processed/stockout_predictions.parquet"),
                daily_path=Path("data/simulated/inventory_daily.parquet"), purchase_path=Path("data/processed/delay/inbound_features.parquet")):
    """Return validation and test cases with visible facts and reference categories, and the settings fitted on train."""
    features = pd.read_parquet(features_path)
    predictions = pd.read_parquet(predictions_path, columns=["scenario", "origin", "id", "tabpfn", "lightgbm"]).rename(columns={"tabpfn": "stockout_risk", "lightgbm": "stockout_risk_lightgbm"})
    daily = pd.read_parquet(daily_path, columns=["scenario", "id", "date", "lost_margin"])
    train = future_lost_margin(daily, features[features.split == "train"][["scenario", "id", "origin", "label", "fc_mean_14", "price"]])
    severe_margin = float(train[train.label == 1].future_lost_margin.quantile(SEVERE_QUANTILE))
    train_value = train.fc_mean_14 / 14 * train.price
    edges = (float(train_value.quantile(1 / 3)), float(train_value.quantile(2 / 3)))
    cases = features[features.split.isin(["validation", "test"])].merge(predictions, on=["scenario", "origin", "id"], validate="one_to_one")
    cases = future_lost_margin(daily, cases)
    recent = pd.read_parquet(purchase_path, columns=["scenario", "supplier_id", "order_date", "supplier_recent_late_rate"]).drop_duplicates(["scenario", "supplier_id", "order_date"])
    cases = cases.sort_values("origin")
    cases = pd.merge_asof(cases, recent.sort_values("order_date"), left_on="origin", right_on="order_date", by=["scenario", "supplier_id"], allow_exact_matches=True).drop(columns="order_date")
    cases["value_band"] = value_bands(cases, edges)
    cases["reference"] = reference_category(cases, severe_margin)
    return cases.sort_values(["scenario", "origin", "id"]).reset_index(drop=True), {"severe_margin": severe_margin, "value_edges": edges}
