"""Build point-in-time tables for predicting synthetic inbound supplier delays at order time."""
import numpy as np
import pandas as pd

from src.stockout.features import supplier_history

FEATURES = ["quantity", "promised_lead_days", "order_month", "cat_code", "supplier_late_rate", "supplier_mean_delay", "supplier_pos_seen",
            "supplier_recent_late_rate", "supplier_recent_pos_seen", "item_late_rate", "item_pos_seen"]
RECENT_DAYS = 30


def recent_supplier(orders, origin):
    """Summarize each supplier's receipts in the last few weeks, which is how disruptions first appear."""
    start = origin - pd.Timedelta(days=RECENT_DAYS)
    seen = orders[(orders.receipt_date < origin) & (orders.receipt_date >= start)]
    return seen.assign(late=seen.delay_days.astype(float) > 0).groupby("supplier_id").late.agg(supplier_recent_late_rate="mean", supplier_recent_pos_seen="size")


def item_history(orders, origin):
    seen = orders[orders.receipt_date < origin]
    return seen.assign(late=seen.delay_days.astype(float) > 0).groupby("id").late.agg(item_late_rate="mean", item_pos_seen="size")


def build_po_features(orders, series, train_end, validation_end, burn_in_end):
    """Return one row per received order in one scenario, with supplier history as of the order date."""
    categories = series.set_index("id").cat_id.astype("category").cat.codes.rename("cat_code")
    placed = orders[(orders.status == "received") & (orders.order_date > burn_in_end)]
    frames = []
    for origin, group in placed.groupby("order_date"):
        known = orders[orders.order_date < origin]
        part = group.merge(supplier_history(known, origin), on="supplier_id", how="left")
        part = part.merge(recent_supplier(known, origin), on="supplier_id", how="left").merge(item_history(known, origin), on="id", how="left")
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True).merge(categories, left_on="id", right_index=True)
    for column in ["supplier_pos_seen", "supplier_recent_pos_seen", "item_pos_seen"]:
        frame[column] = frame[column].fillna(0)
    frame["order_month"] = frame.order_date.dt.month
    frame["delayed"] = (frame.delay_days.astype(float) > 0).astype(int)
    frame["split"] = np.select([frame.order_date > validation_end, frame.order_date > train_end], ["test", "validation"], "train")
    frame["known_before_validation"] = frame.receipt_date <= train_end
    frame["known_before_test"] = frame.receipt_date <= validation_end
    return frame
