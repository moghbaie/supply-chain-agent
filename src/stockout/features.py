"""Build point-in-time features for 14-day stockout risk at each weekly review.

Inputs are limited to what a planner could see at the end of the review day. Unmet demand, true supplier
tiers, scenario names and disruption windows stay hidden.
"""
import numpy as np
import pandas as pd

from src.simulation.engine import purchase_orders_as_of

SUPPLIER_WINDOW_DAYS = 180
FEATURES = ["on_hand", "on_order", "position", "reorder_point", "order_up_to", "ordered_today", "price", "fc_mean_7", "fc_mean_14",
            "fc_mean_28", "fc_q10_sum_14", "fc_q90_sum_14", "fc_q90_max_14", "cover_days", "position_cover_days", "projected_14",
            "projected_14_q90", "open_pos", "units_due_14", "units_overdue", "days_overdue_max", "days_to_next_due", "promised_lead_days",
            "supplier_late_rate", "supplier_mean_delay", "supplier_pos_seen", "sales_mean_28", "sales_zero_share_28", "sold_out_days_28",
            "cat_code"]


def supplier_history(orders, origin):
    """Summarize each supplier's deliveries received in the window before the origin."""
    known = purchase_orders_as_of(orders, origin)
    start = pd.Timestamp(origin) - pd.Timedelta(days=SUPPLIER_WINDOW_DAYS)
    received = known[(known.status == "received") & (known.receipt_date >= start)]
    grouped = received.assign(late=received.delay_days.astype(float) > 0, delay=received.delay_days.astype(float)).groupby("supplier_id")
    return grouped.agg(supplier_late_rate=("late", "mean"), supplier_mean_delay=("delay", "mean"), supplier_pos_seen=("late", "size"))


def open_order_features(orders, origin, horizon):
    """Summarize orders still open at the origin using only their promised dates."""
    origin = pd.Timestamp(origin)
    known = purchase_orders_as_of(orders, origin)
    pending = known[known.status == "open"].copy()
    days_to_due = (pending.promised_date - origin).dt.days
    pending["due_soon"] = pending.quantity.where((days_to_due >= 0) & (days_to_due <= horizon), 0)
    pending["overdue"] = pending.quantity.where(days_to_due < 0, 0)
    pending["days_overdue"] = (-days_to_due).clip(lower=0)
    pending["days_due"] = days_to_due.where(days_to_due >= 0)
    return pending.groupby("id").agg(open_pos=("quantity", "size"), units_due_14=("due_soon", "sum"), units_overdue=("overdue", "sum"),
                                     days_overdue_max=("days_overdue", "max"), days_to_next_due=("days_due", "min"))


def recent_sales(daily, origin, days=28):
    """Summarize sales and sold-out days in the window ending at the origin."""
    origin = pd.Timestamp(origin)
    window = daily[(daily.date > origin - pd.Timedelta(days=days)) & (daily.date <= origin) & daily.store_open]
    return window.groupby("id").agg(sales_mean_28=("sales", "mean"), sales_zero_share_28=("sales", lambda s: (s == 0).mean()),
                                    sold_out_days_28=("on_hand_end", lambda s: (s == 0).sum()))


def build_features(daily, orders, forecasts, series, suppliers, origins, horizon, split_of_origin):
    """Return one row per series per origin, with the label and split attached."""
    categories = series.cat_id.astype("category").cat.codes
    static = pd.DataFrame({"id": series.id, "supplier_id": series.supplier_id, "cat_id": series.cat_id, "cat_code": categories})
    static = static.merge(suppliers[["supplier_id", "promised_lead_days"]])
    by_date = {date: frame for date, frame in daily.groupby("date")}
    orders_by_supplier = orders.sort_values("order_date")
    rows = []
    for origin in origins:
        today = by_date[origin]
        frame = static.merge(today[["id", "on_hand_end", "on_order_end", "reorder_point", "order_up_to", "ordered", "price", "stockout_next_14d"]])
        frame = frame.rename(columns={"on_hand_end": "on_hand", "on_order_end": "on_order", "ordered": "ordered_today", "stockout_next_14d": "label"})
        frame = frame.merge(forecasts[forecasts.origin == origin].drop(columns="origin"), on="id")
        frame = frame.merge(open_order_features(orders_by_supplier, origin, horizon), on="id", how="left")
        frame = frame.merge(supplier_history(orders_by_supplier, origin), on="supplier_id", how="left")
        frame = frame.merge(recent_sales(daily[daily.date > origin - pd.Timedelta(days=28)], origin), on="id", how="left")
        for column in ["open_pos", "units_due_14", "units_overdue", "days_overdue_max", "supplier_pos_seen"]:
            frame[column] = frame[column].fillna(0)
        frame["position"] = frame.on_hand + frame.on_order
        daily_rate = (frame.fc_mean_14 / 14).clip(lower=0.01)
        frame["cover_days"] = frame.on_hand / daily_rate
        frame["position_cover_days"] = frame.position / daily_rate
        frame["projected_14"] = frame.on_hand + frame.units_due_14 - frame.fc_mean_14
        frame["projected_14_q90"] = frame.on_hand + frame.units_due_14 - frame.fc_q90_sum_14
        frame.insert(0, "origin", origin)
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result["split"] = result.origin.map(split_of_origin)
    return result


def split_by_label_end(origins, config, horizon):
    """Assign each origin to the period containing its whole label window, or drop it if the window crosses a boundary."""
    dates = config["dates"]
    train_end, validation_end = pd.Timestamp(dates["train_end"]), pd.Timestamp(dates["validation_end"])
    mapping = {}
    for origin in origins:
        label_end = origin + pd.Timedelta(days=horizon)
        if label_end <= train_end:
            mapping[origin] = "train"
        elif origin > train_end and label_end <= validation_end:
            mapping[origin] = "validation"
        elif origin > validation_end:
            mapping[origin] = "test"
        else:
            mapping[origin] = "purged"
    return mapping


def as_matrix(frame):
    return frame[FEATURES].to_numpy(dtype=np.float32)
