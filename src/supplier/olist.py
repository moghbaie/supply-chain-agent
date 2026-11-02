"""Build point-in-time tables for predicting late customer deliveries on Olist at purchase time.

Delivery dates, order status, approval dates, reviews and payments are never inputs: they happen after purchase or
describe its outcome. History features use only orders already delivered before each purchase.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/olist")
FIRST_PURCHASE = pd.Timestamp("2017-01-01")
LAST_PURCHASE = pd.Timestamp("2018-08-31 23:59:59")
VALIDATION_START = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2018-04-01")
FEATURES = ["estimated_days", "purchase_month", "purchase_weekday", "purchase_hour", "item_count", "seller_count", "total_price", "total_freight",
            "freight_share", "total_weight_g", "total_volume_cm3", "photos_qty", "customer_state_code", "seller_state_code", "same_state",
            "seller_orders_seen", "seller_late_rate", "seller_mean_days", "route_orders_seen", "route_late_rate", "route_mean_days"]


def _read(name, **kwargs):
    return pd.read_csv(RAW / f"{name}.csv", low_memory=False, **kwargs)


def load_tables():
    dates = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    return {
        "orders": _read("olist_orders_dataset", parse_dates=dates),
        "items": _read("olist_order_items_dataset"),
        "customers": _read("olist_customers_dataset"),
        "sellers": _read("olist_sellers_dataset"),
        "products": _read("olist_products_dataset"),
    }


def delivered_orders(orders):
    """Return delivered orders in the study period with their late label."""
    kept = orders[(orders.order_status == "delivered") & orders.order_delivered_customer_date.notna()
                  & orders.order_purchase_timestamp.between(FIRST_PURCHASE, LAST_PURCHASE)].copy()
    kept["late"] = (kept.order_delivered_customer_date.dt.normalize() > kept.order_estimated_delivery_date).astype(int)
    return kept


def as_of_history(rows, events, key, prefix):
    """Attach counts, late rates and delivery days from events delivered before each purchase."""
    ordered = events.sort_values("order_delivered_customer_date")
    grouped = ordered.groupby(key, sort=False)
    history = ordered[[key, "order_delivered_customer_date"]].rename(columns={"order_delivered_customer_date": "known_at"})
    history[f"{prefix}_orders_seen"] = grouped.cumcount() + 1
    history["late_sum"] = grouped.late.cumsum()
    history["days_sum"] = grouped.delivery_days.cumsum()
    merged = pd.merge_asof(rows.sort_values("order_purchase_timestamp"), history.sort_values("known_at"), left_on="order_purchase_timestamp",
                           right_on="known_at", by=key, allow_exact_matches=False)
    merged[f"{prefix}_late_rate"] = merged.late_sum / merged[f"{prefix}_orders_seen"]
    merged[f"{prefix}_mean_days"] = merged.days_sum / merged[f"{prefix}_orders_seen"]
    merged[f"{prefix}_orders_seen"] = merged[f"{prefix}_orders_seen"].fillna(0)
    return merged.drop(columns=["known_at", "late_sum", "days_sum"])


def build_order_features(tables):
    """Return one row per delivered order, with only information known at purchase."""
    items = tables["items"].merge(tables["products"], on="product_id", how="left")
    items["volume"] = items.product_length_cm * items.product_height_cm * items.product_width_cm
    sellers_first = items.sort_values("order_item_id").drop_duplicates("order_id")[["order_id", "seller_id"]]
    totals = items.groupby("order_id").agg(item_count=("order_item_id", "size"), seller_count=("seller_id", "nunique"), total_price=("price", "sum"),
                                           total_freight=("freight_value", "sum"), total_weight_g=("product_weight_g", "sum"),
                                           total_volume_cm3=("volume", "sum"), photos_qty=("product_photos_qty", "mean"))
    orders = delivered_orders(tables["orders"]).merge(totals, on="order_id").merge(sellers_first, on="order_id")
    orders = orders.merge(tables["customers"][["customer_id", "customer_state"]], on="customer_id").merge(tables["sellers"][["seller_id", "seller_state"]], on="seller_id")
    orders["delivery_days"] = (orders.order_delivered_customer_date - orders.order_purchase_timestamp).dt.total_seconds() / 86400
    orders["route"] = orders.seller_state + "-" + orders.customer_state
    orders["estimated_days"] = (orders.order_estimated_delivery_date - orders.order_purchase_timestamp).dt.total_seconds() / 86400
    orders["purchase_month"] = orders.order_purchase_timestamp.dt.month
    orders["purchase_weekday"] = orders.order_purchase_timestamp.dt.weekday
    orders["purchase_hour"] = orders.order_purchase_timestamp.dt.hour
    orders["freight_share"] = orders.total_freight / (orders.total_price + orders.total_freight).replace(0, np.nan)
    orders["same_state"] = (orders.customer_state == orders.seller_state).astype(int)
    states = sorted(set(orders.customer_state) | set(orders.seller_state))
    orders["customer_state_code"] = orders.customer_state.map({s: i for i, s in enumerate(states)})
    orders["seller_state_code"] = orders.seller_state.map({s: i for i, s in enumerate(states)})
    events = orders[["seller_id", "route", "order_delivered_customer_date", "late", "delivery_days"]]
    orders = as_of_history(orders, events, "seller_id", "seller")
    orders = as_of_history(orders, events, "route", "route")
    return orders.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)


def split_orders(orders):
    """Assign splits by purchase date, keeping fit rows only if their outcome was known before the period they predict."""
    frame = orders.copy()
    frame["split"] = np.select([frame.order_purchase_timestamp >= TEST_START, frame.order_purchase_timestamp >= VALIDATION_START],
                               ["test", "validation"], "train")
    frame["known_before_validation"] = frame.order_delivered_customer_date < VALIDATION_START
    frame["known_before_test"] = frame.order_delivered_customer_date < TEST_START
    return frame
