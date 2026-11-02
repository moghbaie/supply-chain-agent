"""Print Olist chronology violations, delivery lateness and monthly coverage, and save them."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RAW = Path("data/raw/olist")
DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"]
orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=DATES)
items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
reviews = pd.read_csv(RAW / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"])
result = {}

print("order_status:", orders.order_status.value_counts().to_dict())
purchase, approved, carrier, customer, estimated = (orders[c] for c in DATES)
violations = {
    "approved_before_purchase": int((approved < purchase).sum()),
    "carrier_before_purchase": int((carrier < purchase).sum()),
    "carrier_before_approved": int((carrier < approved).sum()),
    "customer_before_carrier": int((customer < carrier).sum()),
    "customer_before_purchase": int((customer < purchase).sum()),
    "estimated_before_purchase": int((estimated < purchase).sum()),
}
result["chronology_violations"] = violations
print("chronology violations:", violations)

delivered = orders[(orders.order_status == "delivered") & customer.notna()].copy()
delivered["late"] = delivered.order_delivered_customer_date.dt.normalize() > delivered.order_estimated_delivery_date
delivered["days"] = (delivered.order_delivered_customer_date - delivered.order_purchase_timestamp).dt.total_seconds() / 86400
result["delivered_rows"] = len(delivered)
result["late_rate_pct"] = round(100 * delivered.late.mean(), 2)
result["delivery_days"] = delivered.days.describe().round(2).to_dict()
print(f"delivered with date: {len(delivered):,} | late against estimate: {delivered.late.sum():,} ({result['late_rate_pct']}%)")
print("delivery days:", result["delivery_days"])
print("delivered status but no customer date:", int(((orders.order_status == "delivered") & customer.isna()).sum()))

monthly = delivered.groupby(delivered.order_purchase_timestamp.dt.to_period("M")).agg(orders=("late", "size"), late_pct=("late", lambda s: round(100 * s.mean(), 1)))
all_monthly = orders.groupby(purchase.dt.to_period("M")).size()
monthly = monthly.reindex(all_monthly.index).assign(all_orders=all_monthly)
print(monthly.to_string())
monthly.to_csv("reports/step_02/olist/monthly_delivery.csv")

late_limit = items[items.shipping_limit_date > orders.order_purchase_timestamp.max()]
result["shipping_limit_after_last_purchase"] = len(late_limit)
print(f"\nshipping_limit_date after last purchase: {len(late_limit)} rows")
print(late_limit.merge(orders[["order_id", "order_purchase_timestamp", "order_status"]])[["order_id", "shipping_limit_date", "order_purchase_timestamp", "order_status"]].sort_values("shipping_limit_date").tail(5).to_string(index=False))

merged = reviews.merge(orders[["order_id", "order_delivered_customer_date"]])
print(f"\nreviews created before delivery date: {(merged.review_creation_date < merged.order_delivered_customer_date.dt.normalize()).sum():,} of {len(merged):,}")
print(f"review_id repeated across orders: {reviews.review_id.duplicated().sum():,} | orders with multiple reviews: {(reviews.groupby('order_id').size() > 1).sum():,}")
per_order_sellers = items.groupby("order_id").seller_id.nunique()
result["orders_multi_seller"] = int((per_order_sellers > 1).sum())
print(f"orders with more than one seller: {result['orders_multi_seller']:,} | orders without items: {(~orders.order_id.isin(items.order_id)).sum():,}")
Path("reports/step_02/olist/delivery_checks.json").write_text(json.dumps(result, indent=2, default=str))
