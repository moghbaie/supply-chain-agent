"""Step 2 tests: keys, foreign keys, chronology, required fields and dataset limitations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
M5 = ROOT / "data/raw/m5"
OLIST = ROOT / "data/raw/olist"
ORDER_DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"]
ABSENT_FIELDS = ["inventory", "on_hand", "stock", "purchase_order", "receipt", "lead_time", "reorder"]
UNIQUE_KEYS = [
    (OLIST, "olist_customers_dataset", ["customer_id"]),
    (OLIST, "olist_orders_dataset", ["order_id"]),
    (OLIST, "olist_order_items_dataset", ["order_id", "order_item_id"]),
    (OLIST, "olist_order_payments_dataset", ["order_id", "payment_sequential"]),
    (OLIST, "olist_order_reviews_dataset", ["review_id", "order_id"]),
    (OLIST, "olist_products_dataset", ["product_id"]),
    (OLIST, "olist_sellers_dataset", ["seller_id"]),
    (OLIST, "product_category_name_translation", ["product_category_name"]),
    (M5, "calendar", ["d"]),
    (M5, "calendar", ["date"]),
    (M5, "sell_prices", ["store_id", "item_id", "wm_yr_wk"]),
    (M5, "sales_train_evaluation", ["id"]),
    (M5, "sales_train_evaluation", ["item_id", "store_id"]),
]
OLIST_FOREIGN_KEYS = [
    ("olist_orders_dataset", "customer_id", "olist_customers_dataset", "customer_id"),
    ("olist_order_items_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_items_dataset", "product_id", "olist_products_dataset", "product_id"),
    ("olist_order_items_dataset", "seller_id", "olist_sellers_dataset", "seller_id"),
    ("olist_order_payments_dataset", "order_id", "olist_orders_dataset", "order_id"),
    ("olist_order_reviews_dataset", "order_id", "olist_orders_dataset", "order_id"),
]


def read(folder, name, **kwargs):
    return pd.read_csv(folder / f"{name}.csv", low_memory=False, **kwargs)


@pytest.fixture(scope="module")
def orders():
    return read(OLIST, "olist_orders_dataset", parse_dates=ORDER_DATES)


@pytest.fixture(scope="module")
def calendar():
    return read(M5, "calendar", parse_dates=["date"])


@pytest.fixture(scope="module")
def evaluation():
    return read(M5, "sales_train_evaluation")


@pytest.fixture(scope="module")
def prices():
    return read(M5, "sell_prices")


@pytest.mark.parametrize("folder,name,key", UNIQUE_KEYS, ids=[f"{n}:{'+'.join(k)}" for _, n, k in UNIQUE_KEYS])
def test_unique_key(folder, name, key):
    assert read(folder, name, usecols=key).duplicated().sum() == 0


@pytest.mark.parametrize("child,child_col,parent,parent_col", OLIST_FOREIGN_KEYS, ids=[f"{c}.{k}" for c, k, _, _ in OLIST_FOREIGN_KEYS])
def test_olist_foreign_key(child, child_col, parent, parent_col):
    values = read(OLIST, child, usecols=[child_col])[child_col]
    assert values.isin(read(OLIST, parent, usecols=[parent_col])[parent_col]).all()


def test_m5_price_weeks_exist_in_calendar(prices, calendar):
    assert prices.wm_yr_wk.isin(calendar.wm_yr_wk).all()


def test_m5_every_series_has_price(evaluation, prices):
    priced = set(zip(prices.store_id, prices.item_id))
    assert set(zip(evaluation.store_id, evaluation.item_id)) <= priced


def test_m5_sales_days_exist_in_calendar(evaluation, calendar):
    days = [c for c in evaluation.columns if c.startswith("d_")]
    assert set(days) <= set(calendar.d)
    assert len(days) == 1941


def test_m5_calendar_is_consecutive(calendar):
    assert (calendar.date.diff().dropna() == pd.Timedelta(days=1)).all()


def test_m5_sales_complete_and_nonnegative(evaluation):
    days = [c for c in evaluation.columns if c.startswith("d_")]
    values = evaluation[days]
    assert values.notna().all().all()
    assert (values.to_numpy() >= 0).all()


def test_m5_required_fields_complete(evaluation, prices):
    assert evaluation[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].notna().all().all()
    assert prices.notna().all().all()
    assert (prices.sell_price > 0).all()


def test_m5_no_sales_before_first_price(evaluation, prices, calendar):
    days = [c for c in evaluation.columns if c.startswith("d_")]
    week = calendar.set_index("d").wm_yr_wk.reindex(days).to_numpy()
    first = prices.groupby(["store_id", "item_id"]).wm_yr_wk.min()
    first_price = pd.MultiIndex.from_frame(evaluation[["store_id", "item_id"]]).map(first).to_numpy()
    assert evaluation[days].to_numpy()[week[None, :] < first_price[:, None]].sum() == 0


def test_olist_required_fields_complete(orders):
    assert orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_estimated_delivery_date"]].notna().all().all()


def test_olist_strict_chronology(orders):
    purchase = orders.order_purchase_timestamp
    assert (orders.order_approved_at.dropna() >= purchase[orders.order_approved_at.notna()]).all()
    assert (orders.order_delivered_customer_date.dropna() >= purchase[orders.order_delivered_customer_date.notna()]).all()
    assert (orders.order_estimated_delivery_date >= purchase).all()


def test_olist_documented_chronology_exceptions(orders):
    observed = (
        int((orders.order_delivered_carrier_date < orders.order_purchase_timestamp).sum()),
        int((orders.order_delivered_carrier_date < orders.order_approved_at).sum()),
        int((orders.order_delivered_customer_date < orders.order_delivered_carrier_date).sum()),
    )
    assert observed == (166, 1359, 23)


def test_olist_documented_delivered_without_date(orders):
    assert int(((orders.order_status == "delivered") & orders.order_delivered_customer_date.isna()).sum()) == 8


def test_datasets_stay_separate(evaluation):
    sellers = read(OLIST, "olist_sellers_dataset", usecols=["seller_id"]).seller_id
    assert not set(evaluation.store_id) & set(sellers)
    assert not {p.name for p in M5.glob("*.csv")} & {p.name for p in OLIST.glob("*.csv")}


@pytest.mark.parametrize("folder", [M5, OLIST], ids=["m5", "olist"])
def test_no_real_inventory_or_inbound_fields(folder):
    columns = [c.lower() for path in folder.glob("*.csv") for c in pd.read_csv(path, nrows=0).columns]
    assert not [c for c in columns for field in ABSENT_FIELDS if field in c]


def test_manifest_checksums_match():
    manifest = json.loads((ROOT / "reports/step_02/manifest.json").read_text())
    for dataset, files in manifest.items():
        for name, info in files.items():
            assert hashlib.sha256((ROOT / "data/raw" / dataset / name).read_bytes()).hexdigest() == info["sha256"], name
