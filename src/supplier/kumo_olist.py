"""Build Kumo Relational requests for Olist orders, using only the columns the user approved for upload.

IDs become plain numbers. Delivery dates, order status, reviews, payments, cities, zip codes, geolocation and the
customer's unique ID are never read into a request. The estimated delivery date is sent as days from purchase, because
a timestamp after the prediction time could be treated as future data.
"""
import math

import pandas as pd

from src.supplier.kumo_client import ANCHOR, ENTITY_REF, stamp

FORBIDDEN_TERMS = ["order_delivered", "order_status", "review", "payment", "customer_unique_id", "customer_city", "seller_city", "zip_code", "geolocation", "lat", "lng"]
SCHEMA = {
    "instance_table": {"primary_key": "instance_id", "columns": {
        "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, ENTITY_REF: {"dtype": "int64", "stype": "ID"},
        ANCHOR: {"dtype": "timestamp[us]", "stype": "timestamp", "nullable": False}, "TARGET": {"dtype": "bool", "stype": "categorical"}}},
    "related_tables": {
        "orders": {"primary_key": ["instance_id", "order_key"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "order_key": {"dtype": "int64", "nullable": False},
            "customer_key": {"dtype": "int64", "stype": "ID"}, "estimated_days": {"dtype": "float64", "stype": "numerical"}}},
        "customers": {"primary_key": ["instance_id", "customer_key"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "customer_key": {"dtype": "int64", "nullable": False},
            "customer_state": {"dtype": "string", "stype": "categorical"}}},
        "items": {"primary_key": ["instance_id", "__node_id"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "__node_id": {"dtype": "int64", "stype": "ID", "nullable": False},
            "order_key": {"dtype": "int64", "stype": "ID"}, "item_number": {"dtype": "float64", "stype": "numerical"}, "seller_key": {"dtype": "int64", "stype": "ID"},
            "product_key": {"dtype": "int64", "stype": "ID"}, "price": {"dtype": "float64", "stype": "numerical"}, "freight": {"dtype": "float64", "stype": "numerical"}}},
        "sellers": {"primary_key": ["instance_id", "seller_key"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "seller_key": {"dtype": "int64", "nullable": False},
            "seller_state": {"dtype": "string", "stype": "categorical"}}},
        "products": {"primary_key": ["instance_id", "product_key"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "product_key": {"dtype": "int64", "nullable": False},
            "category": {"dtype": "string", "stype": "categorical"}, "weight_g": {"dtype": "float64", "stype": "numerical"}, "length_cm": {"dtype": "float64", "stype": "numerical"},
            "height_cm": {"dtype": "float64", "stype": "numerical"}, "width_cm": {"dtype": "float64", "stype": "numerical"}, "photos": {"dtype": "float64", "stype": "numerical"}}}},
    "relationships": [
        {"source_columns": [ENTITY_REF], "target_columns": ["order_key"], "target_table": "orders"},
        {"source_table": "orders", "source_columns": ["customer_key"], "target_columns": ["customer_key"], "target_table": "customers"},
        {"source_table": "items", "source_columns": ["order_key"], "target_columns": ["order_key"], "target_table": "orders"},
        {"source_table": "items", "source_columns": ["seller_key"], "target_columns": ["seller_key"], "target_table": "sellers"},
        {"source_table": "items", "source_columns": ["product_key"], "target_columns": ["product_key"], "target_table": "products"}],
}
TASK = {"kind": "binary_classification", "target": {"column_name": "TARGET", "dtype": "bool", "classes": ["false", "true"], "positive_class": "true"},
        "entity_table_names": ["orders"], "anchor_time_column": ANCHOR}


def clean(value):
    """Convert missing values to JSON null."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


class OlistRequestBuilder:
    def __init__(self, tables):
        items = tables["items"]
        self.keys = {
            "order": {v: i for i, v in enumerate(sorted(tables["orders"].order_id))},
            "customer": {v: i for i, v in enumerate(sorted(tables["customers"].customer_id))},
            "seller": {v: i for i, v in enumerate(sorted(tables["sellers"].seller_id))},
            "product": {v: i for i, v in enumerate(sorted(tables["products"].product_id))},
        }
        self.items = {order: group for order, group in items[["order_id", "order_item_id", "seller_id", "product_id", "price", "freight_value"]].groupby("order_id")}
        self.customer_state = tables["customers"].set_index("customer_id").customer_state.to_dict()
        self.seller_state = tables["sellers"].set_index("seller_id").seller_state.to_dict()
        self.products = tables["products"].set_index("product_id")[["product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "product_photos_qty"]]

    def part(self, rows, first_instance, first_node, include_label):
        instance, orders, customers, items, sellers, products = [], [], [], [], [], []
        node = first_node
        for offset, row in enumerate(rows.itertuples(index=False)):
            instance_id = first_instance + offset
            order_key = self.keys["order"][row.order_id]
            customer_key = self.keys["customer"][row.customer_id]
            base = [instance_id, order_key, stamp(row.order_purchase_timestamp)]
            instance.append(base + [bool(row.late)] if include_label else base)
            orders.append([instance_id, order_key, customer_key, float(row.estimated_days)])
            customers.append([instance_id, customer_key, self.customer_state[row.customer_id]])
            seen_sellers, seen_products = set(), set()
            for item in self.items[row.order_id].itertuples(index=False):
                seller_key, product_key = self.keys["seller"][item.seller_id], self.keys["product"][item.product_id]
                items.append([instance_id, node, order_key, float(item.order_item_id), seller_key, product_key, float(item.price), float(item.freight_value)])
                node += 1
                if seller_key not in seen_sellers:
                    sellers.append([instance_id, seller_key, self.seller_state[item.seller_id]])
                    seen_sellers.add(seller_key)
                if product_key not in seen_products:
                    product = self.products.loc[item.product_id]
                    category = product.product_category_name
                    products.append([instance_id, product_key, None if pd.isna(category) else category] + [clean(float(product.iloc[i])) for i in range(1, 6)])
                    seen_products.add(product_key)
        columns = ["instance_id", ENTITY_REF, ANCHOR] + (["TARGET"] if include_label else [])
        return {"instance_table": {"format": "arrays", "columns": columns, "rows": instance}, "related_tables": {
            "orders": {"format": "arrays", "columns": list(SCHEMA["related_tables"]["orders"]["columns"]), "rows": orders},
            "customers": {"format": "arrays", "columns": list(SCHEMA["related_tables"]["customers"]["columns"]), "rows": customers},
            "items": {"format": "arrays", "columns": list(SCHEMA["related_tables"]["items"]["columns"]), "rows": items},
            "sellers": {"format": "arrays", "columns": list(SCHEMA["related_tables"]["sellers"]["columns"]), "rows": sellers},
            "products": {"format": "arrays", "columns": list(SCHEMA["related_tables"]["products"]["columns"]), "rows": products}}}, node

    def request(self, context_rows, predict_rows):
        context, node = self.part(context_rows, 0, 0, True)
        predict, _ = self.part(predict_rows, len(context_rows), node, False)
        return {"model": "kumo-relational", "task": TASK, "schema": SCHEMA, "context": context, "predict": predict,
                "output": {"fields": ["prediction", "probabilities"]}, "inference": {"run_mode": "fast", "use_prediction_time": False}}


def assert_allowed(payload_text, raw_ids):
    """Refuse a payload containing forbidden column names or raw Olist IDs."""
    lowered = payload_text.lower()
    for term in FORBIDDEN_TERMS:
        if f'"{term}' in lowered:
            raise ValueError(f"forbidden term in payload: {term}")
    for value in raw_ids:
        if value in payload_text:
            raise ValueError("raw ID in payload")
