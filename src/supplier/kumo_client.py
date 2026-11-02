"""Build and send Kumo Relational requests for synthetic purchase orders only.

The service's docs do not say it filters related rows by time, so each order gets its own copy of the supplier
history known before its date, and a request with any later row is refused here.
"""
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

URL = "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions"
HISTORY_ROWS = 15
ANCHOR = "__kumo_anchor_time"
ENTITY_REF = "__kumo_entity_ref_0"
ALLOWED_INPUT_COLUMNS = {"quantity", "promised_lead_days", "cat_code", "order_date", "delayed", "supplier_id", "receipt_date", "delay_days"}
SCHEMA = {
    "instance_table": {"primary_key": "instance_id", "columns": {
        "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, ENTITY_REF: {"dtype": "int64", "stype": "ID"},
        ANCHOR: {"dtype": "timestamp[us]", "stype": "timestamp", "nullable": False}, "TARGET": {"dtype": "bool", "stype": "categorical"}}},
    "related_tables": {
        "purchase_orders": {"primary_key": ["instance_id", "order_key"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "order_key": {"dtype": "int64", "nullable": False},
            "supplier_key": {"dtype": "int64", "stype": "categorical"}, "quantity": {"dtype": "float64", "stype": "numerical"},
            "promised_lead_days": {"dtype": "float64", "stype": "numerical"}, "category": {"dtype": "int64", "stype": "categorical"}}},
        "past_deliveries": {"primary_key": ["instance_id", "__node_id"], "columns": {
            "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False}, "__node_id": {"dtype": "int64", "stype": "ID", "nullable": False},
            "order_key": {"dtype": "int64", "stype": "ID"}, "received_at": {"dtype": "timestamp[us]", "stype": "timestamp"},
            "quantity": {"dtype": "float64", "stype": "numerical"}, "delay_days": {"dtype": "float64", "stype": "numerical"}}}},
    "relationships": [{"source_columns": [ENTITY_REF], "target_columns": ["order_key"], "target_table": "purchase_orders"},
                      {"source_table": "past_deliveries", "source_columns": ["order_key"], "target_columns": ["order_key"], "target_table": "purchase_orders"}],
}
TASK = {"kind": "binary_classification", "target": {"column_name": "TARGET", "dtype": "bool", "classes": ["false", "true"], "positive_class": "true"},
        "entity_table_names": ["purchase_orders"], "anchor_time_column": ANCHOR}


def stamp(value):
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


class HistoryIndex:
    """Find each supplier's most recent deliveries received before a date, within one scenario."""

    def __init__(self, orders):
        received = orders[orders.status == "received"].sort_values("receipt_date")
        self.by_key = {key: group for key, group in received.groupby(["scenario", "supplier_id"])}

    def before(self, scenario, supplier_id, date):
        group = self.by_key.get((scenario, supplier_id))
        if group is None:
            return group
        position = group.receipt_date.searchsorted(date, side="left")
        return group.iloc[max(0, position - HISTORY_ROWS):position]


def tables_for(rows, history, first_instance, first_node, include_label, supplier_numbers):
    """Build one request part, refusing any delivery not strictly before its order date."""
    instance, orders, deliveries = [], [], []
    node = first_node
    for offset, row in enumerate(rows.itertuples(index=False)):
        instance_id = first_instance + offset
        order_key = instance_id
        base = [instance_id, order_key, stamp(row.order_date)]
        instance.append(base + [bool(row.delayed)] if include_label else base)
        orders.append([instance_id, order_key, supplier_numbers[row.supplier_id], float(row.quantity), float(row.promised_lead_days), int(row.cat_code)])
        past = history.before(row.scenario, row.supplier_id, row.order_date)
        if past is None:
            continue
        for item in past.itertuples(index=False):
            if not item.receipt_date < row.order_date:
                raise ValueError("delivery at or after the order date")
            deliveries.append([instance_id, node, order_key, stamp(item.receipt_date), float(item.quantity), float(item.delay_days)])
            node += 1
    columns = ["instance_id", ENTITY_REF, ANCHOR] + (["TARGET"] if include_label else [])
    part = {"instance_table": {"format": "arrays", "columns": columns, "rows": instance}, "related_tables": {
        "purchase_orders": {"format": "arrays", "columns": ["instance_id", "order_key", "supplier_key", "quantity", "promised_lead_days", "category"], "rows": orders},
        "past_deliveries": {"format": "arrays", "columns": ["instance_id", "__node_id", "order_key", "received_at", "quantity", "delay_days"], "rows": deliveries}}}
    return part, node


def build_request(context_rows, predict_rows, history, supplier_numbers):
    context, node = tables_for(context_rows, history, 0, 0, True, supplier_numbers)
    predict, _ = tables_for(predict_rows, history, len(context_rows), node, False, supplier_numbers)
    return {"model": "kumo-relational", "task": TASK, "schema": SCHEMA, "context": context, "predict": predict,
            "output": {"fields": ["prediction", "probabilities"]}, "inference": {"run_mode": "fast", "use_prediction_time": False}}


def send(body, key, attempts=4):
    """Post a request and return (status, parsed or text), retrying only on rate limits and server errors."""
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            text = error.read(800).decode(errors="replace")
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                return error.code, text
            time.sleep(10 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                return None, type(error).__name__
            time.sleep(10 * 2 ** attempt)


def positive_probability(item):
    """Read the positive-class probability from one prediction item."""
    probabilities = item["probabilities"]
    if isinstance(probabilities, dict):
        return float(probabilities["true"])
    return float(np.asarray(probabilities, dtype=float)[-1])
