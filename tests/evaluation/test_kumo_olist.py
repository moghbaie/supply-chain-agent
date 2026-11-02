"""Step 6 tests for the Olist Kumo request: approved columns only, no raw IDs, no outcomes. No test calls the API."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.stockout.evaluation import score_probabilities
from src.supplier import olist
from src.supplier.kumo_olist import FORBIDDEN_TERMS, SCHEMA, OlistRequestBuilder, assert_allowed

ROOT = Path(__file__).resolve().parents[2]
APPROVED = {
    "orders": {"instance_id", "order_key", "customer_key", "estimated_days"},
    "customers": {"instance_id", "customer_key", "customer_state"},
    "items": {"instance_id", "__node_id", "order_key", "item_number", "seller_key", "product_key", "price", "freight"},
    "sellers": {"instance_id", "seller_key", "seller_state"},
    "products": {"instance_id", "product_key", "category", "weight_g", "length_cm", "height_cm", "width_cm", "photos"},
}


@pytest.fixture(scope="module")
def tables():
    return olist.load_tables()


@pytest.fixture(scope="module")
def orders(tables):
    return olist.split_orders(olist.build_order_features(tables))


@pytest.fixture(scope="module")
def body(tables, orders):
    context = orders[orders.split.eq("train") & orders.known_before_validation].sample(100, random_state=1)
    rows = orders[orders.split == "validation"].sample(30, random_state=1)
    return OlistRequestBuilder(tables).request(context, rows), context, rows


def test_only_approved_columns_are_sent(body):
    request = body[0]
    for part in ("context", "predict"):
        assert {name: set(table["columns"]) for name, table in request[part]["related_tables"].items()} == APPROVED
    assert {name: set(table["columns"]) for name, table in SCHEMA["related_tables"].items()} == APPROVED


def test_predictions_carry_no_label(body):
    assert "TARGET" not in body[0]["predict"]["instance_table"]["columns"]
    assert "TARGET" in body[0]["context"]["instance_table"]["columns"]


def test_no_raw_ids_or_forbidden_terms(body):
    request, context, rows = body
    text = json.dumps(request)
    assert_allowed(text, list(context.order_id) + list(context.customer_id) + list(context.seller_id) + list(rows.order_id) + list(rows.customer_id) + list(rows.seller_id))


def test_guard_rejects_forbidden_payloads():
    with pytest.raises(ValueError):
        assert_allowed('{"order_delivered_customer_date": 1}', [])
    with pytest.raises(ValueError):
        assert_allowed('{"x": "abc123"}', ["abc123"])
    assert "order_status" in FORBIDDEN_TERMS


def test_estimated_date_is_sent_as_days_not_a_future_timestamp(body):
    columns = [c for table in body[0]["context"]["related_tables"].values() for c in table["columns"]]
    assert "estimated_days" in columns
    assert "order_estimated_delivery_date" not in columns
    timestamp_columns = [c for table in SCHEMA["related_tables"].values() for c, spec in table["columns"].items() if spec.get("stype") == "timestamp"]
    assert timestamp_columns == []


def test_context_outcomes_known_before_predictions(body):
    _, context, rows = body
    assert context.order_delivered_customer_date.max() < rows.order_purchase_timestamp.min()


def test_every_instance_has_its_related_rows(body):
    request = body[0]
    for part in ("context", "predict"):
        ids = {row[0] for row in request[part]["instance_table"]["rows"]}
        for name, table in request[part]["related_tables"].items():
            assert {row[0] for row in table["rows"]} == ids, name


def test_order_key_links_instance_to_orders_and_items(body):
    part = body[0]["context"]
    expected = {row[0]: row[1] for row in part["instance_table"]["rows"]}
    orders_rows = part["related_tables"]["orders"]["rows"]
    assert all(expected[row[0]] == row[1] for row in orders_rows)
    assert all(expected[row[0]] == row[2] for row in part["related_tables"]["items"]["rows"])


def test_instance_ids_do_not_overlap(body):
    request = body[0]
    assert not {r[0] for r in request["context"]["instance_table"]["rows"]} & {r[0] for r in request["predict"]["instance_table"]["rows"]}
    nodes = [r[1] for part in ("context", "predict") for r in request[part]["related_tables"]["items"]["rows"]]
    assert len(nodes) == len(set(nodes))


def test_missing_product_details_become_null(tables, orders):
    builder = OlistRequestBuilder(tables)
    missing = tables["products"][tables["products"].product_category_name.isna()].product_id.iloc[0]
    order_id = tables["items"][tables["items"].product_id == missing].order_id.iloc[0]
    rows = orders[orders.order_id == order_id]
    if rows.empty:
        pytest.skip("that order is outside the study period")
    part, _ = builder.part(rows, 0, 0, False)
    assert any(row[2] is None for row in part["related_tables"]["products"]["rows"])
    json.dumps(part, allow_nan=False)


def test_saved_predictions_complete_and_recompute():
    frame = pd.read_parquet(ROOT / "data/processed/delay/olist_predictions.parquet")
    assert frame.groupby("split").size().to_dict() == {"test": 5000, "validation": 5000}
    assert frame.kumo.between(0, 1).all()
    assert not frame.duplicated(["order_id"]).any()
    saved = pd.read_csv(ROOT / "reports/step_06/kumo_olist_metrics.csv", index_col=0)
    test = frame[frame.split == "test"]
    assert score_probabilities(test.late, test.kumo)["pr_auc"] == pytest.approx(saved[saved.split == "test"].loc["kumo"].pr_auc, abs=1e-4)


def test_all_batches_saved():
    saved = {p.stem for p in (ROOT / "data/processed/delay/kumo_olist_batches").glob("*.json")}
    assert saved == {f"{split}_{i:03d}" for split in ("validation", "test") for i in range(10)}


def test_olist_and_inbound_results_stay_separate():
    olist_frame = pd.read_parquet(ROOT / "data/processed/delay/olist_predictions.parquet")
    inbound_frame = pd.read_parquet(ROOT / "data/processed/delay/inbound_predictions.parquet")
    assert "late" in olist_frame.columns and "late" not in inbound_frame.columns
    assert "delayed" in inbound_frame.columns and "delayed" not in olist_frame.columns
