"""Step 6 tests: relationship integrity, point-in-time joins, unseen entities and leakage. No test calls the API."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stockout.evaluation import score_probabilities
from src.supplier import inbound, olist
from src.supplier.kumo_client import ANCHOR, HISTORY_ROWS, HistoryIndex, build_request, positive_probability, stamp, tables_for

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulated"
FORBIDDEN_OLIST = {"order_delivered_customer_date", "order_status", "delivery_days", "late", "review_score", "payment_value", "order_approved_at"}
FORBIDDEN_INBOUND = {"delay_days", "receipt_date", "actual_lead_days", "delay_prob", "tier", "scenario", "status", "delayed"}


@pytest.fixture(scope="module")
def tables():
    return olist.load_tables()


@pytest.fixture(scope="module")
def orders(tables):
    return olist.split_orders(olist.build_order_features(tables))


@pytest.fixture(scope="module")
def purchase():
    return pd.read_parquet(ROOT / "data/processed/delay/inbound_features.parquet")


@pytest.fixture(scope="module")
def all_orders():
    return pd.read_parquet(SIM / "purchase_orders.parquet")


@pytest.fixture(scope="module")
def supplier_numbers(all_orders):
    return {s: i for i, s in enumerate(sorted(all_orders.supplier_id.unique()))}


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(ROOT / "data/processed/delay/inbound_predictions.parquet")


@pytest.mark.parametrize("child,column,parent,key", [
    ("orders", "customer_id", "customers", "customer_id"), ("items", "order_id", "orders", "order_id"),
    ("items", "seller_id", "sellers", "seller_id"), ("items", "product_id", "products", "product_id")])
def test_olist_relationships_are_intact(tables, child, column, parent, key):
    assert tables[child][column].isin(tables[parent][key]).all()


def test_olist_one_row_per_order(orders):
    assert not orders.order_id.duplicated().any()


def test_inbound_orders_link_to_known_suppliers_and_items(purchase):
    assert purchase.supplier_id.isin(pd.read_parquet(SIM / "suppliers.parquet").supplier_id).all()
    assert purchase.id.isin(pd.read_parquet(SIM / "series.parquet").id).all()


def test_olist_label_matches_dates(orders):
    expected = orders.order_delivered_customer_date.dt.normalize() > orders.order_estimated_delivery_date
    assert (orders.late == expected.astype(int)).all()


def test_olist_features_exclude_outcomes():
    assert not FORBIDDEN_OLIST & set(olist.FEATURES)


def test_inbound_features_exclude_outcomes_and_hidden_settings():
    assert not FORBIDDEN_INBOUND & set(inbound.FEATURES)


def test_as_of_history_uses_only_earlier_deliveries():
    events = pd.DataFrame({"seller_id": ["s", "s"], "order_delivered_customer_date": pd.to_datetime(["2018-01-10", "2018-02-10"]), "late": [1, 0], "delivery_days": [5.0, 7.0]})
    rows = pd.DataFrame({"seller_id": ["s", "s", "s", "new"], "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-10", "2018-02-20", "2018-03-01"])})
    result = olist.as_of_history(rows, events, "seller_id", "seller").sort_values("order_purchase_timestamp")
    assert result.seller_orders_seen.tolist() == [0, 0, 2, 0]
    assert np.isnan(result.seller_late_rate.iloc[0])
    assert result.seller_late_rate.iloc[2] == 0.5
    assert result.seller_mean_days.iloc[2] == 6.0
    assert np.isnan(result.seller_late_rate.iloc[3])


def test_seller_history_recomputes_from_earlier_orders(orders):
    sample = orders[orders.seller_orders_seen > 5].iloc[::4000]
    assert len(sample) > 3
    for row in sample.itertuples():
        earlier = orders[(orders.seller_id == row.seller_id) & (orders.order_delivered_customer_date < row.order_purchase_timestamp)]
        assert len(earlier) == row.seller_orders_seen
        assert earlier.late.mean() == pytest.approx(row.seller_late_rate)


def test_later_deliveries_do_not_change_olist_features(tables):
    original = olist.build_order_features(tables)
    cutoff = pd.Timestamp("2018-01-01")
    changed = {**tables, "orders": tables["orders"].copy()}
    later = changed["orders"].order_delivered_customer_date >= cutoff
    changed["orders"].loc[later, "order_delivered_customer_date"] = changed["orders"].loc[later, "order_delivered_customer_date"] + pd.Timedelta(days=200)
    altered = olist.build_order_features(changed)
    early = original.order_purchase_timestamp < cutoff
    first = original[early].sort_values("order_id")[olist.FEATURES].reset_index(drop=True)
    second = altered[altered.order_id.isin(original[early].order_id)].sort_values("order_id")[olist.FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_olist_splits_are_chronological(orders):
    spans = orders.groupby("split").order_purchase_timestamp.agg(["min", "max"])
    assert spans.loc["train", "max"] < olist.VALIDATION_START <= spans.loc["validation", "min"]
    assert spans.loc["validation", "max"] < olist.TEST_START <= spans.loc["test", "min"]


def test_olist_fit_rows_have_known_outcomes(orders):
    fit = orders[orders.split.eq("train") & orders.known_before_validation]
    assert (fit.order_delivered_customer_date < olist.VALIDATION_START).all()
    assert len(fit) < (orders.split == "train").sum()


def test_olist_new_sellers_have_missing_rates_not_guesses(orders):
    new = orders[orders.seller_orders_seen == 0]
    assert len(new) > 0
    assert new.seller_late_rate.isna().all()


def test_inbound_history_ignores_receipts_after_order_date(all_orders, purchase):
    baseline = all_orders[all_orders.scenario == "baseline"]
    row = purchase[(purchase.scenario == "baseline") & (purchase.split == "test")].iloc[100]
    first = inbound.recent_supplier(baseline[baseline.order_date < row.order_date], row.order_date)
    changed = baseline.copy()
    later = changed.receipt_date >= row.order_date
    changed.loc[later, "delay_days"] = 99
    second = inbound.recent_supplier(changed[changed.order_date < row.order_date], row.order_date)
    pd.testing.assert_frame_equal(first, second)


def test_inbound_fit_rows_known_before_their_period(purchase):
    assert (purchase[purchase.split.eq("train") & purchase.known_before_validation].receipt_date <= pd.Timestamp("2014-12-31")).all()
    assert (purchase[purchase.known_before_test].receipt_date <= pd.Timestamp("2015-12-31")).all()


def test_request_refuses_delivery_at_or_after_order_date(supplier_numbers):
    row = pd.DataFrame({"scenario": ["x"], "supplier_id": ["SUP_00"], "order_date": pd.to_datetime(["2015-01-05"]), "delayed": [0], "quantity": [1], "promised_lead_days": [5], "cat_code": [0]})

    class Leaky:
        def before(self, scenario, supplier_id, date):
            return pd.DataFrame({"receipt_date": pd.to_datetime(["2015-01-05"]), "quantity": [1], "delay_days": [0]})

    with pytest.raises(ValueError):
        tables_for(row, Leaky(), 0, 0, True, supplier_numbers)


def test_history_index_returns_only_recent_earlier_deliveries(all_orders):
    index = HistoryIndex(all_orders)
    date = pd.Timestamp("2015-06-01")
    past = index.before("baseline", "SUP_00", date)
    assert 0 < len(past) <= HISTORY_ROWS
    assert (past.receipt_date < date).all()
    assert past.receipt_date.is_monotonic_increasing


def test_real_request_is_point_in_time_and_label_free_for_predictions(purchase, all_orders, supplier_numbers):
    index = HistoryIndex(all_orders)
    context = purchase[purchase.split.eq("train") & purchase.known_before_validation].sample(50, random_state=1)
    rows = purchase[purchase.split == "validation"].sample(20, random_state=1)
    body = build_request(context, rows, index, supplier_numbers)
    assert "TARGET" not in body["predict"]["instance_table"]["columns"]
    assert "TARGET" in body["context"]["instance_table"]["columns"]
    for part in ("context", "predict"):
        instance = {r[0]: r[2] for r in body[part]["instance_table"]["rows"]}
        columns = body[part]["related_tables"]["past_deliveries"]["columns"]
        received, owner = columns.index("received_at"), columns.index("instance_id")
        deliveries = body[part]["related_tables"]["past_deliveries"]["rows"]
        assert deliveries
        assert all(row[received] < instance[row[owner]] for row in deliveries)


def test_request_ids_do_not_overlap_between_context_and_predictions(purchase, all_orders, supplier_numbers):
    index = HistoryIndex(all_orders)
    body = build_request(purchase.iloc[:10], purchase.iloc[10:15], index, supplier_numbers)
    first = {r[0] for r in body["context"]["instance_table"]["rows"]}
    second = {r[0] for r in body["predict"]["instance_table"]["rows"]}
    assert not first & second
    nodes = [r[1] for part in ("context", "predict") for r in body[part]["related_tables"]["past_deliveries"]["rows"]]
    assert len(nodes) == len(set(nodes))


def test_request_contains_no_names_or_hidden_settings(purchase, all_orders, supplier_numbers):
    index = HistoryIndex(all_orders)
    text = json.dumps(build_request(purchase.iloc[:30], purchase.iloc[30:40], index, supplier_numbers))
    for term in ["FOODS", "HOBBIES", "HOUSEHOLD", "_evaluation", "SUP_", "baseline", "lean_policy", "supplier_disruption", "olist", "tier", "delay_prob", "price"]:
        assert term not in text, term


def test_unseen_supplier_order_is_sent_with_no_history(supplier_numbers, all_orders):
    index = HistoryIndex(all_orders)
    row = pd.DataFrame({"scenario": ["baseline"], "supplier_id": ["SUP_00"], "order_date": pd.to_datetime(["2013-01-01"]), "delayed": [0], "quantity": [5], "promised_lead_days": [18], "cat_code": [1]})
    part, _ = tables_for(row, index, 0, 0, False, supplier_numbers)
    assert part["related_tables"]["past_deliveries"]["rows"] == []
    assert len(part["instance_table"]["rows"]) == 1


def test_supplier_never_seen_in_any_scenario_does_not_crash(supplier_numbers, all_orders):
    index = HistoryIndex(all_orders)
    assert index.before("baseline", "SUP_99", pd.Timestamp("2015-01-01")) is None


def test_kumo_script_never_reads_olist():
    source = (ROOT / "scripts/run_kumo_inbound.py").read_text()
    code = "\n".join(line for line in source.splitlines() if '"olist"' not in line and "Olist data is never" not in line)
    assert "olist" not in code.lower()
    assert "olist" not in (ROOT / "src/supplier/kumo_client.py").read_text().lower()


def test_customer_and_supplier_results_are_kept_separate():
    assert set(pd.read_csv(ROOT / "reports/step_06/baseline_metrics.csv").task) == {"olist_validation", "olist_test", "inbound_validation", "inbound_test"}
    assert "kumo" not in set(pd.read_csv(ROOT / "reports/step_06/baseline_metrics.csv").iloc[:, 0])
    assert (ROOT / "data/processed/delay/olist_labels.parquet").exists()
    assert "order_id" not in pd.read_parquet(ROOT / "data/processed/delay/inbound_predictions.parquet").columns


def test_probability_reader_handles_documented_shape():
    assert positive_probability({"probabilities": {"false": 0.75, "true": 0.25}}) == 0.25


def test_timestamp_format_matches_sample():
    assert stamp("2015-01-05") == "2015-01-05T00:00:00.000000Z"
    assert ANCHOR == "__kumo_anchor_time"


def test_saved_kumo_predictions_are_complete_and_valid(predictions):
    assert predictions.groupby("split").size().to_dict() == {"test": 3000, "validation": 2000}
    assert predictions.kumo.between(0, 1).all()
    assert not predictions.po_id.duplicated().any()


def test_saved_metrics_recompute(predictions):
    metrics = pd.read_csv(ROOT / "reports/step_06/kumo_metrics.csv", index_col=0)
    test = predictions[predictions.split == "test"]
    saved = metrics[metrics.split == "test"].loc["kumo"]
    result = score_probabilities(test.delayed, test.kumo)
    assert result["pr_auc"] == pytest.approx(saved.pr_auc, abs=1e-4)
    assert result["brier"] == pytest.approx(saved.brier, abs=1e-4)


def test_kumo_beats_constant_rate_on_test(predictions):
    test = predictions[predictions.split == "test"]
    assert score_probabilities(test.delayed, test.kumo)["pr_auc"] > score_probabilities(test.delayed, test.constant)["pr_auc"] + 0.1


def test_every_batch_was_saved():
    saved = {p.stem for p in (ROOT / "data/processed/delay/kumo_batches").glob("*.json")}
    assert saved == {f"validation_{i:03d}" for i in range(4)} | {f"test_{i:03d}" for i in range(6)}
