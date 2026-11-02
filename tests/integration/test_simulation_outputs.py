"""Step 3 integration tests on saved simulation outputs and the M5 inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.simulation.demand import closed_store_days, daily_prices, load_m5, select_series
from src.simulation.world import assign_suppliers, build_suppliers, disruption_masks, run_scenario

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulated"
CONFIG = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
SEED = CONFIG["seed"]


@pytest.fixture(scope="module")
def daily():
    return pd.read_parquet(SIM / "inventory_daily.parquet")


@pytest.fixture(scope="module")
def orders():
    return pd.read_parquet(SIM / "purchase_orders.parquet")


@pytest.fixture(scope="module")
def series():
    return pd.read_parquet(SIM / "series.parquet")


@pytest.fixture(scope="module")
def m5():
    return load_m5()


def test_conservation_every_row(daily):
    assert (daily.on_hand_start + daily.received - daily.sales == daily.on_hand_end).all()


def test_stock_carries_to_next_day(daily):
    grouped = daily.groupby(["scenario", "id"], sort=False)
    previous = grouped.on_hand_end.shift(1)
    known = previous.notna()
    assert (daily.on_hand_start[known] == previous[known]).all()


def test_physical_stock_nonnegative(daily):
    assert (daily[["on_hand_start", "on_hand_end", "on_order_end", "sales", "unmet", "received"]] >= 0).all().all()


def test_demand_split_into_sales_and_unmet(daily):
    assert (daily.demand == daily.sales + daily.unmet).all()


def test_rows_complete_and_unique(daily, series):
    days = daily.date.nunique()
    assert len(daily) == len(CONFIG["scenarios"]) * len(series) * days
    assert not daily.duplicated(["scenario", "id", "date"]).any()
    assert daily.drop(columns=["stockout_next_14d"]).notna().all().all()


def test_demand_equals_m5_sales_on_open_days(daily, series, m5):
    _, sales, _, day_columns, dates = m5
    sample = series.id.iloc[::25]
    source = sales.set_index("id")[day_columns].T.set_axis(dates)
    part = daily[(daily.scenario == "baseline") & daily.id.isin(sample) & daily.store_open]
    assert (part.demand.to_numpy() == [source.at[d, i] for d, i in zip(part.date, part.id)]).all()


def test_scenarios_share_the_same_demand(daily):
    wide = daily.pivot_table(index=["id", "date"], columns="scenario", values="demand", observed=True)
    assert wide.nunique(axis=1).eq(1).all()


def test_closed_days_have_no_activity(daily):
    closed = daily[~daily.store_open]
    assert len(closed) > 0
    assert closed[["demand", "sales", "received", "ordered"]].sum().sum() == 0


def test_purchase_order_chronology(orders):
    received = orders[orders.status == "received"]
    assert (received.receipt_date > received.order_date).all()
    assert (received.promised_date > received.order_date).all()
    span = (received.receipt_date - received.order_date).dt.days
    assert (span == received.actual_lead_days).all()
    assert (received.actual_lead_days >= received.promised_lead_days + received.delay_days).all()


def test_purchase_order_ids_unique_and_positive(orders):
    assert not orders.po_id.duplicated().any()
    assert (orders.quantity > 0).all()


def test_open_orders_reveal_no_outcome(orders):
    pending = orders[orders.status == "open"]
    assert len(pending) > 0
    assert pending.receipt_date.isna().all()
    assert pending.delay_days.isna().all()
    assert pending.actual_lead_days.isna().all()
    assert (pending.order_date > pd.Timestamp(CONFIG["dates"]["simulation_end"]) - pd.Timedelta(days=120)).all()


def test_receipts_reconcile_with_purchase_orders(daily, orders):
    in_stock = daily.groupby(["scenario", "id", "date"]).received.sum()
    in_stock = in_stock[in_stock > 0]
    received = orders[orders.status == "received"].groupby(["scenario", "id", "receipt_date"]).quantity.sum()
    received.index = received.index.set_names(in_stock.index.names)
    pd.testing.assert_series_equal(in_stock.sort_index(), received.sort_index(), check_names=False, check_dtype=False)


def test_orders_reconcile_with_daily_ordered(daily, orders):
    placed = daily.groupby("scenario").ordered.sum()
    assert placed.to_dict() == orders.groupby("scenario").quantity.sum().to_dict()


def test_splits_are_chronological_and_contiguous(daily):
    spans = daily.groupby("split").date.agg(["min", "max"])
    order = ["burn_in", "train", "validation", "test"]
    for earlier, later in zip(order, order[1:]):
        assert spans.loc[later, "min"] == spans.loc[earlier, "max"] + pd.Timedelta(days=1)
    assert spans.loc["test", "max"] == pd.Timestamp(CONFIG["dates"]["simulation_end"])


def test_label_matches_recomputed_unmet(daily):
    part = daily[(daily.scenario == "baseline")]
    one = part[part.id == part.id.iloc[0]].reset_index(drop=True)
    for row in [100, 400, 700]:
        expected = float(one.unmet.iloc[row + 1:row + 15].gt(0).any())
        assert one.stockout_next_14d.iloc[row] == expected


def test_label_missing_only_in_final_14_days(daily):
    last_labeled = pd.Timestamp(CONFIG["dates"]["simulation_end"]) - pd.Timedelta(days=14)
    assert daily[daily.date <= last_labeled].stockout_next_14d.notna().all()
    assert daily[daily.date > last_labeled].stockout_next_14d.isna().all()


def test_stockouts_exist_in_every_split_and_scenario(daily):
    active = daily[daily.split != "burn_in"]
    assert active.groupby(["scenario", "split"]).stockout_day.sum().gt(0).all()


def test_lean_policy_has_more_stockouts_and_less_stock(daily):
    means = daily[daily.split != "burn_in"].groupby("scenario")[["stockout_day", "on_hand_end"]].mean()
    assert means.loc["lean_policy", "stockout_day"] > means.loc["baseline", "stockout_day"]
    assert means.loc["lean_policy", "on_hand_end"] < means.loc["baseline", "on_hand_end"]


def test_disruption_raises_delays_only_for_chosen_suppliers_and_windows(orders):
    suppliers = pd.read_parquet(SIM / "suppliers.parquet")
    dates = pd.date_range(CONFIG["dates"]["simulation_start"], CONFIG["dates"]["simulation_end"])
    scenario = CONFIG["scenarios"]["supplier_disruption"]
    masks = disruption_masks(scenario, suppliers, dates, SEED)
    chosen = sorted(masks)
    window_dates = set(dates[masks[chosen[0]]])
    part = orders[(orders.scenario == "supplier_disruption") & (orders.status == "received")]
    hit = part.supplier_id.isin(chosen) & part.order_date.isin(window_dates)
    assert (part[hit].delay_days > 0).mean() == pytest.approx(scenario["delay_prob"], abs=0.05)
    for supplier_id, group in part[~hit].groupby("supplier_id"):
        expected = suppliers.set_index("supplier_id").delay_prob[supplier_id]
        assert (group.delay_days > 0).mean() == pytest.approx(expected, abs=0.04)


def test_each_item_has_one_supplier(series):
    assert series.groupby("item_id").supplier_id.nunique().eq(1).all()


def test_selected_series_meet_rules(series):
    selection = CONFIG["selection"]
    assert len(series) == selection["series_count"]
    assert not series.id.duplicated().any()
    assert (series.longest_zero_run_to_check_end < selection["max_zero_gap_days"]).all()
    assert (series.first_sale <= pd.Timestamp(selection["latest_first_sale"])).all()


def test_selection_ignores_data_after_check_end(m5):
    _, sales, _, day_columns, dates = m5
    changed = sales.copy()
    after = [c for c, d in zip(day_columns, dates) if d > pd.Timestamp(CONFIG["selection"]["gap_check_end"])]
    changed[after] = 0
    original, _, _ = select_series(sales, day_columns, dates, CONFIG, SEED)
    masked, _, _ = select_series(changed, day_columns, dates, CONFIG, SEED)
    assert original.id.tolist() == masked.id.tolist()


def test_small_run_is_reproducible_and_independent_of_series_order(series, m5):
    calendar, sales, prices, day_columns, dates = m5
    subset = series.iloc[:12].reset_index(drop=True)
    suppliers = build_suppliers(CONFIG, SEED)
    closed = closed_store_days(sales, day_columns)

    def simulate(frame):
        demand = sales.set_index("id").loc[frame.id, day_columns].to_numpy(dtype=np.int64)
        price, _ = daily_prices(frame, prices, calendar, dates)
        return run_scenario("baseline", CONFIG, frame, suppliers, dates, demand, closed, price, SEED)

    first, first_orders, *_ = simulate(subset)
    second, second_orders, *_ = simulate(subset)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_orders, second_orders)

    flipped, *_ = simulate(subset.iloc[::-1].reset_index(drop=True))
    key = ["id", "date"]
    pd.testing.assert_frame_equal(first.sort_values(key).reset_index(drop=True), flipped.sort_values(key).reset_index(drop=True))

    saved = pd.read_parquet(SIM / "inventory_daily.parquet", filters=[("scenario", "==", "baseline")])
    saved = saved[saved.id.isin(subset.id)]
    assert (saved.sort_values(key).on_hand_end.to_numpy() == first.sort_values(key).on_hand_end.to_numpy()).all()


def test_supplier_assignment_reproducible(series):
    suppliers = build_suppliers(CONFIG, SEED)
    pd.testing.assert_frame_equal(suppliers, pd.read_parquet(SIM / "suppliers.parquet"), check_dtype=False)
    again = assign_suppliers(series.drop(columns=["supplier_id"]), suppliers, SEED)
    assert again.supplier_id.tolist() == series.supplier_id.tolist()
