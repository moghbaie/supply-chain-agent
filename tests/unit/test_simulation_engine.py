"""Step 3 engine tests: conservation, ordering, labels, leakage and edge cases on small known inputs."""
import numpy as np
import pandas as pd
import pytest

from src.simulation.engine import Policy, draw_delay, purchase_orders_as_of, simulate_series, stock_levels, stockout_within

POLICY = Policy(z=1.28, review_weekday=0, cover_days=14, window_days=56)
MONDAY = "2014-01-06"


def run(demand, is_open=None, delay=0.0, mean_delay=4.0, lead=5, pack=1, history_rate=2, seed=0, start=MONDAY, policy=POLICY):
    demand = np.asarray(demand, dtype=np.int64)
    days = len(demand)
    dates = pd.date_range(start, periods=days)
    is_open = np.ones(days, dtype=bool) if is_open is None else np.asarray(is_open)
    history = np.full(56, history_rate, dtype=np.int64)
    return simulate_series(dates, demand, is_open, history, np.ones(56, dtype=bool), lead, np.full(days, delay),
                           np.full(days, mean_delay), pack, policy, np.random.default_rng(seed))


def conserved(arrays):
    return (arrays["on_hand_start"] + arrays["received"] - arrays["sales"] == arrays["on_hand_end"]).all()


@pytest.mark.parametrize("name,demand", [
    ("zero_demand", [0] * 60),
    ("steady", [2] * 60),
    ("spike_beyond_stock", [1] * 10 + [100000] + [1] * 49),
    ("intermittent", [0, 0, 9, 0, 0, 0, 7] * 9),
], ids=lambda value: value if isinstance(value, str) else "")
def test_conservation_and_nonnegative_stock(name, demand):
    arrays, _ = run(demand, delay=0.4)
    assert conserved(arrays)
    assert (arrays["on_hand_end"] >= 0).all()
    assert (arrays["demand"] == arrays["sales"] + arrays["unmet"]).all()


def test_demand_above_stock_is_lost_not_backordered():
    arrays, _ = run([0, 100000, 0, 0])
    stock = arrays["on_hand_start"][1]
    assert arrays["sales"][1] == stock
    assert arrays["unmet"][1] == 100000 - stock
    assert arrays["on_hand_end"][1] == 0
    assert arrays["unmet"][2] == 0


def test_zero_demand_creates_no_stockout():
    arrays, _ = run([0] * 60)
    assert arrays["unmet"].sum() == 0
    assert arrays["sales"].sum() == 0


def test_on_order_matches_orders_placed_minus_received():
    arrays, orders = run([3] * 200, delay=0.3)
    assert (np.cumsum(arrays["ordered"]) - np.cumsum(arrays["received"]) == arrays["on_order_end"]).all()
    assert arrays["ordered"].sum() == sum(o["quantity"] for o in orders)


def test_orders_only_on_review_day():
    arrays, _ = run([3] * 100)
    weekdays = pd.date_range(MONDAY, periods=100).weekday
    assert (weekdays[arrays["ordered"] > 0] == POLICY.review_weekday).all()
    assert arrays["ordered"].sum() > 0


def test_order_quantity_is_rounded_to_pack():
    _, orders = run([3] * 100, pack=6)
    assert orders and all(o["quantity"] % 6 == 0 and o["quantity"] > 0 for o in orders)


def test_receipt_arrives_on_promised_day_without_delay():
    arrays, orders = run([3] * 60, lead=5, delay=0.0)
    assert orders
    for order in orders:
        if order["receipt_index"] is not None:
            assert order["receipt_index"] == order["order_index"] + 5


def test_late_delivery_is_received_at_promised_plus_delay():
    _, orders = run([3] * 200, lead=5, delay=1.0, mean_delay=4.0)
    received = [o for o in orders if o["receipt_index"] is not None]
    assert received
    for order in received:
        assert order["delay_days"] >= 1
        assert order["receipt_index"] == order["order_index"] + 5 + order["delay_days"]


def test_receipt_on_closed_day_moves_to_next_open_day():
    baseline, orders = run([3] * 60, lead=5)
    first = next(o for o in orders if o["receipt_index"] is not None)
    closed_day = first["receipt_index"]
    is_open = np.ones(60, dtype=bool)
    is_open[closed_day] = False
    arrays, moved = run([3] * 60, is_open=is_open, lead=5)
    assert arrays["received"][closed_day] == 0
    assert moved[0]["receipt_index"] == closed_day + 1
    assert conserved(arrays)


def test_closed_day_has_no_demand_or_sales():
    is_open = np.ones(30, dtype=bool)
    is_open[10] = False
    arrays, _ = run([5] * 30, is_open=is_open)
    assert arrays["demand"][10] == 0
    assert arrays["sales"][10] == 0
    assert arrays["unmet"][10] == 0


def test_no_review_on_closed_monday():
    is_open = np.ones(30, dtype=bool)
    is_open[7] = False
    arrays, _ = run([50] * 30, is_open=is_open)
    assert arrays["ordered"][7] == 0


def test_future_demand_does_not_change_past_decisions():
    early = [3] * 40
    calm, _ = run(early + [0] * 40)
    spike, _ = run(early + [5000] * 40)
    for name in ["on_hand_end", "ordered", "reorder_point", "order_up_to", "on_order_end"]:
        assert (calm[name][:40] == spike[name][:40]).all(), name


def test_reorder_point_ignores_stockout_days():
    sales = np.array([4, 4, 0, 0, 4, 4])
    valid = np.array([True, True, False, False, True, True])
    point, _ = stock_levels(sales, valid, 7, Policy(z=0.0, review_weekday=0, cover_days=0, window_days=6))
    assert point == 4 * 14


def test_reorder_point_is_at_least_one_with_no_sales():
    point, target = stock_levels(np.zeros(10, dtype=int), np.ones(10, dtype=bool), 5, POLICY)
    assert point >= 1
    assert target >= point


def test_no_valid_days_uses_minimum_rate():
    point, _ = stock_levels(np.zeros(5, dtype=int), np.zeros(5, dtype=bool), 5, POLICY)
    assert point == 1


def test_same_seed_repeats_and_other_seed_differs():
    first, first_orders = run([3] * 300, delay=0.5, seed=7)
    second, second_orders = run([3] * 300, delay=0.5, seed=7)
    other, _ = run([3] * 300, delay=0.5, seed=8)
    assert all((first[k] == second[k]).all() for k in first)
    assert first_orders == second_orders
    assert not (first["received"] == other["received"]).all()


def test_delay_probability_bounds():
    rng = np.random.default_rng(0)
    assert {draw_delay(rng, 0.0, 5.0) for _ in range(200)} == {0}
    assert min(draw_delay(rng, 1.0, 5.0) for _ in range(200)) >= 1


def test_delay_mean_matches_setting():
    rng = np.random.default_rng(0)
    assert np.mean([draw_delay(rng, 1.0, 7.0) for _ in range(5000)]) == pytest.approx(7.0, abs=0.2)


def test_label_looks_at_following_days_not_same_day():
    labels = stockout_within([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 2)
    assert labels[:8].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


def test_label_tail_without_full_horizon_is_missing():
    labels = stockout_within([0] * 10, 3)
    assert np.isnan(labels[-3:]).all()
    assert (labels[:-3] == 0).all()


def test_label_when_series_is_shorter_than_horizon():
    assert np.isnan(stockout_within([1, 1], 14)).all()


def test_as_of_view_hides_future_outcomes():
    orders = pd.DataFrame({
        "order_date": pd.to_datetime(["2014-01-01", "2014-01-10", "2014-02-01"]),
        "receipt_date": pd.to_datetime(["2014-01-08", "2014-01-30", "2014-02-09"]),
        "actual_lead_days": pd.array([7, 20, 8], dtype="Int64"),
        "delay_days": pd.array([0, 13, 1], dtype="Int64"),
    })
    view = purchase_orders_as_of(orders, "2014-01-15")
    assert len(view) == 2
    assert view.iloc[0].status == "received"
    assert view.iloc[0].actual_lead_days == 7
    assert view.iloc[1].status == "open"
    assert pd.isna(view.iloc[1].receipt_date)
    assert pd.isna(view.iloc[1].delay_days)
    assert pd.isna(view.iloc[1].actual_lead_days)
