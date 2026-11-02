"""Daily inventory simulation for one series.

Order of events each day: receive purchase orders, serve demand, then review and order.
The policy sees only past sales and promised lead times, never true demand or future delays.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REVIEW_PERIOD_DAYS = 7


@dataclass(frozen=True)
class Policy:
    z: float
    review_weekday: int
    cover_days: int
    window_days: int


def draw_delay(rng, delay_prob, mean_delay):
    if rng.random() >= delay_prob:
        return 0
    return 1 + int(rng.poisson(max(mean_delay - 1.0, 0.0)))


def stock_levels(sales, valid, promised_lead, policy):
    """Return reorder point and order-up-to level from sales on valid days in the window."""
    observed = sales[valid]
    rate = max(float(observed.mean()), 0.01) if observed.size else 0.01
    spread = float(observed.std()) if observed.size > 1 else 0.0
    exposure = promised_lead + REVIEW_PERIOD_DAYS
    reorder_point = max(math.ceil(rate * exposure + policy.z * spread * math.sqrt(exposure)), 1)
    return reorder_point, reorder_point + math.ceil(rate * policy.cover_days)


def simulate_series(dates, demand, is_open, history_sales, history_open, promised_lead, delay_prob, mean_delay,
                    pack_size, policy, rng):
    """Simulate one series and return daily arrays and its purchase orders.

    delay_prob and mean_delay are per-day arrays, applied by order date.
    """
    days = len(dates)
    demand = np.where(is_open, demand, 0)
    window = policy.window_days
    past_sales = np.concatenate([history_sales[-window:], np.zeros(days, dtype=np.int64)])
    past_valid = np.concatenate([history_open[-window:], np.zeros(days, dtype=bool)])
    offset = len(past_sales) - days
    weekdays = pd.DatetimeIndex(dates).weekday

    reorder_point, target = stock_levels(past_sales[:offset], past_valid[:offset], promised_lead, policy)
    on_hand = target
    arrays = {name: np.zeros(days, dtype=np.int64) for name in
              ["on_hand_start", "received", "sales", "unmet", "on_hand_end", "on_order_end", "reorder_point",
               "order_up_to", "ordered"]}
    arrivals = {}
    orders = []
    on_order = 0

    for day in range(days):
        arrays["on_hand_start"][day] = on_hand
        if is_open[day]:
            for arrival_day in [t for t in arrivals if t <= day]:
                for order in arrivals.pop(arrival_day):
                    order["receipt_index"] = day
                    arrays["received"][day] += order["quantity"]
                    on_hand += order["quantity"]
                    on_order -= order["quantity"]

        sold = min(on_hand, int(demand[day]))
        arrays["sales"][day] = sold
        arrays["unmet"][day] = demand[day] - sold
        past_valid[offset + day] = bool(is_open[day] and on_hand > 0)
        on_hand -= sold
        past_sales[offset + day] = sold

        if is_open[day] and weekdays[day] == policy.review_weekday:
            start = offset + day + 1 - window
            reorder_point, target = stock_levels(past_sales[start:offset + day + 1], past_valid[start:offset + day + 1],
                                                 promised_lead, policy)
            position = on_hand + on_order
            if position <= reorder_point:
                quantity = math.ceil((target - position) / pack_size) * pack_size
                delay = draw_delay(rng, delay_prob[day], mean_delay[day])
                order = {"order_index": day, "quantity": quantity, "promised_lead_days": promised_lead,
                         "delay_days": delay, "receipt_index": None}
                arrivals.setdefault(day + promised_lead + delay, []).append(order)
                orders.append(order)
                on_order += quantity
                arrays["ordered"][day] = quantity

        arrays["on_hand_end"][day] = on_hand
        arrays["on_order_end"][day] = on_order
        arrays["reorder_point"][day] = reorder_point
        arrays["order_up_to"][day] = target

    arrays["demand"] = demand
    return arrays, orders


def stockout_within(unmet, horizon):
    """Return whether unmet demand occurs in days t+1..t+horizon, with NaN where the horizon is incomplete."""
    stockout = (np.asarray(unmet) > 0).astype(float)
    running = np.concatenate([[0.0], np.cumsum(stockout)])
    labels = np.full(len(stockout), np.nan)
    if len(stockout) > horizon:
        ends = np.arange(horizon, len(stockout)) + 1
        labels[:len(ends)] = (running[ends] - running[ends - horizon]) > 0
    return labels


def purchase_orders_as_of(purchase_orders, as_of):
    """Return purchase orders as known on a date, hiding outcomes that have not happened yet."""
    as_of = pd.Timestamp(as_of)
    view = purchase_orders[purchase_orders.order_date <= as_of].copy()
    pending = ~(view.receipt_date <= as_of)
    view.loc[pending, "receipt_date"] = pd.NaT
    view.loc[pending, "actual_lead_days"] = pd.NA
    view.loc[pending, "delay_days"] = pd.NA
    view["status"] = np.where(pending, "open", "received")
    return view
