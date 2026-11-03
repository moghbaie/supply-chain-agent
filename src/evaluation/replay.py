"""Replay the unseen test period day by day, so each policy's orders change its later stock, sales and evidence.

Every policy starts from the same stock and open orders and meets the same demand and the same delay for an order placed on the same day.
Each day runs as in the simulator: receive orders, serve demand, then review and order. Decisions see only the past.
"""
import math
import zlib
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.decisions import APPROVE_ALL, simulated_person
from src.simulation.engine import Policy, draw_delay, stock_levels
from src.simulation.world import disruption_masks
from src.stockout.features import SUPPLIER_WINDOW_DAYS, build_features

ARRAYS = ["on_hand_start", "received", "sales", "unmet", "on_hand_end", "on_order_end", "reorder_point", "order_up_to", "ordered", "expedites"]
HISTORY_COLUMNS = ["po_id", "id", "supplier_id", "order_date", "quantity", "promised_lead_days", "promised_date", "delay_days", "receipt_date", "actual_lead_days", "status"]
Z_FOR_Q10_Q90 = 2.563


@dataclass
class World:
    """Everything fixed for one scenario, shared by all policies."""

    scenario: str
    dates: pd.DatetimeIndex
    series: pd.DataFrame
    suppliers: pd.DataFrame
    policy: Policy
    demand: np.ndarray
    is_open: np.ndarray
    price: np.ndarray
    delay_prob: np.ndarray
    mean_delay: np.ndarray
    start_on_hand: np.ndarray
    start_levels: np.ndarray
    past_sales: np.ndarray
    past_valid: np.ndarray
    history_orders: pd.DataFrame
    recent_daily: pd.DataFrame
    pack_size: dict
    costs: dict
    seed: int


def load_world(scenario, config, start, daily=None, orders=None, series=None, suppliers=None):
    """Read the scenario's state at the end of the day before the start date, and its fixed future."""
    start = pd.Timestamp(start)
    daily = pd.read_parquet("data/simulated/inventory_daily.parquet", filters=[("scenario", "==", scenario)]) if daily is None else daily
    orders = pd.read_parquet("data/simulated/purchase_orders.parquet", filters=[("scenario", "==", scenario)]) if orders is None else orders
    series = pd.read_parquet("data/simulated/series.parquet") if series is None else series
    suppliers = pd.read_parquet("data/simulated/suppliers.parquet") if suppliers is None else suppliers
    settings = config["scenarios"][scenario]
    policy = Policy(**{**config["policy"], **settings.get("policy", {})})
    table = lambda column: daily.pivot(index="id", columns="date", values=column).reindex(index=series.id)
    future = sorted(daily.date[daily.date >= start].unique())
    dates = pd.DatetimeIndex(future)
    window = pd.date_range(start - pd.Timedelta(days=policy.window_days), periods=policy.window_days)
    usable = table("store_open").astype(bool)
    received, began = table("received"), table("on_hand_start")
    by_supplier = suppliers.set_index("supplier_id")
    delay_prob = np.zeros((len(series), len(dates)))
    mean_delay = np.zeros((len(series), len(dates)))
    masks = disruption_masks(settings, suppliers, dates, config["seed"])
    for row, supplier_id in enumerate(series.supplier_id):
        supplier = by_supplier.loc[supplier_id]
        delay_prob[row], mean_delay[row] = supplier.delay_prob, supplier.mean_delay_days
        if supplier_id in masks:
            delay_prob[row, masks[supplier_id]] = settings["delay_prob"]
            mean_delay[row, masks[supplier_id]] = supplier.mean_delay_days * settings["mean_delay_multiplier"]
    last = start - pd.Timedelta(days=1)
    history = orders[orders.order_date < start][HISTORY_COLUMNS].copy()
    unknown = history.receipt_date.isna()
    history = history[unknown | (history.receipt_date >= start - pd.Timedelta(days=SUPPLIER_WINDOW_DAYS + 1))]  # older receipts no longer affect any feature
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} orders placed before the start never arrive, so their arrival day is unknown")
    return World(scenario, dates, series, suppliers, policy, table("demand")[dates].to_numpy(dtype=np.int64), usable[dates].to_numpy(), table("price")[dates].to_numpy(dtype=float), delay_prob, mean_delay,
                 table("on_hand_end")[last].to_numpy(dtype=np.int64), np.stack([table("reorder_point")[last].to_numpy(), table("order_up_to")[last].to_numpy()], axis=1),
                 table("sales")[window].to_numpy(dtype=np.int64), (usable[window] & ((began[window] + received[window]) > 0)).to_numpy(),
                 history, daily[(daily.date > start - pd.Timedelta(days=29)) & (daily.date < start)][["id", "date", "sales", "on_hand_end", "store_open"]].copy(), config["pack_size"], config["costs"], config["seed"])


def fixed_delay(world, row, day):
    """Draw the delay for an order by series and day, so any policy ordering that day meets the same delay."""
    key = [world.seed, zlib.crc32(world.scenario.encode()), row, day]
    return draw_delay(np.random.default_rng(key), world.delay_prob[row, day], world.mean_delay[row, day])


def forecast_levels(forecast, lead_days, policy):
    """Use the forecast the way the reorder point uses past sales: expected demand over the exposure, plus safety stock and cover."""
    exposure = lead_days + 7
    mean, quantiles, levels = forecast["mean"], forecast["quantiles"], forecast["levels"]
    low, high = int(np.flatnonzero(np.isclose(levels, 0.1))[0]), int(np.flatnonzero(np.isclose(levels, 0.9))[0])
    results = []
    for row, days in enumerate(np.minimum(exposure, mean.shape[1])):
        spread = (quantiles[row, :days, high] - quantiles[row, :days, low]) / Z_FOR_Q10_Q90
        point = max(math.ceil(float(mean[row, :days].sum()) + policy.z * math.sqrt(float((spread ** 2).sum()))), 1)
        results.append((point, point + math.ceil(float(mean[row].mean()) * policy.cover_days)))
    return results


@dataclass
class Variant:
    """One way of running the same world."""

    name: str
    levels: str = "reorder_point"
    decide: str = None
    conflict_rule: bool = False
    person: str = APPROVE_ALL
    expedite_fraction: float = 0.5


@dataclass
class Replay:
    world: World
    variant: Variant
    scorer: object = None
    delay_lookup: dict = None
    arrays: dict = field(init=False)
    cases: list = field(default_factory=list)
    new_orders: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def __post_init__(self):
        world = self.world
        shape = world.demand.shape
        self.arrays = {name: np.zeros(shape, dtype=np.int64) for name in ARRAYS}
        self.expedite_cost = np.zeros(shape)
        self.on_hand = world.start_on_hand.copy()
        self.levels = world.start_levels.copy()
        self.sales_window = np.concatenate([world.past_sales, np.zeros(shape, dtype=np.int64)], axis=1)
        self.valid_window = np.concatenate([world.past_valid, np.zeros(shape, dtype=bool)], axis=1)
        self.offset = world.past_sales.shape[1]
        self.open = [[] for _ in range(shape[0])]
        self.row_of = {series_id: row for row, series_id in enumerate(world.series.id)}
        self.leads = world.series[["supplier_id"]].merge(world.suppliers[["supplier_id", "promised_lead_days"]], on="supplier_id", how="left").promised_lead_days.to_numpy()
        self.on_order = np.zeros(shape[0], dtype=np.int64)
        self.moved_receipts = {}
        self.ids, self.suppliers, self.packs = world.series.id.tolist(), world.series.supplier_id.tolist(), [world.pack_size[c] for c in world.series.cat_id]
        self.placed = [0] * shape[0]
        pending = world.history_orders[world.history_orders.receipt_date >= world.dates[0]]
        for item in pending.itertuples(index=False):
            row = self.row_of[item.id]
            arrival = int((item.receipt_date - world.dates[0]).days)
            self.open[row].append({"po_id": item.po_id, "quantity": int(item.quantity), "arrival": arrival, "promised": int((item.promised_date - world.dates[0]).days), "new": False, "expedited": False})
            self.on_order[row] += int(item.quantity)

    def _place(self, row, day, quantity):
        world = self.world
        if self.delay_lookup is not None:
            delay = self.delay_lookup[(self.ids[row], world.dates[day])]
        else:
            delay = fixed_delay(world, row, day)
        lead = int(self.leads[row])
        number = self.placed[row]
        self.placed[row] += 1
        item = {"po_id": f"replay|{world.scenario}|{self.variant.name}|{self.ids[row]}|{number:04d}", "id": self.ids[row], "supplier_id": self.suppliers[row], "order_day": day, "quantity": int(quantity),
                "promised_lead_days": lead, "delay_days": int(delay), "receipt_day": None}
        self.new_orders.append(item)
        self.open[row].append({"po_id": item["po_id"], "quantity": int(quantity), "arrival": day + lead + int(delay), "promised": day + lead, "new": True, "expedited": False, "record": item})
        self.on_order[row] += int(quantity)
        self.arrays["ordered"][row, day] += int(quantity)

    def _expedite(self, row, day):
        """Shorten the soonest order not yet expedited, and charge for it. Return False if no order can gain any time."""
        candidates = [item for item in self.open[row] if not item["expedited"] and item["arrival"] > day + 1]
        if not candidates:
            return False
        item = min(candidates, key=lambda order: order["arrival"])
        item["arrival"] = day + max(math.ceil((item["arrival"] - day) * self.variant.expedite_fraction), 1)
        item["expedited"] = True
        if item["new"]:
            item["record"]["delay_days"] = item["arrival"] - item["promised"]
        else:
            self.moved_receipts[item["po_id"]] = (item["arrival"], item["arrival"] - item["promised"])
        costs = self.world.costs
        self.expedite_cost[row, day] += item["quantity"] * self.world.price[row, day] * costs["shipping_rate_of_price"] * (costs["expedite_shipping_multiplier"] - 1) + costs["expedite_fixed_fee"]
        self.arrays["expedites"][row, day] += 1
        return True

    def _start_of_day(self, row, day):
        world, arrays = self.world, self.arrays
        arrays["on_hand_start"][row, day] = self.on_hand[row]
        if world.is_open[row, day]:
            for item in [order for order in self.open[row] if order["arrival"] <= day]:
                self.open[row].remove(item)
                arrays["received"][row, day] += item["quantity"]
                self.on_hand[row] += item["quantity"]
                self.on_order[row] -= item["quantity"]
                if item["new"]:
                    item["record"]["receipt_day"] = day
        demand = world.demand[row, day] if world.is_open[row, day] else 0
        sold = min(self.on_hand[row], demand)
        arrays["sales"][row, day] = sold
        arrays["unmet"][row, day] = demand - sold
        self.valid_window[row, self.offset + day] = bool(world.is_open[row, day] and self.on_hand[row] > 0)
        self.sales_window[row, self.offset + day] = sold
        self.on_hand[row] -= sold

    def _routine(self, row, day, forecast_points):
        world = self.world
        window = world.policy.window_days
        first, last = self.offset + day + 1 - window, self.offset + day + 1
        if forecast_points is not None:
            self.levels[row] = forecast_points[row]
        else:
            self.levels[row] = stock_levels(self.sales_window[row, first:last], self.valid_window[row, first:last], int(self.leads[row]), world.policy)
        position = self.on_hand[row] + self.on_order[row]
        if position <= self.levels[row, 0]:
            pack = self.packs[row]
            self._place(row, day, math.ceil((self.levels[row, 1] - position) / pack) * pack)

    def _orders_frame(self, day):
        """Return all orders known so far, with arrivals still in the future hidden later by the point-in-time view."""
        dates = self.world.dates
        rows = [{"po_id": item["po_id"], "id": item["id"], "supplier_id": item["supplier_id"], "order_date": dates[item["order_day"]], "quantity": item["quantity"], "promised_lead_days": item["promised_lead_days"],
                 "promised_date": dates[item["order_day"]] + pd.Timedelta(days=item["promised_lead_days"]), "delay_days": item["delay_days"], "receipt_date": dates[item["receipt_day"]] if item["receipt_day"] is not None else pd.NaT,
                 "actual_lead_days": item["receipt_day"] - item["order_day"] if item["receipt_day"] is not None else pd.NA, "status": "received" if item["receipt_day"] is not None else "open"} for item in self.new_orders]
        history = self.world.history_orders.copy()
        for po_id, (arrival, delay) in self.moved_receipts.items():
            history.loc[history.po_id == po_id, ["receipt_date", "delay_days"]] = [dates[0] + pd.Timedelta(days=arrival), delay]
        if not rows:
            return history
        frame = pd.DataFrame(rows)
        for column in ["delay_days", "actual_lead_days"]:
            frame[column] = frame[column].astype("Int64")
        return pd.concat([history.astype({"delay_days": "Int64", "actual_lead_days": "Int64"}), frame], ignore_index=True)

    def daily_frame(self, last_day=None):
        """Return the replayed days as a long table."""
        world = self.world
        last_day = world.demand.shape[1] - 1 if last_day is None else last_day
        days, count = last_day + 1, len(world.series)
        frame = {"id": np.repeat(world.series.id.to_numpy(), days), "date": np.tile(world.dates[:days], count), "store_open": world.is_open[:, :days].ravel(), "demand": np.where(world.is_open[:, :days], world.demand[:, :days], 0).ravel(), "price": world.price[:, :days].ravel(),
                 "expedite_cost": self.expedite_cost[:, :days].ravel()}
        frame.update({name: self.arrays[name][:, :days].ravel() for name in ARRAYS})
        return pd.DataFrame(frame)

    def _cases(self, day, forecasts):
        origin = self.world.dates[day]
        replayed = self.daily_frame(day)
        daily = pd.concat([self.world.recent_daily, replayed[["id", "date", "sales", "on_hand_end", "store_open"]]], ignore_index=True)
        window = daily[daily.date > origin - pd.Timedelta(days=29)]
        today = replayed[replayed.date == origin][["id", "on_order_end", "reorder_point", "order_up_to", "ordered", "price"]].assign(stockout_next_14d=np.nan)
        window = window.merge(today, on="id", how="left")
        orders = self._orders_frame(day)
        return build_features(window, orders, forecasts, self.world.series, self.world.suppliers, [origin], 14, {origin: "test"}).assign(scenario=self.world.scenario), orders

    def _review(self, day):
        """Decide on every open series at a review, and apply what the simulated person approves."""
        world, variant = self.world, self.variant
        forecast = forecasts = None
        if self.scorer is not None and (variant.levels == "forecast" or variant.decide):
            forecast, forecasts = self.scorer.forecast(self, day)
        points = forecast_levels(forecast, self.leads, world.policy) if variant.levels == "forecast" else None
        for row in range(len(world.series)):
            if world.is_open[row, day]:
                self._routine(row, day, points)
        self._close_day(day)
        if not variant.decide:
            return
        cases, orders = self._cases(day, forecasts)
        decided = self.scorer.decide(cases, orders, variant)
        self.cases.append(decided)
        for item in decided.itertuples(index=False):
            row = self.row_of[item.id]
            action = simulated_person(item.final, item.open_pos, variant.person)
            if not world.is_open[row, day]:
                action = "none"
            done = False
            if action == "order":
                pack = self.packs[row]
                self._place(row, day, max(math.ceil((self.levels[row, 1] - self.on_hand[row] - self.on_order[row]) / pack), 1) * pack)
                done = True
            elif action == "expedite":
                done = self._expedite(row, day)
            self.actions.append({"id": item.id, "origin": world.dates[day], "final": item.final, "action": action, "done": done})
        self._close_day(day)

    def _close_day(self, day):
        arrays = self.arrays
        arrays["on_hand_end"][:, day] = self.on_hand
        arrays["on_order_end"][:, day] = self.on_order
        arrays["reorder_point"][:, day] = self.levels[:, 0]
        arrays["order_up_to"][:, day] = self.levels[:, 1]

    def run(self):
        world = self.world
        for day, date in enumerate(world.dates):
            for row in range(len(world.series)):
                self._start_of_day(row, day)
            self._close_day(day)
            if date.weekday() == world.policy.review_weekday:
                self._review(day)
        return self

    def orders_frame(self):
        return self._orders_frame(len(self.world.dates) - 1)


def forecast_view(panel, saved_view):
    """Copy a scenario's forecast panel so a policy's sales can overwrite its own test days."""
    view = copy(panel)
    view.demand, view.observed = saved_view.demand.copy(), saved_view.observed.copy()
    return view


def write_days(view, base_observed, replay, day):
    """Show the forecaster this policy's sales through the given day, with sold-out days unobserved."""
    world = replay.world
    positions = pd.DatetimeIndex(view.dates).get_indexer(world.dates[:day + 1])
    view.demand[:, positions] = replay.arrays["sales"][:, :day + 1].astype(np.float32)
    view.observed[:, positions] = base_observed[:, positions] & (replay.arrays["on_hand_end"][:, :day + 1] != 0)
