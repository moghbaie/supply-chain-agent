"""Build synthetic suppliers and scenarios, and run the simulator across selected series."""
import zlib

import numpy as np
import pandas as pd

from src.simulation.engine import Policy, simulate_series, stockout_within


def build_suppliers(config, seed):
    settings = config["suppliers"]
    rng = np.random.default_rng([seed, 1])
    count = settings["count"]
    tiers = list(settings["tiers"])
    tier_counts = np.round([settings["tiers"][t]["share"] * count for t in tiers]).astype(int)
    tier_counts[-1] = count - tier_counts[:-1].sum()
    assigned = rng.permutation(np.repeat(tiers, tier_counts))
    return pd.DataFrame({
        "supplier_id": [f"SUP_{i:02d}" for i in range(count)],
        "tier": assigned,
        "promised_lead_days": rng.integers(settings["promised_lead_days_min"], settings["promised_lead_days_max"] + 1, count),
        "delay_prob": [settings["tiers"][t]["delay_prob"] for t in assigned],
        "mean_delay_days": [settings["tiers"][t]["mean_delay_days"] for t in assigned],
    })


def assign_suppliers(selected, suppliers, seed):
    """Give each item one supplier, shared across stores."""
    items = np.sort(selected.item_id.unique())
    rng = np.random.default_rng([seed, 2])
    mapping = dict(zip(items, rng.choice(suppliers.supplier_id.to_numpy(), len(items))))
    return selected.assign(supplier_id=selected.item_id.map(mapping))


def disruption_masks(scenario, suppliers, dates, seed):
    """Return per-supplier boolean arrays marking order dates inside a disruption window."""
    if "windows" not in scenario:
        return {}
    in_window = np.zeros(len(dates), dtype=bool)
    for start, end in scenario["windows"]:
        in_window |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    rng = np.random.default_rng([seed, 3])
    chosen = rng.choice(suppliers.supplier_id.to_numpy(), scenario["disrupted_supplier_count"], replace=False)
    return {str(supplier_id): in_window for supplier_id in chosen}


def split_of(dates, config):
    settings = config["dates"]
    return np.select(
        [dates <= pd.Timestamp(settings["burn_in_end"]), dates <= pd.Timestamp(settings["train_end"]),
         dates <= pd.Timestamp(settings["validation_end"])],
        ["burn_in", "train", "validation"], "test")


def run_scenario(name, config, selected, suppliers, all_dates, demand, closed, price, seed):
    """Run one scenario for every selected series; series order does not affect random draws."""
    settings = config["dates"]
    start = pd.Timestamp(settings["simulation_start"])
    end = pd.Timestamp(settings["simulation_end"])
    in_sim = (all_dates >= start) & (all_dates <= end)
    before = all_dates < start
    dates = all_dates[in_sim]
    scenario = config["scenarios"][name]
    policy = Policy(**{**config["policy"], **scenario.get("policy", {})})
    masks = disruption_masks(scenario, suppliers, dates, seed)
    costs = config["costs"]
    splits = split_of(dates, config)
    by_supplier = suppliers.set_index("supplier_id")
    scenario_key = zlib.crc32(name.encode())
    frames, order_rows = [], []

    for row, series in enumerate(selected.itertuples(index=False)):
        supplier = by_supplier.loc[series.supplier_id]
        delay_prob = np.full(len(dates), supplier.delay_prob)
        mean_delay = np.full(len(dates), supplier.mean_delay_days)
        if series.supplier_id in masks:
            mask = masks[series.supplier_id]
            delay_prob[mask] = scenario["delay_prob"]
            mean_delay[mask] = supplier.mean_delay_days * scenario["mean_delay_multiplier"]
        is_open = ~closed.loc[series.store_id].to_numpy()
        rng = np.random.default_rng([seed, scenario_key, zlib.crc32(series.id.encode())])
        arrays, orders = simulate_series(
            dates, demand[row, in_sim], is_open[in_sim], demand[row, before], is_open[before],
            int(supplier.promised_lead_days), delay_prob, mean_delay, config["pack_size"][series.cat_id], policy, rng)

        daily_price = price[row, in_sim]
        frame = pd.DataFrame(arrays)
        frame.insert(0, "date", dates)
        frame.insert(0, "id", series.id)
        frame["store_open"] = is_open[in_sim]
        frame["split"] = splits
        frame["price"] = daily_price
        frame["stockout_day"] = frame.unmet > 0
        frame["stockout_next_14d"] = stockout_within(arrays["unmet"], config["labels"]["stockout_horizon_days"])
        frame["holding_cost"] = frame.on_hand_end * daily_price * costs["holding_rate_per_year"] / 365
        frame["lost_margin"] = frame.unmet * daily_price * costs["margin_rate"]
        frame["shipping_cost"] = frame.received * daily_price * costs["shipping_rate_of_price"]
        frames.append(frame)

        for number, order in enumerate(orders):
            received = order["receipt_index"] is not None
            order_date = dates[order["order_index"]]
            order_rows.append({
                "po_id": f"{name}|{series.id}|{number:04d}", "id": series.id, "supplier_id": series.supplier_id,
                "order_date": order_date, "quantity": order["quantity"],
                "promised_lead_days": order["promised_lead_days"],
                "promised_date": order_date + pd.Timedelta(days=order["promised_lead_days"]),
                "delay_days": order["delay_days"] if received else pd.NA,
                "receipt_date": dates[order["receipt_index"]] if received else pd.NaT,
                "actual_lead_days": order["receipt_index"] - order["order_index"] if received else pd.NA,
                "status": "received" if received else "open",
            })

    daily = pd.concat(frames, ignore_index=True)
    daily.insert(0, "scenario", name)
    orders_frame = pd.DataFrame(order_rows)
    orders_frame.insert(0, "scenario", name)
    for column in ["delay_days", "actual_lead_days"]:
        orders_frame[column] = orders_frame[column].astype("Int64")
    return daily, orders_frame, policy, sorted(masks)
