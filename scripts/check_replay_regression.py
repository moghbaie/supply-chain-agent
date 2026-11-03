"""Check that the replay reproduces the saved simulation when given the saved delays. Calls no model unless asked."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.evaluation.replay import Replay, Variant, load_world

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30)
NEVER = 10000  # an order the saved run never received by its last day
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
for scenario in config["scenarios"]:
    daily = pd.read_parquet("data/simulated/inventory_daily.parquet", filters=[("scenario", "==", scenario)])
    orders = pd.read_parquet("data/simulated/purchase_orders.parquet", filters=[("scenario", "==", scenario)])
    world = load_world(scenario, config, settings["start"], daily=daily, orders=orders)
    print(f"\n{scenario}: {world.demand.shape[0]} series, {world.demand.shape[1]} days {world.dates[0].date()} to {world.dates[-1].date()}; open orders at start {int((world.history_orders.receipt_date >= world.dates[0]).sum())}; start on hand {int(world.start_on_hand.sum()):,}; policy {world.policy}")
    lookup = {(item.id, item.order_date): int(item.delay_days) if pd.notna(item.delay_days) else NEVER for item in orders.itertuples(index=False)}
    replay = Replay(world, Variant("regression"), delay_lookup=lookup).run()
    mine = replay.daily_frame().sort_values(["id", "date"]).reset_index(drop=True)
    saved = daily[daily.date >= settings["start"]].sort_values(["id", "date"]).reset_index(drop=True)
    for column in ["on_hand_start", "received", "sales", "unmet", "on_hand_end", "on_order_end", "reorder_point", "order_up_to", "ordered"]:
        different = int((mine[column].to_numpy() != saved[column].to_numpy()).sum())
        print(f"  {column}: {different} of {len(mine):,} rows differ")
    fresh = Replay(world, Variant("fixed_delays")).run().daily_frame()
    print(f"  with the replay's own delay draws: unmet {int(fresh.unmet.sum()):,} against saved {int(saved.unmet.sum()):,}; ordered {int(fresh.ordered.sum()):,} against {int(saved.ordered.sum()):,}")
