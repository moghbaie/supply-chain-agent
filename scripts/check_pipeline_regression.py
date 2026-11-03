"""Run a shadow agent whose every recommendation is declined, so its state stays the saved one, and compare its evidence with the saved test rows."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.decisions import DECLINE_ALL
from src.evaluation.replay import Replay, Variant, load_world
from src.evaluation.scoring import ORDER, SCORES, build_scorer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30)
DAYS, NEVER = 40, 10000
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
scenarios = list(config["scenarios"])
scorer = build_scorer(settings, scenarios)
print(f"GPU in use after loading {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
saved = pd.read_parquet("data/processed/escalation/classified_test.parquet")
saved[SCORES] = np.load("data/processed/escalation/ultrabert/test_fine_tuned.npy")
saved["ultrabert"] = ORDER[saved[SCORES].to_numpy().argmax(axis=1)]
COLUMNS = ["on_hand", "on_order", "reorder_point", "order_up_to", "ordered_today", "price", "fc_mean_7", "fc_mean_14", "fc_mean_28", "fc_q10_sum_14", "fc_q90_sum_14", "fc_q90_max_14", "cover_days", "open_pos", "units_due_14", "units_overdue", "days_overdue_max", "days_to_next_due",
           "supplier_late_rate", "supplier_mean_delay", "supplier_pos_seen", "sales_mean_28", "sales_zero_share_28", "sold_out_days_28", "supplier_recent_late_rate", "stockout_risk", "stockout_risk_lightgbm", *SCORES]
for scenario in scenarios:
    orders = pd.read_parquet("data/simulated/purchase_orders.parquet", filters=[("scenario", "==", scenario)])
    world = load_world(scenario, config, settings["start"], orders=orders)
    world.dates, world.demand = world.dates[:DAYS], world.demand[:, :DAYS]
    lookup = {(item.id, item.order_date): int(item.delay_days) if pd.notna(item.delay_days) else NEVER for item in orders.itertuples(index=False)}
    started = time.time()
    replay = Replay(world, Variant("shadow", decide="ultrabert", person=DECLINE_ALL), scorer, delay_lookup=lookup).run()
    cases = pd.concat(replay.cases, ignore_index=True)
    print(f"\n{scenario}: {len(cases):,} cases over {cases.origin.nunique()} reviews in {time.time() - started:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    both = cases.merge(saved[saved.scenario == scenario], on=["id", "origin"], suffixes=("", "_saved"), validate="one_to_one")
    print(f"matched rows {len(both):,}")
    rows = []
    for column in COLUMNS:
        mine, theirs = both[column].astype(float), both[f"{column}_saved"].astype(float)
        close = np.isclose(mine, theirs, atol=1e-3, rtol=1e-3, equal_nan=True)
        rows.append({"column": column, "rows_not_close": int((~close).sum()), "largest_gap": float((mine - theirs).abs().max())})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"UltraBERT category same on {int((both.suggested == both.ultrabert).sum()):,} of {len(both):,}; value band same on {int((both.value_band == both.value_band_saved).sum()):,}")
