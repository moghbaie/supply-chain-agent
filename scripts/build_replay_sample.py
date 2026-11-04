"""Draw the replay cases the LLM reviews, with the orders and delay risks its tools read. Runs locally."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.supplier.inbound import FEATURES, build_po_features, fit_delay_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RUNS = Path("data/processed/evaluation")
OUT = Path("data/processed/agent")
VARIANT = "agent_conflict"
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
SEED, PER_CATEGORY = settings["seed"], settings["llm_sample"]["cases_per_category"]
START = pd.Timestamp(settings["start"])
dates = config["dates"]

purchase = pd.read_parquet("data/processed/delay/inbound_features.parquet")
model = fit_delay_model(purchase[purchase.split.isin(["train", "validation"]) & purchase.known_before_test], SEED)
saved = pd.read_parquet(OUT / "delay_scores.parquet").query("period == 'test'")
rows = purchase.merge(saved[["po_id", "delay_risk"]], on="po_id")
print(f"delay model on the saved test orders: {len(rows):,} orders, largest gap from saved scores {np.abs(model.predict(rows[FEATURES].to_numpy(dtype=float)) - rows.delay_risk).max():.2e}")

history = pd.read_parquet("data/simulated/purchase_orders.parquet")
series = pd.read_parquet("data/simulated/series.parquet")
cases, orders, scores = [], [], [saved[["po_id", "delay_risk"]]]
for scenario in config["scenarios"]:
    replayed = pd.read_parquet(RUNS / scenario / f"{VARIANT}_orders.parquet")
    cases.append(pd.read_parquet(RUNS / scenario / f"{VARIANT}_cases.parquet"))
    # Replay saves only recent history, so rebuild the full order book, keeping receipts the replay moved.
    before = history[(history.scenario == scenario) & (history.order_date < START)].drop(columns="scenario")
    before = before[~before.po_id.isin(replayed.po_id)]
    book = pd.concat([before.astype({"delay_days": "Int64", "actual_lead_days": "Int64"}), replayed], ignore_index=True)
    orders.append(replayed.assign(scenario=scenario))
    # Features use only what was received before each order date. Open orders are scored like the rest.
    built = build_po_features(book.assign(status="received"), series, pd.Timestamp(dates["train_end"]), pd.Timestamp(dates["validation_end"]), pd.Timestamp(dates["burn_in_end"]))
    new = built[built.po_id.str.startswith("replay|")]
    assert len(new) == replayed.po_id.str.startswith("replay|").sum()
    scores.append(pd.DataFrame({"po_id": new.po_id, "delay_risk": model.predict(new[FEATURES].to_numpy(dtype=float))}))
    print(f"{scenario}: {len(book):,} orders in the book, {len(new):,} replay orders scored, mean delay risk {scores[-1].delay_risk.mean():.4f}, late {new.delayed.mean():.4f}")
cases = pd.concat(cases, ignore_index=True)
# The key's reference holds the code-only result here: the sample measures agreement with it, not accuracy.
cases = cases.assign(reference=cases.final, ultrabert=cases.suggested)
sample = pd.concat([group.sample(min(PER_CATEGORY, len(group)), random_state=SEED) for _, group in cases.groupby("final", sort=True)]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
sample.to_parquet(OUT / "replay_cases.parquet", index=False)
pd.concat(orders, ignore_index=True).to_parquet(OUT / "replay_orders.parquet", index=False)
pd.concat(scores, ignore_index=True).assign(period="replay").to_parquet(OUT / "replay_delay_scores.parquet", index=False)
print("population", cases.final.value_counts().to_dict())
print("sample", len(sample), sample.final.value_counts().to_dict(), sample.scenario.value_counts().to_dict(), "series", sample.id.nunique(), "origins", sample.origin.min().date(), sample.origin.max().date())
print("conflict moved in sample", int((sample.reason == "stockout models conflict").sum()), "suggested", sample.suggested.value_counts().to_dict())
print(sample[["scenario", "origin", "id", "on_hand", "on_order", "stockout_risk", "stockout_risk_lightgbm", "suggested", "final"]].head(10).round(3).to_string(index=False))
