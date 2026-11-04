"""Score the saved replay runs: service, cost, workload, errors, paired differences by series and example recommendations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_colwidth", 60)
RUNS = Path("data/processed/evaluation")
OUT = Path("reports/step_10")
ROUNDS = 1000
COMPARISONS = [("forecast_driven", "reorder_point"), ("policy_layer", "reorder_point"), ("agent_conflict", "reorder_point"), ("agent_conflict", "forecast_driven"), ("agent_conflict", "policy_layer"),
               ("agent_conflict_decline_review", "reorder_point"), ("agent_conflict_decline_review", "forecast_driven"), ("agent_conflict_weak_expedite", "reorder_point"), ("agent_conflict_weak_expedite", "forecast_driven")]
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
COSTS = config["costs"]


def with_costs(daily):
    return daily.assign(holding_cost=daily.on_hand_end * daily.price * COSTS["holding_rate_per_year"] / 365, lost_margin=daily.unmet * daily.price * COSTS["margin_rate"], shipping_cost=daily.received * daily.price * COSTS["shipping_rate_of_price"],
                        total_cost=lambda f: f.holding_cost + f.lost_margin + f.shipping_cost + f.expedite_cost)


def score(daily, cases, actions):
    """Return one row of results for a run. Cost excludes the purchase price of units and the time of reviewers."""
    opened = daily[daily.store_open]
    last = daily[daily.date == daily.date.max()]
    row = {"stockout_day_pct": 100 * (opened.unmet > 0).mean(), "unmet_units": int(daily.unmet.sum()), "fill_pct": 100 * daily.sales.sum() / daily.demand.sum(),
           "holding_cost": daily.holding_cost.sum(), "lost_margin": daily.lost_margin.sum(), "shipping_cost": daily.shipping_cost.sum(), "expedite_cost": daily.expedite_cost.sum(), "total_cost": daily.total_cost.sum(),
           "ordered_units": int(daily.ordered.sum()), "orders": int((daily.ordered > 0).sum()), "expedites": int(daily.expedites.sum()),
           "end_stock_value": float((last.on_hand_end * last.price).sum()), "end_on_order_value": float((last.on_order_end * last.price).sum()),
           "cases": 0, "action_cases": 0, "human_review": 0, "overrides": 0, "conflict_moved": 0, "weak_evidence": 0, "expedite_no_effect": 0, "unknown_risk": 0}
    if cases is not None:
        row.update(cases=len(cases), action_cases=int((cases.final != "monitor").sum()), human_review=int((cases.final == "human_review").sum()), overrides=int(((cases.reason != "") & (cases.reason != "stockout models conflict")).sum()),
                   conflict_moved=int((cases.reason == "stockout models conflict").sum()), weak_evidence=int(cases.weak_evidence.sum()), expedite_no_effect=int(((actions.action == "expedite") & ~actions.done).sum()),
                   unknown_risk=int(cases[["stockout_risk", "stockout_risk_lightgbm"]].isna().any(axis=1).sum()))
    return row


def paired(left, right, column, rng):
    """Bootstrap whole series, so a series' days stay together, and return the median and 95 percent interval of left minus right."""
    gap = left.groupby("id")[column].sum() - right.groupby("id")[column].sum()
    picks = rng.integers(0, len(gap), (ROUNDS, len(gap)))
    totals = gap.to_numpy()[picks].sum(axis=1)
    return float(gap.sum()), float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


rows, runs = [], {}
for scenario in config["scenarios"]:
    for variant in settings["variants"]:
        daily = with_costs(pd.read_parquet(RUNS / scenario / f"{variant}_daily.parquet"))
        path = RUNS / scenario / f"{variant}_cases.parquet"
        cases = pd.read_parquet(path) if path.exists() else None
        actions = pd.read_parquet(RUNS / scenario / f"{variant}_actions.parquet") if path.exists() else None
        runs[scenario, variant] = daily, cases, actions
        rows.append({"scenario": scenario, "variant": variant, **score(daily, cases, actions)})
table = pd.DataFrame(rows)
table.to_csv(OUT / "replay_metrics.csv", index=False)
for scenario, group in table.groupby("scenario", sort=False):
    print(f"\n=== {scenario}: service and cost (cost excludes unit purchase price and review time)")
    print(group[["variant", "stockout_day_pct", "unmet_units", "fill_pct", "holding_cost", "lost_margin", "shipping_cost", "expedite_cost", "total_cost"]].round(2).to_string(index=False))
    print(f"--- {scenario}: orders, ending position and workload")
    print(group[["variant", "orders", "ordered_units", "expedites", "end_stock_value", "end_on_order_value", "action_cases", "human_review", "overrides", "conflict_moved", "weak_evidence", "expedite_no_effect", "unknown_risk"]].round(0).to_string(index=False))

rng = np.random.default_rng(settings["seed"])
differences = []
for scenario in config["scenarios"]:
    for left, right in COMPARISONS:
        for column in ["total_cost", "unmet"]:
            total, low, high = paired(runs[scenario, left][0], runs[scenario, right][0], column, rng)
            differences.append({"scenario": scenario, "left": left, "right": right, "measure": column, "difference": total, "low": low, "high": high, "clear": bool(low > 0 or high < 0)})
differences = pd.DataFrame(differences)
differences.to_csv(OUT / "replay_differences.csv", index=False)
print(f"\n=== Paired differences, series bootstrap, {ROUNDS} rounds, left minus right; negative is better")
print(differences.round(1).to_string(index=False))

print("\n=== Share of each run's total cost by part")
for (scenario, variant), (daily, _, _) in runs.items():
    if variant in ("reorder_point", "agent_conflict"):
        parts = daily[["holding_cost", "lost_margin", "shipping_cost", "expedite_cost"]].sum()
        print(scenario, variant, (100 * parts / parts.sum()).round(1).to_dict())

print("\n=== Why declining human review matters: agent_conflict against decline_review")
for scenario in config["scenarios"]:
    base, cases, _ = runs[scenario, "agent_conflict"]
    declined, _, _ = runs[scenario, "agent_conflict_decline_review"]
    gap = (declined.groupby("id").unmet.sum() - base.groupby("id").unmet.sum()).sort_values(ascending=False)
    reviewed = set(cases[cases.final == "human_review"].id)
    print(f"{scenario}: extra unmet {int(gap.sum())}; series with any change {int((gap != 0).sum())}; series ever sent to human review {len(reviewed)}; extra unmet in those {int(gap[gap.index.isin(reviewed)].sum())}; largest 5 series {gap.head(5).to_dict()}")
    print(f"   human-review cases' mean 14-day forecast {cases[cases.final == 'human_review'].fc_mean_14.mean():.2f}, all cases {cases.fc_mean_14.mean():.2f}; mean price {cases[cases.final == 'human_review'].price.mean():.2f}, all {cases.price.mean():.2f}")

print("\n=== What the cost result depends on, agent with conflict rule against forecast-driven. Decisions never see these costs, so no rerun is needed")
sensitivity = []
for scenario in config["scenarios"]:
    agent, _, _ = runs[scenario, "agent_conflict"]
    rival, _, _ = runs[scenario, "forecast_driven"]
    extra, avoided = agent.total_cost.sum() - rival.total_cost.sum(), rival.unmet.sum() - agent.unmet.sum()
    fees = agent.expedites.sum() * COSTS["expedite_fixed_fee"]
    sensitivity.append({"scenario": scenario, "extra_cost": extra, "fees_in_extra": fees, "extra_cost_with_no_fee": extra - fees, "unmet_units_avoided": avoided, "extra_cost_per_unit_avoided": extra / avoided, "same_with_no_fee": (extra - fees) / avoided, "margin_counted_per_lost_unit": rival.lost_margin.sum() / rival.unmet.sum()})
sensitivity = pd.DataFrame(sensitivity)
sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)
print(sensitivity.round(2).T.to_string())

print("\n=== Example recommendations, first case per final category in each scenario, agent_conflict")
columns = ["origin", "id", "on_hand", "on_order", "cover_days", "open_pos", "fc_mean_14", "stockout_risk", "stockout_risk_lightgbm", "suggested", "final", "reason"]
examples = []
for scenario in config["scenarios"]:
    _, cases, actions = runs[scenario, "agent_conflict"]
    joined = cases.assign(action=actions.action.to_numpy(), done=actions.done.to_numpy(), scenario=scenario)
    examples.append(joined.groupby("final", sort=True).head(1)[["scenario"] + columns + ["action", "done"]])
examples = pd.concat(examples, ignore_index=True)
examples.to_csv(OUT / "example_recommendations.csv", index=False)
print(examples.round(3).to_string(index=False))
