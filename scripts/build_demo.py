"""Build the Step 10 demo page from saved results: the pipeline traced on real cases, and the replay's numbers. Reads local files only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.agent.audit import audit
from src.evaluation.decisions import simulated_person

RUNS = Path("data/processed/evaluation")
TRACES = Path("data/processed/agent/traces/smollm_untuned/replay")
REPORT = Path("reports/step_10")
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
FEE = config["costs"]["expedite_fixed_fee"]
SCENARIOS = {"baseline": "Baseline", "lean_policy": "Lean policy", "supplier_disruption": "Supplier disruption"}
LABELS = {"reorder_point": "Reorder point", "forecast_driven": "Forecast-driven", "policy_layer": "Policy rule", "agent": "Agent", "agent_conflict": "Agent with conflict rule", "agent_conflict_decline_review": "Conflict rule, review declined", "agent_conflict_weak_expedite": "Conflict rule, weak expedites"}
WINDOW = pd.Timedelta(days=14)
START_CASE = "replay-0003"  # has a model conflict, open orders and a stockout risk worth reading

metrics = pd.read_csv(REPORT / "replay_metrics.csv")
runs = {scenario: [{"key": row.variant, "label": LABELS[row.variant], "holding": row.holding_cost, "lost_margin": row.lost_margin, "shipping": row.shipping_cost, "expedite_shipping": row.expedite_cost - FEE * row.expedites, "expedites": row.expedites, "unmet": row.unmet_units, "stockout_pct": row.stockout_day_pct, "fill_pct": row.fill_pct,
                    "action_cases": row.action_cases, "human_review": row.human_review, "weak": row.weak_evidence} for row in metrics[metrics.scenario == scenario].itertuples()] for scenario in SCENARIOS}
differences = {}
table = pd.read_csv(REPORT / "replay_differences.csv")
for scenario in SCENARIOS:
    chosen = table[(table.scenario == scenario) & table.left.isin(["forecast_driven", "agent_conflict"])]
    differences[scenario] = [{"label": f"{LABELS[row.left]} minus {LABELS[row.right].lower()}", "measure": "Cost, $" if row.measure == "total_cost" else "Unmet units", "difference": row.difference, "low": row.low, "high": row.high, "clear": bool(row.clear)} for row in chosen.itertuples()]

key = pd.read_parquet(TRACES / "key.parquet").rename(columns={"reference": "code_final"})
sample = pd.read_parquet("data/processed/agent/replay_cases.parquet")
assert (key.id.to_numpy() == sample.id.to_numpy()).all()
key["open_pos"] = sample.open_pos.to_numpy()
daily = {s: pd.read_parquet(RUNS / s / "agent_conflict_daily.parquet", columns=["id", "date", "on_hand_end", "unmet", "received"]) for s in SCENARIOS}
routine = {s: pd.read_parquet(RUNS / s / "reorder_point_daily.parquet", columns=["id", "date", "unmet"]) for s in SCENARIOS}
cases, problem_counts, changed, same_as_suggestion, factual, failed = [], {}, 0, 0, 0, 0
for row in key.itertuples():
    trace = json.loads((TRACES / f"{row.case_id}.json").read_text(encoding="utf-8"))
    recommendation = trace["recommendation"] or {}
    explanation = recommendation.get("explanation", "")
    problems = audit(explanation, trace["evidence"]) if explanation else []
    for problem in problems:
        problem_counts[problem] = problem_counts.get(problem, 0) + 1
    factual += any(name != "cites_no_numbers" for name in problems)
    final = trace["review"]["final_category"]
    changed += final != row.code_final
    same_as_suggestion += recommendation.get("category") == row.ultrabert
    failed += trace["status"] != "submitted"
    origin = pd.Timestamp(row.origin)
    days = daily[row.scenario][(daily[row.scenario].id == row.id) & (daily[row.scenario].date >= origin - WINDOW) & (daily[row.scenario].date <= origin + WINDOW)].sort_values("date")
    after = (routine[row.scenario].id == row.id) & (routine[row.scenario].date > origin) & (routine[row.scenario].date <= origin + WINDOW)
    cases.append({"case_id": row.case_id, "scenario_label": SCENARIOS[row.scenario], "id": row.id.replace("_evaluation", ""), "origin": str(origin.date()), "code_final": row.code_final, "ultrabert": row.ultrabert, "final": final, "model": recommendation.get("category"), "uncertainty": recommendation.get("uncertainty"), "explanation": explanation, "flags": trace["review"]["flags"], "problems": problems,
                  "turns": len(trace["turns"]), "seconds": trace["seconds"], "calls": [call["name"] for call in trace["tool_calls"]], "evidence": trace["evidence"], "person_action": simulated_person(final, row.open_pos), "unmet_after": int(days[days.date > origin].unmet.sum()), "unmet_after_routine": int(routine[row.scenario][after].unmet.sum()),
                  "days": [{"date": str(item.date.date()), "on_hand": int(item.on_hand_end), "unmet": int(item.unmet), "received": int(item.received)} for item in days.itertuples()]})
audit_names = {"cites_no_numbers": "Cites no numbers", "says_high_risk_but_both_models_under_20": "Says high risk, both models under 20%", "says_no_open_orders_but_one_exists": "Says no open orders, one exists", "says_it_conflicts_with_the_suggestion_it_followed": "Says it conflicts with the suggestion it followed", "stockout_risk_number_not_from_either_model": "Risk number from neither model", "says_delayed_but_nothing_overdue": "Says delayed, nothing overdue",
               "describes_an_open_order_that_does_not_exist": "Describes an open order that does not exist", "says_low_risk_but_a_model_at_50_or_more": "Says low risk, a model at 50% or more", "says_demand_exceeds_stock_but_it_does_not": "Says demand exceeds stock, it does not"}
llm = {"facts": [["Chose UltraBERT's category", f"{same_as_suggestion} of {len(cases)}"], ["Final same as code only", f"{len(cases) - changed} of {len(cases)}"], ["Orders placed", "0"], ["Runs with no valid answer", str(failed)]],
       "audit": [["Any factual rule", factual]] + sorted(([audit_names[name], count] for name, count in problem_counts.items()), key=lambda item: -item[1])}
data = {"scenarios": [{"key": key_, "label": label} for key_, label in SCENARIOS.items()], "runs": runs, "differences": differences, "cases": cases, "llm": llm, "start_case": START_CASE}
template = Path("scripts/demo_template.html").read_text(encoding="utf-8")
payload = json.dumps(data, default=lambda value: value.item()).replace("</", "<\\/")
target = REPORT / "demo.html"
target.write_text(template.replace("/*DATA*/null", payload), encoding="utf-8")
print(f"wrote {target}, {target.stat().st_size / 1024:.0f} KB, {len(cases)} cases; same as UltraBERT {same_as_suggestion}, changed {changed}, factual {factual}, failed {failed}")
