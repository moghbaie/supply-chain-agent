"""Audit every test explanation against its own evidence, and check how often code applied the policy's conflict rule. Calls no API."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.agent.audit import audit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_colwidth", 70)
TRACES = Path("data/processed/agent/traces/smollm_untuned/test")
rows = []
for path in sorted(TRACES.glob("test-*.json")):
    trace = json.loads(path.read_text(encoding="utf-8"))
    risk = trace["evidence"]["get_stockout_risk"]
    rows.append({"case_id": trace["case_id"], "final": trace["review"]["final_category"], "problems": audit(trace["recommendation"]["explanation"], trace["evidence"]), "explanation": trace["recommendation"]["explanation"], "gap": risk["models_differ_by_points"], "max_risk": max(risk["tabpfn_risk_percent"], risk["lightgbm_risk_percent"])})
frame = pd.DataFrame(rows)
print(f"explanations {len(frame)}; with at least one rule broken {int(frame.problems.map(bool).sum())}; distinct explanations {frame.explanation.nunique()}")
print(frame.explode("problems").problems.value_counts().to_string())
factual = frame.problems.map(lambda found: any(name != "cites_no_numbers" for name in found))
print(f"with a factual rule broken, leaving out cites_no_numbers: {int(factual.sum())}")
print("repeated word for word:", {text[:60]: int(count) for text, count in frame.explanation.value_counts().items() if count > 1})
print("\nBY FINAL CATEGORY\n", frame.assign(any_problem=frame.problems.map(bool), factual=factual).groupby("final")[["any_problem", "factual"]].agg(["sum", "size"]).to_string())
print("\nFINAL MONITOR WITH A FACTUAL ERROR (no person reviews these)\n", frame[factual & (frame.final == "monitor")][["case_id", "problems", "max_risk"]].to_string(index=False))
print("\nPOLICY CONFLICT RULE: stockout models differ by 30 points or more")
print(frame[frame.gap >= 30][["case_id", "final", "gap", "max_risk"]].to_string(index=False))
frame.assign(problems=frame.problems.map("; ".join)).drop(columns="explanation").to_csv("reports/step_08/explanation_audit.csv", index=False)
