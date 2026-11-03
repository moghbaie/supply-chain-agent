"""Score the agent on the saved test traces against UltraBERT, the policy and saved Jev answers on the same cases. Calls no API."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.escalation.cases import CATEGORIES
from src.escalation.policy import apply_overrides

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 60)
SEED, ROUNDS = 42, 1000
REPORT = Path("reports/step_08")
DATA = Path("data/processed/agent")
TRACES = DATA / "traces/smollm_untuned/test"
SHORT = ["monitor", "reorder", "expedite", "human"]
REAL_MIX = pd.read_parquet("data/processed/escalation/classified_test.parquet").reference.value_counts().reindex(CATEGORIES)

traces = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(TRACES.glob("test-*.json"))}
key = pd.read_parquet(TRACES / "key.parquet")
cases = pd.read_parquet(DATA / "test_cases.parquet").reset_index(drop=True).assign(case_id=key.case_id)
assert (cases.reference == key.reference).all() and (cases.id == key.id).all()
rows = [{"case_id": case_id, "status": t["status"], "model_choice": (t["recommendation"] or {}).get("category"), "agent": t["review"]["final_category"], "uncertainty": (t["recommendation"] or {}).get("uncertainty"), "approval_required": t["review"]["approval_required"],
         "order_placed": t["review"]["order_placed"], "flags": "; ".join(t["review"]["flags"]), "turns": len(t["turns"]), "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"], "seconds": t["seconds"], "peak_gpu_gb": t["peak_gpu_gb"],
         "malformed": t["malformed_calls"], "untagged": t.get("untagged_calls", 0), "invalid_submissions": t["invalid_submissions"], "bad_tool_calls": sum(not c["ok"] and c["name"] != "submit_recommendation" for c in t["tool_calls"]), "policy_searches": sum(c["name"] == "search_policy" for c in t["tool_calls"]),
         "repeating": bool(t.get("stopped_repeating")), "explanation": (t["recommendation"] or {}).get("explanation")} for case_id, t in traces.items()]
scored = cases.merge(pd.DataFrame(rows), on="case_id", validate="one_to_one")
scored["ultrabert_override"] = apply_overrides(scored, scored.ultrabert)[0]
print(f"traces {len(scored)}, series {scored.id.nunique()}, origins {scored.origin.min().date()} to {scored.origin.max().date()}, scenarios {scored.scenario.value_counts().to_dict()}")
print("reference", scored.reference.value_counts().reindex(CATEGORIES).to_dict())
print("status", scored.status.value_counts().to_dict())

NAMES = {"agent_final": "agent", "agent_model_only": "model_choice", "ultrabert": "ultrabert", "ultrabert_with_overrides": "ultrabert_override", "policy": "policy", "jev_saved": "jev"}


def metrics(reference, choices, weights):
    """Score balanced as run, and reweighted so each reference category counts as often as in the full test set."""
    critical = reference == "human_review"
    matrix = confusion_matrix(reference, choices, labels=CATEGORIES).astype(float)
    per_row = matrix / matrix.sum(axis=1, keepdims=True)
    real = per_row * weights[:, None]
    hits, actual, predicted = np.diag(real), real.sum(axis=1), real.sum(axis=0)
    f1 = np.divide(2 * hits, actual + predicted, out=np.zeros(4), where=actual + predicted > 0)
    quiet = CATEGORIES.index("monitor")
    return {"macro_f1_balanced": f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0), "accuracy_balanced": float((reference == choices).mean()), "critical_recall": float((choices[critical] == "human_review").mean()),
            "critical_acted_on": float((choices[critical] != "monitor").mean()), "false_alarm_share": per_row[quiet, 1:].sum(), "macro_f1_real_mix": f1.mean(), "actions_per_27000": real[:, 1:].sum()}


reference = scored.reference.to_numpy(dtype=object)
weights = REAL_MIX.to_numpy(dtype=float)
table = pd.DataFrame([{"method": name, **metrics(reference, scored[column].to_numpy(dtype=object), weights)} for name, column in NAMES.items()]).set_index("method")
print("\nMETRICS ON THE SAME 100 CASES (synthetic simulator data)\n", table.round(4).T.to_string())
for name, column in NAMES.items():
    print(f"\nCONFUSION {name} (rows reference, columns predicted)\n", pd.DataFrame(confusion_matrix(reference, scored[column].to_numpy(dtype=object), labels=CATEGORIES), index=SHORT, columns=SHORT).to_string())

print("\nAGREEMENT")
print(f"model choice same as UltraBERT: {int((scored.model_choice == scored.ultrabert).sum())} of {len(scored)}")
print(f"final differs from model choice: {int((scored.agent != scored.model_choice).sum())}")
print(f"final same as UltraBERT with overrides: {int((scored.agent == scored.ultrabert_override).sum())}")
print("\nCASES WHERE THE MODEL DIFFERED FROM ULTRABERT OR THE CHECKS CHANGED ITS CHOICE\n", scored[(scored.model_choice != scored.ultrabert) | (scored.agent != scored.model_choice)][["case_id", "reference", "ultrabert", "model_choice", "agent", "flags"]].to_string(index=False))

print("\nFLAGS")
kinds = pd.Series([flag.split(":")[0] for text in scored["flags"] for flag in text.split("; ") if flag], dtype=object)
print(kinds.value_counts().to_dict())
print("override reasons", pd.Series([flag.split(":")[1] for text in scored["flags"] for flag in text.split("; ") if flag.startswith("override")], dtype=object).value_counts().to_dict())
print("weak_evidence by reference", scored[scored["flags"].str.contains("weak_evidence")].reference.value_counts().to_dict())
print("unsupported numbers\n", scored[scored["flags"].str.contains("unsupported")][["case_id", "agent", "flags"]].to_string(index=False))
print("uncertainty", scored.uncertainty.value_counts().to_dict(), "| human_review final", int((scored.agent == "human_review").sum()))

print("\nRUN COST")
print(scored[["turns", "input_tokens", "output_tokens", "seconds", "peak_gpu_gb"]].describe().loc[["mean", "50%", "max", "min"]].round(2).to_string())
print(f"95th percentile seconds {scored.seconds.quantile(.95):.1f}; total tokens {int(scored.input_tokens.sum()):,} in, {int(scored.output_tokens.sum()):,} out; API cost $0 (runs locally)")
print("errors", {c: int(scored[c].sum()) for c in ["malformed", "untagged", "invalid_submissions", "bad_tool_calls", "policy_searches", "repeating"]}, "| cases with untagged", int((scored.untagged > 0).sum()))
print(f"order_placed true {int(scored.order_placed.sum())}; approval missing on an action {int(((scored.agent != 'monitor') & ~scored.approval_required).sum())}; approval set on monitor {int(((scored.agent == 'monitor') & scored.approval_required).sum())}")

print("\nACTUAL STOCKOUT IN THE NEXT 14 DAYS BY FINAL CATEGORY (label, never shown to the agent)")
print(scored.groupby("agent").label.agg(["size", "sum", "mean"]).round(3).to_string())

rng = np.random.default_rng(SEED)
groups = list(scored.groupby("id").indices.values())
pairs = [("agent_final", "ultrabert"), ("agent_final", "ultrabert_with_overrides"), ("agent_final", "policy"), ("agent_final", "jev_saved"), ("ultrabert", "policy")]
gaps = {pair: {"macro_f1_balanced": [], "macro_f1_real_mix": [], "critical_recall": [], "false_alarm_share": []} for pair in pairs}
done = 0
while done < ROUNDS:
    picked = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    truth = reference[picked]
    if any((truth == c).sum() == 0 for c in CATEGORIES):
        continue
    done += 1
    results = {name: metrics(truth, scored[column].to_numpy(dtype=object)[picked], weights) for name, column in NAMES.items() if name in {n for p in pairs for n in p}}
    for pair in pairs:
        for measure in gaps[pair]:
            gaps[pair][measure].append(results[pair[0]][measure] - results[pair[1]][measure])
intervals = {f"{m}|{left}-{right}": [round(float(np.percentile(values, q)), 4) for q in (2.5, 50, 97.5)] for (left, right), by in gaps.items() for m, values in by.items()}
print("\nSERIES BOOTSTRAP OF DIFFERENCES [2.5%, median, 97.5%], 1,000 rounds")
for name, value in intervals.items():
    print(f"{name}: {value}")

table.round(5).to_csv(REPORT / "test_metrics.csv")
scored[["case_id", "scenario", "id", "origin", "reference", "label", "ultrabert", "ultrabert_override", "policy", "jev", "model_choice", "agent", "uncertainty", "approval_required", "order_placed", "flags", "turns", "input_tokens", "output_tokens", "seconds", "malformed", "untagged", "explanation"]].to_csv(REPORT / "test_results.csv", index=False)
(REPORT / "test_bootstrap.json").write_text(json.dumps(intervals, indent=2))
