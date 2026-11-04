"""Compare the LLM agent with the code-only result on the sampled replay cases, and audit its explanations. Calls no API."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.agent.audit import audit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 120)
TRACES = Path("data/processed/agent/traces/smollm_untuned/replay")
OUT = Path("reports/step_10")
key = pd.read_parquet(TRACES / "key.parquet").rename(columns={"reference": "code_final"})
rows = []
for path in sorted(TRACES.glob("replay-*.json")):
    trace = json.loads(path.read_text(encoding="utf-8"))
    review = trace["review"]
    explanation = (trace["recommendation"] or {}).get("explanation", "")
    rows.append({"case_id": trace["case_id"], "status": trace["status"], "model": (trace["recommendation"] or {}).get("category"), "final": review["final_category"], "flags": "; ".join(review["flags"]), "order_placed": review["order_placed"], "approval_required": review["approval_required"],
                 "problems": audit(explanation, trace["evidence"]) if explanation else [], "explanation": explanation, "turns": len(trace["turns"]), "seconds": trace["seconds"], "input_tokens": trace["input_tokens"], "output_tokens": trace["output_tokens"], "malformed": trace["malformed_calls"], "invalid": trace["invalid_submissions"], "peak_gpu_gb": trace.get("peak_gpu_gb"),
                 "policy_calls": sum(call["name"] == "search_policy" for call in trace["tool_calls"])})
frame = key.merge(pd.DataFrame(rows), on="case_id", validate="one_to_one")
print(f"cases {len(frame)}; status {frame.status.value_counts().to_dict()}; orders placed {int(frame.order_placed.sum())}; scenarios {frame.scenario.value_counts().to_dict()}")
print(f"LLM chose UltraBERT's category: {int((frame.model == frame.ultrabert).sum())} of {len(frame)}")
print(f"final equals the code-only result: {int((frame.final == frame.code_final).sum())} of {len(frame)}")
print("\nCONFUSION, code-only (rows) against LLM agent final (columns)\n", pd.crosstab(frame.code_final, frame.final).to_string())
changed = frame[frame.final != frame.code_final]
print(f"\nCASES WHERE THE FINAL RESULT CHANGED: {len(changed)}\n", changed[["case_id", "scenario", "ultrabert", "model", "code_final", "final", "flags"]].to_string(index=False))
print("\nFLAGS\n", frame["flags"].str.split("; ").explode().str.split(":").str[0].replace("", np.nan).value_counts().to_string())
factual = frame.problems.map(lambda found: any(name != "cites_no_numbers" for name in found))
print(f"\nEXPLANATIONS: any rule broken {int(frame.problems.map(bool).sum())}; factual rule broken {int(factual.sum())}; distinct {frame.explanation.nunique()}")
print(frame.explode("problems").problems.value_counts().to_string())
print("\nBY FINAL CATEGORY\n", frame.assign(factual=factual).groupby("final").factual.agg(["size", "sum"]).to_string())
monitor = frame[factual & (frame.final == "monitor")]
print(f"\nfinal monitor with a factual error, which no person reviews: {len(monitor)} of {int((frame.final == 'monitor').sum())}")
print(f"\nRUN COST: seconds median {frame.seconds.median():.0f}, mean {frame.seconds.mean():.0f}, 95th {frame.seconds.quantile(.95):.0f}, max {frame.seconds.max():.0f}; total {frame.seconds.sum():.0f}; turns mean {frame.turns.mean():.1f}; tokens mean {frame.input_tokens.mean():.0f} in, {frame.output_tokens.mean():.0f} out; peak GPU {frame.peak_gpu_gb.max():.2f} GB")
print(f"errors: malformed calls {int(frame.malformed.sum())}, invalid submissions {int(frame.invalid.sum())}, search_policy calls {int(frame.policy_calls.sum())}")
frame.assign(problems=frame.problems.map("; ".join)).to_csv(OUT / "llm_sample_results.csv", index=False)
print("\nFIRST TWO EXPLANATIONS PER FINAL CATEGORY")
for category, group in frame.groupby("final"):
    for item in group.head(2).itertuples():
        print(f"[{item.case_id} {item.scenario} {category}] {item.explanation}")
