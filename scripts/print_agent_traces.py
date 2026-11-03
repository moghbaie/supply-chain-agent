"""Print 10 complete test traces, chosen by rule, and every explanation for reading. Calls no API."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TRACES = Path("data/processed/agent/traces/smollm_untuned/test")
results = pd.read_csv("reports/step_08/test_results.csv").fillna({"flags": ""})

# The first two cases of each category, then the first two with an unsupported number, so the sample is not picked by how it reads.
chosen = list(results.groupby("reference", sort=False).head(2).case_id)
chosen += [case_id for case_id in results[results["flags"].str.contains("unsupported")].case_id if case_id not in chosen][:2]
for case_id in chosen:
    trace = json.loads((TRACES / f"{case_id}.json").read_text(encoding="utf-8"))
    row = results[results.case_id == case_id].iloc[0]
    print("=" * 110)
    print(f"{case_id} | reference {row.reference} (hidden from agent) | UltraBERT {row.ultrabert} | {len(trace['turns'])} turns | {trace['input_tokens']}+{trace['output_tokens']} tokens | {trace['seconds']} s | $0")
    for call in trace["tool_calls"]:
        print(f"  turn {call['turn']} {call['name']} {json.dumps(call['arguments'])[:760]}\n    ok={call['ok']} -> {json.dumps(call['output'])}")
    print(f"  EXPLANATION: {(trace['recommendation'] or {}).get('explanation')}")
    print(f"  MODEL: {(trace['recommendation'] or {}).get('category')} uncertainty={(trace['recommendation'] or {}).get('uncertainty')}")
    print(f"  FINAL: {trace['review']}")

print("=" * 110)
print("ALL EXPLANATIONS")
for row in results.itertuples(index=False):
    print(f"{row.case_id} {row.reference[:8]} final={row.agent[:8]}: {row.explanation}")
