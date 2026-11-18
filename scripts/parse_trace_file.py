"""Rebuild evidence for the complete traces in reports/step_08/test_traces.txt, because the full traces are git-ignored.

Run directly, it checks the parse two ways: the Step 8 audit on parsed evidence must equal the saved explanation_audit.csv,
and, where the local JSON traces exist, parsed evidence must equal the saved evidence exactly. Calls no API.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.tools import REQUIRED_TOOLS

TRACE_FILE = ROOT / "reports/step_08/test_traces.txt"
TRACES = ROOT / "data/processed/agent/traces/smollm_untuned/test"
HEADER = re.compile(r"^(test-\d{4}) \| reference (\w+)")
CALL = re.compile(r"^  turn \d+ (\w+) ")
OUTPUT = re.compile(r"^    ok=(True|False) -> (.*)$")
LISTED = re.compile(r"^(test-\d{4}) \S+ final=\S+: (.*)$")


def parse(path=TRACE_FILE):
    """Return (complete traces keyed by case ID, every explanation keyed by case ID)."""
    traces, explanations, current, tool = {}, {}, None, None
    in_list = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line == "ALL EXPLANATIONS":
            in_list, current = True, None
            continue
        if in_list:
            match = LISTED.match(line)
            if match:
                explanations[match.group(1)] = match.group(2)
            continue
        header = HEADER.match(line)
        if header:
            current = traces.setdefault(header.group(1), {"case_id": header.group(1), "evidence": {}, "explanation": None, "final": None})
            continue
        if current is None:
            continue
        call = CALL.match(line)
        if call:
            tool = call.group(1)
            continue
        output = OUTPUT.match(line)
        if output and tool in REQUIRED_TOOLS and output.group(1) == "True":
            current["evidence"][tool] = json.loads(output.group(2))
        elif line.startswith("  EXPLANATION: "):
            current["explanation"] = line[len("  EXPLANATION: "):]
        elif line.startswith("  FINAL: "):
            current["final"] = ast.literal_eval(line[len("  FINAL: "):])["final_category"]
    complete = {key: value for key, value in traces.items() if set(value["evidence"]) == REQUIRED_TOOLS and value["explanation"]}
    return complete, explanations


def main():
    import pandas as pd

    from src.agent.audit import audit

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    complete, explanations = parse()
    print(f"complete traces {len(complete)}, each with all {len(REQUIRED_TOOLS)} tool results; explanations listed {len(explanations)}")
    saved = pd.read_csv(ROOT / "reports/step_08/explanation_audit.csv").fillna({"problems": ""}).set_index("case_id")
    same = 0
    for case_id, trace in complete.items():
        found = "; ".join(audit(trace["explanation"], trace["evidence"]))
        agree = found == saved.loc[case_id, "problems"]
        same += agree
        print(f"  {case_id} final={trace['final']:24} audit={found or '-':55} saved={saved.loc[case_id, 'problems'] or '-':55} {'same' if agree else 'DIFFERENT'}")
    print(f"Step 8 audit on parsed evidence equals the saved audit: {same} of {len(complete)}")
    if TRACES.exists():
        exact = sum(json.loads((TRACES / f"{case_id}.json").read_text(encoding="utf-8"))["evidence"] == trace["evidence"] for case_id, trace in complete.items())
        print(f"parsed evidence equals the local JSON trace evidence: {exact} of {len(complete)}")
    else:
        print("local JSON traces not present, so the exact evidence check was skipped")


if __name__ == "__main__":
    main()
