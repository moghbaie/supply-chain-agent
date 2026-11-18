"""Measure how much of each explanation the judge can check, from the text alone. Needs no evidence, so it says nothing about whether claims are true.

Reads the 100 explanations in reports/step_08/test_results.csv, which is committed. Calls no API.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.agent.judge import clauses, extract_claims, sentences


def coverage(frame):
    rows, unbound = [], []
    for case_id, explanation in zip(frame.case_id, frame.explanation):
        claims = extract_claims(case_id, explanation)
        numbers = [c for c in claims if c.kind == "number"]
        unbound += [(case_id, c.text, c.clause) for c in numbers if c.group is None]
        rows.append({"case_id": case_id, "clauses": sum(len(clauses(s)) for s in sentences(explanation)), "numbers": len(numbers), "bound": sum(c.group is not None for c in numbers),
                     "open_orders": sum(c.kind == "open_orders" for c in claims), "value_band": sum(c.kind == "value_band" for c in claims), "qualitative": sum(c.kind == "qualitative" for c in claims)})
    return pd.DataFrame(rows), unbound


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    frame = pd.read_csv(ROOT / "reports/step_08/test_results.csv")
    table, unbound = coverage(frame)
    numbers, bound = int(table.numbers.sum()), int(table.bound.sum())
    print(f"explanations {len(table)}")
    print(f"clauses {int(table.clauses.sum())}, median {table.clauses.median():g} per explanation")
    print(f"numbers stated {numbers}; bound to a named field {bound} ({100 * bound / numbers:.1f}%)")
    print(f"explanations with no number {int((table.numbers == 0).sum())}")
    print(f"open-order claims checked by code {int(table.open_orders.sum())}; value-band claims {int(table.value_band.sum())}")
    print(f"clauses sent to the LLM judge {int(table.qualitative.sum())}, in {int((table.qualitative > 0).sum())} explanations")
    print("\nUNBOUND NUMBERS (still checked against every tool result)")
    for case_id, text, clause in unbound:
        print(f"  {case_id} {text!r:8} in: {clause}")


if __name__ == "__main__":
    main()
