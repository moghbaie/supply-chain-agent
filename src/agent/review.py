"""Deterministic checks on the agent's recommendation. They run in code, after the model, and the model cannot skip them."""
import re

import numpy as np
import pandas as pd

from src.agent.tools import REQUIRED_TOOLS
from src.escalation.policy import apply_overrides, requires_approval

LOW_RISK_PERCENT = 10.0
ALWAYS_ALLOWED_NUMBERS = {14.0, 28.0, 30.0}
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in(value):
    """Collect every number in a tool result, in any nesting."""
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float)):
        return {float(value)}
    if isinstance(value, dict):
        return set().union(*(numbers_in(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(numbers_in(v) for v in value)) if value else set()
    return set()


def unsupported_numbers(explanation, evidence):
    """Return numbers in the explanation that no tool reported. Rounding to a whole number is allowed."""
    known = numbers_in(evidence)
    missing = []
    for text in NUMBER.findall(explanation):
        number = float(text.replace(",", ""))
        if number not in ALWAYS_ALLOWED_NUMBERS and not any(abs(number - k) < 1e-9 or round(k) == number for k in known):
            missing.append(number)
    return missing


def review(category, explanation, evidence, risks, open_orders):
    """Return the final category, whether a person must approve, and a flag for each problem found."""
    flags = []
    missing_tools = sorted(REQUIRED_TOOLS - set(evidence))
    if missing_tools:
        flags.append("missing_evidence:" + ",".join(missing_tools))
    bad_numbers = unsupported_numbers(explanation, evidence)
    if bad_numbers:
        flags.append("unsupported_numbers:" + ",".join(map(str, bad_numbers)))
    known_risks = [r for r in risks if r is not None]
    frame = pd.DataFrame({"stockout_risk": [max(known_risks) / 100 if known_risks else np.nan], "open_pos": [open_orders if open_orders is not None else np.nan]})
    changed, reason = apply_overrides(frame, [category])
    final = str(changed[0])
    if reason[0]:
        flags.append(f"override:{reason[0]}")
    if missing_tools or (bad_numbers and final == "monitor"):
        final = "human_review"
    if final not in ("monitor", "human_review") and known_risks and max(known_risks) < LOW_RISK_PERCENT:
        flags.append("weak_evidence:action with both stockout risks under 10 percent")
    return {"final_category": final, "approval_required": bool(requires_approval([final])[0]), "order_placed": False, "flags": flags}
