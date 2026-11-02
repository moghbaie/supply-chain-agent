"""Deterministic escalation policy, and overrides applied after any classifier.

Nothing here places an order: every action other than monitor still needs a person's approval.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODEL_DISAGREEMENT = 0.30
OVERRIDE_RISK = 0.80


@dataclass(frozen=True)
class PolicyThresholds:
    act: float
    critical: float


def policy_baseline(cases, thresholds):
    """Choose a category from risk, open orders, value and model agreement."""
    risk = cases.stockout_risk
    disagree = (risk - cases.stockout_risk_lightgbm).abs() >= MODEL_DISAGREEMENT
    critical = (risk >= thresholds.critical) & (cases.value_band == "high")
    act = risk >= thresholds.act
    return np.select([critical | (disagree & (risk >= thresholds.act)), act & (cases.open_pos > 0), act & (cases.open_pos == 0)],
                     ["human_review", "expedite_recommendation", "reorder_recommendation"], "monitor")


def apply_overrides(cases, choices):
    """Correct classifier choices that conflict with known facts, and return choices with a reason for each change."""
    result = pd.Series(np.asarray(choices, dtype=object), index=cases.index)
    reason = pd.Series("", index=cases.index, dtype=object)
    for mask, value, why in [
        ((result == "monitor") & (cases.stockout_risk >= OVERRIDE_RISK), "human_review", "high risk marked monitor"),
        ((result == "reorder_recommendation") & (cases.open_pos > 0), "expedite_recommendation", "reorder with an order on the way"),
        ((result == "expedite_recommendation") & (cases.open_pos == 0), "reorder_recommendation", "expedite with no open order"),
    ]:
        result[mask] = value
        reason[mask] = why
    return result.to_numpy(dtype=object), reason.to_numpy(dtype=object)


def requires_approval(choices):
    """Every action other than monitor needs a person's approval before any purchase."""
    return np.asarray(choices) != "monitor"
