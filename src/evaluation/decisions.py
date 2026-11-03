"""Turn a category into a simulated action. A person's approval is simulated here, and nothing leaves the simulator."""
import numpy as np

from src.escalation.policy import MODEL_DISAGREEMENT, apply_overrides, requires_approval

APPROVE_ALL, DECLINE_REVIEW, DECLINE_ALL = "approve_all", "decline_review", "decline_all"
LOW_RISK = 0.10


def code_checks(cases, choices, conflict_rule=False):
    """Apply the agent's checks in code and return final categories with a reason for each change."""
    final, reason = apply_overrides(cases, choices)
    final, reason = final.copy(), reason.copy()
    gap = (cases.stockout_risk - cases.stockout_risk_lightgbm).abs().to_numpy()
    unknown = cases[["stockout_risk", "stockout_risk_lightgbm"]].isna().any(axis=1).to_numpy()
    if conflict_rule:
        conflict = (gap >= MODEL_DISAGREEMENT) & (final != "human_review")
        final[conflict], reason[conflict] = "human_review", "stockout models conflict"
    final[unknown], reason[unknown] = "human_review", "stockout risk unavailable"
    return final, reason


def weak_evidence(cases, final):
    """Mark actions where both stockout models put the risk under 10 percent."""
    return (np.isin(final, ["reorder_recommendation", "expedite_recommendation"])) & (cases[["stockout_risk", "stockout_risk_lightgbm"]].max(axis=1).to_numpy() < LOW_RISK)


def simulated_person(category, open_orders, person=APPROVE_ALL):
    """Return the action a simulated person approves: none, order or expedite."""
    if not requires_approval([category])[0] or person == DECLINE_ALL:
        return "none"
    if category == "human_review":
        if person == DECLINE_REVIEW:
            return "none"
        return "expedite" if open_orders > 0 else "order"
    return "order" if category == "reorder_recommendation" else "expedite"
