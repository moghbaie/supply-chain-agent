"""Rule-based audit of explanations against the evidence in a trace. Each rule is narrow, so the counts are a lower bound on errors."""
import re

from src.agent.review import NUMBER

NO_ORDERS = re.compile(r"no open orders?\b(?! that)|open orders? (?:is|are) empty|no pending orders?", re.I)
ORDER_CLAIM = re.compile(r"open orders? (?:is|are) (?:not expected|delayed|due|expected|pending|on track|at risk)|open orders? (?:may|that may)", re.I)
DELAYED = re.compile(r"open orders? (?:is|are) delayed", re.I)
HIGH_RISK = re.compile(r"high risk of stockout|stockout risk is (?:high|likely)|risk of stockout is high", re.I)
LOW_RISK = re.compile(r"low risk of stockout|stockout risk is low|risk of stockout is low", re.I)
RISK_NUMBER = re.compile(r"(?:stockout risk of|risk of stockout is|probability of stockout \(|stockout \()\s*(\d+(?:\.\d+)?)%", re.I)
EXCEEDS_STOCK = re.compile(r"(?:demand|forecast)[^.]*(?:exceeding|higher than)[^.]*(?:stock|on-hand)", re.I)
HIGH_RISK_LIMIT, LOW_RISK_LIMIT = 20.0, 50.0


def audit(explanation, evidence):
    """Return the name of each rule the explanation breaks."""
    stock = evidence.get("get_inventory_status", {})
    risk = evidence.get("get_stockout_risk", {})
    orders = evidence.get("get_open_orders", {})
    forecast = evidence.get("get_demand_forecast", {})
    suggestion = evidence.get("get_escalation_suggestion", {})
    risks = [value for value in (risk.get("tabpfn_risk_percent"), risk.get("lightgbm_risk_percent")) if value is not None]
    listed = orders.get("open_orders", [])
    found = []
    if listed and NO_ORDERS.search(explanation):
        found.append("says_no_open_orders_but_one_exists")
    if not listed and ORDER_CLAIM.search(explanation):
        found.append("describes_an_open_order_that_does_not_exist")
    if listed and DELAYED.search(explanation) and not orders.get("units_overdue"):
        found.append("says_delayed_but_nothing_overdue")
    if risks and HIGH_RISK.search(explanation) and max(risks) < HIGH_RISK_LIMIT:
        found.append("says_high_risk_but_both_models_under_20")
    if risks and LOW_RISK.search(explanation) and max(risks) >= LOW_RISK_LIMIT:
        found.append("says_low_risk_but_a_model_at_50_or_more")
    if risks:
        for text in RISK_NUMBER.findall(explanation):
            if not any(abs(float(text) - value) < 1 for value in risks):
                found.append("stockout_risk_number_not_from_either_model")
                break
    if EXCEEDS_STOCK.search(explanation) and forecast.get("forecast_units_next_14_days") is not None and stock.get("on_hand_units") is not None and forecast["forecast_units_next_14_days"] <= stock["on_hand_units"]:
        found.append("says_demand_exceeds_stock_but_it_does_not")
    if "conflicts with the escalation suggestion" in explanation and suggestion.get("suggested_category") == "human_review":
        found.append("says_it_conflicts_with_the_suggestion_it_followed")
    if not NUMBER.search(explanation):
        found.append("cites_no_numbers")
    return found
