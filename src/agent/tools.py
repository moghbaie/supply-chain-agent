"""Typed tools the agent can call. Each reads saved point-in-time results and can be called on its own.

The store is built from an allowed column list, so outcomes, scenario names and item IDs cannot reach a tool result.
No tool can place, change or cancel a purchase order.
"""
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agent.retrieval import PolicyIndex
from src.escalation.cases import CATEGORIES
from src.simulation.engine import purchase_orders_as_of

CASE_COLUMNS = ["on_hand", "on_order", "cover_days", "open_pos", "ordered_today", "value_band", "sales_mean_28", "sold_out_days_28", "fc_mean_14", "fc_q10_sum_14", "fc_q90_sum_14",
                "stockout_risk", "stockout_risk_lightgbm", "days_to_next_due", "units_due_14", "units_overdue", "days_overdue_max", "supplier_late_rate", "supplier_recent_late_rate"]
SCORE_COLUMNS = [f"p_{category}" for category in CATEGORIES]
REQUIRED_TOOLS = {"get_inventory_status", "get_demand_forecast", "get_stockout_risk", "get_open_orders", "get_escalation_suggestion"}
Category = Literal["monitor", "reorder_recommendation", "expedite_recommendation", "human_review"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseRequest(Strict):
    case_id: str = Field(min_length=1, description="The case being reviewed")


class PolicyRequest(Strict):
    query: str = Field(min_length=2, max_length=200, description="Words describing the rule to find")


class Recommendation(Strict):
    category: Category = Field(description="The recommended category")
    explanation: str = Field(min_length=10, max_length=700, description="Why, citing only values from tool results")
    uncertainty: Literal["low", "medium", "high"] = Field(description="How unsettled the evidence is")


def clean(value, digits=1):
    """Round a number and turn a missing value into None."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def percent(value):
    number = clean(value, 6)
    return None if number is None else round(100 * number, 1)


def with_missing(values):
    """List missing fields explicitly, so the agent can say what is unknown."""
    return {**values, "missing_fields": [key for key, value in values.items() if value is None]}


@dataclass
class ToolBox:
    """Serve evidence for registered cases, identified only by an anonymous case ID."""

    cases: pd.DataFrame
    open_orders: dict = field(default_factory=dict)
    policy: PolicyIndex = field(default_factory=PolicyIndex)

    def _case(self, case_id):
        if case_id not in self.cases.index:
            raise KeyError(f"unknown case_id {case_id}")
        return self.cases.loc[case_id]

    def get_inventory_status(self, case_id):
        row = self._case(case_id)
        return with_missing({"on_hand_units": clean(row.on_hand, 0), "on_order_units": clean(row.on_order, 0), "days_of_cover": clean(row.cover_days), "open_orders": clean(row.open_pos, 0),
                             "ordered_at_this_review": bool(row.ordered_today > 0), "sales_per_day_last_28_days": clean(row.sales_mean_28, 2), "sold_out_days_last_28_days": clean(row.sold_out_days_28, 0), "value_band": row.value_band})

    def get_demand_forecast(self, case_id):
        row = self._case(case_id)
        return with_missing({"forecast_units_next_14_days": clean(row.fc_mean_14), "low_estimate_units": clean(row.fc_q10_sum_14), "high_estimate_units": clean(row.fc_q90_sum_14), "model": "Timestral-28M"})

    def get_stockout_risk(self, case_id):
        row = self._case(case_id)
        first, second = percent(row.stockout_risk), percent(row.stockout_risk_lightgbm)
        gap = None if first is None or second is None else round(abs(first - second), 1)
        return with_missing({"tabpfn_risk_percent": first, "lightgbm_risk_percent": second, "models_differ_by_points": gap})

    def get_open_orders(self, case_id):
        row = self._case(case_id)
        return with_missing({"open_orders": self.open_orders.get(case_id, []), "days_to_next_delivery": clean(row.days_to_next_due, 0), "units_due_next_14_days": clean(row.units_due_14, 0), "units_overdue": clean(row.units_overdue, 0),
                             "days_overdue": clean(row.days_overdue_max, 0), "supplier_late_percent_long_run": percent(row.supplier_late_rate), "supplier_late_percent_last_30_days": percent(row.supplier_recent_late_rate)})

    def get_escalation_suggestion(self, case_id):
        scores = self._case(case_id)[SCORE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("escalation score unavailable")
        return {"suggested_category": CATEGORIES[int(scores.argmax())], "probabilities_percent": {c: round(100 * float(s), 1) for c, s in zip(CATEGORIES, scores)}, "model": "fine-tuned UltraBERT"}

    def search_policy(self, query):
        return {"passages": self.policy.search(query)}


TOOLS = {"get_inventory_status": (CaseRequest, "Get stock on hand, days of cover, recent sales and the value band."),
         "get_demand_forecast": (CaseRequest, "Get forecast demand for the next 14 days with low and high estimates."),
         "get_stockout_risk": (CaseRequest, "Get the chance of a stockout in the next 14 days from two models."),
         "get_open_orders": (CaseRequest, "Get open purchase orders, their delay risk and the supplier's late rates."),
         "get_escalation_suggestion": (CaseRequest, "Get the escalation classifier's suggested category and probabilities."),
         "search_policy": (PolicyRequest, "Find project policy rules about categories, approvals and conflicting evidence."),
         "submit_recommendation": (Recommendation, "Submit the final recommendation for a person to review. This places no order.")}


def tool_schemas(hidden=()):
    """Describe the tools in the format chat templates expect, leaving out any hidden ones."""
    schemas = []
    for name, (model, description) in TOOLS.items():
        if name in hidden:
            continue
        parameters = model.model_json_schema()
        parameters.pop("title", None)
        for item in parameters["properties"].values():
            item.pop("title", None)
        schemas.append({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
    return schemas


def call_tool(box, name, arguments):
    """Validate a call and run it, returning a result or an error the agent can read. Failures never become evidence."""
    if name not in TOOLS or name == "submit_recommendation":
        return {"ok": False, "error": f"unknown tool {name}. Available tools only read evidence, and orders cannot be placed."}
    try:
        request = TOOLS[name][0](**(arguments if isinstance(arguments, dict) else {}))
        return {"ok": True, "result": getattr(box, name)(**request.model_dump())}
    except ValidationError as error:
        return {"ok": False, "error": f"invalid arguments for {name}: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())}
    except Exception as error:
        return {"ok": False, "error": f"{name} failed: {error}"}


def open_orders_at(orders, origin, delay_scores):
    """List orders open at a review date using promised dates only, with delay risk scored when each was placed."""
    pending = purchase_orders_as_of(orders, origin)
    pending = pending[pending.status == "open"].sort_values("promised_date")
    return [{"units": clean(item.quantity, 0), "days_until_promised": int((item.promised_date - pd.Timestamp(origin)).days), "delay_risk_percent": percent(delay_scores.get(item.po_id))} for item in pending.itertuples(index=False)]


def build_toolbox(cases, orders, delay_scores, prefix="case"):
    """Register cases under anonymous IDs. Return the toolbox and a separate key table, which the agent never sees."""
    cases = cases.reset_index(drop=True)
    ids = [f"{prefix}-{position:04d}" for position in range(len(cases))]
    table = cases[CASE_COLUMNS + SCORE_COLUMNS].set_axis(ids)
    lookup = {key: group for key, group in orders.groupby(["scenario", "id"])}
    empty = orders.iloc[:0]
    listed = {case_id: open_orders_at(lookup.get((row.scenario, row.id), empty), row.origin, delay_scores) for case_id, row in zip(ids, cases.itertuples(index=False))}
    return ToolBox(table, listed), cases.assign(case_id=ids)
