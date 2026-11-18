"""Step 11 tests: claim splitting, number binding, verdicts and tolerances, open orders, value band, the committed traces,
the LLM interface with a scripted model, reply parsing, position-bias orders, agreement scoring and coverage. No test calls an API."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent.audit import audit
from src.agent.judge import (DEFINITIONS, ERROR_VERDICTS, TOOL_ORDER, Claim, check_number, clauses, extract_claims, flatten, judge_explanation, judge_prompt, parse_reply,
                             sentences, stated_numbers, summarize, tool_orders)
from src.agent.review import unsupported_numbers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.judge_agreement import cohen_kappa, score  # noqa: E402
from scripts.judge_coverage import coverage  # noqa: E402
from scripts.parse_trace_file import parse  # noqa: E402

TRACES = ROOT / "data/processed/agent/traces/smollm_untuned/test"


def evidence(**changes):
    """A small, complete set of tool results. Keyword names are tool.field, such as risk__tabpfn_risk_percent."""
    base = {
        "get_inventory_status": {"on_hand_units": 30.0, "on_order_units": 28.0, "days_of_cover": 23.6, "open_orders": 1.0, "ordered_at_this_review": False, "sales_per_day_last_28_days": 1.39,
                                 "sold_out_days_last_28_days": 3.0, "value_band": "high", "missing_fields": []},
        "get_demand_forecast": {"forecast_units_next_14_days": 17.8, "low_estimate_units": 0.0, "high_estimate_units": 44.6, "model": "Timestral-28M", "missing_fields": []},
        "get_stockout_risk": {"tabpfn_risk_percent": 0.2, "lightgbm_risk_percent": 0.3, "models_differ_by_points": 0.1, "missing_fields": []},
        "get_open_orders": {"open_orders": [{"units": 28.0, "days_until_promised": 4, "delay_risk_percent": 5.2}], "days_to_next_delivery": 4.0, "units_due_next_14_days": 28.0, "units_overdue": 0.0,
                            "days_overdue": 0.0, "supplier_late_percent_long_run": 3.6, "supplier_late_percent_last_30_days": 5.3, "missing_fields": []},
        "get_escalation_suggestion": {"suggested_category": "monitor", "probabilities_percent": {"monitor": 94.3, "reorder_recommendation": 0.0, "expedite_recommendation": 5.0, "human_review": 0.6},
                                      "model": "fine-tuned UltraBERT"},
    }
    tools = {"inventory": "get_inventory_status", "forecast": "get_demand_forecast", "risk": "get_stockout_risk", "orders": "get_open_orders", "suggestion": "get_escalation_suggestion"}
    for key, value in changes.items():
        tool, name = key.split("__")
        base[tools[tool]][name] = value
    return base


def numbers(text, ev=None):
    """(stated text, group, verdict) for each number claim."""
    found = judge_explanation("case", text, ev or evidence())
    return [(c.text, c.group, c.verdict) for c in found if c.kind == "number"]


def first_number(text, ev=None):
    return numbers(text, ev)[0]


class Scripted:
    """Stands in for the LLM: returns preset replies and records each prompt."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, messages, tools=None):
        self.prompts.append(messages)
        return {"text": self.replies.pop(0)}


# Splitting -------------------------------------------------------------------------------------------------------------

def test_sentences_do_not_split_at_decimals():
    assert sentences("The risk is 5.2%. Orders are 3.0 units. Done.") == ["The risk is 5.2%.", "Orders are 3.0 units.", "Done."]


def test_clauses_split_at_commas_outside_parentheses():
    assert clauses("Risk is high (90.9% from tabpfn, 96.0% from lightgbm), and there is an open order.") == ["Risk is high (90.9% from tabpfn, 96.0% from lightgbm)", "and there is an open order"]


def test_clauses_keep_respectively_with_its_list():
    assert clauses("Late rates are 4.5% and 5.0% for long run and last 30 days, respectively.") == ["Late rates are 4.5% and 5.0% for long run and last 30 days, respectively"]


@pytest.mark.parametrize("text", ["over the next 14 days", "over the last 30 days", "sales in the last 28 days", "over 14 days", "for long-run and last-30-days"])
def test_horizons_are_not_claims(text):
    assert stated_numbers(f"Forecast of 5 units {text}") == [("5", 5.0, 12, 13, "units")]


# Binding: patterns taken from real explanations ------------------------------------------------------------------------

@pytest.mark.parametrize("text, group", [
    ("The inventory status shows 30 units on hand and 28 on order", "on_hand"),
    ("The inventory is currently at 30 units with a days of cover of 23.6 days", "on_hand"),
    ("with a days of cover of 23.6", "cover"),
    ("a low estimate of 0 units and a high estimate of 44.6 units", "low_estimate"),
    ("The tabpfn model indicates a 0.2% risk, while the lightgbm model suggests a 0.3% risk", "tabpfn"),
    ("the stockout risk is 0.3% (lightgbm) and 0.2% (tabpfn)", "lightgbm"),
    ("with a tabpfn risk of 0.2% and a lightgbm risk of 0.3%", "tabpfn"),
    ("the supplier's late rates are 3.6% long run and 5.3% last 30 days", "late_long"),
    ("the supplier's late rates are 5.3% for the last 30 days and 3.6% for the long run", "late_recent"),
    ("the supplier's late rates are 3.6% and 5.3% for long run and last 30 days, respectively", "late_long"),
    ("The suggested category is 'monitor' with a high probability of 94.3%", "prob_monitor"),
    ("a high probability of stockout (0.3%)", "stockout"),
    ("and a 5.0% probability of expediting", "prob_expedite"),
    ("there is 1 open order with a delay risk of 5.2%", "order_count"),
    ("The open order of 28 units is due in 4 days", "order_units"),
    ("The open orders show 28 units due in 4 days", "order_units"),
    ("The next delivery is in 4 days", "order_days"),
    ("The stockout risk models differ by 0.1 points", "gap"),
    ("with 0.0 units overdue", "order_units"),
    ("with 1.39 sales per day last 28 days", "sales_rate"),
    ("(days_until_promised is 4, with delay_risk_percent 5.2)", "order_days"),
])
def test_binding_patterns_from_real_explanations(text, group):
    stated, bound, verdict = first_number(text)
    assert bound == group, text
    assert verdict == "supported", text


def test_second_number_in_a_pair_binds_to_its_own_model():
    found = numbers("with a tabpfn risk of 0.2% and a lightgbm risk of 0.3%")
    assert found == [("0.2", "tabpfn", "supported"), ("0.3", "lightgbm", "supported")]


def test_number_after_value_band_stays_unbound_and_is_checked_everywhere():
    assert first_number("the medium value band of 30 units") == ("30", None, "unverified")
    assert first_number("the medium value band of 99 units") == ("99", None, "unsupported")


def test_a_number_near_a_value_band_still_binds_to_the_forecast():
    assert first_number("The demand forecast indicates a low value band with 17.8 units expected") == ("17.8", "forecast", "supported")


# Number verdicts and tolerances ----------------------------------------------------------------------------------------

def test_whole_number_may_round_or_truncate():
    assert first_number("a forecast of 18 units") == ("18", "forecast", "supported")
    assert first_number("a forecast of 17 units") == ("17", "forecast", "supported")
    assert first_number("a forecast of 19 units") == ("19", "forecast", "wrong_value")


def test_decimal_must_match_to_its_last_digit():
    assert first_number("a forecast of 17.8 units") == ("17.8", "forecast", "supported")
    assert first_number("a forecast of 17.9 units") == ("17.9", "forecast", "wrong_value")


def test_percent_scale_error_is_wrong_value():
    assert first_number("The stockout risk is 20% according to both models") == ("20", "stockout", "wrong_value")


def test_decimal_belonging_to_another_field_is_misattributed():
    claim = judge_explanation("case", "a stockout risk of 94.3%", evidence())[0]
    assert claim.verdict == "misattributed" and "probabilities_percent.monitor" in claim.detail


def test_whole_number_is_never_misattributed():
    """A whole number can equal another field by chance: here 5 is the expedite probability, but the claim names a unit count."""
    assert first_number("with 5 units on hand") == ("5", "on_hand", "wrong_value")


def test_zero_is_never_misattributed():
    assert first_number("a forecast of 0.0 units") == ("0.0", "forecast", "wrong_value")


def test_late_rate_with_one_qualifier_applies_it_to_both_numbers():
    """test-0007: '37.5% and 41.5% over the last 30 days' when 37.5% is the long-run rate."""
    found = numbers("the late rates are 3.6% and 5.3% over the last 30 days")
    assert found == [("3.6", "late_recent", "misattributed"), ("5.3", "late_recent", "supported")]


def test_both_models_must_match_when_the_text_says_both():
    assert first_number("The stockout risk is 0.2% from both models") == ("0.2", "stockout", "wrong_value")
    assert first_number("The stockout risk is 0.2% from both models", evidence(risk__lightgbm_risk_percent=0.2)) == ("0.2", "stockout", "supported")


def test_missing_field_makes_a_stated_value_unsupported():
    claim = judge_explanation("case", "a forecast of 17.8 units", evidence(forecast__forecast_units_next_14_days=None))[0]
    assert claim.verdict == "unsupported" and "missing" in claim.detail


def test_missing_tool_makes_a_bound_number_unsupported():
    ev = evidence()
    del ev["get_demand_forecast"]
    assert first_number("a forecast of 17.8 units", ev)[2] == "unsupported"


def test_flatten_names_order_fields_and_totals():
    flat = flatten(evidence())
    assert flat["orders.open_orders[0].units"] == 28.0 and flat["orders.count"] == 1.0 and flat["orders.total_units"] == 28.0
    assert "inventory.ordered_at_this_review" not in flat


# Open orders and value band --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, listed, overdue, verdict", [
    ("There are no open orders", [], 0.0, "supported"),
    ("There are no open orders", [{"units": 8.0}], 8.0, "contradicted"),
    ("The open orders are empty", [{"units": 8.0}], 0.0, "contradicted"),
    ("The open orders are delayed", [{"units": 8.0}], 0.0, "contradicted"),
    ("The open orders are delayed", [{"units": 8.0}], 8.0, "supported"),
    ("The open orders are not expected to arrive in time", [], 0.0, "contradicted"),
    ("The open order is due in 4 days", [{"units": 8.0}], 0.0, "supported"),
])
def test_open_order_claims(text, listed, overdue, verdict):
    found = [c for c in judge_explanation("case", text, evidence(orders__open_orders=listed, orders__units_overdue=overdue)) if c.kind == "open_orders"]
    assert len(found) == 1 and found[0].verdict == verdict


@pytest.mark.parametrize("text, verdict", [("The inventory is in the high value band", "supported"), ("The inventory is in the low value band", "contradicted")])
def test_value_band(text, verdict):
    found = [c for c in judge_explanation("case", text, evidence()) if c.kind == "value_band"]
    assert [c.verdict for c in found] == [verdict]


def test_relative_band_wording_is_not_a_band_claim():
    assert not [c for c in extract_claims("case", "The inventory is below the high value band") if c.kind == "value_band"]


# Qualitative claims ----------------------------------------------------------------------------------------------------

def test_qualitative_claims_are_not_checked_without_an_llm():
    found = judge_explanation("case", "The evidence suggests a low risk of stockout, and the supplier's late rates are not a concern.", evidence())
    assert [(c.kind, c.verdict) for c in found] == [("qualitative", "not_checked"), ("qualitative", "not_checked")]


def test_code_checked_wording_is_not_sent_to_the_llm():
    found = extract_claims("case", "The demand forecast indicates a low estimate of 0 units and a high estimate of 44.6 units, with a high probability of 94.3%.")
    assert not [c for c in found if c.kind == "qualitative"]


# The committed traces --------------------------------------------------------------------------------------------------

def test_parsed_traces_reproduce_the_step_8_audit():
    complete, explanations = parse()
    saved = pd.read_csv(ROOT / "reports/step_08/explanation_audit.csv").fillna({"problems": ""}).set_index("case_id")
    assert len(complete) == 10 and len(explanations) == 100
    for case_id, trace in complete.items():
        assert "; ".join(audit(trace["explanation"], trace["evidence"])) == saved.loc[case_id, "problems"], case_id


@pytest.mark.skipif(not TRACES.exists(), reason="full traces are git-ignored")
def test_parsed_evidence_equals_the_local_traces():
    complete, _ = parse()
    for case_id, trace in complete.items():
        assert json.loads((TRACES / f"{case_id}.json").read_text(encoding="utf-8"))["evidence"] == trace["evidence"]


def test_judge_on_the_ten_committed_traces():
    complete, _ = parse()
    flagged = {case_id: sorted(f"{c.verdict}:{c.text}" for c in judge_explanation(case_id, t["explanation"], t["evidence"]) if c.verdict in ERROR_VERDICTS) for case_id, t in complete.items()}
    assert {k: v for k, v in flagged.items() if v} == {
        "test-0001": ["wrong_value:20"],
        "test-0002": ["contradicted:no open orders"],
        "test-0003": ["misattributed:67.7"],
        "test-0006": ["contradicted:open orders are delayed"],
        "test-0007": ["misattributed:37.5"],
        "test-0041": ["wrong_value:100"],
        "test-0087": ["contradicted:an open order exists", "wrong_value:70"],
    }


NUMBER_RULES = {"says_no_open_orders_but_one_exists", "describes_an_open_order_that_does_not_exist", "says_delayed_but_nothing_overdue", "stockout_risk_number_not_from_either_model"}


def test_judge_finds_every_number_or_open_order_error_the_rules_find():
    complete, _ = parse()
    for case_id, trace in complete.items():
        if NUMBER_RULES & set(audit(trace["explanation"], trace["evidence"])):
            assert summarize(case_id, judge_explanation(case_id, trace["explanation"], trace["evidence"]))["code_error"], case_id


def test_errors_the_live_check_misses():
    """test-0007: every number exists somewhere in the evidence, so the Step 8 live check passes it; binding shows 37.5 is the wrong rate."""
    complete, _ = parse()
    trace = complete["test-0007"]
    assert unsupported_numbers(trace["explanation"], trace["evidence"]) == []
    assert summarize("test-0007", judge_explanation("test-0007", trace["explanation"], trace["evidence"]))["has_error"]


# LLM interface ---------------------------------------------------------------------------------------------------------

def test_llm_judges_each_qualitative_claim_once_per_order():
    llm = Scripted(['{"verdict": "contradicted", "reason": "both under 20%"}', '{"verdict": "supported", "reason": "x"}'])
    found = judge_explanation("case", "The evidence suggests a high risk of stockout.", evidence(), llm=llm, orders=2)
    assert [(c.verdict, c.actual, c.detail) for c in found] == [("contradicted", ["contradicted", "supported"], "both under 20%")]
    assert len(llm.prompts) == 2


def test_llm_is_not_asked_about_numbers():
    llm = Scripted([])
    judge_explanation("case", "a forecast of 17.8 units", evidence(), llm=llm)
    assert llm.prompts == []


@pytest.mark.parametrize("reply, parsed", [
    ('{"verdict": "supported", "reason": "ok"}', ("supported", "ok")),
    ('```json\n{"verdict": "unclear", "reason": "x"}\n```', ("unclear", "x")),
    ('Sure! {"verdict": "supported"}', None),
    ('{"verdict": "probably"}', None),
    ("[]", None),
    ("not json", None),
])
def test_reply_parsing_is_strict(reply, parsed):
    assert parse_reply(reply) == parsed


def test_bad_reply_becomes_judge_error():
    found = judge_explanation("case", "The evidence suggests a high risk of stockout.", evidence(), llm=Scripted(["I think it is supported"]))
    assert found[0].verdict == "judge_error"


def test_prompt_holds_one_claim_the_definitions_and_the_evidence_but_not_the_explanation():
    explanation = "The evidence suggests a high risk of stockout. The open orders are on track."
    claim = [c for c in extract_claims("case", explanation) if c.kind == "qualitative"][0]
    system, user = (m["content"] for m in judge_prompt(claim, evidence()))
    assert DEFINITIONS in system and "only this part: \"The evidence suggests a high risk of stockout\"" in user
    assert "on track" not in user, "the prompt must not hold the rest of the explanation"
    assert all(tool in user for tool in TOOL_ORDER)


def test_prompt_never_carries_the_agent_category():
    """The judge gets the claim and the tool results only; nothing passes the agent's own category in."""
    claim = Claim("case", 0, 0, "x", "qualitative", "The stock is sufficient", sentence_text="The stock is sufficient.")
    text = json.dumps(judge_prompt(claim, evidence()))
    assert "human_review\"" not in text.replace('"human_review": 0.6', "") and "recommendation\":" not in text


def test_prompt_follows_the_given_tool_order():
    claim = Claim("case", 0, 0, "x", "qualitative", "x", sentence_text="x")
    order = list(reversed(TOOL_ORDER))
    user = judge_prompt(claim, evidence(), order)[1]["content"]
    assert [user.index(tool) for tool in order] == sorted(user.index(tool) for tool in order)


def test_tool_orders_are_distinct_and_start_with_the_fixed_order():
    orders = tool_orders(3)
    assert orders[0] == TOOL_ORDER and len({tuple(o) for o in orders}) == 3 and all(sorted(o) == sorted(TOOL_ORDER) for o in orders)
    assert tool_orders(3) == orders


# Agreement and coverage ------------------------------------------------------------------------------------------------

def test_kappa_matches_sklearn_without_weights():
    from sklearn.metrics import cohen_kappa_score
    judge = np.array([True, True, False, False, True, False, False, True])
    human = np.array([True, False, False, False, True, True, False, True])
    assert cohen_kappa(judge, human, np.ones(8)) == pytest.approx(cohen_kappa_score(judge, human))


def test_unflagged_claims_are_weighted_up_for_recall():
    frame = pd.DataFrame({"flagged": [True, True, False, False], "label": ["error", "ok", "error", "ok"], "weight": [1.0, 1.0, 10.0, 10.0]})
    result = score(frame)
    assert result["precision"] == 0.5 and result["recall_weighted"] == pytest.approx(1 / 11)


def test_coverage_regression_guard():
    """At least 95% of stated numbers bind to a named field on the 100 committed explanations (99.5% when written)."""
    table, unbound = coverage(pd.read_csv(ROOT / "reports/step_08/test_results.csv"))
    assert table.bound.sum() / table.numbers.sum() >= 0.95
    assert len(table) == 100
