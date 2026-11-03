"""Step 8 tests: tool contracts, missing data, failures, unsupported claims, invalid actions and approval boundaries. A scripted model stands in for the LLM, and no test calls an API."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agent.audit import audit
from src.agent.retrieval import PolicyIndex
from src.agent.review import review, unsupported_numbers
from src.agent.runner import HIDDEN_SUGGESTION, MAX_TURNS, SYSTEM_NO_SUGGESTION, parse_tool_calls, run_case
from src.agent.tools import CASE_COLUMNS, REQUIRED_TOOLS, SCORE_COLUMNS, TOOLS, ToolBox, build_toolbox, call_tool, open_orders_at, tool_schemas

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/agent"
TRACES = DATA / "traces/smollm_untuned/test"
FORBIDDEN = {"reference", "label", "future_lost_margin", "scenario", "id", "cat_id", "jev", "policy", "price", "ultrabert"}
GOOD = {"category": "monitor", "explanation": "Stock of 50 units covers the forecast of 14 units.", "uncertainty": "low"}
REQUIRED_CALLS = [{"name": name, "arguments": {"case_id": "case-0000"}} for name in sorted(REQUIRED_TOOLS)]


def case(**changes):
    base = {"on_hand": 50, "on_order": 0, "cover_days": 50.0, "open_pos": 0, "ordered_today": 0, "value_band": "low", "sales_mean_28": 1.0, "sold_out_days_28": 0, "fc_mean_14": 14.0, "fc_q10_sum_14": 5.0, "fc_q90_sum_14": 25.0, "stockout_risk": 0.02,
            "stockout_risk_lightgbm": 0.03, "days_to_next_due": np.nan, "units_due_14": 0, "units_overdue": 0, "days_overdue_max": 0, "supplier_late_rate": 0.1, "supplier_recent_late_rate": 0.1, "p_monitor": 0.7, "p_reorder_recommendation": 0.1, "p_expedite_recommendation": 0.1, "p_human_review": 0.1}
    return {**base, **changes}


def toolbox(**changes):
    return ToolBox(pd.DataFrame([case(**changes)], index=["case-0000"]), {"case-0000": []})


class ScriptedModel:
    """Reply with fixed texts and record the tools it was offered."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.turn = 0
        self.tools = None
        self.prompts = []

    def generate(self, messages, tools):
        self.tools, self.prompts = tools, messages
        text = self.replies[min(self.turn, len(self.replies) - 1)]
        self.turn += 1
        return {"text": text, "input_tokens": 10, "output_tokens": 5, "seconds": 0.0}


def tagged(*calls):
    return "".join(f"<tool_call>{json.dumps(call)}</tool_call>" for call in calls)


def submit(**changes):
    return {"name": "submit_recommendation", "arguments": {**GOOD, **changes}}


def orders_table(**changes):
    base = {"po_id": 1, "scenario": "baseline", "id": "item", "order_date": "2016-01-01", "promised_date": "2016-01-20", "receipt_date": "2016-01-25", "quantity": 10.0, "actual_lead_days": 24.0, "delay_days": 5.0}
    frame = pd.DataFrame([{**base, **changes}])
    for column in ["order_date", "promised_date", "receipt_date"]:
        frame[column] = pd.to_datetime(frame[column])
    return frame


# Tool contracts
@pytest.mark.parametrize("name", TOOLS)
def test_every_tool_schema_forbids_extra_fields(name):
    assert TOOLS[name][0].model_json_schema()["additionalProperties"] is False


def test_no_tool_can_change_an_order():
    for name in TOOLS:
        assert not any(word in name for word in ["place", "create", "cancel", "update", "purchase"])


def test_extra_argument_is_refused():
    result = call_tool(toolbox(), "get_inventory_status", {"case_id": "case-0000", "force": True})
    assert result["ok"] is False and "force" in result["error"]


def test_unknown_tool_is_refused():
    result = call_tool(toolbox(), "place_purchase_order", {"case_id": "case-0000"})
    assert result["ok"] is False and "orders cannot be placed" in result["error"]


def test_unknown_case_is_an_error_not_evidence():
    result = call_tool(toolbox(), "get_inventory_status", {"case_id": "case-9999"})
    assert result["ok"] is False and "unknown case_id" in result["error"]


def test_allowed_columns_exclude_outcomes_and_identity():
    assert not FORBIDDEN & set(CASE_COLUMNS + SCORE_COLUMNS)


def test_build_toolbox_drops_identity_from_the_store():
    row = {**case(), "scenario": "baseline", "id": "SECRET_ITEM", "origin": pd.Timestamp("2016-02-01"), "reference": "monitor", "label": 1}
    box, keyed = build_toolbox(pd.DataFrame([row]), orders_table(id="other"), {}, prefix="test")
    assert list(box.cases.index) == ["test-0000"] and not FORBIDDEN & set(box.cases.columns)
    assert "SECRET_ITEM" not in json.dumps(call_tool(box, "get_inventory_status", {"case_id": "test-0000"}))
    assert keyed.case_id[0] == "test-0000"


# Missing data
def test_missing_value_is_none_and_listed():
    result = call_tool(toolbox(on_hand=np.nan), "get_inventory_status", {"case_id": "case-0000"})["result"]
    assert result["on_hand_units"] is None and "on_hand_units" in result["missing_fields"]


def test_one_missing_risk_leaves_the_gap_unknown():
    result = call_tool(toolbox(stockout_risk=np.nan), "get_stockout_risk", {"case_id": "case-0000"})["result"]
    assert result["tabpfn_risk_percent"] is None and result["models_differ_by_points"] is None and result["lightgbm_risk_percent"] == 3.0


def test_missing_escalation_score_is_an_error():
    result = call_tool(toolbox(p_monitor=np.nan), "get_escalation_suggestion", {"case_id": "case-0000"})
    assert result["ok"] is False and "unavailable" in result["error"]


def test_all_risks_missing_does_not_raise_and_cannot_pick_monitor_with_missing_tools():
    outcome = review("monitor", "Stock looks fine today.", {"get_inventory_status": {}}, [None, None], None)
    assert outcome["final_category"] == "human_review" and outcome["flags"][0].startswith("missing_evidence")


# Tool failures
def test_failed_tool_stays_out_of_the_evidence_and_the_case_goes_to_a_person():
    box = toolbox(p_monitor=np.nan)
    trace = run_case(ScriptedModel([tagged(*REQUIRED_CALLS), tagged(submit())]), box, "case-0000", max_turns=3)
    assert "get_escalation_suggestion" not in trace["evidence"]
    assert trace["status"] == "failed" and trace["recommendation"] is None
    assert trace["review"]["final_category"] == "human_review" and trace["review"]["approval_required"] is True


def test_tool_error_reaches_the_model_so_it_can_say_what_is_unknown():
    model = ScriptedModel([tagged(*REQUIRED_CALLS), tagged(submit())])
    run_case(model, toolbox(p_monitor=np.nan), "case-0000", max_turns=2)
    assert any("unavailable" in message["content"] for message in model.prompts if message["role"] == "tool")


def test_model_that_never_submits_goes_to_a_person_at_the_turn_limit():
    model = ScriptedModel([f"thinking {i}" for i in range(20)])
    trace = run_case(model, toolbox(), "case-0000")
    assert len(trace["turns"]) == MAX_TURNS and trace["status"] == "failed" and trace["review"]["final_category"] == "human_review"


def test_model_repeating_itself_stops_early_and_goes_to_a_person():
    trace = run_case(ScriptedModel(["same"]), toolbox(), "case-0000")
    assert trace["stopped_repeating"] is True and len(trace["turns"]) == 2 and trace["review"]["final_category"] == "human_review"


# Unsupported claims
def test_invented_number_is_found():
    assert unsupported_numbers("The risk is 80 percent.", {"tool": {"risk": 3.1}}) == [80.0]


def test_cited_and_rounded_numbers_pass():
    assert unsupported_numbers("Risk 3.1, or about 3, over 14 days.", {"tool": {"risk": 3.1}}) == []


def test_invented_number_on_monitor_goes_to_a_person():
    evidence = {name: {"value": 5.0} for name in REQUIRED_TOOLS}
    outcome = review("monitor", "The risk is 80 percent here.", evidence, [1.0, 1.0], 0)
    assert outcome["final_category"] == "human_review" and "unsupported_numbers:80.0" in outcome["flags"]


def test_invented_number_on_an_action_is_flagged_and_keeps_approval():
    evidence = {name: {"value": 5.0} for name in REQUIRED_TOOLS}
    outcome = review("reorder_recommendation", "The risk is 80 percent here.", evidence, [50.0, 50.0], 0)
    assert outcome["final_category"] == "reorder_recommendation" and outcome["approval_required"] is True and "unsupported_numbers:80.0" in outcome["flags"]


def test_known_limit_numeric_check_misses_a_real_number_used_wrongly():
    """A classifier probability called a stockout risk passes the live check. The audit finds it, and only after the run."""
    evidence = {"get_stockout_risk": {"tabpfn_risk_percent": 3.1, "lightgbm_risk_percent": 3.9}, "get_escalation_suggestion": {"probabilities_percent": {"reorder_recommendation": 67.7}}}
    text = "The evidence suggests a stockout risk of 67.7% today."
    assert unsupported_numbers(text, evidence) == []
    assert "stockout_risk_number_not_from_either_model" in audit(text, evidence)


@pytest.mark.parametrize("text, evidence, rule", [
    ("There are no open orders here.", {"get_open_orders": {"open_orders": [{"units": 8}]}}, "says_no_open_orders_but_one_exists"),
    ("The open orders are not expected to arrive.", {"get_open_orders": {"open_orders": []}}, "describes_an_open_order_that_does_not_exist"),
    ("The open orders are delayed now.", {"get_open_orders": {"open_orders": [{"units": 8}], "units_overdue": 0}}, "says_delayed_but_nothing_overdue"),
    ("There is a high risk of stockout.", {"get_stockout_risk": {"tabpfn_risk_percent": 3.0, "lightgbm_risk_percent": 4.0}}, "says_high_risk_but_both_models_under_20"),
    ("There is a low risk of stockout.", {"get_stockout_risk": {"tabpfn_risk_percent": 3.0, "lightgbm_risk_percent": 60.0}}, "says_low_risk_but_a_model_at_50_or_more"),
    ("The forecast demand exceeding current stock levels.", {"get_demand_forecast": {"forecast_units_next_14_days": 5.0}, "get_inventory_status": {"on_hand_units": 9.0}}, "says_demand_exceeds_stock_but_it_does_not"),
    ("The evidence conflicts with the escalation suggestion, so review.", {"get_escalation_suggestion": {"suggested_category": "human_review"}}, "says_it_conflicts_with_the_suggestion_it_followed"),
    ("Stock looks fine.", {}, "cites_no_numbers"),
])
def test_audit_rule_fires(text, evidence, rule):
    assert rule in audit(text, evidence)


def test_audit_accepts_a_true_explanation():
    evidence = {"get_stockout_risk": {"tabpfn_risk_percent": 90.9, "lightgbm_risk_percent": 96.0}, "get_open_orders": {"open_orders": [{"units": 36}], "units_overdue": 0}}
    assert audit("There is a high risk of stockout (90.9% from one model), and an open order may arrive late.", evidence) == []


def test_audit_reads_no_orders_that_would_delay_as_a_different_claim():
    assert audit("There are no open orders that would delay delivery by 2 days.", {"get_open_orders": {"open_orders": [{"units": 8}]}}) == []


# Invalid actions
@pytest.mark.parametrize("change", [{"category": "place_order"}, {"category": "reorder"}, {"explanation": "short"}, {"explanation": "x" * 701}, {"uncertainty": "none"}, {"quantity": 500}])
def test_invalid_submission_is_refused_and_counted(change):
    trace = run_case(ScriptedModel([tagged(*REQUIRED_CALLS), tagged(submit(**change))]), toolbox(), "case-0000", max_turns=2)
    assert trace["invalid_submissions"] == 1 and trace["recommendation"] is None and trace["review"]["final_category"] == "human_review"


def test_submission_before_any_evidence_is_refused():
    trace = run_case(ScriptedModel([tagged(submit())]), toolbox(), "case-0000", max_turns=2)
    assert trace["invalid_submissions"] >= 1 and trace["recommendation"] is None


def test_submission_in_the_same_turn_as_the_tool_calls_is_refused():
    """Results come back only after the turn, so the model could not have read them."""
    trace = run_case(ScriptedModel([tagged(*REQUIRED_CALLS, submit())]), toolbox(), "case-0000", max_turns=1)
    assert trace["invalid_submissions"] == 1 and trace["recommendation"] is None


def test_submit_cannot_be_called_as_an_ordinary_tool():
    assert call_tool(toolbox(), "submit_recommendation", GOOD)["ok"] is False


@pytest.mark.parametrize("choice, open_orders, risk, final", [("reorder_recommendation", 1, 50.0, "expedite_recommendation"), ("expedite_recommendation", 0, 50.0, "reorder_recommendation"), ("monitor", 0, 90.0, "human_review")])
def test_choices_that_conflict_with_facts_are_corrected(choice, open_orders, risk, final):
    evidence = {name: {} for name in REQUIRED_TOOLS}
    outcome = review(choice, "Plain text with no figures.", evidence, [risk, risk], open_orders)
    assert outcome["final_category"] == final and any(flag.startswith("override") for flag in outcome["flags"])


# UltraBERT edge cases from Step 7: it suggested reorder on both, so the checks must raise a flag
@pytest.mark.parametrize("name", ["healthy_stock", "zero_demand_no_stock"])
def test_odd_ultrabert_reorders_are_flagged_and_keep_approval(name):
    edge = pd.read_csv(ROOT / "reports/step_07/edge_cases.csv").set_index("case").loc[name]
    assert edge.ultrabert_fine_tuned == "reorder_recommendation"
    evidence = {tool: {} for tool in REQUIRED_TOOLS}
    outcome = review(edge.ultrabert_fine_tuned, "Plain text with no figures.", evidence, [100 * edge.stockout_risk, 100 * edge.stockout_risk_lightgbm], edge.open_pos)
    assert outcome["approval_required"] is True and any(flag.startswith("weak_evidence") for flag in outcome["flags"])


# Approval boundaries
@pytest.mark.parametrize("category, needs", [("monitor", False), ("reorder_recommendation", True), ("expedite_recommendation", True), ("human_review", True)])
def test_every_action_needs_approval_and_no_order_is_placed(category, needs):
    evidence = {name: {} for name in REQUIRED_TOOLS}
    outcome = review(category, "Plain text with no figures.", evidence, [50.0, 50.0], 1 if category == "expedite_recommendation" else 0)
    assert outcome["approval_required"] is needs and outcome["order_placed"] is False


def test_policy_states_the_approval_boundary():
    text = {p["id"]: p["text"] for p in PolicyIndex().passages}["approval"]
    assert "cannot place" in text and "approval" in text


# Point-in-time open orders
def test_order_placed_after_the_review_is_hidden():
    assert open_orders_at(orders_table(order_date="2016-03-01"), "2016-02-01", {}) == []


def test_order_received_before_the_review_is_not_open():
    assert open_orders_at(orders_table(receipt_date="2016-01-15"), "2016-02-01", {}) == []


def test_order_received_after_the_review_shows_as_open_without_its_outcome():
    listed = open_orders_at(orders_table(), "2016-01-10", {1: 0.25})
    assert listed == [{"units": 10.0, "days_until_promised": 10, "delay_risk_percent": 25.0}]


def test_overdue_order_has_negative_days():
    assert open_orders_at(orders_table(receipt_date="2016-03-01"), "2016-02-01", {})[0]["days_until_promised"] == -12


# Retrieval
@pytest.mark.parametrize("query, expected", [("models differ conflicting risk", "conflicting_risk"), ("who approves a purchase", "approval"), ("tool failed value missing", "missing_evidence"), ("reorder with an open order", "order_consistency")])
def test_policy_search_finds_the_right_rule(query, expected):
    assert expected in [item["id"] for item in PolicyIndex().search(query)]


def test_policy_search_finds_nothing_for_unrelated_words():
    assert PolicyIndex().search("zebra xylophone") == []


# Parsing
def test_parser_reads_tagged_json():
    assert parse_tool_calls(tagged({"name": "a", "arguments": {"x": 1}})) == ([{"name": "a", "arguments": {"x": 1}}], 0, 0)


def test_parser_reads_an_unclosed_tag():
    calls, malformed, _ = parse_tool_calls('<tool_call>{"name": "a", "arguments": {}}')
    assert calls[0]["name"] == "a" and malformed == 0


def test_parser_counts_untagged_and_python_style_calls():
    calls, malformed, untagged = parse_tool_calls("{'name': 'a', 'arguments': {'x': 1}}")
    assert calls[0]["arguments"] == {"x": 1} and malformed == 0 and untagged == 1


@pytest.mark.parametrize("text", ["<tool_call>{broken</tool_call>", "<tool_call>__import__('os').system('echo hacked')</tool_call>", "<tool_call>[1, 2]</tool_call>"])
def test_parser_counts_bad_calls_and_runs_nothing(text):
    calls, malformed, _ = parse_tool_calls(text)
    assert calls == [] and malformed == 1


# The version without the suggestion
def test_hidden_tool_is_not_offered_or_mentioned_and_is_refused():
    calls = [call for call in REQUIRED_CALLS if call["name"] not in HIDDEN_SUGGESTION]
    model = ScriptedModel([tagged({"name": "get_escalation_suggestion", "arguments": {"case_id": "case-0000"}}, *calls), tagged(submit())])
    trace = run_case(model, toolbox(), "case-0000", hidden=HIDDEN_SUGGESTION)
    assert "get_escalation_suggestion" not in [schema["function"]["name"] for schema in model.tools]
    assert "get_escalation_suggestion" not in SYSTEM_NO_SUGGESTION and "get_escalation_suggestion" not in trace["evidence"]
    assert trace["status"] == "submitted"


def test_schemas_list_every_tool_when_nothing_is_hidden():
    assert {schema["function"]["name"] for schema in tool_schemas()} == set(TOOLS)


# A full run with the scripted model
def test_complete_run_produces_a_checked_trace():
    trace = run_case(ScriptedModel([tagged(*REQUIRED_CALLS), tagged(submit())]), toolbox(), "case-0000")
    assert trace["status"] == "submitted" and trace["review"]["final_category"] == "monitor" and trace["review"]["flags"] == []
    assert set(trace["evidence"]) == REQUIRED_TOOLS and trace["input_tokens"] == 20 and trace["output_tokens"] == 10


# The saved test run
@pytest.fixture(scope="module")
def saved():
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(TRACES.glob("test-*.json"))}


@pytest.fixture(scope="module")
def results():
    return pd.read_csv(ROOT / "reports/step_08/test_results.csv").fillna({"flags": ""})


def test_saved_run_has_25_cases_of_each_category(results):
    assert len(results) == 100 and (results.reference.value_counts() == 25).all()


def test_saved_run_never_placed_an_order_or_skipped_approval(saved):
    for trace in saved.values():
        outcome = trace["review"]
        assert outcome["order_placed"] is False and outcome["approval_required"] is (outcome["final_category"] != "monitor")


def test_saved_evidence_matches_what_the_tools_return_now(saved):
    orders = pd.read_parquet(ROOT / "data/simulated/purchase_orders.parquet")
    scores = pd.read_parquet(DATA / "delay_scores.parquet")
    delay = scores[scores.period == "test"].set_index("po_id").delay_risk.to_dict()
    box, _ = build_toolbox(pd.read_parquet(DATA / "test_cases.parquet"), orders, delay, prefix="test")
    for case_id, trace in saved.items():
        for name, evidence in trace["evidence"].items():
            assert evidence == json.loads(json.dumps(getattr(box, name)(case_id), default=str)), (case_id, name)


def test_saved_tool_results_hold_no_identity_or_outcome(saved):
    key = pd.read_parquet(TRACES / "key.parquet").set_index("case_id")
    for case_id, trace in saved.items():
        text = json.dumps([call["output"] for call in trace["tool_calls"]])
        assert key.id[case_id] not in text and key.scenario[case_id] not in text and "reference" not in text


def test_saved_recommendations_passed_through_the_checks(saved):
    for trace in saved.values():
        recommendation = trace["recommendation"]
        assert set(trace["evidence"]) >= REQUIRED_TOOLS
        outcome = review(recommendation["category"], recommendation["explanation"], trace["evidence"], [trace["evidence"]["get_stockout_risk"]["tabpfn_risk_percent"], trace["evidence"]["get_stockout_risk"]["lightgbm_risk_percent"]], trace["evidence"]["get_inventory_status"]["open_orders"])
        assert outcome == trace["review"]


def test_saved_metrics_reproduce_from_the_results(results):
    table = pd.read_csv(ROOT / "reports/step_08/test_metrics.csv").set_index("method")
    assert table.loc["agent_final", "accuracy_balanced"] == pytest.approx((results.agent == results.reference).mean())
    assert table.loc["ultrabert", "accuracy_balanced"] == pytest.approx((results.ultrabert == results.reference).mean())
    assert table.loc["policy", "critical_recall"] == pytest.approx((results.policy[results.reference == "human_review"] == "human_review").mean())
