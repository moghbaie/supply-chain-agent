"""Step 7 tests: category validity, conflicting signals, critical cases and policy overrides. No test calls the API."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.escalation import jev
from src.escalation.cases import CATEGORIES, VISIBLE_COLUMNS, future_lost_margin, reference_category, value_bands
from src.escalation.jev import CRITERIA, FORBIDDEN_TERMS, assert_clean, build_request, case_state, classify
from src.escalation.policy import MODEL_DISAGREEMENT, OVERRIDE_RISK, PolicyThresholds, apply_overrides, policy_baseline, requires_approval

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/escalation"
REPORT = ROOT / "reports/step_07"
THRESHOLDS = PolicyThresholds(act=0.1, critical=0.9)


def case(**changes):
    base = {"on_hand": 50, "on_order": 0, "fc_mean_14": 14.0, "fc_q90_sum_14": 25.0, "cover_days": 50.0, "days_to_next_due": np.nan, "units_due_14": 0.0, "units_overdue": 0.0, "days_overdue_max": 0.0, "open_pos": 0.0,
            "ordered_today": 0, "supplier_late_rate": 0.1, "supplier_recent_late_rate": 0.1, "stockout_risk": 0.02, "stockout_risk_lightgbm": 0.02, "value_band": "low"}
    return pd.DataFrame([{**base, **changes}])


@pytest.fixture(scope="module")
def test_cases():
    return pd.read_parquet(OUT / "classified_test.parquet")


def test_categories_match_the_plan():
    assert set(CATEGORIES) == {"monitor", "reorder_recommendation", "expedite_recommendation", "human_review"}
    assert set(CRITERIA) == set(CATEGORIES)


def test_saved_choices_are_valid_categories(test_cases):
    for column in ["jev", "jev_override", "policy", "reference"]:
        assert set(test_cases[column]) <= set(CATEGORIES), column


def test_every_test_case_was_scored(test_cases):
    assert len(test_cases) == 27000
    assert test_cases.jev.notna().all()


def test_probabilities_cover_all_categories_and_sum_to_one(test_cases):
    for text in test_cases.jev_probabilities.iloc[::3000]:
        values = json.loads(text)
        assert set(values) == set(CATEGORIES)
        assert sum(values.values()) == pytest.approx(1.0, abs=1e-3)


def test_choice_is_the_most_likely_category(test_cases):
    for row in test_cases.iloc[::3000].itertuples():
        values = json.loads(row.jev_probabilities)
        assert row.jev == max(values, key=values.get)


def test_reference_rules():
    frame = pd.DataFrame({"label": [0, 1, 1, 1], "future_lost_margin": [0.0, 30.0, 5.0, 5.0], "open_pos": [1, 1, 1, 0]})
    assert list(reference_category(frame, 24.0)) == ["monitor", "human_review", "expedite_recommendation", "reorder_recommendation"]


def test_future_margin_excludes_the_review_day():
    daily = pd.DataFrame({"scenario": "s", "id": "a", "date": pd.date_range("2015-01-01", periods=5), "lost_margin": [100.0, 1.0, 2.0, 0.0, 0.0]})
    origin = pd.DataFrame({"scenario": ["s"], "id": ["a"], "origin": [pd.Timestamp("2015-01-01")]})
    assert future_lost_margin(daily, origin, horizon=2).future_lost_margin.iloc[0] == 3.0


def test_request_state_has_no_hidden_or_identifying_data(test_cases):
    for row in test_cases.iloc[::2500].itertuples(index=False):
        body = build_request(row)
        assert_clean(body)
        assert body["model"] == jev.MODEL
    state_text = json.dumps(build_request(test_cases.iloc[0])["state"])
    for term in ["label", "reference", "future", "unmet", "origin", "2016"]:
        assert term not in state_text, term


def test_request_guard_rejects_forbidden_terms():
    with pytest.raises(ValueError):
        assert_clean({"state": {"item": "FOODS_1_001"}})
    assert "price" in FORBIDDEN_TERMS


def test_stock_comparison_is_computed_in_code():
    assert case_state(next(case(cover_days=5.0, days_to_next_due=9.0).itertuples(index=False)))["item"]["inbound"]["stock_lasts_until_next_delivery"] is False
    assert case_state(next(case(cover_days=12.0, days_to_next_due=9.0).itertuples(index=False)))["item"]["inbound"]["stock_lasts_until_next_delivery"] is True
    assert case_state(next(case().itertuples(index=False)))["item"]["inbound"]["stock_lasts_until_next_delivery"] is None


def test_value_band_hides_raw_price():
    frame = pd.DataFrame({"fc_mean_14": [14.0, 14.0, 14.0], "price": [1.0, 3.0, 9.0]})
    assert list(value_bands(frame, (2.0, 4.0))) == ["low", "medium", "high"]


def test_visible_columns_exclude_outcomes():
    assert not {"label", "reference", "future_lost_margin", "scenario", "unmet", "delay_days"} & set(VISIBLE_COLUMNS)


def test_failed_jev_call_is_an_error_not_a_category(monkeypatch):
    def fail(body, key):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(jev, "send", fail)
    with pytest.raises(RuntimeError):
        classify(case(), "unused", workers=1)


def test_sending_to_jev_is_turned_off():
    assert jev.SENDING_ENABLED is False
    with pytest.raises(RuntimeError, match="turned off"):
        jev.send({"state": {}}, "unused")


def test_classify_keeps_case_order(monkeypatch):
    def echo(body, key):
        units = body["state"]["item"]["stock"]["units_on_hand"]
        choice = "monitor" if units == 1 else "human_review"
        return {"answers": {"category": {"choice": choice, "probabilities": {choice: 1.0}}}, "usage": {"input_tokens": 7}}
    monkeypatch.setattr(jev, "send", echo)
    frame = pd.concat([case(on_hand=1), case(on_hand=2), case(on_hand=1)], ignore_index=True)
    assert [a[0] for a in classify(frame, "unused", workers=3)] == ["monitor", "human_review", "monitor"]


def test_policy_returns_valid_categories_for_any_case(test_cases):
    assert set(policy_baseline(test_cases, THRESHOLDS)) <= set(CATEGORIES)


def test_policy_empty_with_no_order_recommends_reorder():
    assert policy_baseline(case(stockout_risk=0.9, stockout_risk_lightgbm=0.9), THRESHOLDS)[0] == "reorder_recommendation"


def test_policy_order_on_the_way_recommends_expedite():
    assert policy_baseline(case(open_pos=1.0, stockout_risk=0.5, stockout_risk_lightgbm=0.5), THRESHOLDS)[0] == "expedite_recommendation"


def test_policy_low_risk_monitors():
    assert policy_baseline(case(), THRESHOLDS)[0] == "monitor"


def test_policy_critical_high_value_goes_to_a_person():
    assert policy_baseline(case(stockout_risk=0.95, stockout_risk_lightgbm=0.95, value_band="high"), THRESHOLDS)[0] == "human_review"


def test_policy_conflicting_models_go_to_a_person():
    assert policy_baseline(case(stockout_risk=0.5, stockout_risk_lightgbm=0.5 - MODEL_DISAGREEMENT), THRESHOLDS)[0] == "human_review"


def test_override_high_risk_cannot_stay_on_monitor():
    choices, reasons = apply_overrides(case(stockout_risk=OVERRIDE_RISK), ["monitor"])
    assert choices[0] == "human_review"
    assert reasons[0]


def test_override_reorder_with_order_on_the_way_becomes_expedite():
    assert apply_overrides(case(open_pos=2.0), ["reorder_recommendation"])[0][0] == "expedite_recommendation"


def test_override_expedite_with_no_order_becomes_reorder():
    assert apply_overrides(case(open_pos=0.0), ["expedite_recommendation"])[0][0] == "reorder_recommendation"


def test_override_leaves_consistent_choices_alone():
    choices, reasons = apply_overrides(case(open_pos=1.0), ["expedite_recommendation"])
    assert choices[0] == "expedite_recommendation"
    assert reasons[0] == ""


def test_every_action_needs_approval():
    assert list(requires_approval(CATEGORIES)) == [False, True, True, True]


def test_saved_actions_are_flagged_for_approval(test_cases):
    assert (test_cases.requires_approval == (test_cases.jev_override != "monitor")).all()


def test_saved_metrics_recompute(test_cases):
    saved = pd.read_csv(REPORT / "metrics_test.csv", index_col=0)
    for name, column in [("jev", "jev"), ("policy_baseline", "policy")]:
        assert f1_score(test_cases.reference, test_cases[column], labels=CATEGORIES, average="macro", zero_division=0) == pytest.approx(saved.loc[name, "macro_f1"], abs=1e-4)


def test_spend_stayed_within_limit():
    assert json.loads((OUT / "usage.json").read_text())["usd"] < 5.0


def test_policy_thresholds_came_from_validation():
    assert json.loads((OUT / "policy_thresholds.json").read_text()) == {"act": 0.1, "critical": 0.9}


def test_edge_cases_were_recorded():
    frame = pd.read_csv(REPORT / "edge_cases.csv")
    assert len(frame) == 10
    assert set(frame.jev) <= set(CATEGORIES)
    assert frame.set_index("case").loc["healthy_stock", "jev"] == "monitor"
