"""Step 10 tests on the replay engine and its saved runs. A stand-in scorer replaces the models, so nothing here needs the GPU or a hosted API."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agent.review import review
from src.agent.runner import _finish
from src.evaluation.decisions import APPROVE_ALL, DECLINE_ALL, DECLINE_REVIEW, code_checks, simulated_person, weak_evidence
from src.evaluation.replay import Replay, Variant, forecast_levels, load_world, write_days
from src.forecasting.models import LEVELS
from src.stockout.forecasts import summarize_forecast

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/processed/evaluation"
CONFIG = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
SETTINGS = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
SCENARIOS = list(CONFIG["scenarios"])
VARIANTS = list(SETTINGS["variants"])
DECIDING = [name for name in VARIANTS if SETTINGS["variants"][name].get("decide")]
START = SETTINGS["start"]
POSITIONS = {level: int(np.flatnonzero(np.isclose(LEVELS, level))[0]) for level in (0.1, 0.9)}
ALL_RUNS = [(s, v) for s in SCENARIOS for v in VARIANTS]
SERIES_COUNT, DAYS = 30, 45


class StandInScorer:
    """Use recent sales as the forecast and a fixed rule as the decision, and record what each review saw."""

    def __init__(self, rule):
        self.rule = rule
        self.seen = []

    def forecast(self, replay, day):
        start = max(0, day - 27)
        rate = replay.arrays["sales"][:, start:day + 1].mean(axis=1, keepdims=True) + 0.1
        mean = np.repeat(rate, 28, axis=1).astype(float)
        quantiles = np.stack([mean * (0.5 + level) for level in LEVELS], axis=2)
        forecast = {"mean": mean, "quantiles": quantiles, "levels": LEVELS}
        return forecast, summarize_forecast(replay.world.series.id.to_numpy(), replay.world.dates[day], forecast, POSITIONS)

    def decide(self, cases, orders, variant):
        cases = cases.copy()
        cases["stockout_risk"], cases["stockout_risk_lightgbm"] = 0.5, 0.5
        cases["suggested"] = self.rule(cases)
        cases["final"], cases["reason"] = code_checks(cases, cases.suggested, variant.conflict_rule)
        self.seen.append(cases)
        return cases


def always(category):
    return lambda cases: np.array([category] * len(cases), dtype=object)


def read_saved(path, ids=None):
    filters = [("id", "in", ids)] if ids is not None else None
    return pd.read_parquet(path, filters=filters)


@pytest.fixture(scope="module")
def small_inputs():
    """Keep 30 series, so a replay takes moments."""
    series = pd.read_parquet(ROOT / "data/simulated/series.parquet").sort_values("id").iloc[::16].head(SERIES_COUNT).reset_index(drop=True)
    daily = pd.read_parquet(ROOT / "data/simulated/inventory_daily.parquet", filters=[("scenario", "==", "baseline")])
    orders = pd.read_parquet(ROOT / "data/simulated/purchase_orders.parquet", filters=[("scenario", "==", "baseline")])
    return daily[daily.id.isin(series.id)], orders[orders.id.isin(series.id)], series


@pytest.fixture
def world(small_inputs):
    daily, orders, series = small_inputs
    loaded = load_world("baseline", CONFIG, START, daily=daily, orders=orders, series=series)
    return replace(loaded, dates=loaded.dates[:DAYS], demand=loaded.demand[:, :DAYS].copy(), is_open=loaded.is_open.copy(), delay_prob=loaded.delay_prob.copy(), mean_delay=loaded.mean_delay.copy())


def run(world, name="test", scorer=None, **options):
    return Replay(world, Variant(name, **options), scorer).run()


def assert_conserved(daily):
    assert (daily.on_hand_start + daily.received - daily.sales == daily.on_hand_end).all()
    assert (daily.sales + daily.unmet == daily.demand).all()
    assert (daily[["on_hand_start", "on_hand_end", "on_order_end", "sales", "unmet", "received", "ordered"]] >= 0).all().all()


@pytest.fixture(scope="module")
def saved_daily():
    return {key: pd.read_parquet(RUNS / key[0] / f"{key[1]}_daily.parquet") for key in ALL_RUNS}


# Integration: the finished runs.
@pytest.mark.parametrize("scenario, variant", ALL_RUNS)
def test_every_run_is_complete_and_conserves_stock(saved_daily, scenario, variant):
    daily = saved_daily[scenario, variant]
    assert len(daily) == 71500 and daily.id.nunique() == 500 and daily.date.nunique() == 143
    assert daily.date.min() == pd.Timestamp("2016-01-01") and daily.date.max() == pd.Timestamp("2016-05-22")
    assert not daily.isna().any().any()
    assert_conserved(daily)


@pytest.mark.parametrize("scenario, variant", [key for key in ALL_RUNS if key[1] in DECIDING])
def test_every_review_decided_every_series(scenario, variant):
    cases = pd.read_parquet(RUNS / scenario / f"{variant}_cases.parquet", columns=["origin", "id", "final"])
    actions = pd.read_parquet(RUNS / scenario / f"{variant}_actions.parquet")
    assert len(cases) == 10000 and cases.origin.nunique() == 20 and (cases.origin.dt.day_of_week == 0).all()
    assert not cases.duplicated(["origin", "id"]).any()
    assert (cases.final.to_numpy() == actions.final.to_numpy()).all()


def test_saved_metrics_match_the_daily_rows(saved_daily):
    metrics = pd.read_csv(ROOT / "reports/step_10/replay_metrics.csv").set_index(["scenario", "variant"])
    costs = CONFIG["costs"]
    for key, daily in saved_daily.items():
        total = (daily.on_hand_end * daily.price * costs["holding_rate_per_year"] / 365 + daily.unmet * daily.price * costs["margin_rate"] + daily.received * daily.price * costs["shipping_rate_of_price"] + daily.expedite_cost).sum()
        assert metrics.loc[key, "total_cost"] == pytest.approx(total)
        assert metrics.loc[key, "unmet_units"] == daily.unmet.sum()
        row = metrics.loc[key]
        assert row.total_cost == pytest.approx(row.holding_cost + row.lost_margin + row.shipping_cost + row.expedite_cost)


# Identical constraints.
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_runs_face_the_same_world(saved_daily, scenario):
    first = saved_daily[scenario, VARIANTS[0]].sort_values(["id", "date"]).reset_index(drop=True)
    opening = first[first.date == first.date.min()]
    for variant in VARIANTS[1:]:
        other = saved_daily[scenario, variant].sort_values(["id", "date"]).reset_index(drop=True)
        for column in ["id", "date", "store_open", "demand", "price"]:
            assert (first[column].to_numpy() == other[column].to_numpy()).all()
        assert (opening.on_hand_start.to_numpy() == other[other.date == other.date.min()].on_hand_start.to_numpy()).all()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_series_and_day_meets_the_same_delay(scenario):
    """Orders no one expedited, placed for the same series on the same day, get the same draw in every run."""
    left = pd.read_parquet(RUNS / scenario / "reorder_point_orders.parquet")
    right = pd.read_parquet(RUNS / scenario / "forecast_driven_orders.parquet")
    new = lambda frame: frame[frame.po_id.str.startswith("replay|")]
    both = new(left).merge(new(right), on=["id", "order_date"])
    assert len(both) > 500
    assert (both.delay_days_x == both.delay_days_y).all()


def test_expedite_settings_differ_only_where_stated():
    specs = SETTINGS["variants"]
    assert specs["agent_conflict_weak_expedite"] == {**specs["agent_conflict"], "expedite": "weak"}
    assert specs["agent_conflict_decline_review"] == {**specs["agent_conflict"], "person": "decline_review"}
    assert SETTINGS["expedite"]["remaining_time_fraction"] == 0.5 and SETTINGS["expedite"]["weak_remaining_time_fraction"] == 0.75


# Policy compliance.
@pytest.mark.parametrize("category, open_orders, person, action", [
    ("monitor", 1, APPROVE_ALL, "none"), ("reorder_recommendation", 0, APPROVE_ALL, "order"), ("expedite_recommendation", 1, APPROVE_ALL, "expedite"),
    ("human_review", 1, APPROVE_ALL, "expedite"), ("human_review", 0, APPROVE_ALL, "order"), ("human_review", 1, DECLINE_REVIEW, "none"), ("reorder_recommendation", 0, DECLINE_REVIEW, "order"),
    ("reorder_recommendation", 0, DECLINE_ALL, "none"), ("expedite_recommendation", 1, DECLINE_ALL, "none"), ("human_review", 1, DECLINE_ALL, "none")])
def test_simulated_person(category, open_orders, person, action):
    assert simulated_person(category, open_orders, person) == action


@pytest.mark.parametrize("category", ["reorder_recommendation", "expedite_recommendation", "human_review"])
def test_declining_everything_equals_the_reorder_point_run(world, category):
    """The agent can add to the routine only through a person's approval."""
    base = run(world).daily_frame()
    declined = run(world, scorer=StandInScorer(always(category)), decide="stand_in", person=DECLINE_ALL)
    frame = declined.daily_frame()
    pd.testing.assert_frame_equal(base, frame)
    assert {a["action"] for a in declined.actions} == {"none"}


def test_monitor_changes_nothing(world):
    base = run(world).daily_frame()
    watched = run(world, scorer=StandInScorer(always("monitor")), decide="stand_in").daily_frame()
    pd.testing.assert_frame_equal(base, watched)


@pytest.mark.parametrize("scenario, variant", [key for key in ALL_RUNS if key[1] in DECIDING])
def test_saved_actions_follow_the_approval_rules(scenario, variant):
    actions = pd.read_parquet(RUNS / scenario / f"{variant}_actions.parquet")
    assert (actions[actions.final == "monitor"].action == "none").all()
    assert (actions[actions.final != "monitor"].action != "none").sum() > 0
    assert not actions[actions.action == "none"].done.any()
    if SETTINGS["variants"][variant].get("person") == DECLINE_REVIEW:
        assert (actions[actions.final == "human_review"].action == "none").all()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_orders_beyond_the_routine_trace_to_approved_actions(saved_daily, scenario):
    """Series untouched by any approved action behave exactly as they do in the plain routine."""
    actions = pd.read_parquet(RUNS / scenario / "agent_conflict_decline_review_actions.parquet")
    touched = set(actions[actions.done].id)
    plain, agent = saved_daily[scenario, "reorder_point"], saved_daily[scenario, "agent_conflict_decline_review"]
    keep = sorted(set(plain.id) - touched)
    assert keep
    columns = ["id", "date", "on_hand_end", "ordered", "unmet", "expedites"]
    left = plain[plain.id.isin(keep)][columns].sort_values(["id", "date"]).reset_index(drop=True)
    right = agent[agent.id.isin(keep)][columns].sort_values(["id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_conflict_rule_on_saved_cases(scenario):
    conflict = pd.read_parquet(RUNS / scenario / "agent_conflict_cases.parquet")
    without = pd.read_parquet(RUNS / scenario / "agent_cases.parquet")
    wide = (conflict.stockout_risk - conflict.stockout_risk_lightgbm).abs() >= 0.30
    assert wide.sum() > 0 and (conflict[wide].final == "human_review").all()
    assert (without[(without.stockout_risk - without.stockout_risk_lightgbm).abs() >= 0.30].final != "human_review").any()
    moved = conflict.reason == "stockout models conflict"
    assert moved.sum() > 0 and wide[moved].all()


def test_code_checks_rule_is_off_by_default():
    cases = pd.DataFrame({"stockout_risk": [0.1, 0.9, np.nan], "stockout_risk_lightgbm": [0.6, 0.85, 0.5], "open_pos": [1, 0, 0]})
    choices = np.array(["expedite_recommendation", "reorder_recommendation", "monitor"], dtype=object)
    plain, _ = code_checks(cases, choices)
    ruled, reason = code_checks(cases, choices, conflict_rule=True)
    assert list(plain) == ["expedite_recommendation", "reorder_recommendation", "human_review"]
    assert list(ruled) == ["human_review", "reorder_recommendation", "human_review"]
    assert list(reason) == ["stockout models conflict", "", "stockout risk unavailable"]


def test_weak_evidence_marks_only_actions_under_ten_percent():
    cases = pd.DataFrame({"stockout_risk": [0.05, 0.05, 0.5, 0.05], "stockout_risk_lightgbm": [0.09, 0.2, 0.05, 0.05]})
    assert list(weak_evidence(cases, np.array(["reorder_recommendation", "reorder_recommendation", "expedite_recommendation", "monitor"], dtype=object))) == [True, False, False, False]


@pytest.mark.parametrize("risks, category, expected, flagged", [
    ([10.0, 45.0], "monitor", "human_review", True), ([10.0, 45.0], "expedite_recommendation", "human_review", True), ([10.0, 39.9], "expedite_recommendation", "expedite_recommendation", False),
    ([None, 45.0], "expedite_recommendation", "human_review", True)])
def test_live_review_applies_the_conflict_rule(risks, category, expected, flagged):
    evidence = {name: {} for name in ["get_inventory_status", "get_demand_forecast", "get_stockout_risk", "get_open_orders", "get_escalation_suggestion"]}
    result = review(category, "Plain text.", evidence, risks, 1, conflict_rule=True)
    assert result["final_category"] == expected
    assert any(flag.startswith("conflict:") for flag in result["flags"]) == flagged
    assert result["order_placed"] is False and result["approval_required"] is True


def test_live_review_default_keeps_the_step_8_result():
    evidence = {name: {} for name in ["get_inventory_status", "get_demand_forecast", "get_stockout_risk", "get_open_orders", "get_escalation_suggestion"]}
    result = review("expedite_recommendation", "Plain text.", evidence, [10.0, 45.0], 1)
    assert result["final_category"] == "expedite_recommendation" and result["flags"] == []


def test_llm_sample_traces_never_place_orders():
    folder = ROOT / "data/processed/agent/traces/smollm_untuned/replay"
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("replay-*.json")]
    if not traces:
        pytest.skip("LLM sample not run")
    for trace in traces:
        assert trace["review"]["order_placed"] is False
        assert trace["review"]["approval_required"] == (trace["review"]["final_category"] != "monitor")
        risk = trace["evidence"].get("get_stockout_risk", {})
        values = [risk.get("tabpfn_risk_percent"), risk.get("lightgbm_risk_percent")]
        if None not in values and abs(values[0] - values[1]) >= 30:
            assert trace["review"]["final_category"] == "human_review"


# Recovery.
def test_finished_runs_are_not_repeated():
    target = RUNS / "baseline/reorder_point_daily.parquet"
    before = target.stat().st_mtime_ns
    result = subprocess.run([sys.executable, "-W", "ignore", "scripts/run_replay.py", "--scenarios", "baseline", "--variants", "reorder_point"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "to run: 0 of 1" in result.stdout
    assert target.stat().st_mtime_ns == before


def test_run_counts_as_finished_only_when_its_last_file_is_saved():
    """Resume looks for the daily file, so it must be written after the others."""
    source = (ROOT / "scripts/run_replay.py").read_text()
    assert max(source.index('_orders.parquet"'), source.index('_cases.parquet"'), source.index('_actions.parquet"')) < source.index('_daily.parquet", index=False')


def test_model_failure_stops_the_run_rather_than_inventing_a_decision(world):
    class Broken(StandInScorer):
        def decide(self, cases, orders, variant):
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        run(world, scorer=Broken(always("monitor")), decide="stand_in")


def test_order_that_never_arrives_is_refused(small_inputs):
    daily, orders, series = small_inputs
    broken = orders.copy()
    broken.loc[broken.index[0], "receipt_date"] = pd.NaT
    with pytest.raises(ValueError, match="never arrive"):
        load_world("baseline", CONFIG, START, daily=daily, orders=broken, series=series)


def test_agent_with_no_answer_goes_to_a_person():
    result = _finish({"evidence": {}}, None, conflict_rule=True)
    assert result["review"]["final_category"] == "human_review" and result["review"]["order_placed"] is False


# Regression.
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_replay_with_saved_delays_reproduces_the_simulation(scenario):
    daily = pd.read_parquet(ROOT / "data/simulated/inventory_daily.parquet", filters=[("scenario", "==", scenario)])
    orders = pd.read_parquet(ROOT / "data/simulated/purchase_orders.parquet", filters=[("scenario", "==", scenario)])
    world = load_world(scenario, CONFIG, START, daily=daily, orders=orders)
    lookup = {(item.id, item.order_date): int(item.delay_days) if pd.notna(item.delay_days) else 10000 for item in orders.itertuples(index=False)}
    mine = Replay(world, Variant("regression"), delay_lookup=lookup).run().daily_frame().sort_values(["id", "date"]).reset_index(drop=True)
    saved = daily[daily.date >= START].sort_values(["id", "date"]).reset_index(drop=True)
    for column in ["on_hand_start", "received", "sales", "unmet", "on_hand_end", "on_order_end", "reorder_point", "order_up_to", "ordered"]:
        assert (mine[column].to_numpy() == saved[column].to_numpy()).all(), column


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_saved_routine_run_repeats_exactly(saved_daily, scenario):
    world = load_world(scenario, CONFIG, START)
    again = Replay(world, Variant("reorder_point")).run().daily_frame()
    pd.testing.assert_frame_equal(again, saved_daily[scenario, "reorder_point"], check_dtype=False)


def test_decisions_repeat_exactly(world):
    rule = lambda cases: np.where(cases.cover_days < 10, "expedite_recommendation", "monitor").astype(object)
    first = run(world, scorer=StandInScorer(rule), decide="stand_in")
    second = run(world, scorer=StandInScorer(rule), decide="stand_in")
    pd.testing.assert_frame_equal(first.daily_frame(), second.daily_frame())
    assert first.actions == second.actions


# Expedite and order mechanics.
def test_expedite_halves_the_wait_and_charges_once(world):
    replay = Replay(world, Variant("test"))
    replay.open[0] = []
    replay._place(0, 3, 12)
    item = replay.open[0][0]
    item["arrival"] = 13
    assert replay._expedite(0, 3) is True
    assert item["arrival"] == 8 and item["expedited"] is True
    costs = world.costs
    assert replay.expedite_cost[0, 3] == pytest.approx(12 * world.price[0, 3] * costs["shipping_rate_of_price"] * (costs["expedite_shipping_multiplier"] - 1) + costs["expedite_fixed_fee"])
    charged = replay.expedite_cost[0, 3]
    assert replay._expedite(0, 3) is False
    assert replay.expedite_cost[0, 3] == charged and item["arrival"] == 8


def test_weak_expedite_saves_less_time(world):
    replay = Replay(world, Variant("test", expedite_fraction=0.75))
    replay.open[0] = []
    replay._place(0, 3, 12)
    replay.open[0][0]["arrival"] = 13
    replay._expedite(0, 3)
    assert replay.open[0][0]["arrival"] == 11


def test_expedite_with_nothing_to_gain_costs_nothing(world):
    replay = Replay(world, Variant("test"))
    replay.open[0] = []
    assert replay._expedite(0, 3) is False
    replay._place(0, 3, 12)
    replay.open[0][0]["arrival"] = 4
    assert replay._expedite(0, 3) is False
    assert replay.expedite_cost[0].sum() == 0 and replay.arrays["expedites"][0].sum() == 0


def test_approved_order_uses_whole_packs(world):
    replay = run(world, scorer=StandInScorer(lambda cases: np.where(cases.open_pos == 0, "reorder_recommendation", "monitor").astype(object)), decide="stand_in")
    assert replay.new_orders
    for item in replay.new_orders:
        pack = replay.packs[replay.row_of[item["id"]]]
        assert item["quantity"] > 0 and item["quantity"] % pack == 0


def test_forecast_levels_add_safety_for_uncertainty():
    policy = SimpleNamespace(z=1.28, cover_days=14)
    mean = np.full((1, 28), 2.0)
    narrow = {"mean": mean, "quantiles": np.stack([mean * (1 + (level - 0.5) * 0.1) for level in LEVELS], axis=2), "levels": LEVELS}
    wide = {"mean": mean, "quantiles": np.stack([mean * (1 + (level - 0.5) * 2) for level in LEVELS], axis=2), "levels": LEVELS}
    (low_point, low_up), (high_point, high_up) = forecast_levels(narrow, np.array([7]), policy)[0], forecast_levels(wide, np.array([7]), policy)[0]
    assert low_point >= 28 and high_point > low_point
    assert low_up - low_point == high_up - high_point == 28


# Stress.
def test_zero_demand(world):
    world.demand[:] = 0
    daily = run(world).daily_frame()
    assert_conserved(daily)
    assert daily.unmet.sum() == 0 and daily.sales.sum() == 0


def test_demand_spike(world):
    world.demand[:] = world.demand * 50 + 5
    daily = run(world, scorer=StandInScorer(always("expedite_recommendation")), decide="stand_in").daily_frame()
    assert_conserved(daily)
    assert daily.unmet.sum() > 0


def test_every_supplier_delayed(world):
    calm = run(world).daily_frame().unmet.sum()
    world.delay_prob[:] = 1.0
    world.mean_delay[:] = 30.0
    daily = run(world).daily_frame()
    assert_conserved(daily)
    assert daily.unmet.sum() >= calm


def test_store_closed_throughout(world):
    world.is_open[:] = False
    replay = run(world, scorer=StandInScorer(always("reorder_recommendation")), decide="stand_in")
    daily = replay.daily_frame()
    assert_conserved(daily)
    assert daily.sales.sum() == 0 and daily.received.sum() == 0 and daily.ordered.sum() == 0
    assert {a["action"] for a in replay.actions} == {"none"}


def test_missing_stockout_scores_go_to_a_person(world):
    class Missing(StandInScorer):
        def decide(self, cases, orders, variant):
            cases = cases.copy()
            cases["stockout_risk"], cases["stockout_risk_lightgbm"] = np.nan, np.nan
            cases["suggested"] = self.rule(cases)
            cases["final"], cases["reason"] = code_checks(cases, cases.suggested, variant.conflict_rule)
            return cases

    replay = run(world, scorer=Missing(always("monitor")), decide="stand_in", person=DECLINE_ALL)
    assert {a["final"] for a in replay.actions} == {"human_review"}
    assert {a["action"] for a in replay.actions} == {"none"}


# No future leakage.
def test_changing_future_demand_leaves_earlier_decisions_alone(world, small_inputs):
    cut = 24  # the fourth review is on day 24
    assert world.dates[cut].weekday() == 0
    rule = lambda cases: np.where(cases.cover_days < 10, "expedite_recommendation", "monitor").astype(object)
    first = run(world, scorer=(early := StandInScorer(rule)), decide="stand_in")
    changed = replace(world, demand=world.demand.copy())
    changed.demand[:, cut + 1:] = changed.demand[:, cut + 1:] * 7 + 3
    second = run(changed, scorer=(late := StandInScorer(rule)), decide="stand_in")
    pd.testing.assert_frame_equal(first.daily_frame(cut), second.daily_frame(cut))
    assert len(early.seen) >= 4
    for left, right in list(zip(early.seen, late.seen))[:4]:
        assert left.origin.iloc[0] <= world.dates[cut]
        pd.testing.assert_frame_equal(left, right)
    assert first.daily_frame().unmet.sum() != second.daily_frame().unmet.sum()


def test_hidden_delays_do_not_reach_the_cases_before_arrival(world):
    """An order's delay is only known once its promised date passes, so it cannot change what a review sees before then."""
    rule = always("monitor")
    world.delay_prob[:] = 0.0
    calm = StandInScorer(rule)
    run(world, scorer=calm, decide="stand_in")
    late = replace(world, delay_prob=np.ones_like(world.delay_prob), mean_delay=np.full_like(world.mean_delay, 10.0))
    delayed = StandInScorer(rule)
    run(late, scorer=delayed, decide="stand_in")
    columns = ["id", "origin", "on_hand", "on_order", "open_pos", "units_due_14", "days_to_next_due", "units_overdue"]
    pd.testing.assert_frame_equal(calm.seen[0][columns], delayed.seen[0][columns])
    # A week later, series whose promised lead is over a week still have every order on the way in both runs.
    long_lead = world.series.merge(world.suppliers[["supplier_id", "promised_lead_days"]], on="supplier_id").query("promised_lead_days > 7").id
    assert len(long_lead)
    pick = lambda frame: frame[frame.id.isin(long_lead)][["id", "origin", "open_pos", "days_to_next_due"]].reset_index(drop=True)
    pending_at_start = set(world.history_orders[world.history_orders.receipt_date >= world.dates[0]].id)
    fresh = [i for i in long_lead if i not in pending_at_start]
    assert pick(calm.seen[1])[lambda f: f.id.isin(fresh)].reset_index(drop=True).equals(pick(delayed.seen[1])[lambda f: f.id.isin(fresh)].reset_index(drop=True))


def test_forecaster_sees_only_days_already_played(world):
    replay = run(world)
    day = 10
    dates = pd.date_range(world.dates[0], periods=DAYS)
    view = SimpleNamespace(dates=dates, demand=np.full((SERIES_COUNT, DAYS), -1.0, dtype=np.float32), observed=np.zeros((SERIES_COUNT, DAYS), dtype=bool))
    write_days(view, np.ones((SERIES_COUNT, DAYS), dtype=bool), replay, day)
    assert (view.demand[:, day + 1:] == -1).all() and not view.observed[:, day + 1:].any()
    assert (view.demand[:, :day + 1] == replay.arrays["sales"][:, :day + 1]).all()
    assert (view.observed[:, :day + 1] == (replay.arrays["on_hand_end"][:, :day + 1] != 0)).all()
