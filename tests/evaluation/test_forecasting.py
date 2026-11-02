"""Step 4 tests: future leakage, horizon alignment, valid intervals, stable inference and baseline comparison."""
import json
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting.metrics import pinball, score, summarize
from src.forecasting.models import LEVELS, GbmForecaster, TimestralRunner, gbm_features, last_same_weekday_index, mean_from_quantiles, seasonal_naive
from src.forecasting.panel import DemandPanel, monday_origins
from src.simulation.demand import daily_prices, load_m5

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/forecasting.yaml").read_text())
CHOSEN = json.loads((ROOT / "reports/step_04/chosen_model.json").read_text())
HORIZON = CONFIG["horizon_days"]
ORIGIN = pd.Timestamp("2015-03-02")


@pytest.fixture(scope="module")
def panel():
    return DemandPanel()


@pytest.fixture(scope="module")
def runner():
    return TimestralRunner(CHOSEN["size"], CONFIG["batch_size"])


@pytest.fixture(scope="module")
def price(panel):
    calendar, _, prices, _, dates = load_m5()
    return daily_prices(panel.series, prices, calendar, dates)[0]


@pytest.fixture(scope="module")
def saved():
    return pd.read_parquet(ROOT / "data/processed/forecasts/test.parquet")


def with_changed_future(panel, origin):
    """Return a panel copy whose demand after the origin is replaced with a huge value."""
    changed = copy(panel)
    changed.demand = panel.demand.copy()
    changed.demand[:, panel.day_index[origin] + 1:] = 99999
    return changed


def test_context_ends_at_origin(panel):
    assert panel.context(ORIGIN).shape[1] == panel.day_index[ORIGIN] + 1


def test_context_marks_prelaunch_and_closed_days_missing(panel):
    context = panel.context(ORIGIN)
    assert np.array_equal(np.isfinite(context), panel.observed[:, :context.shape[1]])
    christmas = panel.day_index[pd.Timestamp("2014-12-25")]
    assert np.isnan(context[:, christmas]).any()


def test_timestral_ignores_future_demand(panel, runner):
    original = runner.forecast(panel, ORIGIN, HORIZON)
    changed = runner.forecast(with_changed_future(panel, ORIGIN), ORIGIN, HORIZON)
    assert np.array_equal(original["quantiles"], changed["quantiles"])


def test_seasonal_naive_ignores_future_demand(panel):
    original = seasonal_naive(panel, ORIGIN, HORIZON, 56)
    changed = seasonal_naive(with_changed_future(panel, ORIGIN), ORIGIN, HORIZON, 56)
    assert np.array_equal(original["mean"], changed["mean"])
    assert np.array_equal(original["quantiles"], changed["quantiles"])


def test_gbm_features_ignore_future_demand(panel, price):
    original = gbm_features(panel, price, ORIGIN, HORIZON)
    changed = gbm_features(with_changed_future(panel, ORIGIN), price, ORIGIN, HORIZON)
    pd.testing.assert_frame_equal(original, changed)


def test_gbm_uses_only_price_known_at_origin(panel, price):
    later = price.copy()
    later[:, panel.day_index[ORIGIN] + 1:] = 12345.0
    pd.testing.assert_frame_equal(gbm_features(panel, price, ORIGIN, HORIZON), gbm_features(panel, later, ORIGIN, HORIZON))


def test_gbm_training_ignores_data_after_cutoff(panel, price):
    settings = {**CONFIG, "lightgbm": {**CONFIG["lightgbm"], "train_first_origin": "2014-09-01", "num_boost_round": 5}}
    cutoff = pd.Timestamp("2014-12-31")
    changed = with_changed_future(panel, cutoff)
    first = GbmForecaster(panel, price, settings, 42).fit(cutoff)
    second = GbmForecaster(changed, price, settings, 42).fit(cutoff)
    assert first.train_rows == second.train_rows > 0
    origin_before_cutoff = pd.Timestamp("2014-12-29")
    first_output = first.forecast(origin_before_cutoff)
    second_output = second.forecast(origin_before_cutoff)
    assert np.array_equal(first_output["mean"], second_output["mean"])
    assert np.array_equal(first_output["quantiles"], second_output["quantiles"])


def test_training_origins_finish_before_cutoff():
    origins = monday_origins("2013-01-08", "2014-12-31", HORIZON)
    assert (origins + pd.Timedelta(days=HORIZON) <= pd.Timestamp("2014-12-31")).all()
    assert (origins.weekday == 0).all()
    assert len(origins) == 100


def test_validation_and_test_targets_do_not_overlap():
    validation = monday_origins(CONFIG["validation"]["first_target"], CONFIG["validation"]["last_target"], HORIZON)
    test = monday_origins(CONFIG["test"]["first_target"], CONFIG["test"]["last_target"], HORIZON)
    assert validation[-1] + pd.Timedelta(days=HORIZON) < test[0]
    assert (len(validation), len(test)) == (48, 16)


def test_actuals_align_with_horizon(panel):
    actual, _ = panel.actuals(ORIGIN, HORIZON)
    start = panel.day_index[ORIGIN] + 1
    assert actual.shape == (len(panel.series), HORIZON)
    assert np.array_equal(actual[:, 0], panel.demand[:, start])
    assert np.array_equal(actual[:, -1], panel.demand[:, start + HORIZON - 1])


def test_saved_actuals_match_m5_dates(saved, panel):
    sample = saved[(saved.model == "timestral") & (saved.origin == saved.origin.min())]
    position = {series_id: i for i, series_id in enumerate(panel.series.id)}
    for row in sample.iloc[::1777].itertuples():
        target = row.origin + pd.Timedelta(days=int(row.horizon))
        assert row.actual == panel.demand[position[row.id], panel.day_index[target]]


def test_saved_forecasts_are_complete(saved):
    assert not saved[["mean", "q10", "q50", "q90", "actual"]].isna().any().any()
    assert saved.groupby(["model", "origin", "id"]).horizon.nunique().eq(HORIZON).all()
    assert saved.groupby("model").origin.nunique().eq(16).all()


def test_seasonal_index_matches_target_weekday():
    for step in range(1, 29):
        index = last_same_weekday_index(1000, step)
        assert index <= 1000
        assert (1000 + step - index) % 7 == 0
        assert 1000 - index < 7


def test_timestral_shape_and_levels(panel, runner):
    forecast = runner.forecast(panel, ORIGIN, HORIZON)
    assert forecast["quantiles"].shape == (len(panel.series), HORIZON, len(LEVELS))
    assert forecast["mean"].shape == (len(panel.series), HORIZON)


def test_timestral_quantiles_ordered_and_nonnegative(panel, runner):
    quantiles = runner.forecast(panel, ORIGIN, HORIZON)["quantiles"]
    assert (quantiles >= 0).all()
    assert (np.diff(quantiles, axis=-1) >= -1e-5).all()


def test_saved_intervals_valid(saved):
    assert (saved.q10 <= saved.q50 + 1e-5).all()
    assert (saved.q50 <= saved.q90 + 1e-5).all()
    assert (saved[["q10", "q50", "q90", "mean"]] >= 0).all().all()


def test_mean_stays_inside_forecast_range(panel, runner):
    forecast = runner.forecast(panel, ORIGIN, HORIZON)
    assert (forecast["mean"] >= forecast["quantiles"][..., 0] - 1e-4).all()
    assert (forecast["mean"] <= forecast["quantiles"][..., -1] + 1e-4).all()


def test_inference_is_stable_across_repeats(panel, runner):
    assert np.array_equal(runner.forecast(panel, ORIGIN, HORIZON)["quantiles"], runner.forecast(panel, ORIGIN, HORIZON)["quantiles"])


def test_inference_barely_changes_with_batch_size(panel):
    large = TimestralRunner(CHOSEN["size"], 64).forecast(panel, ORIGIN, HORIZON)["quantiles"]
    small = TimestralRunner(CHOSEN["size"], 16).forecast(panel, ORIGIN, HORIZON)["quantiles"]
    assert np.abs(large - small).max() < 0.01


def test_mean_from_quantiles_on_symmetric_values():
    assert mean_from_quantiles(LEVELS - 0.5) == pytest.approx(0.0, abs=1e-9)
    assert mean_from_quantiles(np.full(len(LEVELS), 3.0)) == pytest.approx(3.0)


def test_metrics_on_known_values():
    actual = np.array([[10.0, 0.0]])
    seen = np.array([[True, True]])
    forecast = {"levels": np.array([0.1, 0.5, 0.9]), "quantiles": np.array([[[6.0, 8.0, 14.0], [0.0, 1.0, 2.0]]]), "mean": np.array([[9.0, 1.0]])}
    result = summarize(score(actual, seen, forecast, [0.1, 0.5, 0.9]))
    assert result["wape_median_pct"] == pytest.approx(30.0)
    assert result["wape_mean_pct"] == pytest.approx(20.0)
    assert result["bias_mean_pct"] == pytest.approx(0.0)
    assert result["bias_median_pct"] == pytest.approx(-10.0)
    assert result["coverage_80_pct"] == pytest.approx(100.0)


def test_metrics_skip_unobserved_days():
    actual = np.array([[10.0, 500.0]])
    seen = np.array([[True, False]])
    forecast = {"levels": np.array([0.1, 0.5, 0.9]), "quantiles": np.array([[[8.0, 10.0, 12.0], [0.0, 0.0, 0.0]]]), "mean": np.array([[10.0, 0.0]])}
    assert summarize(score(actual, seen, forecast, [0.1, 0.5, 0.9]))["wape_mean_pct"] == 0.0


def test_pinball_penalizes_each_side_by_level():
    assert pinball(np.array(10.0), np.array(8.0), 0.9) == pytest.approx(1.8)
    assert pinball(np.array(8.0), np.array(10.0), 0.9) == pytest.approx(0.2)


def test_timestral_beats_baselines_on_test_pinball():
    comparison = json.loads((ROOT / "reports/step_04/comparison.json").read_text())
    for rival in ["seasonal_naive", "lightgbm"]:
        assert comparison[rival]["ci95_high"] < 1.0, rival


def test_chosen_variant_follows_selection_rule():
    board = pd.read_csv(ROOT / "reports/step_04/selection_checkpoint.csv")
    limit = board.pinball_per_unit.min() * (1 + CONFIG["selection"]["tolerance_pct"] / 100)
    close = board[board.pinball_per_unit <= limit]
    assert CHOSEN["size"] == min(close["size"], key=CONFIG["selection"]["sizes"].index)
    assert CHOSEN["covariates"] == close[close["size"] == CHOSEN["size"]].sort_values("pinball_per_unit").iloc[0].covariates
