"""Step 5 tests: label correctness, feature availability, chronological splits, missing values and class imbalance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting.panel import DemandPanel
from src.stockout.baselines import ConstantRate, CoverRule, SmallLogistic
from src.stockout.evaluation import expected_calibration_error, score_probabilities
from src.stockout.features import FEATURES, as_matrix, build_features, open_order_features, split_by_label_end, supplier_history
from src.stockout.forecasts import scenario_panel
from src.stockout.tabpfn_model import TabPFNStockout

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
HORIZON = CONFIG["labels"]["stockout_horizon_days"]
SIM = ROOT / "data/simulated"
ORIGIN = pd.Timestamp("2015-03-02")
HIDDEN = {"unmet", "stockout_day", "demand", "lost_margin", "scenario", "tier", "delay_prob", "mean_delay_days", "receipt_date", "actual_lead_days", "label"}


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(ROOT / "data/processed/stockout_features.parquet")


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(ROOT / "data/processed/stockout_predictions.parquet")


@pytest.fixture(scope="module")
def daily():
    return pd.read_parquet(SIM / "inventory_daily.parquet", filters=[("scenario", "==", "baseline")])


@pytest.fixture(scope="module")
def orders():
    return pd.read_parquet(SIM / "purchase_orders.parquet", filters=[("scenario", "==", "baseline")])


def test_label_matches_unmet_in_next_14_days(features, daily):
    part = features[(features.scenario == "baseline") & (features.origin == ORIGIN)].iloc[::25]
    for row in part.itertuples():
        window = daily[(daily.id == row.id) & (daily.date > ORIGIN) & (daily.date <= ORIGIN + pd.Timedelta(days=HORIZON))]
        assert len(window) == HORIZON
        assert row.label == float((window.unmet > 0).any()), row.id


def test_label_ignores_stockout_on_the_review_day(daily):
    one = daily[daily.id == daily.id.iloc[0]].reset_index(drop=True)
    event = one.index[one.stockout_day][0]
    assert one.stockout_next_14d.iloc[event - 1] == 1.0
    following = one.iloc[event + 1:event + HORIZON + 1]
    assert one.stockout_next_14d.iloc[event] == float(following.stockout_day.any())


def test_labels_are_binary_and_complete(features):
    assert set(features.label.unique()) <= {0.0, 1.0}


def test_features_exclude_hidden_information():
    assert not HIDDEN & set(FEATURES)


def test_features_ignore_everything_after_the_origin(features, daily, orders):
    forecasts = pd.read_parquet(ROOT / "data/processed/stockout_forecasts/baseline.parquet")
    forecasts = forecasts[forecasts.origin == ORIGIN]
    series = pd.read_parquet(SIM / "series.parquet")
    suppliers = pd.read_parquet(SIM / "suppliers.parquet")
    mapping = {ORIGIN: "validation"}
    original = build_features(daily, orders, forecasts, series, suppliers, [ORIGIN], HORIZON, mapping)

    changed_daily = daily.copy()
    later = changed_daily.date > ORIGIN
    changed_daily.loc[later, ["sales", "on_hand_end", "demand", "unmet"]] = 99999
    changed_orders = orders[orders.order_date <= ORIGIN].copy()
    late = changed_orders.receipt_date > ORIGIN
    changed_orders.loc[late, "delay_days"] = 77
    changed_orders.loc[late, "receipt_date"] = changed_orders.loc[late, "receipt_date"] + pd.Timedelta(days=50)
    altered = build_features(changed_daily, changed_orders, forecasts, series, suppliers, [ORIGIN], HORIZON, mapping)
    pd.testing.assert_frame_equal(original[FEATURES], altered[FEATURES])


def test_open_order_features_on_known_case():
    orders = pd.DataFrame({
        "id": ["a", "a", "a", "b"], "supplier_id": "s", "quantity": [10, 20, 40, 5],
        "order_date": pd.to_datetime(["2015-01-01", "2015-01-20", "2015-01-25", "2015-01-01"]),
        "promised_date": pd.to_datetime(["2015-01-25", "2015-02-05", "2015-03-01", "2015-01-10"]),
        "receipt_date": pd.to_datetime(["2015-02-03", "2015-02-10", "2015-03-09", "2015-01-10"]),
        "actual_lead_days": pd.array([33, 21, 43, 9], dtype="Int64"), "delay_days": pd.array([9, 5, 8, 0], dtype="Int64"),
    })
    result = open_order_features(orders, "2015-02-01", 14).loc["a"]
    assert result.open_pos == 3
    assert result.units_overdue == 10
    assert result.days_overdue_max == 7
    assert result.units_due_14 == 20
    assert result.days_to_next_due == 4
    assert "b" not in open_order_features(orders, "2015-02-01", 14).index


def test_supplier_history_uses_only_receipts_before_origin():
    orders = pd.DataFrame({
        "id": "a", "supplier_id": "s", "quantity": 1, "order_date": pd.to_datetime(["2015-01-01", "2015-01-05"]),
        "promised_date": pd.to_datetime(["2015-01-05", "2015-01-10"]), "receipt_date": pd.to_datetime(["2015-01-05", "2015-03-01"]),
        "actual_lead_days": pd.array([4, 55], dtype="Int64"), "delay_days": pd.array([0, 50], dtype="Int64"),
    })
    result = supplier_history(orders, "2015-02-01").loc["s"]
    assert result.supplier_pos_seen == 1
    assert result.supplier_late_rate == 0.0
    assert result.supplier_mean_delay == 0.0


def test_scenario_panel_marks_sold_out_days_and_uses_no_unmet(daily):
    panel = DemandPanel()
    view = scenario_panel(panel, daily)
    position = panel.day_index[ORIGIN]
    today = daily[daily.date == ORIGIN].set_index("id").reindex(panel.series.id)
    assert np.array_equal(view.demand[:, position], today.sales.to_numpy(dtype=np.float32))
    assert not view.observed[:, position][today.on_hand_end.to_numpy() == 0].any()
    assert np.array_equal(panel.demand[:, 100], view.demand[:, 100])

    changed = daily.copy()
    changed["unmet"] = 12345
    changed["demand"] = 12345
    other = scenario_panel(panel, changed)
    assert np.array_equal(view.demand, other.demand)
    assert np.array_equal(view.observed, other.observed)


def test_splits_keep_whole_label_windows_inside_their_period(features):
    dates = CONFIG["dates"]
    train_end, validation_end = pd.Timestamp(dates["train_end"]), pd.Timestamp(dates["validation_end"])
    window = pd.Timedelta(days=HORIZON)
    spans = features.groupby("split").origin.agg(["min", "max"])
    assert spans.loc["train", "max"] + window <= train_end
    assert spans.loc["validation", "min"] > train_end
    assert spans.loc["validation", "max"] + window <= validation_end
    assert spans.loc["test", "min"] > validation_end
    assert spans.loc["test", "max"] + window <= pd.Timestamp(dates["simulation_end"])


def test_burn_in_is_excluded(features):
    assert features.origin.min() > pd.Timestamp(CONFIG["dates"]["burn_in_end"])


def test_split_function_purges_boundary_origins():
    mapping = split_by_label_end(pd.to_datetime(["2014-12-15", "2014-12-22", "2015-01-05", "2015-12-28", "2016-01-04"]), CONFIG, HORIZON)
    assert list(mapping.values()) == ["train", "purged", "validation", "purged", "test"]


def test_predictions_never_use_purged_rows(predictions):
    assert set(predictions.split) == {"validation", "test"}


def test_rows_unique_and_complete(features):
    assert not features.duplicated(["scenario", "id", "origin"]).any()
    assert features.groupby(["scenario", "origin"]).size().eq(500).all()


def test_only_documented_feature_has_missing_values(features):
    missing = features[FEATURES].isna().sum()
    assert list(missing[missing > 0].index) == ["days_to_next_due"]


def test_next_due_missing_only_without_upcoming_orders(features):
    empty = features[features.days_to_next_due.isna()]
    assert (empty.units_due_14 == 0).all()
    assert (features[features.open_pos == 0].days_to_next_due.isna()).all()


def test_no_infinite_feature_values(features):
    assert not np.isinf(as_matrix(features)).any()


def test_tabpfn_handles_missing_values_and_is_reproducible(features):
    train = features[features.split == "train"]
    rows = features[features.split == "validation"].iloc[:20]
    assert rows.days_to_next_due.isna().any()
    first = TabPFNStockout(300, 1, 20, 42).fit(train).predict(rows)
    second = TabPFNStockout(300, 1, 20, 42).fit(train).predict(rows)
    assert np.allclose(first, second, atol=1e-5)
    assert ((first >= 0) & (first <= 1)).all()


def test_class_imbalance_is_present_and_within_expected_range(features):
    rates = features[features.split != "purged"].groupby(["scenario", "split"]).label.mean()
    assert ((rates >= 0.05) & (rates <= 0.20)).all()


def test_training_sample_preserves_positive_rate_roughly(features):
    train = features[features.split == "train"]
    sample = train.sample(3000, random_state=42)
    assert abs(sample.label.mean() - train.label.mean()) < 0.015


def test_all_saved_probabilities_valid(predictions):
    probabilities = predictions[["constant", "cover_rule", "logistic", "lightgbm", "tabpfn"]]
    assert probabilities.notna().all().all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all().all()


def test_saved_predictions_have_all_rows(predictions):
    assert predictions.groupby("split").size().to_dict() == {"test": 27000, "validation": 75000}


def test_saved_metrics_recompute_from_predictions(predictions):
    test = predictions[predictions.split == "test"]
    metrics = pd.read_csv(ROOT / "reports/step_05/metrics.csv", index_col=0)
    saved = metrics[metrics.period == "test"].loc["tabpfn"]
    result = score_probabilities(test.label, test.tabpfn)
    assert result["pr_auc"] == pytest.approx(saved.pr_auc, abs=1e-4)
    assert result["brier"] == pytest.approx(saved.brier, abs=1e-4)


def test_tabpfn_beats_simple_baselines_on_test():
    comparison = json.loads((ROOT / "reports/step_05/comparison.json").read_text())
    for rival in ["cover_rule", "logistic"]:
        assert comparison[rival]["pr_auc"]["ci95_low"] > 0, rival
        assert comparison[rival]["brier"]["ci95_high"] < 0, rival


def test_tabpfn_calibration_within_acceptable_gap(predictions):
    test = predictions[predictions.split == "test"]
    assert expected_calibration_error(test.label.to_numpy(), test.tabpfn.to_numpy()) < 0.03


def test_calibration_metric_on_known_values():
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(labels, np.array([0.0, 0.0, 1.0, 1.0]), bins=2) == pytest.approx(0.0)
    assert expected_calibration_error(labels, np.array([0.25, 0.25, 0.75, 0.75]), bins=2) == pytest.approx(0.25)


def test_constant_baseline_has_no_skill():
    frame = pd.DataFrame({"label": [0.0, 1.0, 0.0, 1.0]})
    result = score_probabilities(frame.label, ConstantRate().fit(frame).predict(frame))
    assert result["roc_auc"] == 0.5
    assert result["brier"] == pytest.approx(0.25)


def test_cover_rule_risk_never_rises_with_more_cover(features):
    train = features[features.split == "train"]
    model = CoverRule().fit(train)
    grid = pd.DataFrame({"cover_days": np.linspace(0, 120, 200)})
    assert (np.diff(model.predict(grid)) <= 1e-12).all()


def test_logistic_handles_extreme_values(features):
    train = features[features.split == "train"].iloc[:2000]
    model = SmallLogistic().fit(train)
    extreme = train.iloc[:3].copy()
    extreme["cover_days"] = [0.0, 1e6, 5.0]
    assert np.isfinite(model.predict(extreme)).all()
