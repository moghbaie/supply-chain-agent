"""Run the saved model pipeline on replayed states: forecast, stockout risk, then a category. Runs locally and sends nothing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.escalation.cases import CATEGORIES, build_cases, value_bands
from src.escalation.policy import PolicyThresholds, policy_baseline
from src.escalation.ultrabert import FineTunedClassifier
from src.evaluation.decisions import code_checks, weak_evidence
from src.evaluation.replay import forecast_view, write_days
from src.forecasting.models import LEVELS, TimestralRunner
from src.forecasting.panel import DemandPanel
from src.stockout.baselines import Boosted
from src.stockout.forecasts import scenario_panel, summarize_forecast
from src.stockout.tabpfn_model import TabPFNStockout
from src.supplier.inbound import recent_supplier

ORDER = np.array(CATEGORIES, dtype=object)
SCORES = [f"p_{c}" for c in CATEGORIES]
HORIZON = 28
LEVEL_POSITIONS = {level: int(np.flatnonzero(np.isclose(LEVELS, level))[0]) for level in (0.1, 0.9)}


class ModelScorer:
    """Hold the fitted models once, and keep each replay's own forecast history and supplier state."""

    def __init__(self, panel, runner, covariates, risk_models, classifier, thresholds, value_edges, saved_daily, start, recent_start):
        self.panel, self.runner, self.covariates = panel, runner, covariates
        self.tabpfn, self.lightgbm = risk_models
        self.classifier, self.thresholds, self.value_edges = classifier, thresholds, value_edges
        self.saved_daily, self.start, self.recent_start = saved_daily, pd.Timestamp(start), recent_start
        self.views, self.carry = {}, {}

    def forecast(self, replay, day):
        key = id(replay)
        if key not in self.views:
            before = self.saved_daily[(self.saved_daily.scenario == replay.world.scenario) & (self.saved_daily.date < self.start)]
            self.views[key] = forecast_view(self.panel, scenario_panel(self.panel, before))
        view = self.views[key]
        write_days(view, self.panel.observed, replay, day)
        origin = replay.world.dates[day]
        forecast = self.runner.forecast(view, origin, HORIZON, self.covariates)
        return forecast, summarize_forecast(self.panel.series.id.to_numpy(), origin, forecast, LEVEL_POSITIONS)

    def _recent_rate(self, cases, orders, variant):
        """Use the supplier's late rate over the last 30 days, keeping its last known value when no receipts arrived."""
        key = (cases.scenario.iloc[0], variant.name)
        carried = self.carry.setdefault(key, dict(self.recent_start[cases.scenario.iloc[0]]))
        carried.update(recent_supplier(orders, cases.origin.iloc[0]).supplier_recent_late_rate.dropna().to_dict())
        return cases.supplier_id.map(carried)

    def decide(self, cases, orders, variant):
        cases = cases.copy()
        cases["supplier_recent_late_rate"] = self._recent_rate(cases, orders, variant)
        cases["stockout_risk"] = self.tabpfn.predict(cases)
        cases["stockout_risk_lightgbm"] = self.lightgbm.predict(cases)
        cases["value_band"] = value_bands(cases, self.value_edges)
        if variant.decide == "ultrabert":
            scores = self.classifier.predict(cases)
            cases[SCORES] = scores
            cases["suggested"] = ORDER[scores.argmax(axis=1)]
        else:
            cases["suggested"] = policy_baseline(cases, self.thresholds)
        cases["final"], cases["reason"] = code_checks(cases, cases.suggested, variant.conflict_rule)
        cases["weak_evidence"] = weak_evidence(cases, cases.final.to_numpy(dtype=object))
        return cases

    def forget(self, replay):
        self.views.pop(id(replay), None)
        torch.cuda.empty_cache()


def load_classifier(seed):
    classifier = FineTunedClassifier(8, 1e-5, seed)
    classifier.model.load_state_dict(torch.load("data/processed/escalation/ultrabert/best_model/weights.pt", map_location="cuda", weights_only=True))
    return classifier


def build_scorer(settings, scenarios):
    """Fit the Step 5 stockout models on the same data as their test run, and load the chosen forecaster and classifier."""
    seed = settings["seed"]
    data = pd.read_parquet("data/processed/stockout_features.parquet")
    history = pd.concat([data[data.split == "train"], data[data.split == "validation"]], ignore_index=True)  # the row order of the Step 5 fit, which the fits depend on
    tabpfn, booster = TabPFNStockout(3000, 2, 5000, seed).fit(history), Boosted(seed).fit(history)
    _, fitted = build_cases()
    recent = pd.read_parquet("data/processed/delay/inbound_features.parquet", columns=["scenario", "supplier_id", "order_date", "supplier_recent_late_rate"])
    recent = recent[recent.order_date < settings["start"]].sort_values("order_date").groupby(["scenario", "supplier_id"]).supplier_recent_late_rate.last()
    chosen = json.loads(Path("reports/step_04/chosen_model.json").read_text())
    forecast_config = yaml.safe_load(Path("configs/forecasting.yaml").read_text())
    saved_daily = pd.read_parquet("data/simulated/inventory_daily.parquet", columns=["scenario", "id", "date", "sales", "on_hand_end"])
    thresholds = PolicyThresholds(**json.loads(Path("data/processed/escalation/policy_thresholds.json").read_text()))
    return ModelScorer(DemandPanel(), TimestralRunner(chosen["size"], forecast_config["batch_size"]), chosen["covariates"], (tabpfn, booster), load_classifier(seed), thresholds, fitted["value_edges"], saved_daily, settings["start"], {s: recent.loc[s].to_dict() for s in scenarios})
