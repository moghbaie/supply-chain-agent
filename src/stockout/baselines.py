"""Baselines for 14-day stockout risk, from a constant rate up to gradient boosting."""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stockout.features import FEATURES, as_matrix

LOGISTIC_FEATURES = ["cover_days", "position_cover_days", "projected_14", "projected_14_q90", "sold_out_days_28", "units_overdue", "supplier_late_rate"]
LIGHTGBM_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "deterministic": True, "force_row_wise": True}


class ConstantRate:
    def fit(self, frame):
        self.rate = float(frame.label.mean())
        return self

    def predict(self, frame):
        return np.full(len(frame), self.rate)


class CoverRule:
    """Convert days of stock cover into a probability with a monotonic fit."""

    def fit(self, frame):
        self.model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(frame.cover_days, frame.label)
        return self

    def predict(self, frame):
        return self.model.predict(frame.cover_days)


class SmallLogistic:
    def fit(self, frame):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.model.fit(self._inputs(frame), frame.label)
        return self

    @staticmethod
    def _inputs(frame):
        return np.arcsinh(frame[LOGISTIC_FEATURES].to_numpy(dtype=float))

    def predict(self, frame):
        return self.model.predict_proba(self._inputs(frame))[:, 1]


class Boosted:
    def __init__(self, seed, rounds=300):
        self.seed, self.rounds = seed, rounds

    def fit(self, frame):
        dataset = lgb.Dataset(as_matrix(frame), frame.label.to_numpy(), feature_name=FEATURES, free_raw_data=False)
        self.model = lgb.train({**LIGHTGBM_PARAMS, "seed": self.seed}, dataset, self.rounds)
        return self

    def predict(self, frame):
        return self.model.predict(as_matrix(frame))
