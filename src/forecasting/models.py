"""Forecasters with a shared output: quantiles at LEVELS plus a mean, shaped by series and horizon."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

from src.forecasting.panel import monday_origins

LEVELS = np.array([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99])
GBM_LEVELS = [0.1, 0.5, 0.9]
FEATURES = ["horizon", "target_weekday", "target_month", "target_event", "target_snap", "price", "same_weekday_last",
            "same_weekday_mean_4", "mean_7", "mean_28", "mean_56", "std_28", "zero_share_28", "cat_code", "state_code"]


def mean_from_quantiles(quantiles, levels=LEVELS):
    """Approximate the mean by weighting each quantile with the probability it represents."""
    edges = np.concatenate([[0.0], (levels[1:] + levels[:-1]) / 2, [1.0]])
    return (quantiles * np.diff(edges)).sum(axis=-1)


class TimestralRunner:
    def __init__(self, size, batch_size, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device)
        self.batch_size = batch_size

    def forecast(self, panel, origin, horizon, covariates="none"):
        context = panel.context(origin)
        known = panel.calendar_covariates(origin, horizon) if covariates == "calendar" else None
        parts = []
        with torch.no_grad():
            for first in range(0, len(context), self.batch_size):
                arguments = {"known_covariates": known[first:first + self.batch_size]} if known is not None else {}
                output = self.model.forecast(context[first:first + self.batch_size], prediction_length=horizon,
                                             clip_nonnegative=True, **arguments)
                parts.append(output.quantiles.detach().cpu().numpy()[:, 0])
        quantiles = np.concatenate(parts)
        return {"levels": LEVELS, "quantiles": quantiles, "mean": mean_from_quantiles(quantiles)}


def last_same_weekday_index(origin_index, horizon_step):
    """Return the latest index at or before the origin that shares the target day's weekday."""
    return origin_index + horizon_step - 7 * int(np.ceil(horizon_step / 7))


def seasonal_naive(panel, origin, horizon, window_days):
    """Repeat last week's same-weekday demand, with empirical quantiles from the recent window."""
    history = panel.context(origin)
    origin_index = history.shape[1] - 1
    mean = np.stack([np.nan_to_num(history[:, last_same_weekday_index(origin_index, step)]) for step in range(1, horizon + 1)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recent = np.nanquantile(history[:, -window_days:], LEVELS, axis=1).T
    quantiles = np.repeat(recent[:, None, :], horizon, axis=1)
    return {"levels": LEVELS, "quantiles": quantiles, "mean": mean}


def gbm_features(panel, price, origin, horizon):
    """Build one row per series and horizon day from data known at the origin plus the calendar."""
    history = panel.context(origin)
    origin_index = history.shape[1] - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = {
            "price": price[:, origin_index],
            "mean_7": np.nanmean(history[:, -7:], axis=1),
            "mean_28": np.nanmean(history[:, -28:], axis=1),
            "mean_56": np.nanmean(history[:, -56:], axis=1),
            "std_28": np.nanstd(history[:, -28:], axis=1),
            "zero_share_28": np.nanmean(np.where(np.isnan(history[:, -28:]), np.nan, history[:, -28:] == 0), axis=1),
        }
    categories = panel.series.cat_id.astype("category").cat.codes.to_numpy()
    states = panel.series.state_id.astype("category").cat.codes.to_numpy()
    snap = {state: panel.calendar[f"snap_{state}"].to_numpy() for state in panel.series.state_id.unique()}
    frames = []
    for step in range(1, horizon + 1):
        target = origin_index + step
        same = [last_same_weekday_index(origin_index, step) - 7 * k for k in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            same_mean = np.nanmean(history[:, same], axis=1)
        frames.append(pd.DataFrame({
            **base, "horizon": step, "target_weekday": panel.calendar.wday.iloc[target], "target_month": panel.calendar.month.iloc[target],
            "target_event": int(pd.notna(panel.calendar.event_name_1.iloc[target])),
            "target_snap": [snap[s][target] for s in panel.series.state_id], "same_weekday_last": history[:, same[0]],
            "same_weekday_mean_4": same_mean, "cat_code": categories, "state_code": states}))
    return pd.concat(frames, ignore_index=True)[FEATURES]


class GbmForecaster:
    """Direct multi-horizon LightGBM: a Poisson mean plus quantile models, trained only on targets up to a cutoff."""

    def __init__(self, panel, price, config, seed):
        self.panel, self.price, self.settings, self.seed = panel, price, config["lightgbm"], seed
        self.horizon = config["horizon_days"]
        self.boosters = {}
        self.train_rows = 0

    def fit(self, cutoff):
        origins = monday_origins(pd.Timestamp(self.settings["train_first_origin"]) + pd.Timedelta(days=1), cutoff, self.horizon)
        features, targets, weights = [], [], []
        for origin in origins:
            truth, seen = self.panel.actuals(origin, self.horizon)
            features.append(gbm_features(self.panel, self.price, origin, self.horizon))
            targets.append(truth.T.reshape(-1))
            weights.append(seen.T.reshape(-1))
        keep = np.concatenate(weights)
        data = pd.concat(features, ignore_index=True)[keep]
        target = np.concatenate(targets)[keep]
        self.train_rows = len(data)
        dataset = lgb.Dataset(data, target, free_raw_data=False)
        common = {**self.settings["params"], "seed": self.seed, "deterministic": True, "force_row_wise": True}
        rounds = self.settings["num_boost_round"]
        self.boosters = {"mean": lgb.train({**common, "objective": "poisson"}, dataset, rounds)}
        for level in GBM_LEVELS:
            self.boosters[level] = lgb.train({**common, "objective": "quantile", "alpha": level}, dataset, rounds)
        return self

    def forecast(self, origin):
        rows = gbm_features(self.panel, self.price, origin, self.horizon)
        count = len(self.panel.series)
        reshape = lambda values: values.reshape(self.horizon, count).T
        quantiles = np.stack([np.clip(reshape(self.boosters[level].predict(rows)), 0, None) for level in GBM_LEVELS], axis=-1)
        return {"levels": np.array(GBM_LEVELS), "quantiles": np.sort(quantiles, axis=-1), "mean": reshape(self.boosters["mean"].predict(rows))}
