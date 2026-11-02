"""Build point-in-time demand contexts and weekly forecast origins from M5 for the selected series."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.demand import closed_store_days, load_m5

SERIES_PATH = Path("data/simulated/series.parquet")


class DemandPanel:
    """Daily demand for the selected series, where days before first sale and closed days are unobserved."""

    def __init__(self):
        self.calendar, sales, _, day_columns, self.dates = load_m5()
        self.series = pd.read_parquet(SERIES_PATH)
        self.demand = sales.set_index("id").loc[self.series.id, day_columns].to_numpy(dtype=np.float32)
        closed = closed_store_days(sales, day_columns).loc[self.series.store_id].to_numpy()
        launched = np.arange(len(self.dates))[None, :] >= (self.demand > 0).argmax(axis=1)[:, None]
        self.observed = launched & ~closed
        self.day_index = {date: i for i, date in enumerate(self.dates)}

    def context(self, origin):
        """Return demand through the origin date, with NaN on unobserved days."""
        end = self.day_index[pd.Timestamp(origin)] + 1
        return np.where(self.observed[:, :end], self.demand[:, :end], np.nan)

    def actuals(self, origin, horizon):
        start = self.day_index[pd.Timestamp(origin)] + 1
        return self.demand[:, start:start + horizon], self.observed[:, start:start + horizon]

    def calendar_covariates(self, origin, horizon):
        """Return covariates known in advance, shaped (series, covariates, context + horizon)."""
        end = self.day_index[pd.Timestamp(origin)] + 1 + horizon
        days = self.calendar.iloc[:end]
        weekday = 2 * np.pi * days.wday.to_numpy() / 7
        has_event = days.event_name_1.notna().to_numpy(dtype=np.float32)
        shared = np.stack([np.sin(weekday), np.cos(weekday), has_event]).astype(np.float32)
        snap = np.stack([days[f"snap_{state}"].to_numpy(dtype=np.float32) for state in self.series.state_id])
        stacked = np.repeat(shared[None, :, :], len(self.series), axis=0)
        return np.concatenate([stacked, snap[:, None, :]], axis=1)


def monday_origins(first_target, last_target, horizon):
    """Return Mondays whose full forecast horizon falls within the first and last target dates."""
    first = pd.Timestamp(first_target) - pd.Timedelta(days=1)
    last = pd.Timestamp(last_target) - pd.Timedelta(days=horizon)
    return pd.date_range(first, last, freq="W-MON")
