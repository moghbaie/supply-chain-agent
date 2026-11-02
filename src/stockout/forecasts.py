"""Build Timestral forecasts from what an analyst could see in each simulated scenario."""
from copy import copy

import numpy as np
import pandas as pd


def scenario_panel(panel, daily):
    """Return a panel whose history is simulated sales, with sold-out days marked unobserved.

    Unmet demand is hidden from analysts, so a day that ends with no stock counts as unobserved
    because its sales may understate demand.
    """
    columns = pd.DatetimeIndex(panel.dates)
    sales = daily.pivot(index="id", columns="date", values="sales").reindex(index=panel.series.id)
    sold_out = daily.pivot(index="id", columns="date", values="on_hand_end").reindex(index=panel.series.id) == 0
    positions = columns.get_indexer(sales.columns)
    view = copy(panel)
    view.demand = panel.demand.copy()
    view.observed = panel.observed.copy()
    view.demand[:, positions] = sales.to_numpy(dtype=np.float32)
    view.observed[:, positions] &= ~sold_out.to_numpy()
    return view


def summarize_forecast(series_ids, origin, forecast, level_positions):
    """Reduce a 28-day forecast to the totals that the stockout model uses."""
    mean, quantiles = forecast["mean"], forecast["quantiles"]
    return pd.DataFrame({
        "id": series_ids, "origin": origin,
        "fc_mean_7": mean[:, :7].sum(axis=1), "fc_mean_14": mean[:, :14].sum(axis=1), "fc_mean_28": mean.sum(axis=1),
        "fc_q10_sum_14": quantiles[:, :14, level_positions[0.1]].sum(axis=1),
        "fc_q90_sum_14": quantiles[:, :14, level_positions[0.9]].sum(axis=1),
        "fc_q90_max_14": quantiles[:, :14, level_positions[0.9]].max(axis=1),
    })
