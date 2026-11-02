"""Accuracy metrics computed only on observed days."""
import numpy as np


def quantile_at(forecast, level):
    position = np.flatnonzero(np.isclose(forecast["levels"], level))
    if position.size != 1:
        raise ValueError(f"level {level} not in forecast")
    return forecast["quantiles"][..., position[0]]


def pinball(actual, predicted, level):
    error = actual - predicted
    return np.maximum(level * error, (level - 1) * error)


def score(actual, seen, forecast, levels, steps=None):
    """Return WAPE and bias for mean and median, pinball loss and 80% interval coverage."""
    steps = slice(None) if steps is None else steps
    truth = actual[:, steps]
    mask = seen[:, steps]
    total = truth[mask].sum()
    median = quantile_at(forecast, 0.5)[:, steps]
    mean = forecast["mean"][:, steps]
    low = quantile_at(forecast, 0.1)[:, steps]
    high = quantile_at(forecast, 0.9)[:, steps]
    return {
        "units": float(total),
        "abs_error_median": float(np.abs(median - truth)[mask].sum()),
        "abs_error_mean": float(np.abs(mean - truth)[mask].sum()),
        "signed_error_median": float((median - truth)[mask].sum()),
        "signed_error_mean": float((mean - truth)[mask].sum()),
        "pinball": float(sum(pinball(truth, quantile_at(forecast, level)[:, steps], level)[mask].sum() for level in levels)),
        "covered": float(((truth >= low) & (truth <= high))[mask].sum()),
        "days": float(mask.sum()),
    }


def summarize(totals):
    """Convert summed scores into percentages."""
    units = totals["units"]
    return {
        "wape_median_pct": 100 * totals["abs_error_median"] / units,
        "wape_mean_pct": 100 * totals["abs_error_mean"] / units,
        "bias_median_pct": 100 * totals["signed_error_median"] / units,
        "bias_mean_pct": 100 * totals["signed_error_mean"] / units,
        "pinball_per_unit": totals["pinball"] / units,
        "coverage_80_pct": 100 * totals["covered"] / totals["days"],
    }


def cumulative_wape(actual, seen, mean, days):
    """Return WAPE of each series' total demand over the first given days, the quantity reorder decisions use."""
    truth = np.where(seen[:, :days], actual[:, :days], 0).sum(axis=1)
    predicted = np.where(seen[:, :days], mean[:, :days], 0).sum(axis=1)
    return np.abs(predicted - truth), truth
