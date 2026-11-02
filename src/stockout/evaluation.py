"""Metrics for stockout probabilities: ranking, accuracy of probabilities and calibration."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

TOP_SHARE = 0.05


def expected_calibration_error(labels, probabilities, bins=10):
    """Return the row-weighted gap between predicted and observed rates across equal-size bins."""
    order = np.argsort(probabilities, kind="stable")
    gap = 0.0
    for chunk in np.array_split(order, bins):
        gap += len(chunk) * abs(probabilities[chunk].mean() - labels[chunk].mean())
    return gap / len(labels)


def reliability_table(labels, probabilities, bins=10):
    """Return predicted and observed rates by equal-size probability bin."""
    order = np.argsort(probabilities, kind="stable")
    rows = [{"bin": i + 1, "rows": len(chunk), "predicted": probabilities[chunk].mean(), "observed": labels[chunk].mean()}
            for i, chunk in enumerate(np.array_split(order, bins))]
    return pd.DataFrame(rows)


def score_probabilities(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    top = np.argsort(-probabilities, kind="stable")[:max(1, int(TOP_SHARE * len(labels)))]
    varies = probabilities.std() > 0
    return {
        "rows": len(labels), "positive_pct": 100 * labels.mean(),
        "pr_auc": average_precision_score(labels, probabilities), "roc_auc": roc_auc_score(labels, probabilities) if varies else 0.5,
        "brier": brier_score_loss(labels, probabilities), "log_loss": log_loss(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities), "mean_predicted_pct": 100 * probabilities.mean(),
        "precision_top5_pct": 100 * labels[top].mean(), "recall_top5_pct": 100 * labels[top].sum() / labels.sum(),
    }


def bootstrap_difference(frame, model_a, model_b, seed, draws=500):
    """Compare two models by resampling series, and return mean differences with 95% intervals."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in frame.groupby("id")]
    labels, a, b = frame.label.to_numpy(), frame[model_a].to_numpy(), frame[model_b].to_numpy()
    positions = {index: i for i, index in enumerate(frame.index)}
    arrays = [np.array([positions[i] for i in group]) for group in groups]
    differences = {"pr_auc": [], "brier": []}
    for _ in range(draws):
        rows = np.concatenate([arrays[i] for i in rng.integers(0, len(arrays), len(arrays))])
        differences["pr_auc"].append(average_precision_score(labels[rows], a[rows]) - average_precision_score(labels[rows], b[rows]))
        differences["brier"].append(brier_score_loss(labels[rows], a[rows]) - brier_score_loss(labels[rows], b[rows]))
    return {key: {"point": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5))}
            for key, values in differences.items()}
