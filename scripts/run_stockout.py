"""Score stockout models on validation, refit on train plus validation, and score the test period once."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.stockout.baselines import Boosted, ConstantRate, CoverRule, SmallLogistic
from src.stockout.evaluation import bootstrap_difference, reliability_table, score_probabilities
from src.stockout.features import FEATURES
from src.stockout.tabpfn_model import TabPFNStockout

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 20, "display.max_colwidth", 30)
SEED, TRAIN_ROWS, MEMBERS, CHUNK = 42, 3000, 2, 5000
EXTRA_SEEDS = [1, 2]
REPORT = Path("reports/step_05")
INK, MUTED, SURFACE, GRID, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3", "#2a78d6"
data = pd.read_parquet("data/processed/stockout_features.parquet")
train = data[data.split == "train"]
validation = data[data.split == "validation"].copy()
test = data[data.split == "test"].copy()
print(f"rows: train {len(train):,}, validation {len(validation):,}, test {len(test):,}, purged {int((data.split == 'purged').sum()):,}", flush=True)


def build_models():
    return {"constant": ConstantRate(), "cover_rule": CoverRule(), "logistic": SmallLogistic(), "lightgbm": Boosted(SEED),
            "tabpfn": TabPFNStockout(TRAIN_ROWS, MEMBERS, CHUNK, SEED)}


def run(fit_frame, eval_frame, name):
    results, seconds = {}, {}
    for model_name, model in build_models().items():
        started = time.time()
        model.fit(fit_frame)
        eval_frame[model_name] = model.predict(eval_frame)
        seconds[model_name] = round(time.time() - started, 1)
        results[model_name] = score_probabilities(eval_frame.label, eval_frame[model_name])
        print(f"  {name} {model_name}: {results[model_name]['pr_auc']:.4f} PR-AUC, {seconds[model_name]}s", flush=True)
    table = pd.DataFrame(results).T.assign(seconds=pd.Series(seconds))
    print(f"\n{name.upper()} overall\n", table.round(4).to_string(), flush=True)
    return table


validation_table = run(train, validation, "validation")
stability = []
for seed in EXTRA_SEEDS:
    subset = validation.sample(5000, random_state=SEED)
    probabilities = TabPFNStockout(TRAIN_ROWS, MEMBERS, CHUNK, seed).fit(train).predict(subset)
    stability.append({"train_sample_seed": seed, **score_probabilities(subset.label, probabilities)})
stability.append({"train_sample_seed": SEED, **score_probabilities(subset.label, subset.tabpfn)})
stability = pd.DataFrame(stability)[["train_sample_seed", "pr_auc", "brier", "ece"]]
print("\nTabPFN stability across training samples (same 5,000 validation rows)\n", stability.round(4).to_string(index=False), flush=True)

history = pd.concat([train, data[data.split == "validation"]], ignore_index=True)
test_table = run(history, test, "test")
per_scenario = []
for scenario, group in test.groupby("scenario", sort=False):
    for model_name in ["cover_rule", "logistic", "lightgbm", "tabpfn"]:
        per_scenario.append({"scenario": scenario, "model": model_name, **score_probabilities(group.label, group[model_name])})
per_scenario = pd.DataFrame(per_scenario)
print("\nTEST by scenario\n", per_scenario[["scenario", "model", "positive_pct", "pr_auc", "roc_auc", "brier", "ece", "mean_predicted_pct", "recall_top5_pct"]].round(4).to_string(index=False))

comparison = {rival: bootstrap_difference(test, "tabpfn", rival, SEED) for rival in ["cover_rule", "logistic", "lightgbm"]}
print("\nTabPFN minus rival on test, series bootstrap (PR-AUC above 0 and Brier below 0 favor TabPFN)\n", json.dumps(comparison, indent=1))

print("\nRELIABILITY on test")
tables = {name: reliability_table(test.label.to_numpy(), test[name].to_numpy()) for name in ["cover_rule", "lightgbm", "tabpfn"]}
for name, table in tables.items():
    print(name, "\n", table.round(3).to_string(index=False))

print("\n20 PREDICTIONS (test, baseline, 10 highest and 10 spread across the rest)")
base = test[test.scenario == "baseline"].sort_values("tabpfn", ascending=False)
picks = pd.concat([base.head(10), base.iloc[np.linspace(10, len(base) - 1, 10).astype(int)]])
preview = picks[["origin", "id", "on_hand", "on_order", "fc_mean_14", "cover_days", "units_overdue", "tabpfn", "lightgbm", "cover_rule", "label"]].assign(id=lambda d: d.id.str.replace("_evaluation", ""))
print(preview.round(3).to_string(index=False))

booster = Boosted(SEED).fit(history)
importance = pd.Series(booster.model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print("\nLightGBM feature importance, top 12 (share of gain %)\n", importance.div(importance.sum()).mul(100).head(12).round(2).to_string())

REPORT.mkdir(parents=True, exist_ok=True)
pd.concat([validation_table.assign(period="validation"), test_table.assign(period="test")]).round(5).to_csv(REPORT / "metrics.csv")
per_scenario.round(5).to_csv(REPORT / "metrics_by_scenario.csv", index=False)
stability.round(5).to_csv(REPORT / "tabpfn_stability.csv", index=False)
preview.to_csv(REPORT / "predictions_preview.csv", index=False)
(REPORT / "comparison.json").write_text(json.dumps(comparison, indent=2))
(REPORT / "settings.json").write_text(json.dumps({"tabpfn_version": "v2", "train_rows": TRAIN_ROWS, "members": MEMBERS, "seed": SEED}, indent=2))
for name, table in tables.items():
    table.round(5).to_csv(REPORT / f"reliability_{name}.csv", index=False)
columns = ["scenario", "origin", "id", "split", "label", "constant", "cover_rule", "logistic", "lightgbm", "tabpfn"]
pd.concat([validation[columns], test[columns]], ignore_index=True).to_parquet("data/processed/stockout_predictions.parquet", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True, facecolor=SURFACE)
for ax, (name, table) in zip(axes, tables.items()):
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect")
    ax.plot(table.predicted, table.observed, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label="Observed")
    ax.set_title(f"{name} (ECE {test_table.loc[name, 'ece']:.4f})", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Predicted probability", fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Observed stockout rate", fontsize=9, color=MUTED)
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
fig.tight_layout()
fig.savefig(REPORT / "calibration.png", dpi=130)
print("peak VRAM MB:", round(torch.cuda.max_memory_allocated() / 1024**2))
