"""Score synthetic purchase orders with Kumo Relational and compare local models on the same rows.

Only synthetic purchase orders are sent. Olist data is never read here. Each batch is saved, so a rerun resumes.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.stockout.evaluation import reliability_table, score_probabilities
from src.supplier import inbound
from src.supplier.kumo_client import HistoryIndex, build_request, positive_probability, send

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30)
SEED, CONTEXT_ROWS, BATCH_ROWS = 42, 1000, 500  # the hosted API rejects more than 1,000 context rows
SCORE_ROWS = {"validation": 2000, "test": 3000}
REPORT = Path("reports/step_06")
CHECKPOINT = Path("data/processed/delay/kumo_batches")
CHECKPOINT.mkdir(parents=True, exist_ok=True)
INK, MUTED, SURFACE, GRID, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3", "#2a78d6"
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "verbosity": -1, "deterministic": True, "force_row_wise": True, "seed": SEED}
key = dotenv_values(".env.secrets")["KUMO_API_KEY"]
data = pd.read_parquet("data/processed/delay/inbound_features.parquet")
all_orders = pd.read_parquet("data/simulated/purchase_orders.parquet")
supplier_numbers = {s: i for i, s in enumerate(sorted(all_orders.supplier_id.unique()))}
history = HistoryIndex(all_orders)
fit_frames = {"validation": data[data.split.eq("train") & data.known_before_validation],
              "test": data[data.split.isin(["train", "validation"]) & data.known_before_test]}


def kumo_scores(split, rows, context):
    scores, usage = [], {"calls": 0, "bytes": 0, "seconds": 0.0}
    for number, first in enumerate(range(0, len(rows), BATCH_ROWS)):
        target = CHECKPOINT / f"{split}_{number:03d}.json"
        if target.exists():
            scores.extend(json.loads(target.read_text()))
            print(f"  {split} batch {number}: already saved", flush=True)
            continue
        batch = rows.iloc[first:first + BATCH_ROWS]
        body = build_request(context, batch, history, supplier_numbers)
        payload = json.dumps(body)
        for term in ["FOODS", "HOBBIES", "HOUSEHOLD", "SUP_", "baseline", "supplier_disruption", "lean_policy", "olist", "tier"]:
            if term in payload:
                raise ValueError(f"forbidden term in payload: {term}")
        started = time.time()
        status, response = send(body, key)
        elapsed = time.time() - started
        if status != 200:
            raise SystemExit(f"stopping at {split} batch {number}: HTTP {status} {str(response)[:300]}")
        items = sorted(response["predictions"], key=lambda item: item["row_index"])
        if len(items) != len(batch):
            raise SystemExit(f"stopping: expected {len(batch)} predictions, got {len(items)}")
        values = [positive_probability(item) for item in items]
        target.write_text(json.dumps(values))
        scores.extend(values)
        usage["calls"] += 1
        usage["bytes"] += len(payload)
        usage["seconds"] += elapsed
        print(f"  {split} batch {number}: {len(payload) / 1e6:.2f} MB, {elapsed:.1f}s, mean {np.mean(values):.3f}", flush=True)
    return np.array(scores), usage


def local_scores(fit, rows):
    dataset = lgb.Dataset(fit[inbound.FEATURES].to_numpy(dtype=float), fit.delayed.to_numpy(), feature_name=inbound.FEATURES, free_raw_data=False)
    boosted = lgb.train(PARAMS, dataset, 300).predict(rows[inbound.FEATURES].to_numpy(dtype=float))
    known = fit[fit.supplier_late_rate.notna()]
    rule = IsotonicRegression(out_of_bounds="clip").fit(known.supplier_late_rate, known.delayed)
    fallback = np.where(rows.supplier_late_rate.notna(), rule.predict(rows.supplier_late_rate.fillna(0)), known.delayed.mean())
    return {"constant": np.full(len(rows), fit.delayed.mean()), "supplier_rule": fallback, "lightgbm": boosted}


def bootstrap(frame, first, second, draws=1000):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(frame.id.to_numpy() == series) for series in frame.id.unique()]
    labels, a, b = frame.delayed.to_numpy(), frame[first].to_numpy(), frame[second].to_numpy()
    pr, brier = [], []
    for _ in range(draws):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        pr.append(average_precision_score(labels[rows], a[rows]) - average_precision_score(labels[rows], b[rows]))
        brier.append(brier_score_loss(labels[rows], a[rows]) - brier_score_loss(labels[rows], b[rows]))
    return {"pr_auc": [round(float(np.percentile(pr, q)), 4) for q in (2.5, 50, 97.5)], "brier": [round(float(np.percentile(brier, q)), 5) for q in (2.5, 50, 97.5)]}


results, saved, usage_total = [], {}, {}
for split, count in SCORE_ROWS.items():
    rows = data[data.split == split].sample(count, random_state=SEED).sort_values(["order_date", "po_id"]).reset_index(drop=True)
    context = fit_frames[split].sample(CONTEXT_ROWS, random_state=SEED)
    print(f"\n{split}: context {len(context):,} (latest order {context.order_date.max().date()}, latest receipt {context.receipt_date.max().date()}), scoring {len(rows):,}, positive {100 * rows.delayed.mean():.2f}%", flush=True)
    values, usage = kumo_scores(split, rows, context)
    usage_total[split] = usage
    rows["kumo"] = values
    for name, predicted in local_scores(fit_frames[split], rows).items():
        rows[name] = predicted
    table = pd.DataFrame({m: score_probabilities(rows.delayed, rows[m]) for m in ["constant", "supplier_rule", "lightgbm", "kumo"]}).T.assign(split=split)
    print(table[["rows", "positive_pct", "pr_auc", "roc_auc", "brier", "ece", "mean_predicted_pct", "recall_top5_pct"]].round(4).to_string(), flush=True)
    results.append(table)
    saved[split] = rows

test = saved["test"]
print("\nTEST by scenario (scenario used for scoring only, never sent)")
scenario_rows = []
for scenario, group in test.groupby("scenario", sort=False):
    for model in ["supplier_rule", "lightgbm", "kumo"]:
        scenario_rows.append({"scenario": scenario, "model": model, **{k: v for k, v in score_probabilities(group.delayed, group[model]).items() if k in ("rows", "positive_pct", "pr_auc", "roc_auc", "brier", "ece")}})
scenario_table = pd.DataFrame(scenario_rows)
print(scenario_table.round(4).to_string(index=False))
comparison = {rival: bootstrap(test, "kumo", rival) for rival in ["supplier_rule", "lightgbm"]}
print("\nKumo minus rival on test, series bootstrap [2.5%, median, 97.5%]; PR-AUC above 0 and Brier below 0 favor Kumo:\n", json.dumps(comparison))

new_items = test[test.item_pos_seen == 0]
print(f"\nUNSEEN ENTITIES: test orders for items with no received order history: {len(new_items)}")
print("suppliers in test not in context:", sorted(set(test.supplier_id) - set(fit_frames['test'].supplier_id)))

print("\n20 PREDICTIONS (test)")
preview = test.sort_values("kumo", ascending=False).iloc[np.linspace(0, len(test) - 1, 20).astype(int)]
print(preview[["order_date", "supplier_id", "quantity", "promised_lead_days", "supplier_late_rate", "supplier_recent_late_rate", "kumo", "lightgbm", "supplier_rule", "delayed", "delay_days"]].round(3).to_string(index=False))
tables = {m: reliability_table(test.delayed.to_numpy(), test[m].to_numpy()) for m in ["supplier_rule", "lightgbm", "kumo"]}
print("\nRELIABILITY, Kumo, test\n", tables["kumo"].round(3).to_string(index=False))

pd.concat(results).round(5).to_csv(REPORT / "kumo_metrics.csv")
scenario_table.round(5).to_csv(REPORT / "kumo_metrics_by_scenario.csv", index=False)
preview.to_csv(REPORT / "kumo_predictions_preview.csv", index=False)
(REPORT / "kumo_comparison.json").write_text(json.dumps(comparison, indent=2))
(REPORT / "kumo_usage.json").write_text(json.dumps(usage_total, indent=2))
pd.concat([saved["validation"], test], ignore_index=True)[["scenario", "po_id", "split", "order_date", "supplier_id", "delayed", "constant", "supplier_rule", "lightgbm", "kumo"]].to_parquet("data/processed/delay/inbound_predictions.parquet", index=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True, facecolor=SURFACE)
for ax, (name, table) in zip(axes, tables.items()):
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect")
    ax.plot(table.predicted, table.observed, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label="Observed")
    ax.set_title(f"{name} (ECE {results[-1].loc[name, 'ece']:.4f})", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Predicted probability", fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.8)
axes[0].set_ylabel("Observed delay rate", fontsize=9, color=MUTED)
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
fig.tight_layout()
fig.savefig(REPORT / "kumo_calibration.png", dpi=130)
print("usage:", usage_total)
