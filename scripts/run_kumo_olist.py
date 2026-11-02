"""Score Olist late deliveries with Kumo Relational and compare local models on the same rows. Each batch is saved."""
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
from src.supplier import olist
from src.supplier.kumo_client import positive_probability, send
from src.supplier.kumo_olist import OlistRequestBuilder, assert_allowed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30)
SEED, CONTEXT_ROWS, BATCH_ROWS, SCORE_ROWS = 42, 1000, 500, 5000
REPORT = Path("reports/step_06")
CHECKPOINT = Path("data/processed/delay/kumo_olist_batches")
CHECKPOINT.mkdir(parents=True, exist_ok=True)
INK, MUTED, SURFACE, GRID, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3", "#2a78d6"
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "verbosity": -1, "deterministic": True, "force_row_wise": True, "seed": SEED}
key = dotenv_values(".env.secrets")["KUMO_API_KEY"]
tables = olist.load_tables()
orders = olist.split_orders(olist.build_order_features(tables))
builder = OlistRequestBuilder(tables)
fit_frames = {"validation": orders[orders.split.eq("train") & orders.known_before_validation],
              "test": orders[orders.split.isin(["train", "validation"]) & orders.known_before_test]}


def kumo_scores(split, rows, context):
    scores, usage = [], {"calls": 0, "bytes": 0, "seconds": 0.0}
    for number, first in enumerate(range(0, len(rows), BATCH_ROWS)):
        target = CHECKPOINT / f"{split}_{number:03d}.json"
        if target.exists():
            scores.extend(json.loads(target.read_text()))
            continue
        batch = rows.iloc[first:first + BATCH_ROWS]
        body = builder.request(context, batch)
        payload = json.dumps(body)
        assert_allowed(payload, list(batch.order_id) + list(batch.customer_id) + list(batch.seller_id) + list(context.order_id))
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
    dataset = lgb.Dataset(fit[olist.FEATURES].to_numpy(dtype=float), fit.late.to_numpy(), feature_name=olist.FEATURES, free_raw_data=False)
    boosted = lgb.train(PARAMS, dataset, 300).predict(rows[olist.FEATURES].to_numpy(dtype=float))
    known = fit[fit.seller_late_rate.notna()]
    rule = IsotonicRegression(out_of_bounds="clip").fit(known.seller_late_rate, known.late)
    fallback = np.where(rows.seller_late_rate.notna(), rule.predict(rows.seller_late_rate.fillna(0)), known.late.mean())
    return {"constant": np.full(len(rows), fit.late.mean()), "seller_rule": fallback, "lightgbm": boosted}


def bootstrap(frame, first, second, draws=1000):
    rng = np.random.default_rng(SEED)
    groups = list(frame.groupby("seller_id").indices.values())
    labels, a, b = frame.late.to_numpy(), frame[first].to_numpy(), frame[second].to_numpy()
    pr, brier = [], []
    for _ in range(draws):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        pr.append(average_precision_score(labels[rows], a[rows]) - average_precision_score(labels[rows], b[rows]))
        brier.append(brier_score_loss(labels[rows], a[rows]) - brier_score_loss(labels[rows], b[rows]))
    return {"pr_auc": [round(float(np.percentile(pr, q)), 4) for q in (2.5, 50, 97.5)], "brier": [round(float(np.percentile(brier, q)), 5) for q in (2.5, 50, 97.5)]}


results, saved, usage_total = [], {}, {}
for split in ["validation", "test"]:
    rows = orders[orders.split == split].sample(SCORE_ROWS, random_state=SEED).sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    context = fit_frames[split].sample(CONTEXT_ROWS, random_state=SEED)
    print(f"\n{split}: context {len(context):,} (late {100 * context.late.mean():.2f}%, last delivered {context.order_delivered_customer_date.max().date()}), scoring {len(rows):,}, late {100 * rows.late.mean():.2f}%", flush=True)
    values, usage = kumo_scores(split, rows, context)
    usage_total[split] = usage
    rows["kumo"] = values
    for name, predicted in local_scores(fit_frames[split], rows).items():
        rows[name] = predicted
    table = pd.DataFrame({m: score_probabilities(rows.late, rows[m]) for m in ["constant", "seller_rule", "lightgbm", "kumo"]}).T.assign(split=split)
    print(table[["rows", "positive_pct", "pr_auc", "roc_auc", "brier", "ece", "mean_predicted_pct", "recall_top5_pct"]].round(4).to_string(), flush=True)
    results.append(table)
    saved[split] = rows

test = saved["test"]
comparison = {rival: bootstrap(test, "kumo", rival) for rival in ["seller_rule", "lightgbm"]}
print("\nKumo minus rival on test, seller bootstrap [2.5%, median, 97.5%]; PR-AUC above 0 and Brier below 0 favor Kumo:\n", json.dumps(comparison))
context_sellers = set(fit_frames["test"].sample(CONTEXT_ROWS, random_state=SEED).seller_id)
unseen = ~test.seller_id.isin(context_sellers).to_numpy()
no_history = (test.seller_orders_seen == 0).to_numpy()
print(f"\nUNSEEN ENTITIES on test: sellers absent from Kumo's examples {int(unseen.sum()):,} orders; sellers with no delivered history {int(no_history.sum()):,} orders")
for label, mask in (("absent from examples", unseen), ("no delivered history", no_history)):
    print(label, {m: {k: round(float(v), 4) for k, v in score_probabilities(test.late[mask], test[m][mask]).items() if k in ("rows", "positive_pct", "pr_auc", "brier")} for m in ["lightgbm", "kumo"]})
print("\nmonthly test results (Kumo):")
monthly = test.groupby(test.order_purchase_timestamp.dt.to_period("M")).agg(orders=("late", "size"), late_pct=("late", lambda s: 100 * s.mean()), kumo_pred_pct=("kumo", lambda s: 100 * s.mean()), lightgbm_pred_pct=("lightgbm", lambda s: 100 * s.mean())).round(2)
print(monthly.to_string())
print("\n20 PREDICTIONS (test, IDs shortened)")
preview = test.sort_values("kumo", ascending=False).iloc[np.linspace(0, len(test) - 1, 20).astype(int)]
print(preview[["order_purchase_timestamp", "seller_state", "customer_state", "item_count", "total_price", "estimated_days", "seller_late_rate", "kumo", "lightgbm", "seller_rule", "late"]].round(3).to_string(index=False))
tables_rel = {m: reliability_table(test.late.to_numpy(), test[m].to_numpy()) for m in ["seller_rule", "lightgbm", "kumo"]}
print("\nRELIABILITY, Kumo, test\n", tables_rel["kumo"].round(3).to_string(index=False))

pd.concat(results).round(5).to_csv(REPORT / "kumo_olist_metrics.csv")
preview.drop(columns=[c for c in preview.columns if c.endswith("_id")]).to_csv(REPORT / "kumo_olist_predictions_preview.csv", index=False)
(REPORT / "kumo_olist_comparison.json").write_text(json.dumps(comparison, indent=2))
(REPORT / "kumo_olist_usage.json").write_text(json.dumps(usage_total, indent=2))
pd.concat([saved["validation"], test], ignore_index=True)[["order_id", "seller_id", "split", "order_purchase_timestamp", "late", "seller_orders_seen", "constant", "seller_rule", "lightgbm", "kumo"]].to_parquet("data/processed/delay/olist_predictions.parquet", index=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True, facecolor=SURFACE)
for ax, (name, table) in zip(axes, tables_rel.items()):
    upper = max(table.predicted.max(), table.observed.max()) * 1.1
    ax.plot([0, upper], [0, upper], color=MUTED, linewidth=1, linestyle="--", label="Perfect")
    ax.plot(table.predicted, table.observed, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label="Observed")
    ax.set_title(f"{name} (ECE {results[-1].loc[name, 'ece']:.4f})", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Predicted probability", fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.8)
axes[0].set_ylabel("Observed late rate", fontsize=9, color=MUTED)
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
fig.tight_layout()
fig.savefig(REPORT / "kumo_olist_calibration.png", dpi=130)
print("usage:", usage_total)
