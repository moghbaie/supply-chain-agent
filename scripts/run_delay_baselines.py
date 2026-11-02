"""Build the delay-risk tables and score local baselines. Nothing here leaves this machine."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

from src.stockout.evaluation import reliability_table, score_probabilities
from src.supplier import inbound, olist

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 24)
SEED = 42
REPORT = Path("reports/step_06")
OUT = Path("data/processed/delay")
REPORT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "verbosity": -1, "deterministic": True, "force_row_wise": True, "seed": SEED}


def boosted(fit, rows, features, target):
    dataset = lgb.Dataset(fit[features].to_numpy(dtype=float), fit[target].to_numpy(), feature_name=features, free_raw_data=False)
    return lgb.train(PARAMS, dataset, 300).predict(rows[features].to_numpy(dtype=float))


def single_feature(fit, rows, column, target, increasing=True):
    known = fit[fit[column].notna()]
    fitted = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(known[column], known[target])
    return np.where(rows[column].notna(), fitted.predict(rows[column].fillna(0)), float(known[target].mean()))


def score_all(name, fit, rows, features, target, rule_column, increasing=True):
    predictions = {"constant": np.full(len(rows), fit[target].mean()), "single_rule": single_feature(fit, rows, rule_column, target, increasing),
                   "lightgbm": boosted(fit, rows, features, target)}
    table = pd.DataFrame({model: score_probabilities(rows[target], values) for model, values in predictions.items()}).T
    print(f"\n{name}: fit {len(fit):,}, score {len(rows):,}, rule on {rule_column}\n",
          table[["rows", "positive_pct", "pr_auc", "roc_auc", "brier", "ece", "mean_predicted_pct", "recall_top5_pct"]].round(4).to_string())
    return table.assign(task=name), predictions


print("===== OLIST: real customer deliveries, predicted at purchase time")
tables = olist.load_tables()
orders = olist.split_orders(olist.build_order_features(tables))
print("relationship schema: orders.customer_id -> customers | items.order_id -> orders | items.seller_id -> sellers | items.product_id -> products")
print("rows:", {k: len(v) for k, v in tables.items()}, "| delivered orders used:", len(orders))
print("split sizes and late rates:\n", orders.groupby("split").agg(orders=("late", "size"), late_pct=("late", lambda s: round(100 * s.mean(), 2)),
                                                                  first=("order_purchase_timestamp", "min"), last=("order_purchase_timestamp", "max")).to_string())
print("fit rows known before validation:", int((orders.split.eq("train") & orders.known_before_validation).sum()), "of", int(orders.split.eq("train").sum()))
print("fit rows known before test:", int(orders.known_before_test.sum()))
print("\nexample join, 5 orders:")
print(orders[["order_id", "order_purchase_timestamp", "seller_state", "customer_state", "item_count", "total_price", "estimated_days", "seller_orders_seen", "seller_late_rate", "route_late_rate", "late"]]
      .assign(order_id=lambda d: d.order_id.str[:8]).iloc[20000:20005].round(2).to_string(index=False))
print("\nmissing feature share %:", orders[olist.FEATURES].isna().mean().mul(100).round(2).loc[lambda s: s > 0].to_dict())
fit = orders[orders.split.eq("train") & orders.known_before_validation]
validation = orders[orders.split == "validation"]
history = orders[orders.known_before_test & orders.split.isin(["train", "validation"])]
test = orders[orders.split == "test"]
results = []
table, _ = score_all("olist_validation", fit, validation, olist.FEATURES, "late", "seller_late_rate")
results.append(table)
print("\nsellers not seen in fit data at test time: scored separately below")
table, predicted = score_all("olist_test", history, test, olist.FEATURES, "late", "seller_late_rate")
results.append(table)
new_seller = test.seller_orders_seen.to_numpy() == 0
print(f"test orders from sellers with no delivered history: {int(new_seller.sum()):,} ({100 * new_seller.mean():.2f}%)")
print("LightGBM on those orders:", {k: round(v, 4) for k, v in score_probabilities(test.late[new_seller], predicted["lightgbm"][new_seller]).items() if k in ("rows", "positive_pct", "pr_auc", "brier")})
print("monthly late rate on test:", test.groupby(test.order_purchase_timestamp.dt.to_period("M")).late.mean().mul(100).round(2).to_dict())
print("reliability, LightGBM, test:\n", reliability_table(test.late.to_numpy(), predicted["lightgbm"]).round(3).to_string(index=False))
orders[["order_id", "split", "late", "order_purchase_timestamp"]].to_parquet(OUT / "olist_labels.parquet", index=False)

print("\n===== SYNTHETIC: inbound supplier delays, predicted at order time")
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
dates = {k: pd.Timestamp(config["dates"][k]) for k in ["train_end", "validation_end", "burn_in_end"]}
all_orders = pd.read_parquet("data/simulated/purchase_orders.parquet")
series = pd.read_parquet("data/simulated/series.parquet")
frames = [inbound.build_po_features(all_orders[all_orders.scenario == s], series, dates["train_end"], dates["validation_end"], dates["burn_in_end"]).assign(scenario=s) for s in config["scenarios"]]
purchase = pd.concat(frames, ignore_index=True)
print("rows:", len(purchase), "| delayed % by scenario and split:\n", purchase.pivot_table(index="scenario", columns="split", values="delayed", aggfunc="mean", observed=True).mul(100).round(2).to_string())
fit = purchase[purchase.split.eq("train") & purchase.known_before_validation]
validation = purchase[purchase.split == "validation"]
history = purchase[purchase.split.isin(["train", "validation"]) & purchase.known_before_test]
test = purchase[purchase.split == "test"]
table, _ = score_all("inbound_validation", fit, validation, inbound.FEATURES, "delayed", "supplier_late_rate")
results.append(table)
table, predicted = score_all("inbound_test", history, test, inbound.FEATURES, "delayed", "supplier_late_rate")
results.append(table)
for scenario, group in test.groupby("scenario", sort=False):
    position = test.index.get_indexer(group.index)
    print(f"  {scenario} LightGBM:", {k: round(v, 4) for k, v in score_probabilities(group.delayed, predicted["lightgbm"][position]).items() if k in ("positive_pct", "pr_auc", "brier")})
truth = pd.read_parquet("data/simulated/suppliers.parquet").set_index("supplier_id").delay_prob
oracle = test.supplier_id.map(truth).to_numpy()
print("best possible baseline-scenario check, true supplier delay probability (hidden from models):", {k: round(v, 4) for k, v in score_probabilities(test.delayed[test.scenario == "baseline"], oracle[(test.scenario == "baseline").to_numpy()]).items() if k in ("pr_auc", "brier")})
pd.concat(results).round(5).to_csv(REPORT / "baseline_metrics.csv")
purchase.to_parquet(OUT / "inbound_features.parquet", index=False)

print("\n===== WHAT WOULD BE SENT TO NVIDIA FOR OLIST (preview only, nothing was sent)")
columns = {"instance table (order)": ["order_id", "customer_id", "order_purchase_timestamp", "order_estimated_delivery_date", "label for context rows only"],
           "customers": ["customer_id", "customer_state"], "items": ["order_id", "order_item_id", "seller_id", "product_id", "price", "freight_value"],
           "sellers": ["seller_id", "seller_state"], "products": ["product_id", "category", "weight", "dimensions", "photos"]}
print(json.dumps(columns, indent=1))
print("excluded: customer_unique_id, cities, zip codes, geolocation, reviews, payments, delivery dates, order status")
(REPORT / "planned_upload_columns.json").write_text(json.dumps(columns, indent=2))
