"""Prepare what the agent's tools read: delay risk for every order, and the agent's development and test cases. Runs locally."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.ultrabert import FineTunedClassifier
from src.supplier.inbound import FEATURES, fit_delay_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 30)
SEED, DEV_PER_CATEGORY, TEST_PER_CATEGORY = 42, 10, 50
DEV_FIRST_ORIGIN = pd.Timestamp("2015-10-01")
OUT = Path("data/processed/agent")
OUT.mkdir(parents=True, exist_ok=True)
SCORES = [f"p_{c}" for c in CATEGORIES]


def stratified(frame, per_category):
    return pd.concat([group.sample(min(per_category, len(group)), random_state=SEED) for _, group in frame.groupby("reference", sort=False)]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


purchase = pd.read_parquet("data/processed/delay/inbound_features.parquet")
train_fit = purchase[purchase.split.eq("train") & purchase.known_before_validation]
test_fit = purchase[purchase.split.isin(["train", "validation"]) & purchase.known_before_test]
scores = []
for name, fit, target in [("dev", train_fit, ~purchase.known_before_validation), ("test", test_fit, ~purchase.known_before_test)]:
    rows = purchase[target]
    scores.append(pd.DataFrame({"po_id": rows.po_id, "period": name, "delay_risk": fit_delay_model(fit, SEED).predict(rows[FEATURES].to_numpy(dtype=float))}))
    print(f"{name}: fit {len(fit):,} orders, last receipt {fit.receipt_date.max().date()}; scored {len(rows):,} orders, first order {rows.order_date.min().date()}, mean risk {scores[-1].delay_risk.mean():.4f}, actual late {rows.delayed.mean():.4f}")
pd.concat(scores, ignore_index=True).to_parquet(OUT / "delay_scores.parquet", index=False)

test = pd.read_parquet("data/processed/escalation/classified_test.parquet")
test[SCORES] = np.load("data/processed/escalation/ultrabert/test_fine_tuned.npy")
choices = pd.read_parquet("data/processed/escalation/ultrabert/test_choices.parquet")
assert (choices.id.to_numpy() == test.id.to_numpy()).all()
test["ultrabert"] = choices.ultrabert_fine_tuned.to_numpy()
evaluation = stratified(test, TEST_PER_CATEGORY)
evaluation.to_parquet(OUT / "test_cases.parquet", index=False)

cases, _ = build_cases()
dev = stratified(cases[(cases.split == "validation") & (cases.origin >= DEV_FIRST_ORIGIN)], DEV_PER_CATEGORY)
classifier = FineTunedClassifier(8, 1e-5, SEED)
classifier.model.load_state_dict(torch.load("data/processed/escalation/ultrabert/best_model/weights.pt", map_location="cuda", weights_only=True))
dev[SCORES] = classifier.predict(dev)
dev["ultrabert"] = np.array(CATEGORIES, dtype=object)[dev[SCORES].to_numpy().argmax(axis=1)]
del classifier
torch.cuda.empty_cache()
dev.to_parquet(OUT / "dev_cases.parquet", index=False)

print("dev cases", len(dev), dev.reference.value_counts().to_dict(), "origins", dev.origin.min().date(), dev.origin.max().date())
print("test cases", len(evaluation), evaluation.reference.value_counts().to_dict(), "origins", evaluation.origin.min().date(), evaluation.origin.max().date(), "series", evaluation.id.nunique())
print("all test cases", test.reference.value_counts().to_dict())
