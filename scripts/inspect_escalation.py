"""Print escalation intervals, scenario results and conflicting-signal behavior, and send a few hand-built edge cases to Jev."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.metrics import confusion_matrix, f1_score

from src.escalation.cases import CATEGORIES
from src.escalation.jev import USD_PER_INPUT_TOKEN, classify
from src.escalation.policy import MODEL_DISAGREEMENT, PolicyThresholds, apply_overrides, policy_baseline

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 44)
SEED = 42
REPORT = Path("reports/step_07")
OUT = Path("data/processed/escalation")
INK, MUTED, SURFACE, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
test = pd.read_parquet(OUT / "classified_test.parquet")
thresholds = PolicyThresholds(**json.loads((OUT / "policy_thresholds.json").read_text()))


def macro(reference, choices):
    return f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0)


def critical(reference, choices):
    mask = reference == "human_review"
    return float((choices[mask] == "human_review").mean()) if mask.any() else float("nan")


rng = np.random.default_rng(SEED)
groups = list(test.groupby("id").indices.values())
reference, jev, policy = (test[c].to_numpy(dtype=object) for c in ["reference", "jev", "policy"])
macro_gap, critical_gap = [], []
for _ in range(1000):
    rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    macro_gap.append(macro(reference[rows], jev[rows]) - macro(reference[rows], policy[rows]))
    critical_gap.append(critical(reference[rows], jev[rows]) - critical(reference[rows], policy[rows]))
comparison = {"macro_f1_jev_minus_policy": [round(float(np.percentile(macro_gap, q)), 4) for q in (2.5, 50, 97.5)],
              "critical_recall_jev_minus_policy": [round(float(np.percentile(critical_gap, q)), 4) for q in (2.5, 50, 97.5)]}
print("Jev minus policy on test, series bootstrap [2.5%, median, 97.5%]:", json.dumps(comparison))

rows = []
for scenario, group in test.groupby("scenario", sort=False):
    truth = group.reference.to_numpy(dtype=object)
    for model in ["jev", "policy"]:
        rows.append({"scenario": scenario, "model": model, "macro_f1": macro(truth, group[model].to_numpy(dtype=object)), "critical_recall": critical(truth, group[model].to_numpy(dtype=object)),
                     "human_review_cases": int((group[model] == "human_review").sum()), "true_critical": int((truth == "human_review").sum())})
print("\nBY SCENARIO\n", pd.DataFrame(rows).round(4).to_string(index=False))

print("\nREVIEW WORKLOAD on test")
for model in ["jev", "policy"]:
    choice = test[model]
    acted = choice != "monitor"
    print(f"{model:7} sent to a person or recommendation {int(acted.sum()):,} ({100 * acted.mean():.2f}%) | human review {int((choice == 'human_review').sum()):,} | precision of any action {100 * (test.reference[acted] != 'monitor').mean():.2f}%")

conflict = (test.stockout_risk - test.stockout_risk_lightgbm).abs() >= MODEL_DISAGREEMENT
print(f"\nCONFLICTING SIGNALS: models differ by {MODEL_DISAGREEMENT}+ on {int(conflict.sum())} test cases, reference {test.reference[conflict].value_counts().to_dict()}")
print({m: test[m][conflict].value_counts().to_dict() for m in ["jev", "policy"]})
print("late-supplier and low-risk conflict, recent late rate 50%+ with stockout risk under 10%:", {m: test[m][(test.supplier_recent_late_rate >= 0.5) & (test.stockout_risk < 0.1)].value_counts().to_dict() for m in ["jev", "policy"]})

base = {"on_hand": 50, "on_order": 0, "fc_mean_14": 14.0, "fc_q90_sum_14": 25.0, "cover_days": 50.0, "days_to_next_due": np.nan, "units_due_14": 0.0, "units_overdue": 0.0, "days_overdue_max": 0.0, "open_pos": 0.0,
        "ordered_today": 0, "supplier_late_rate": 0.1, "supplier_recent_late_rate": 0.1, "stockout_risk": 0.02, "stockout_risk_lightgbm": 0.02, "value_band": "low"}
crafted = {
    "healthy_stock": {},
    "empty_no_order_high_risk": {"on_hand": 0, "cover_days": 0.0, "stockout_risk": 0.95, "stockout_risk_lightgbm": 0.95},
    "empty_order_late_high_risk": {"on_hand": 0, "cover_days": 0.0, "on_order": 40, "open_pos": 1.0, "days_to_next_due": 9.0, "units_due_14": 40.0, "stockout_risk": 0.95, "stockout_risk_lightgbm": 0.92},
    "high_value_high_risk": {"on_hand": 2, "cover_days": 2.0, "on_order": 40, "open_pos": 1.0, "days_to_next_due": 9.0, "units_due_14": 40.0, "stockout_risk": 0.9, "stockout_risk_lightgbm": 0.9, "value_band": "high"},
    "models_conflict": {"on_hand": 6, "cover_days": 6.0, "stockout_risk": 0.85, "stockout_risk_lightgbm": 0.10, "value_band": "medium"},
    "risk_low_but_cover_low": {"on_hand": 1, "cover_days": 1.0, "stockout_risk": 0.03, "stockout_risk_lightgbm": 0.03},
    "risk_high_but_cover_high": {"on_hand": 200, "cover_days": 200.0, "stockout_risk": 0.95, "stockout_risk_lightgbm": 0.95},
    "order_overdue": {"on_hand": 3, "cover_days": 3.0, "on_order": 30, "open_pos": 1.0, "days_to_next_due": np.nan, "units_overdue": 30.0, "days_overdue_max": 12.0, "stockout_risk": 0.7, "stockout_risk_lightgbm": 0.75},
    "disrupted_supplier": {"on_hand": 12, "cover_days": 12.0, "on_order": 20, "open_pos": 1.0, "days_to_next_due": 10.0, "units_due_14": 20.0, "supplier_recent_late_rate": 0.9, "supplier_late_rate": 0.15, "stockout_risk": 0.4, "stockout_risk_lightgbm": 0.45},
    "zero_demand_no_stock": {"on_hand": 0, "cover_days": 0.0, "fc_mean_14": 0.0, "fc_q90_sum_14": 0.0, "stockout_risk": 0.05, "stockout_risk_lightgbm": 0.05},
}
frame = pd.DataFrame([{**base, **changes} for changes in crafted.values()], index=list(crafted))
answers = classify(frame, dotenv_values(".env.secrets")["TYPESAFE_API_KEY"], workers=4)
tokens = sum(a[2] for a in answers)
usage = json.loads((OUT / "usage.json").read_text())
usage["input_tokens"] += tokens
usage["usd"] = usage["input_tokens"] * USD_PER_INPUT_TOKEN
(OUT / "usage.json").write_text(json.dumps(usage))
frame["jev"] = [a[0] for a in answers]
frame["jev_top_probability"] = [round(max(a[1].values()), 3) for a in answers]
frame["jev_override"], frame["override_reason"] = apply_overrides(frame, frame.jev)
frame["policy"] = policy_baseline(frame, thresholds)
print("\nHAND-BUILT EDGE CASES (synthetic)\n", frame[["jev", "jev_top_probability", "jev_override", "override_reason", "policy"]].to_string())
print(f"total Jev spend about ${usage['usd']:.4f} of $5.00, {usage['input_tokens']:,} tokens")

pd.DataFrame(rows).round(5).to_csv(REPORT / "metrics_by_scenario.csv", index=False)
frame.reset_index(names="case").to_csv(REPORT / "edge_cases.csv", index=False)
(REPORT / "comparison.json").write_text(json.dumps(comparison, indent=2))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
for ax, model in zip(axes, ["jev", "policy"]):
    matrix = confusion_matrix(reference, test[model].to_numpy(dtype=object), labels=CATEGORIES, normalize="true")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    labels = ["monitor", "reorder", "expedite", "human"]
    ax.set_xticks(range(4), labels, fontsize=8, color=MUTED)
    ax.set_yticks(range(4), labels, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("Reference", fontsize=9, color=MUTED)
    ax.set_title(model, fontsize=10, color=INK, loc="left")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100 * matrix[i, j]:.0f}%", ha="center", va="center", fontsize=9, color="#ffffff" if matrix[i, j] > 0.55 else INK)
    ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(REPORT / "confusion.png", dpi=130)
