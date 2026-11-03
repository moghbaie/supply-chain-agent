"""Add UltraBERT to the Step 7 comparison, on the same test cases as Jev and the policy. Runs locally and calls no API."""
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
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from src.escalation.cases import CATEGORIES
from src.escalation.jev import CRITERIA
from src.escalation.policy import MODEL_DISAGREEMENT, apply_overrides
from src.escalation.ultrabert import FineTunedClassifier, zero_shot_probabilities

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 44)
SEED, ROUNDS = 42, 1000
REPORT = Path("reports/step_07")
OUT = Path("data/processed/escalation")
MODELS = OUT / "ultrabert"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
FINE, ZERO = "ultrabert_fine_tuned", "ultrabert_zero_shot"
test = pd.read_parquet(OUT / "classified_test.parquet")
ZERO_ORDER = np.array(list(CRITERIA), dtype=object)
ORDER = np.array(CATEGORIES, dtype=object)


def metrics(reference, choices):
    critical = reference == "human_review"
    return {"macro_f1": f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0),
            "accuracy": float((reference == choices).mean()),
            "critical_recall": float((choices[critical] == "human_review").mean()),
            "critical_acted_on": float((choices[critical] != "monitor").mean()),
            "any_stockout_acted_on": float((choices[reference != "monitor"] != "monitor").mean()),
            "false_alarm_share": float((choices[reference == "monitor"] != "monitor").mean()),
            **{f"recall_{c}": recall_score(reference == c, choices == c, zero_division=0) for c in CATEGORIES}}


def fast_scores(reference, choices):
    """Return macro-F1 and critical recall from category codes, quickly enough for a bootstrap."""
    matrix = np.bincount(4 * reference + choices, minlength=16).reshape(4, 4)
    hits, actual, predicted = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)
    f1 = np.divide(2 * hits, actual + predicted, out=np.zeros(4), where=actual + predicted > 0)
    critical = CATEGORIES.index("human_review")
    return f1.mean(), hits[critical] / actual[critical] if actual[critical] else np.nan


def interval(values):
    return [round(float(np.percentile(values, q)), 4) for q in (2.5, 50, 97.5)]


fine_probabilities = np.load(MODELS / "test_fine_tuned.npy")
zero_similarities = np.load(MODELS / "test_zero_shot.npy")
test[FINE] = ORDER[fine_probabilities.argmax(axis=1)]
test[ZERO] = ZERO_ORDER[zero_similarities.argmax(axis=1)]
test[f"{FINE}_override"], test[f"{FINE}_reason"] = apply_overrides(test, test[FINE])
test[f"{ZERO}_override"], test[f"{ZERO}_reason"] = apply_overrides(test, test[ZERO])
reference = test.reference.to_numpy(dtype=object)
print(f"test rows {len(test):,}, probabilities {fine_probabilities.shape}, similarities {zero_similarities.shape}")
print(f"fine-tuned probability sums {fine_probabilities.sum(axis=1).min():.4f} to {fine_probabilities.sum(axis=1).max():.4f}; similarity range {zero_similarities.min():.3f} to {zero_similarities.max():.3f}")

names = {"jev": "jev", "policy_baseline": "policy", FINE: FINE, f"{FINE}_with_overrides": f"{FINE}_override", ZERO: ZERO, f"{ZERO}_with_overrides": f"{ZERO}_override"}
table = pd.DataFrame([{"model": name, **metrics(reference, test[column].to_numpy(dtype=object))} for name, column in names.items()]).set_index("model")
print("\nTEST METRICS\n", table.round(4).T.to_string())
print("\nPREDICTED COUNTS\n", pd.DataFrame({name: test[column].value_counts().reindex(CATEGORIES).fillna(0).astype(int) for name, column in names.items()}).T.to_string())
print("reference counts", test.reference.value_counts().reindex(CATEGORIES).to_dict())
for name in [FINE, f"{FINE}_override", ZERO]:
    print(f"\nCONFUSION {name} (rows reference, columns predicted)\n", pd.DataFrame(confusion_matrix(reference, test[name].to_numpy(dtype=object), labels=CATEGORIES), index=CATEGORIES, columns=[c[:8] for c in CATEGORIES]).to_string())
print("\noverride reasons, fine-tuned:", test[f"{FINE}_reason"].value_counts().to_dict())
print("override reasons, zero-shot:", test[f"{ZERO}_reason"].value_counts().to_dict())

print("\nREVIEW WORKLOAD on test")
for name, column in names.items():
    acted = test[column] != "monitor"
    print(f"{name:38} action {int(acted.sum()):>6,} ({100 * acted.mean():5.2f}%) | human review {int((test[column] == 'human_review').sum()):>5,} | precision of any action {100 * (test.reference[acted] != 'monitor').mean():.2f}%")

conflict = (test.stockout_risk - test.stockout_risk_lightgbm).abs() >= MODEL_DISAGREEMENT
print(f"\nCONFLICTING SIGNALS: {int(conflict.sum())} cases, reference {test.reference[conflict].value_counts().to_dict()}")
print({m: test[m][conflict].value_counts().to_dict() for m in ["jev", "policy", FINE, ZERO]})

scenario_rows = []
for scenario, group in test.groupby("scenario", sort=False):
    truth = group.reference.to_numpy(dtype=object)
    for name in [FINE, ZERO]:
        choices = group[name].to_numpy(dtype=object)
        scored = metrics(truth, choices)
        scenario_rows.append({"scenario": scenario, "model": name, "macro_f1": scored["macro_f1"], "critical_recall": scored["critical_recall"], "human_review_cases": int((choices == "human_review").sum()), "true_critical": int((truth == "human_review").sum())})
print("\nBY SCENARIO\n", pd.DataFrame(scenario_rows).round(4).to_string(index=False))

code = {c: i for i, c in enumerate(CATEGORIES)}
encoded = {name: test[column].map(code).to_numpy() for name, column in names.items()}
truth_codes = test.reference.map(code).to_numpy()
for name in names:
    assert abs(fast_scores(truth_codes, encoded[name])[0] - table.loc[name, "macro_f1"]) < 1e-9
pairs = [(FINE, "policy_baseline"), (FINE, "jev"), (f"{FINE}_with_overrides", "policy_baseline"), (f"{FINE}_with_overrides", "jev"), (ZERO, "policy_baseline"), (ZERO, "jev")]
rng = np.random.default_rng(SEED)
groups = list(test.groupby("id").indices.values())
gaps = {pair: ([], []) for pair in pairs}
false_alarm_gap, caught_gap = [], []
for _ in range(ROUNDS):
    rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    scores = {name: fast_scores(truth_codes[rows], encoded[name][rows]) for name in names}
    for pair in pairs:
        gaps[pair][0].append(scores[pair[0]][0] - scores[pair[1]][0])
        gaps[pair][1].append(scores[pair[0]][1] - scores[pair[1]][1])
    quiet = truth_codes[rows] == code["monitor"]
    acted_fine, acted_policy = encoded[FINE][rows] != code["monitor"], encoded["policy_baseline"][rows] != code["monitor"]
    false_alarm_gap.append(acted_fine[quiet].mean() - acted_policy[quiet].mean())
    caught_gap.append(acted_fine[~quiet].mean() - acted_policy[~quiet].mean())
added = {f"false_alarm_share_{FINE}_minus_policy_baseline": interval(false_alarm_gap), f"any_stockout_acted_on_{FINE}_minus_policy_baseline": interval(caught_gap)}
print(f"\nfine-tuned minus policy: false alarm share {interval(false_alarm_gap)}, any stockout acted on {interval(caught_gap)}")
print("\nSERIES BOOTSTRAP [2.5%, median, 97.5%]")
for (left, right), (macro_gap, critical_gap) in gaps.items():
    added[f"macro_f1_{left}_minus_{right}"] = interval(macro_gap)
    added[f"critical_recall_{left}_minus_{right}"] = interval(critical_gap)
    print(f"{left} minus {right}: macro-F1 {interval(macro_gap)}, critical recall {interval(critical_gap)}")

edge = pd.read_csv(REPORT / "edge_cases.csv")
classifier = FineTunedClassifier(8, 1e-5, SEED)
classifier.model.load_state_dict(torch.load(MODELS / "best_model/weights.pt", map_location="cuda", weights_only=True))
edge_probabilities = classifier.predict(edge)
repeat = classifier.predict(test.iloc[:500])
print("\nDETERMINISM: rescored 500 test cases, same choice %:", 100 * float((ORDER[repeat.argmax(axis=1)] == test[FINE].to_numpy(dtype=object)[:500]).mean()), "| largest probability change", float(np.abs(repeat - fine_probabilities[:500]).max()))
del classifier
torch.cuda.empty_cache()
edge[FINE] = ORDER[edge_probabilities.argmax(axis=1)]
edge[f"{FINE}_top_probability"] = edge_probabilities.max(axis=1).round(3)
edge[ZERO] = ZERO_ORDER[zero_shot_probabilities(edge, list(CRITERIA.values())).argmax(axis=1)]
print("\nHAND-BUILT EDGE CASES (synthetic, Jev column is the saved answer)\n", edge[["case", "jev", "policy", FINE, f"{FINE}_top_probability", ZERO]].to_string(index=False))

existing = pd.read_csv(REPORT / "metrics_test.csv")
keep = existing[~existing.model.str.startswith("ultrabert")]
pd.concat([keep, table.loc[[n for n in names if n.startswith("ultrabert")]].reset_index()]).round(5).to_csv(REPORT / "metrics_test.csv", index=False)
existing = pd.read_csv(REPORT / "metrics_by_scenario.csv")
pd.concat([existing[~existing.model.str.startswith("ultrabert")], pd.DataFrame(scenario_rows).round(5)]).to_csv(REPORT / "metrics_by_scenario.csv", index=False)
saved = json.loads((REPORT / "comparison.json").read_text())
(REPORT / "comparison.json").write_text(json.dumps({**{k: v for k, v in saved.items() if "ultrabert" not in k}, **added}, indent=2))
edge.to_csv(REPORT / "edge_cases.csv", index=False)
test[["id", "scenario", "origin", "reference", FINE, f"{FINE}_override", ZERO, f"{ZERO}_override"]].to_parquet(MODELS / "test_choices.parquet", index=False)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), facecolor=SURFACE)
for ax, (model, title) in zip(axes, [("jev", "Jev"), ("policy", "Policy"), (FINE, "UltraBERT fine-tuned"), (ZERO, "UltraBERT zero-shot")]):
    matrix = confusion_matrix(reference, test[model].to_numpy(dtype=object), labels=CATEGORIES, normalize="true")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    labels = ["monitor", "reorder", "expedite", "human"]
    ax.set_xticks(range(4), labels, fontsize=8, color=MUTED)
    ax.set_yticks(range(4), labels, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("Reference", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100 * matrix[i, j]:.0f}%", ha="center", va="center", fontsize=9, color="#ffffff" if matrix[i, j] > 0.55 else INK)
    ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(REPORT / "confusion.png", dpi=130)
