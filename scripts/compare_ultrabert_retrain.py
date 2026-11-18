"""Compare retrained and original UltraBERT with the policy. Decision rules are fitted on the tune half only, then applied to check and test."""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.policy import OVERRIDE_RISK, PolicyThresholds, policy_baseline

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200, "display.max_columns", 30)
SEED, ROUNDS, LATE_FIRST = 42, 1000, pd.Timestamp("2015-10-01")
GRID = np.arange(-6.0, 1.01, 0.5)
MONITOR, REORDER, EXPEDITE, HUMAN = range(4)
CODE = {c: i for i, c in enumerate(CATEGORIES)}
OLD_LATE = Path(sys.argv[1])
NEW = ROOT / "data/processed/escalation/ultrabert_retrain"


def prepare(frame):
    return {"truth": frame.reference.astype(str).map(CODE).to_numpy(), "risk": frame.stockout_risk.to_numpy(), "open_pos": frame.open_pos.to_numpy(),
            "id": frame.id.astype(str).to_numpy(), "scenario": frame.scenario.astype(str).to_numpy()}


def decide(log_p, data, offsets):
    choice = (log_p + np.array([0.0, *offsets])).argmax(axis=1)
    high = (choice == MONITOR) & (data["risk"] >= OVERRIDE_RISK)
    reorder = (choice == REORDER) & (data["open_pos"] > 0)
    expedite = (choice == EXPEDITE) & (data["open_pos"] == 0)
    choice[high], choice[reorder], choice[expedite] = HUMAN, EXPEDITE, REORDER
    return choice


def scores(t, p):
    m = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    hits, actual, pred = np.diag(m), m.sum(1), m.sum(0)
    f1 = np.divide(2 * hits, actual + pred, out=np.zeros(4), where=actual + pred > 0)
    acted, stock = p != MONITOR, t != MONITOR
    return {"macro_f1": f1.mean(), "accuracy": hits.sum() / m.sum(), "critical_recall": hits[HUMAN] / max(actual[HUMAN], 1), "critical_not_monitor": acted[t == HUMAN].mean(),
            "stockouts_acted": acted[stock].mean(), "false_alarms": acted[~stock].mean(), "action_precision": stock[acted].mean(), "actions_per_1000": 1000 * acted.mean(),
            "reorder_precision": hits[REORDER] / max(pred[REORDER], 1), "reorder_recall": hits[REORDER] / max(actual[REORDER], 1),
            "expedite_precision": hits[EXPEDITE] / max(pred[EXPEDITE], 1), "expedite_recall": hits[EXPEDITE] / max(actual[EXPEDITE], 1),
            "reorder_predicted": pred[REORDER], "expedite_predicted": pred[EXPEDITE], "human_predicted": pred[HUMAN]}


def match_policy(log_p, data, policy):
    """Catch the most stockouts without exceeding the policy's false alarms or human review count on the given rows."""
    limit = scores(data["truth"], policy)
    rows = [{"offsets": o, **scores(data["truth"], decide(log_p, data, o))} for o in itertools.product(GRID, repeat=3)]
    table = pd.DataFrame(rows)
    allowed = table[(table.false_alarms <= limit["false_alarms"]) & (table.human_predicted <= limit["human_predicted"])]
    return allowed.sort_values(["stockouts_acted", "macro_f1"], ascending=False).iloc[0].offsets


def report(name, data, decisions, action_scores):
    truth = data["truth"]
    print(f"\n===== {name}: {len(truth):,} cases, true counts {np.bincount(truth, minlength=4).tolist()} {CATEGORIES}")
    print(pd.DataFrame({n: scores(truth, d) for n, d in decisions.items()}).round(4).to_string())
    print("\nVariance across series (error rate, points squared)")
    print(pd.Series({n: pd.Series(100 * (d != truth)).groupby(data["id"]).mean().var(ddof=0) for n, d in decisions.items()}).round(1).to_string())
    print("\nBy scenario")
    rows = []
    for scenario in np.unique(data["scenario"]):
        part = data["scenario"] == scenario
        for n, d in decisions.items():
            s = scores(truth[part], d[part])
            rows.append({"scenario": scenario, "rule": n, **{k: s[k] for k in ["macro_f1", "critical_recall", "stockouts_acted", "false_alarms"]}})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    stock = (truth != MONITOR).astype(int)
    print("\nRanking stockouts")
    for n, value in action_scores.items():
        print(f"  {n}: AUROC {roc_auc_score(stock, value):.4f}, AP {average_precision_score(stock, value):.4f}")
    groups = [np.flatnonzero(data["id"] == s) for s in np.unique(data["id"])]
    draw = np.random.default_rng(SEED)
    measures = ["macro_f1", "critical_recall", "critical_not_monitor", "stockouts_acted", "false_alarms", "action_precision"]
    samples = {n: {m: [] for m in measures} for n in decisions}
    ranks = {n: [] for n in action_scores}
    for _ in range(ROUNDS):
        pick = np.concatenate([groups[i] for i in draw.integers(0, len(groups), len(groups))])
        for n, d in decisions.items():
            s = scores(truth[pick], d[pick])
            for m in measures:
                samples[n][m].append(s[m])
        for n, value in action_scores.items():
            ranks[n].append(roc_auc_score(stock[pick], value[pick]))
    print("\nBootstrap SD x100")
    print(pd.DataFrame({n: {m: 100 * float(np.std(v[m], ddof=1)) for m in measures} for n, v in samples.items()}).round(3).to_string())
    print("\nBootstrap gaps [2.5, 50, 97.5]")
    for left, right in [("Retrained", "Original"), ("Retrained matched", "Original matched"), ("Retrained", "Policy"), ("Retrained matched", "Policy")]:
        print(f"{left} minus {right}:", {m: np.round(np.percentile(np.array(samples[left][m]) - np.array(samples[right][m]), [2.5, 50, 97.5]), 4).tolist() for m in measures})
    print("AUROC gap retrained minus original", np.round(np.percentile(np.array(ranks["Retrained"]) - np.array(ranks["Original"]), [2.5, 50, 97.5]), 4).tolist())
    print("AUROC gap retrained minus TabPFN risk", np.round(np.percentile(np.array(ranks["Retrained"]) - np.array(ranks["TabPFN risk"]), [2.5, 50, 97.5]), 4).tolist())


cases, _ = build_cases()
year = cases[cases.split == "validation"]
late = year[year.origin >= LATE_FIRST].copy()
in_dev = late.index.isin(late.sample(3000, random_state=SEED).index)
late = late.reset_index(drop=True)
series = np.sort(late.id.astype(str).unique())
tune_series = set(np.random.default_rng(SEED).choice(series, len(series) // 2, replace=False))
is_tune = late.id.astype(str).isin(tune_series).to_numpy()
check_mask = ~is_tune & ~in_dev
old_late, new_late = np.load(OLD_LATE), np.load(NEW / "late_2015_probabilities.npy")
assert old_late.shape == new_late.shape == (len(late), 4)
log_old_late, log_new_late = np.log(np.clip(old_late, 1e-9, 1)), np.log(np.clip(new_late, 1e-9, 1))
late_data = prepare(late)
policy_late = pd.Series(policy_baseline(late, PolicyThresholds(0.1, 0.9))).map(CODE).to_numpy()
take = lambda data, mask: {k: v[mask] for k, v in data.items()}
tune_data = take(late_data, is_tune)
old_offsets = match_policy(log_old_late[is_tune], tune_data, policy_late[is_tune])
new_offsets = match_policy(log_new_late[is_tune], tune_data, policy_late[is_tune])
print("matched offsets (reorder, expedite, human): original", old_offsets, "retrained", new_offsets)
ZERO = (0.0, 0.0, 0.0)


def bundle(old_log, new_log, data, policy):
    return {"Original": decide(old_log, data, ZERO), "Original matched": decide(old_log, data, old_offsets), "Retrained": decide(new_log, data, ZERO),
            "Retrained matched": decide(new_log, data, new_offsets), "Policy": policy}


check_data = take(late_data, check_mask)
report("LATE 2015 CHECK HALF", check_data, bundle(log_old_late[check_mask], log_new_late[check_mask], check_data, policy_late[check_mask]),
       {"Original": 1 - old_late[check_mask][:, 0], "Retrained": 1 - new_late[check_mask][:, 0], "TabPFN risk": check_data["risk"]})

test = pd.read_parquet(ROOT / "data/processed/escalation/classified_test.parquet")
old_test, new_test = np.load(ROOT / "data/processed/escalation/ultrabert/test_fine_tuned.npy"), np.load(NEW / "test_probabilities.npy")
assert old_test.shape == new_test.shape == (len(test), 4)
test_data = prepare(test)
report("TEST", test_data, bundle(np.log(np.clip(old_test, 1e-9, 1)), np.log(np.clip(new_test, 1e-9, 1)), test_data, test.policy.astype(str).map(CODE).to_numpy()),
       {"Original": 1 - old_test[:, 0], "Retrained": 1 - new_test[:, 0], "TabPFN risk": test_data["risk"]})
