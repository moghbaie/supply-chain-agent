"""Compare Jev and the deterministic policy on escalation cases. Usage: run_escalation.py validation|test"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.jev import CRITERIA, INSTRUCTIONS, MODEL, USD_PER_INPUT_TOKEN, build_request, classify
from src.escalation.policy import PolicyThresholds, apply_overrides, policy_baseline, requires_approval

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 175, "display.max_columns", 30, "display.max_colwidth", 40)
SEED, VALIDATION_ROWS, TEST_ROWS, BUDGET_USD = 42, 2000, 27000, 5.0
ORDER_CHECK_ROWS = {"validation": 500, "test": 2000}
REPORT = Path("reports/step_07")
OUT = Path("data/processed/escalation")
THRESHOLDS = OUT / "policy_thresholds.json"
USAGE = OUT / "usage.json"
stage = sys.argv[1]
key = dotenv_values(".env.secrets")["TYPESAFE_API_KEY"]
cases, settings = build_cases()


def metrics(reference, choices):
    critical = reference == "human_review"
    return {"macro_f1": f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0),
            "accuracy": float((reference == choices).mean()),
            "critical_recall": float((choices[critical] == "human_review").mean()),
            "critical_acted_on": float((choices[critical] != "monitor").mean()),
            "any_stockout_acted_on": float((choices[reference != "monitor"] != "monitor").mean()),
            "false_alarm_share": float((choices[reference == "monitor"] != "monitor").mean()),
            **{f"recall_{c}": recall_score(reference == c, choices == c, zero_division=0) for c in CATEGORIES}}


def show(name, reference, choices):
    print(f"\n{name}: predicted counts {pd.Series(choices).value_counts().reindex(CATEGORIES).fillna(0).astype(int).to_dict()}")
    print("confusion (rows reference, columns predicted):\n", pd.DataFrame(confusion_matrix(reference, choices, labels=CATEGORIES), index=CATEGORIES, columns=[c[:8] for c in CATEGORIES]).to_string())
    return {"model": name, **metrics(reference, choices)}


def spend(tokens):
    used = json.loads(USAGE.read_text()) if USAGE.exists() else {"input_tokens": 0}
    used["input_tokens"] += tokens
    used["usd"] = used["input_tokens"] * USD_PER_INPUT_TOKEN
    USAGE.write_text(json.dumps(used))
    print(f"tokens this run {tokens:,}, total {used['input_tokens']:,}, about ${used['usd']:.4f} of ${BUDGET_USD}", flush=True)
    if used["usd"] > BUDGET_USD:
        raise SystemExit("budget exceeded")
    return used


rows = cases[cases.split == stage]
if stage == "validation":
    sample = rows.sample(VALIDATION_ROWS, random_state=SEED).reset_index(drop=True)
    grid = [(a, c) for a in np.arange(0.10, 0.61, 0.05) for c in np.arange(0.40, 0.96, 0.05)]
    scored = [(f1_score(rows.reference, policy_baseline(rows, PolicyThresholds(a, c)), labels=CATEGORIES, average="macro", zero_division=0), a, c) for a, c in grid]
    best, act, critical = max(scored)
    THRESHOLDS.write_text(json.dumps({"act": round(float(act), 2), "critical": round(float(critical), 2)}))
    print(f"policy thresholds fitted on all validation rows: act {act:.2f}, critical {critical:.2f}, macro-F1 {best:.4f}")
    print("example request state:\n", json.dumps(build_request(sample.iloc[0])["state"]))
    print("question:", INSTRUCTIONS, "| criteria:", json.dumps(CRITERIA))
else:
    sample = rows.sample(TEST_ROWS, random_state=SEED).reset_index(drop=True)
thresholds = PolicyThresholds(**json.loads(THRESHOLDS.read_text()))
print(f"\n{stage}: rows {len(sample):,}, reference {sample.reference.value_counts().to_dict()}", flush=True)

started = time.time()
answers = classify(sample, key, on_progress=lambda n: print(f"  {n} done, {time.time() - started:.0f}s", flush=True))
spend(sum(a[2] for a in answers))
print(f"Jev: {len(answers):,} requests in {time.time() - started:.0f}s, mean input tokens {np.mean([a[2] for a in answers]):.0f}")
sample["jev"] = [a[0] for a in answers]
sample["jev_probabilities"] = [json.dumps(a[1]) for a in answers]
if not set(sample.jev) <= set(CATEGORIES):
    raise SystemExit(f"invalid category returned: {set(sample.jev) - set(CATEGORIES)}")
sample["jev_override"], sample["override_reason"] = apply_overrides(sample, sample.jev)
sample["policy"] = policy_baseline(sample, thresholds)
sample["requires_approval"] = requires_approval(sample.jev_override)
reference = sample.reference.to_numpy()
table = pd.DataFrame([show("jev", reference, sample.jev.to_numpy(dtype=object)), show("jev_with_overrides", reference, sample.jev_override), show("policy_baseline", reference, sample.policy.to_numpy(dtype=object))]).set_index("model")
print(f"\n{stage.upper()} METRICS\n", table.round(4).T.to_string())
print("\noverride reasons:", sample.override_reason.value_counts().to_dict())

subset = sample.iloc[:ORDER_CHECK_ROWS[stage]]
reversed_criteria = dict(reversed(list(CRITERIA.items())))
second = classify(subset, key, criteria=reversed_criteria)
spend(sum(a[2] for a in second))
other = np.array([a[0] for a in second], dtype=object)
print(f"\nOPTION ORDER CHECK ({len(subset)} cases): same choice {100 * (other == subset.jev.to_numpy()).mean():.2f}% | macro-F1 original {metrics(subset.reference.to_numpy(), subset.jev.to_numpy(dtype=object))['macro_f1']:.4f}, reversed {metrics(subset.reference.to_numpy(), other)['macro_f1']:.4f}")
print("counts original", subset.jev.value_counts().to_dict(), "| reversed", pd.Series(other).value_counts().to_dict())
if stage == "validation":
    repeat = classify(subset.iloc[:100], key)
    spend(sum(a[2] for a in repeat))
    print("repeat stability, 100 cases, same choice %:", 100 * float(np.mean([a[0] == b for a, b in zip(repeat, subset.jev)])))

columns = ["origin", "on_hand", "on_order", "cover_days", "days_to_next_due", "open_pos", "units_overdue", "fc_mean_14", "stockout_risk", "value_band", "reference", "jev", "jev_override", "policy"]
print(f"\n20 CLASSIFIED CASES ({stage}, 12 stockouts and 8 others)")
preview = pd.concat([sample[sample.reference != "monitor"].head(12), sample[sample.reference == "monitor"].sort_values("stockout_risk", ascending=False).head(8)])[columns]
print(preview.round(2).to_string(index=False))
table.round(5).to_csv(REPORT / f"metrics_{stage}.csv")
preview.to_csv(REPORT / f"classified_{stage}.csv", index=False)
sample.drop(columns=["supplier_id"]).to_parquet(OUT / f"classified_{stage}.parquet", index=False)
(REPORT / "settings.json").write_text(json.dumps({**settings, "policy_thresholds": json.loads(THRESHOLDS.read_text()), "model": MODEL, "instructions": INSTRUCTIONS, "criteria": CRITERIA}, indent=2))
