"""Score the judge against hand labels. Calls no API.

Fill the `label` column of reports/step_11/claims_to_label_<tag>.csv with: error, ok, or skip (cannot tell). That file has no judge verdict,
so labels are blind. The sample is every flagged claim plus a random sample of the rest, so unflagged rows are weighted up to the
number they stand for when recall and kappa are computed.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.agent.judge import ERROR_VERDICTS

OUT = ROOT / "reports/step_11"
LABELS = {"error", "ok"}


def cohen_kappa(judge, human, weight):
    """Weighted Cohen's kappa for two yes/no raters."""
    total = weight.sum()
    observed = weight[judge == human].sum() / total
    p_judge, p_human = weight[judge].sum() / total, weight[human].sum() / total
    expected = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def score(frame):
    """Precision uses every flagged claim, all of which are labelled. Recall and kappa weight unflagged claims up to their population."""
    judge, human, weight = frame.flagged.to_numpy(), (frame.label == "error").to_numpy(), frame.weight.to_numpy()
    true_positive = weight[judge & human].sum()
    flagged, errors = weight[judge].sum(), weight[human].sum()
    return {"labelled": len(frame), "flagged": int(judge.sum()), "human_errors": int(human.sum()),
            "precision": true_positive / flagged if flagged else np.nan, "recall_weighted": true_positive / errors if errors else np.nan,
            "kappa_weighted": cohen_kappa(judge, human, weight)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="such as traces_qwen")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    claims = pd.read_csv(OUT / f"claims_{args.tag}.csv")
    labels = pd.read_csv(OUT / f"claims_to_label_{args.tag}.csv").fillna({"label": ""})
    sample = json.loads((OUT / f"label_sample_{args.tag}.json").read_text(encoding="utf-8"))
    labels["label"] = labels.label.str.strip().str.lower()
    unknown = sorted(set(labels.label) - LABELS - {"", "skip"})
    if unknown:
        raise SystemExit(f"unknown labels {unknown}; use error, ok or skip")
    frame = labels[labels.label.isin(LABELS)].merge(claims[["claim_id", "verdict"]], on="claim_id", validate="one_to_one")
    frame["flagged"] = frame.verdict.isin(ERROR_VERDICTS)
    frame["weight"] = np.where(frame.flagged, 1.0, sample["unflagged_total"] / max(sample["unflagged_sampled"], 1))
    print(f"labels: {len(frame)} usable, {int((labels.label == 'skip').sum())} skipped, {int((labels.label == '').sum())} blank")
    missing = int((labels.label == "").sum())
    if missing:
        print("WARNING: blank labels remain, so these numbers are not final")
    rows = {"all": score(frame), "code layer": score(frame[frame.kind != "qualitative"])}
    qualitative = frame[frame.kind == "qualitative"]
    if len(qualitative):
        rows["LLM layer"] = score(qualitative)
        print("LLM verdicts among labelled qualitative claims:", qualitative.verdict.value_counts().to_dict())
    print(pd.DataFrame(rows).T.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nDISAGREEMENTS\n", frame[frame.flagged != (frame.label == "error")][["claim_id", "kind", "text", "verdict", "label", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
