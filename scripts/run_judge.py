"""Judge agent explanations claim by claim and compare with the Step 8 audit. Runs locally; calls no API.

--source traces   all 100 test traces in data/processed (git-ignored)
--source report   the 10 complete traces committed in reports/step_08/test_traces.txt
--llm none        code layer only; qualitative claims are not_checked
--llm qwen        Qwen3-4B judges qualitative claims (needs a GPU). The agent was SmolLM3, so the default judge is a different family.
--orders K        ask the LLM K times per claim with tool results in different orders, to measure position bias
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.parse_trace_file import parse
from src.agent.audit import audit
from src.agent.judge import ERROR_VERDICTS, judge_explanation, summarize

TRACES = ROOT / "data/processed/agent/traces/smollm_untuned/test"
OUT = ROOT / "reports/step_11"
LABEL_COLUMNS = ["claim_id", "case_id", "kind", "text", "clause", "group", "stated"]


def load(source):
    if source == "report":
        complete, _ = parse()
        return [(c["case_id"], c["explanation"], c["evidence"], c["final"]) for c in complete.values()]
    rows = []
    for path in sorted(TRACES.glob("test-*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows.append((trace["case_id"], trace["recommendation"]["explanation"], trace["evidence"], trace["review"]["final_category"]))
    return rows


def label_sample(claims, size, seed):
    """All flagged claims plus a random sample of the rest, shuffled, with the judge's verdict left out so labels are blind."""
    flagged = claims[claims.verdict.isin(ERROR_VERDICTS)]
    rest = claims.drop(flagged.index)
    picked = pd.concat([flagged, rest.sample(min(size, len(rest)), random_state=seed)])
    picked = picked.sample(frac=1, random_state=seed)[LABEL_COLUMNS].assign(label="", note="")
    return picked, len(rest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["traces", "report"], default="report")
    parser.add_argument("--llm", choices=["none", "qwen", "smollm"], default="none")
    parser.add_argument("--orders", type=int, default=1)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 200, "display.max_colwidth", 90, "display.max_rows", 500)
    tag = f"{args.source}_{args.llm}"
    llm = None
    if args.llm != "none":
        from src.agent.llm import LocalLLM
        llm = LocalLLM(args.llm, max_new_tokens=120)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = load(args.source)
    claims, explanations = [], []
    started = time.time()
    for position, (case_id, explanation, evidence, final) in enumerate(cases, 1):
        found = judge_explanation(case_id, explanation, evidence, llm=llm, orders=args.orders)
        if llm is not None:
            asked = sum(c.kind == "qualitative" for c in found)
            minutes = (time.time() - started) / 60
            print(f"  [{position}/{len(cases)}] {case_id}: {asked} qualitative claims; {minutes:.1f} min so far, about {minutes / position * (len(cases) - position):.0f} min left", flush=True)
        claims += [c.row() for c in found]
        rules = [name for name in audit(explanation, evidence) if name != "cites_no_numbers"]
        explanations.append({**summarize(case_id, found), "final": final, "step8_rules": "; ".join(rules), "step8_error": bool(rules), "explanation": explanation})
    claims = pd.DataFrame(claims)
    explanations = pd.DataFrame(explanations)

    print(f"source {args.source}, judge {args.llm}, orders {args.orders}: {len(explanations)} explanations, {len(claims)} claims in {claims.groupby(['case_id', 'sentence', 'clause_index']).ngroups} clauses")
    print("\nCLAIMS BY KIND AND VERDICT\n", pd.crosstab(claims.kind, claims.verdict, margins=True, margins_name="all").to_string())
    print(f"\nexplanations with a factual error: judge {int(explanations.has_error.sum())} (code layer {int(explanations.code_error.sum())}), Step 8 rules {int(explanations.step8_error.sum())}")
    print("\nPER EXPLANATION, where either finds an error\n", explanations[explanations.has_error | explanations.step8_error][["case_id", "final", "error_claims", "step8_rules"]].to_string(index=False))
    missed = explanations[explanations.step8_error & ~explanations.code_error]
    print(f"\nStep 8 rule hits with no code-layer error: {len(missed)}")
    if len(missed):
        print(missed[["case_id", "step8_rules", "error_claims"]].to_string(index=False))
    new = explanations[explanations.code_error & ~explanations.step8_error]
    print(f"code-layer errors no Step 8 rule found: {len(new)} explanations ({', '.join(new.case_id)})")
    monitor = explanations[explanations.has_error & (explanations.final == "monitor")]
    print(f"errors on a final monitor case, which no person reviews: {len(monitor)} ({', '.join(monitor.case_id)})")
    print("\nFLAGGED CLAIMS\n", claims[claims.verdict.isin(ERROR_VERDICTS)][["case_id", "kind", "text", "group", "actual", "verdict", "detail"]].to_string(index=False))
    if args.orders > 1 and llm is not None:
        asked = claims[claims.kind == "qualitative"]
        changed = asked.actual.map(lambda text: len(set(text.split(";"))) > 1)
        print(f"\nPOSITION BIAS: {int(changed.sum())} of {len(asked)} qualitative verdicts change when only the order of tool results changes")
        print(asked[changed][["case_id", "text", "actual"]].to_string(index=False))

    claims.to_csv(OUT / f"claims_{tag}.csv", index=False)
    explanations.drop(columns="explanation").to_csv(OUT / f"explanations_{tag}.csv", index=False)
    sample, rest = label_sample(claims, args.sample, args.seed)
    sample.to_csv(OUT / f"claims_to_label_{tag}.csv", index=False)
    (OUT / f"label_sample_{tag}.json").write_text(json.dumps({"flagged": int(claims.verdict.isin(ERROR_VERDICTS).sum()), "unflagged_total": rest, "unflagged_sampled": min(args.sample, rest), "seed": args.seed}), encoding="utf-8")
    print(f"\nwrote claims_{tag}.csv, explanations_{tag}.csv, claims_to_label_{tag}.csv ({len(sample)} claims, judge verdict hidden) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
