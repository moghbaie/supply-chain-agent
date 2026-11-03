"""Run the untuned agent on saved cases, one trace file per case, so an interrupted run resumes. Runs locally."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.agent.llm import LocalLLM
from src.agent.runner import run_case
from src.agent.tools import build_toolbox

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--cases", choices=["dev", "test"], required=True)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--model", default="smollm")
parser.add_argument("--name", default="smollm_untuned")
args = parser.parse_args()

DATA = Path("data/processed/agent")
OUT = DATA / "traces" / args.name / args.cases
OUT.mkdir(parents=True, exist_ok=True)
cases = pd.read_parquet(DATA / f"{args.cases}_cases.parquet")
scores = pd.read_parquet(DATA / "delay_scores.parquet")
delay = scores[scores.period == args.cases].set_index("po_id").delay_risk.to_dict()
orders = pd.read_parquet("data/simulated/purchase_orders.parquet")
box, keyed = build_toolbox(cases, orders, delay, prefix=args.cases)
keyed[["case_id", "scenario", "id", "origin", "reference", "ultrabert"]].to_parquet(OUT / "key.parquet", index=False)
chosen = keyed if args.limit is None else keyed.head(args.limit)
pending = [case_id for case_id in chosen.case_id if not (OUT / f"{case_id}.json").exists()]
print(f"{args.cases}: {len(chosen)} chosen, {len(pending)} to run", flush=True)
if not pending:
    sys.exit(0)

llm = LocalLLM(args.model)
torch.cuda.reset_peak_memory_stats()
started = time.time()
for number, case_id in enumerate(pending, 1):
    try:
        trace = run_case(llm, box, case_id)
    except (torch.cuda.OutOfMemoryError, ValueError) as error:
        torch.cuda.empty_cache()
        trace = {"case_id": case_id, "status": "error", "error": f"{type(error).__name__}: {str(error)[:200]}", "turns": [], "tool_calls": [], "evidence": {}, "input_tokens": 0, "output_tokens": 0, "malformed_calls": 0, "invalid_submissions": 0, "seconds": None, "recommendation": None,
                 "review": {"final_category": "human_review", "approval_required": True, "order_placed": False, "flags": ["agent_failed:run error"]}}
    trace["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    (OUT / f"{case_id}.json").write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    picked = (trace["recommendation"] or {}).get("category")
    print(f"{number}/{len(pending)} {case_id} {trace['status']} model={picked} final={trace['review']['final_category']} turns={len(trace['turns'])} tokens={trace['input_tokens']}+{trace['output_tokens']} {trace['seconds']}s flags={trace['review']['flags']} elapsed={time.time() - started:.0f}s", flush=True)
print(f"done in {time.time() - started:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
