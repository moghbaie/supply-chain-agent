"""Replay the test period for every policy and scenario, saving each finished run so an interrupted replay resumes. Runs locally."""
import argparse
import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import yaml

from src.evaluation.decisions import APPROVE_ALL
from src.evaluation.replay import Replay, Variant, load_world
from src.evaluation.scoring import build_scorer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="*", default=None)
parser.add_argument("--scenarios", nargs="*", default=None)
parser.add_argument("--last-day", type=int, default=None, help="Stop after this many days, for a trial. Nothing is saved.")
args = parser.parse_args()

OUT = Path("data/processed/evaluation")
OUT.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
settings = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
SEED = settings["seed"]
scenarios = args.scenarios or list(config["scenarios"])
names = args.variants or list(settings["variants"])


def make_variant(name):
    spec = settings["variants"][name]
    fraction = settings["expedite"]["weak_remaining_time_fraction" if spec.get("expedite") == "weak" else "remaining_time_fraction"]
    return Variant(name, spec["levels"], spec.get("decide"), spec.get("conflict_rule", False), spec.get("person", APPROVE_ALL), fraction)


todo = [(s, n) for s in scenarios for n in names if args.last_day or not (OUT / s / f"{n}_daily.parquet").exists()]
print(f"to run: {len(todo)} of {len(scenarios) * len(names)}", flush=True)
if not todo:
    sys.exit(0)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

scorer = None
if any(settings["variants"][n].get("decide") or settings["variants"][n]["levels"] == "forecast" for _, n in todo):
    started = time.time()
    scorer = build_scorer(settings, scenarios)
    print(f"models ready in {time.time() - started:.0f}s; GPU in use {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

for scenario, name in todo:
    started = time.time()
    world = load_world(scenario, config, settings["start"])
    if args.last_day:
        world.dates, world.demand = world.dates[:args.last_day], world.demand[:, :args.last_day]
    replay = Replay(world, make_variant(name), scorer).run()
    if scorer:
        scorer.forget(replay)
    daily = replay.daily_frame()
    summary = f"unmet {int(daily.unmet.sum()):,}, ordered {int(daily.ordered.sum()):,}, expedites {int(daily.expedites.sum()):,}"
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"{scenario} {name}: {len(daily):,} rows, {summary}, {time.time() - started:.0f}s, peak GPU {peak:.2f} GB", flush=True)
    if args.last_day:
        continue
    target = OUT / scenario
    target.mkdir(exist_ok=True)
    replay.orders_frame().astype({"delay_days": "Int64", "actual_lead_days": "Int64"}).to_parquet(target / f"{name}_orders.parquet", index=False)
    if replay.cases:
        pd.concat(replay.cases, ignore_index=True).to_parquet(target / f"{name}_cases.parquet", index=False)
        pd.DataFrame(replay.actions).to_parquet(target / f"{name}_actions.parquet", index=False)
    daily.to_parquet(target / f"{name}_daily.parquet", index=False)
print("done", flush=True)
