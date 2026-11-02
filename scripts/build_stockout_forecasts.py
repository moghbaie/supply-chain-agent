"""Generate weekly Timestral-28M forecasts for each scenario, saving progress per scenario."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.forecasting.models import LEVELS, TimestralRunner
from src.forecasting.panel import DemandPanel, monday_origins
from src.stockout.forecasts import scenario_panel, summarize_forecast

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("data/processed/stockout_forecasts")
OUT.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(Path("configs/simulation.yaml").read_text())
forecast_config = yaml.safe_load(Path("configs/forecasting.yaml").read_text())
chosen = json.loads(Path("reports/step_04/chosen_model.json").read_text())
horizon = forecast_config["horizon_days"]
label_days = config["labels"]["stockout_horizon_days"]
level_positions = {level: int(np.flatnonzero(np.isclose(LEVELS, level))[0]) for level in (0.1, 0.9)}

first_target = pd.Timestamp(config["dates"]["burn_in_end"]) + pd.Timedelta(days=2)
origins = monday_origins(first_target, config["dates"]["simulation_end"], label_days)
print(f"origins: {len(origins)} ({origins[0].date()} to {origins[-1].date()}) | model: {chosen}", flush=True)

panel = DemandPanel()
runner = TimestralRunner(chosen["size"], forecast_config["batch_size"])
daily = pd.read_parquet("data/simulated/inventory_daily.parquet", columns=["scenario", "id", "date", "sales", "on_hand_end"])

for scenario in config["scenarios"]:
    target = OUT / f"{scenario}.parquet"
    if target.exists():
        print(f"{scenario}: already finished", flush=True)
        continue
    view = scenario_panel(panel, daily[daily.scenario == scenario])
    print(f"{scenario}: unobserved share of simulation days {100 * (1 - view.observed[:, -1241:].mean()):.2f}%", flush=True)
    started = time.time()
    frames = []
    for number, origin in enumerate(origins, 1):
        forecast = runner.forecast(view, origin, horizon, chosen["covariates"])
        frames.append(summarize_forecast(panel.series.id.to_numpy(), origin, forecast, level_positions))
        if number % 20 == 0 or number == len(origins):
            print(f"  {scenario}: date {number}/{len(origins)}, {time.time() - started:.0f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
print("done", flush=True)
