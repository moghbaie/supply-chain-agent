"""Select a Timestral variant on validation, then evaluate it and the baselines on validation and test."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.forecasting.metrics import cumulative_wape, pinball, quantile_at, score, summarize
from src.forecasting.models import GbmForecaster, TimestralRunner, seasonal_naive
from src.forecasting.panel import DemandPanel, monday_origins
from src.simulation.demand import daily_prices, load_m5

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 20)
REPORT = Path("reports/step_04")
OUT = Path("data/processed/forecasts")
config = yaml.safe_load(Path("configs/forecasting.yaml").read_text())
seed, horizon, levels = config["seed"], config["horizon_days"], config["report_levels"]
REPORT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def add(totals, values):
    for key, value in values.items():
        totals[key] += value


panel = DemandPanel()
calendar, _, prices, _, dates = load_m5()
price, _ = daily_prices(panel.series, prices, calendar, dates)
validation = monday_origins(config["validation"]["first_target"], config["validation"]["last_target"], horizon)
test = monday_origins(config["test"]["first_target"], config["test"]["last_target"], horizon)
print(f"origins: validation {len(validation)} ({validation[0].date()} to {validation[-1].date()}), test {len(test)} ({test[0].date()} to {test[-1].date()})", flush=True)

selection = config["selection"]
chosen_origins = validation[::selection["every_nth_validation_origin"]]
checkpoint = REPORT / "selection_checkpoint.csv"
rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
finished = {(row["size"], row["covariates"]) for row in rows}
print(f"variants already finished: {sorted(finished)}", flush=True)
for size, covariates in selection["variants"]:
    if (size, covariates) in finished:
        continue
    runner = TimestralRunner(size, config["batch_size"])
    totals = defaultdict(float)
    started = time.time()
    for number, origin in enumerate(chosen_origins, 1):
        actual, seen = panel.actuals(origin, horizon)
        add(totals, score(actual, seen, runner.forecast(panel, origin, horizon, covariates), levels))
        print(f"  {size}/{covariates}: date {number}/{len(chosen_origins)}, {time.time() - started:.0f}s", flush=True)
    rows.append({"size": size, "covariates": covariates, **summarize(totals), "seconds": time.time() - started})
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(checkpoint, index=False)
    del runner
    torch.cuda.empty_cache()
board = pd.DataFrame(rows)
board.round(3).to_csv(REPORT / "validation_selection.csv", index=False)
best = board.pinball_per_unit.min()
close = board[board.pinball_per_unit <= best * (1 + selection["tolerance_pct"] / 100)]
order = {size: i for i, size in enumerate(selection["sizes"])}
winner = close.assign(rank=close["size"].map(order)).sort_values(["rank", "pinball_per_unit"]).iloc[0]
print("\nVALIDATION SELECTION\n", board.round(3).to_string(index=False))
print(f"\nchosen: Timestral-{winner['size']} covariates={winner['covariates']} (pinball {winner.pinball_per_unit:.4f}, best {best:.4f})", flush=True)
(REPORT / "chosen_model.json").write_text(json.dumps({"size": winner["size"], "covariates": winner["covariates"]}))

timestral = TimestralRunner(winner["size"], config["batch_size"])
timer = time.time()
gbm_validation = GbmForecaster(panel, price, config, seed).fit(pd.Timestamp("2014-12-31"))
print(f"LightGBM (train targets through 2014): {gbm_validation.train_rows:,} rows, {time.time() - timer:.0f}s", flush=True)
timer = time.time()
gbm_test = GbmForecaster(panel, price, config, seed).fit(pd.Timestamp("2015-12-31"))
print(f"LightGBM (train targets through 2015): {gbm_test.train_rows:,} rows, {time.time() - timer:.0f}s", flush=True)


def evaluate(name, origins, gbm):
    models = {
        "seasonal_naive": lambda o: seasonal_naive(panel, o, horizon, config["empirical_window_days"]),
        "lightgbm": gbm.forecast,
        "timestral": lambda o: timestral.forecast(panel, o, horizon, winner["covariates"]),
    }
    results, saved = [], []
    per_series = {m: np.zeros(len(panel.series)) for m in models}
    cumulative = {m: {14: [0.0, 0.0], 28: [0.0, 0.0]} for m in models}
    buckets = {"all": None, "days_1_14": np.arange(14), "days_15_28": np.arange(14, 28)}
    stats = {m: {b: defaultdict(float) for b in buckets} for m in models}
    by_category = {m: {c: defaultdict(float) for c in panel.series.cat_id.unique()} for m in models}
    seconds = defaultdict(float)
    for number, origin in enumerate(origins, 1):
        print(f"  {name}: date {number}/{len(origins)}", flush=True)
        actual, seen = panel.actuals(origin, horizon)
        for model, predict in models.items():
            tick = time.time()
            forecast = predict(origin)
            seconds[model] += time.time() - tick
            for bucket, steps in buckets.items():
                add(stats[model][bucket], score(actual, seen, forecast, levels, steps))
            for category in by_category[model]:
                rows_mask = panel.series.cat_id.to_numpy() == category
                part = {**forecast, "quantiles": forecast["quantiles"][rows_mask], "mean": forecast["mean"][rows_mask]}
                add(by_category[model][category], score(actual[rows_mask], seen[rows_mask], part, levels))
            per_series[model] += sum(np.where(seen, pinball(actual, quantile_at(forecast, level), level), 0).sum(axis=1) for level in levels)
            for days in (14, 28):
                errors, truth = cumulative_wape(actual, seen, forecast["mean"], days)
                cumulative[model][days][0] += errors.sum()
                cumulative[model][days][1] += truth.sum()
            if model == "timestral" or name == "test":
                frame = pd.DataFrame({
                    "id": np.repeat(panel.series.id.to_numpy(), horizon), "origin": origin,
                    "horizon": np.tile(np.arange(1, horizon + 1), len(panel.series)), "model": model,
                    "actual": actual.reshape(-1), "observed": seen.reshape(-1), "mean": forecast["mean"].reshape(-1),
                    "q10": quantile_at(forecast, 0.1).reshape(-1), "q50": quantile_at(forecast, 0.5).reshape(-1),
                    "q90": quantile_at(forecast, 0.9).reshape(-1)})
                saved.append(frame)
    for model in models:
        for bucket in buckets:
            results.append({"period": name, "model": model, "days": bucket, **summarize(stats[model][bucket])})
        results[-3]["wape_cum14_pct"] = 100 * cumulative[model][14][0] / cumulative[model][14][1]
        results[-3]["wape_cum28_pct"] = 100 * cumulative[model][28][0] / cumulative[model][28][1]
        results[-3]["seconds"] = seconds[model]
    categories = pd.DataFrame([{"period": name, "model": m, "category": c, **summarize(v)} for m, group in by_category.items() for c, v in group.items()])
    return pd.DataFrame(results), categories, per_series, pd.concat(saved, ignore_index=True)


validation_results, validation_categories, _, validation_saved = evaluate("validation", validation, gbm_validation)
print("\nVALIDATION (all weekly origins)\n", validation_results.round(3).to_string(index=False), flush=True)
test_results, test_categories, test_series, test_saved = evaluate("test", test, gbm_test)
print("\nTEST\n", test_results.round(3).to_string(index=False))
print("\nTEST by category\n", test_categories.drop(columns="period").round(3).sort_values(["category", "model"]).to_string(index=False))

rng = np.random.default_rng(seed)
comparison = {}
for rival in ["seasonal_naive", "lightgbm"]:
    samples = rng.integers(0, len(panel.series), (2000, len(panel.series)))
    ratio = test_series["timestral"][samples].sum(axis=1) / test_series[rival][samples].sum(axis=1)
    comparison[rival] = {"pinball_ratio": float(test_series["timestral"].sum() / test_series[rival].sum()),
                         "ci95_low": float(np.percentile(ratio, 2.5)), "ci95_high": float(np.percentile(ratio, 97.5)),
                         "series_where_timestral_better_pct": float(100 * (test_series["timestral"] < test_series[rival]).mean())}
print("\nTimestral vs baselines, pinball ratio on test (below 1 favors Timestral), series bootstrap:\n", json.dumps(comparison, indent=1))

pd.concat([validation_results, test_results]).round(4).to_csv(REPORT / "metrics.csv", index=False)
pd.concat([validation_categories, test_categories]).round(4).to_csv(REPORT / "metrics_by_category.csv", index=False)
(REPORT / "comparison.json").write_text(json.dumps(comparison, indent=2))
validation_saved.to_parquet(OUT / "validation.parquet", index=False)
test_saved.to_parquet(OUT / "test.parquet", index=False)
print({p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUT.glob("*.parquet")})
print("peak VRAM MB:", round(torch.cuda.max_memory_allocated() / 1024**2))
