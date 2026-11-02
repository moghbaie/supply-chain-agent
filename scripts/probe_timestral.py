"""Measure Timestral sizes on one validation origin: runtime, memory and a first accuracy check."""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from timestral import TimestralForecaster

from src.forecasting.panel import DemandPanel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ORIGIN = "2015-03-02"
HORIZON = 28
BATCH = 64
panel = DemandPanel()
context = panel.context(ORIGIN)
actual, observed = panel.actuals(ORIGIN, HORIZON)
print("context shape:", context.shape, "| observed share:", round(float(np.isfinite(context).mean()), 3), "| actual units:", int(actual.sum()))

results = {}
for size in ["5M", "28M", "123M", "209M"]:
    for device, batch in [("cuda", BATCH), ("cuda", 8), ("cpu", BATCH)]:
        key = f"{size}/{device}/batch{batch}"
        if size in {"5M", "28M"} and batch == 8:
            continue
        if device == "cpu" and size in {"123M", "209M"}:
            sample = 16
        else:
            sample = len(context)
        try:
            torch.cuda.reset_peak_memory_stats()
            model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device)
            started = time.time()
            parts = []
            with torch.no_grad():
                for first in range(0, sample, batch):
                    forecast = model.forecast(context[first:first + batch], prediction_length=HORIZON, clip_nonnegative=True)
                    parts.append(forecast.quantiles.detach().cpu().numpy())
            quantiles = np.concatenate(parts)
            elapsed = time.time() - started
            median = quantiles[:, 0, :, quantiles.shape[-1] // 2]
            mean = quantiles[:, 0].mean(axis=-1)
            truth = actual[:sample]
            results[key] = {
                "series": sample, "seconds": round(elapsed, 1), "ms_per_series": round(1000 * elapsed / sample, 1),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else None,
                "shape": list(quantiles.shape), "wape_median_pct": round(100 * float(np.abs(median - truth).sum() / truth.sum()), 2),
                "wape_mean_pct": round(100 * float(np.abs(mean - truth).sum() / truth.sum()), 2),
                "bias_mean_pct": round(100 * float((mean - truth).sum() / truth.sum()), 2),
                "bias_median_pct": round(100 * float((median - truth).sum() / truth.sum()), 2),
            }
            print(key, results[key], flush=True)
            del model
        except RuntimeError as exc:
            results[key] = {"error": str(exc)[:120]}
            print(key, "FAILED:", str(exc)[:120], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if size in {"123M", "209M"} and device == "cuda" and "error" not in results[key]:
            break

Path("reports/step_04").mkdir(parents=True, exist_ok=True)
Path("reports/step_04/probe.json").write_text(json.dumps(results, indent=2))
