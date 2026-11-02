"""Measure TabPFN v2 runtime, memory and validation accuracy across training sizes."""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from src.stockout.features import as_matrix

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEED = 42
PREDICT_ROWS = 5000
data = pd.read_parquet("data/processed/stockout_features.parquet")
train = data[data.split == "train"]
validation = data[data.split == "validation"].sample(PREDICT_ROWS, random_state=SEED)
X_valid, y_valid = as_matrix(validation), validation.label.to_numpy()
print(f"validation sample {len(validation):,}, positive rate {y_valid.mean():.4f}", flush=True)

results = []
for size, members, chunk in [(1000, 1, 5000), (1000, 2, 5000), (3000, 1, 5000), (3000, 2, 5000), (10000, 1, 5000)]:
    sample = train.sample(size, random_state=SEED)
    try:
        torch.cuda.reset_peak_memory_stats()
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda", random_state=SEED, n_estimators=members)
        started = time.time()
        model.fit(as_matrix(sample), sample.label.to_numpy())
        fit_seconds = time.time() - started
        started = time.time()
        probability = np.concatenate([model.predict_proba(X_valid[i:i + chunk])[:, 1] for i in range(0, len(X_valid), chunk)])
        results.append({"train_rows": size, "members": members, "fit_s": round(fit_seconds, 1), "predict_s": round(time.time() - started, 1),
                        "ms_per_row": round(1000 * (time.time() - started) / len(X_valid), 2),
                        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
                        "pr_auc": round(average_precision_score(y_valid, probability), 4), "brier": round(brier_score_loss(y_valid, probability), 5)})
    except Exception as exc:
        results.append({"train_rows": size, "members": members, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    print(results[-1], flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    Path("reports/step_05/tabpfn_probe.json").write_text(json.dumps(results, indent=2))
