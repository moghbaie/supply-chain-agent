"""Fine-tune UltraBERT on 2015 cases, then score it and zero-shot embedding on the Step 7 test cases. Runs locally and resumes."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.jev import CRITERIA
from src.escalation.ultrabert import FineTunedClassifier, balanced_training_rows, class_weights, zero_shot_probabilities

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 20)
SEED, EPOCHS, BATCH, LEARNING_RATE, TRAIN_ROWS, MONITOR_PER_STOCKOUT, DEV_ROWS = 42, 2, 8, 1e-5, 8000, 1, 3000
TRAIN_LAST_ORIGIN, DEV_FIRST_ORIGIN = pd.Timestamp("2015-09-14"), pd.Timestamp("2015-10-01")
OUT = Path("data/processed/escalation/ultrabert")
OUT.mkdir(parents=True, exist_ok=True)
BEST = OUT / "best_model"
STATE = OUT / "training_state.json"
REPORT = Path("reports/step_07")
started = time.time()


def macro(reference, choices):
    return float(f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0))


cases, _ = build_cases()
year = cases[cases.split == "validation"]
pool = balanced_training_rows(year[year.origin <= TRAIN_LAST_ORIGIN], MONITOR_PER_STOCKOUT, SEED)
train = pool.sample(min(TRAIN_ROWS, len(pool)), random_state=SEED).reset_index(drop=True)
dev = year[year.origin >= DEV_FIRST_ORIGIN].sample(DEV_ROWS, random_state=SEED).reset_index(drop=True)
print(f"train {len(train):,} (last origin {train.origin.max().date()}), mix {train.reference.value_counts().to_dict()}", flush=True)
print(f"dev {len(dev):,} (first origin {dev.origin.min().date()}), mix {dev.reference.value_counts().to_dict()}", flush=True)
weights = class_weights(train.reference)
print("class weights:", dict(zip(CATEGORIES, np.round(weights, 2))), flush=True)

if STATE.exists() and json.loads(STATE.read_text()).get("done"):
    training = json.loads(STATE.read_text())
    classifier = FineTunedClassifier(BATCH, LEARNING_RATE, SEED)
    classifier.model.load_state_dict(torch.load(BEST / "weights.pt", map_location="cuda", weights_only=True))
else:
    classifier = FineTunedClassifier(BATCH, LEARNING_RATE, SEED)
    BEST.mkdir(exist_ok=True)
    training = {"epochs": [], "best_macro_f1": -1.0, "best_epoch": None}
    for epoch in range(EPOCHS):
        tick = time.time()
        classifier.fit_epoch(train, weights, epoch)
        probabilities = classifier.predict(dev)
        choices = np.array(CATEGORIES, dtype=object)[probabilities.argmax(axis=1)]
        score = macro(dev.reference.to_numpy(dtype=object), choices)
        training["epochs"].append({"epoch": epoch, "dev_macro_f1": score, "seconds": round(time.time() - tick), "counts": pd.Series(choices).value_counts().to_dict()})
        print(f"epoch {epoch}: dev macro-F1 {score:.4f}, {time.time() - tick:.0f}s, {training['epochs'][-1]['counts']}", flush=True)
        if score > training["best_macro_f1"]:
            training.update(best_macro_f1=score, best_epoch=epoch)
            torch.save(classifier.model.state_dict(), BEST / "weights.pt")
        STATE.write_text(json.dumps(training))
    classifier.model.load_state_dict(torch.load(BEST / "weights.pt", map_location="cuda", weights_only=True))
    training["done"] = True
    training["loss_history"] = classifier.history
    STATE.write_text(json.dumps(training, indent=2))
print(f"best epoch {training['best_epoch']}, dev macro-F1 {training['best_macro_f1']:.4f}, elapsed {time.time() - started:.0f}s", flush=True)

test = pd.read_parquet("data/processed/escalation/classified_test.parquet")
target = OUT / "test_fine_tuned.npy"
if not target.exists():
    tick = time.time()
    np.save(target, classifier.predict(test, batch_size=32))
    print(f"test fine-tuned scored in {time.time() - tick:.0f}s", flush=True)
del classifier
torch.cuda.empty_cache()
target_embed = OUT / "test_zero_shot.npy"
if not target_embed.exists():
    tick = time.time()
    np.save(target_embed, zero_shot_probabilities(test, list(CRITERIA.values())))
    print(f"test zero-shot scored in {time.time() - tick:.0f}s", flush=True)
print(f"done, total {time.time() - started:.0f}s", flush=True)
