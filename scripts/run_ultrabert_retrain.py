"""Retrain UltraBERT with stronger regularization and a more realistic mix. Leaves the agent's saved model untouched, runs locally and resumes."""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.ultrabert import ENCODER, FineTunedClassifier, balanced_training_rows, case_texts, class_weights

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SMOKE = "--smoke" in sys.argv
SEED, MAX_EPOCHS, PATIENCE, BATCH, ACCUMULATE = 42, 4, 1, 8, 2
PEAK_LR, WARMUP_SHARE, WEIGHT_DECAY, CLASSIFIER_DROPOUT, ATTENTION_DROPOUT = 2e-5, 0.1, 0.05, 0.1, 0.0
TRAIN_ROWS, MONITOR_PER_STOCKOUT = 16000, 3
TRAIN_LAST_ORIGIN, LATE_FIRST_ORIGIN = pd.Timestamp("2015-09-14"), pd.Timestamp("2015-10-01")
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "data/processed/escalation/ultrabert_retrain"
OUT.mkdir(parents=True, exist_ok=True)
STATE, BEST, LAST = OUT / "training_state.json", OUT / "best_weights.pt", OUT / "last_checkpoint.pt"
ORDER = np.array(CATEGORIES, dtype=object)
if SMOKE:
    MAX_EPOCHS, TRAIN_ROWS = 2, 48


class RetrainClassifier(FineTunedClassifier):
    def __init__(self):
        torch.manual_seed(SEED)
        self.batch_size, self.device, self.seed = BATCH, "cuda", SEED
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.model = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=len(CATEGORIES), attn_implementation="sdpa",
                                                                        classifier_dropout=CLASSIFIER_DROPOUT, attention_dropout=ATTENTION_DROPOUT).to(self.device)
        assert self.model.config.classifier_dropout == CLASSIFIER_DROPOUT and self.model.drop.p == CLASSIFIER_DROPOUT
        self.history = []


def tune_rows(late):
    """Use the same series split as the decision-rule work, so the other half stays unused here."""
    series = np.sort(late.id.astype(str).unique())
    tune_series = set(np.random.default_rng(SEED).choice(series, len(series) // 2, replace=False))
    return late[late.id.astype(str).isin(tune_series)].reset_index(drop=True)


def macro(reference, choices):
    return float(f1_score(reference, choices, labels=CATEGORIES, average="macro", zero_division=0))


cases, _ = build_cases()
year = cases[cases.split == "validation"]
pool = balanced_training_rows(year[year.origin <= TRAIN_LAST_ORIGIN], MONITOR_PER_STOCKOUT, SEED)
train = pool.sample(min(TRAIN_ROWS, len(pool)), random_state=SEED).reset_index(drop=True)
late = year[year.origin >= LATE_FIRST_ORIGIN].reset_index(drop=True)
selection = tune_rows(late)
if SMOKE:
    selection = selection.head(64)
weights = class_weights(train.reference)
print(f"train {len(train):,}, last origin {train.origin.max().date()}, mix {train.reference.value_counts().to_dict()}", flush=True)
print(f"selection {len(selection):,}, first origin {selection.origin.min().date()}, mix {selection.reference.value_counts().to_dict()}", flush=True)
print("class weights", dict(zip(CATEGORIES, np.round(weights, 2).tolist())), flush=True)

classifier = RetrainClassifier()
model = classifier.model
steps_per_epoch = math.ceil(math.ceil(len(train) / BATCH) / ACCUMULATE)
total_steps = steps_per_epoch * MAX_EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda")
training = {"epochs": [], "best_macro_f1": -1.0, "best_epoch": None, "done": False}
if STATE.exists() and LAST.exists():
    training = json.loads(STATE.read_text())
    saved = torch.load(LAST, map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    torch.set_rng_state(saved["rng"].cpu())
    torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    print(f"resuming after epoch {training['epochs'][-1]['epoch']}", flush=True)

texts = case_texts(train)
target = np.array([CATEGORIES.index(c) for c in train.reference])
weight = torch.tensor(weights, dtype=torch.float32, device="cuda")
started = time.time()
while not training["done"]:
    epoch = len(training["epochs"])
    tick, running, count = time.time(), 0.0, 0
    order = np.random.default_rng(SEED + epoch).permutation(len(train))
    model.train()
    optimizer.zero_grad()
    batches = [order[first:first + BATCH] for first in range(0, len(order), BATCH)]
    for number, picks in enumerate(batches, 1):
        encoded = classifier._encode([texts[i] for i in picks])
        labels = torch.tensor(target[picks], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(**encoded).logits, labels, weight=weight)
        scaler.scale(loss / ACCUMULATE).backward()
        running, count = running + float(loss), count + 1
        if number % ACCUMULATE == 0 or number == len(batches):
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        if number % 250 == 0 or number == len(batches):
            print(f"  epoch {epoch} batch {number}/{len(batches)}, loss {running / count:.3f}, lr {scheduler.get_last_lr()[0]:.2e}, {time.time() - tick:.0f}s", flush=True)
            running, count = 0.0, 0
    probabilities = classifier.predict(selection, batch_size=32)
    choices = ORDER[probabilities.argmax(axis=1)]
    score = macro(selection.reference.to_numpy(dtype=object), choices)
    counts = pd.Series(choices).value_counts().to_dict()
    training["epochs"].append({"epoch": epoch, "selection_macro_f1": score, "seconds": round(time.time() - tick), "counts": counts})
    print(f"epoch {epoch}: selection macro-F1 {score:.4f}, {time.time() - tick:.0f}s, {counts}", flush=True)
    if score > training["best_macro_f1"]:
        training.update(best_macro_f1=score, best_epoch=epoch)
        torch.save(model.state_dict(), BEST)
    if epoch + 1 >= MAX_EPOCHS or epoch - training["best_epoch"] >= PATIENCE:
        training["done"] = True
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, LAST)
    STATE.write_text(json.dumps(training, indent=2))
    model.train()
print(f"best epoch {training['best_epoch']}, selection macro-F1 {training['best_macro_f1']:.4f}", flush=True)
if SMOKE:
    sys.exit(0)

LAST.unlink(missing_ok=True)
model.load_state_dict(torch.load(BEST, map_location="cuda", weights_only=True))
late_path, test_path = OUT / "late_2015_probabilities.npy", OUT / "test_probabilities.npy"
if not late_path.exists():
    tick = time.time()
    np.save(late_path, classifier.predict(late, batch_size=32))
    print(f"late 2015 scored in {time.time() - tick:.0f}s", flush=True)
if not test_path.exists():
    tick = time.time()
    test = pd.read_parquet("data/processed/escalation/classified_test.parquet")
    np.save(test_path, classifier.predict(test, batch_size=32))
    print(f"test scored in {time.time() - tick:.0f}s", flush=True)
print("done", flush=True)
