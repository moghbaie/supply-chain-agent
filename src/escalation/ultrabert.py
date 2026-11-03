"""Local UltraBERT classifiers for escalation cases. Nothing here sends data anywhere.

Cases are written as compact sentences carrying the same facts Jev saw, because JSON punctuation used most of the tokens.
"""
import math
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.escalation.cases import CATEGORIES

ENCODER = "dmis-lab/UltraBERT-150M"
EMBEDDER = "dmis-lab/UltraBERT-Embed-150M"
MAX_TOKENS = 110
EMBED_PROMPT = "Instruct: Choose the inventory review outcome\nQuery: "


def _value(number, digits=0):
    if number is None or (isinstance(number, float) and math.isnan(number)):
        return "unknown"
    return f"{float(number):.{digits}f}"


def case_text(row):
    """Write one case as a sentence, with comparisons already computed."""
    if row.days_to_next_due is None or (isinstance(row.days_to_next_due, float) and math.isnan(row.days_to_next_due)):
        lasts = "no delivery date due"
    else:
        lasts = "stock lasts until delivery" if row.cover_days >= row.days_to_next_due else "stock runs out before delivery"
    return (f"On hand {int(row.on_hand)} units, {_value(row.cover_days, 1)} days cover. On order {int(row.on_order)} in {int(row.open_pos)} orders, "
            f"new order {'yes' if row.ordered_today > 0 else 'no'}. Forecast {_value(row.fc_mean_14)} units, high {_value(row.fc_q90_sum_14)}. "
            f"Next delivery {_value(row.days_to_next_due)} days, {_value(row.units_due_14)} units due, {_value(row.units_overdue)} overdue by {_value(row.days_overdue_max)} days; {lasts}. "
            f"Supplier late {_value(100 * row.supplier_late_rate)}% long run, {_value(100 * row.supplier_recent_late_rate)}% recent. "
            f"Stockout chance {_value(100 * row.stockout_risk)}% and {_value(100 * row.stockout_risk_lightgbm)}%. Value {row.value_band}.")


def case_texts(frame):
    return [case_text(row) for row in frame.itertuples(index=False)]


def class_weights(reference):
    """Weight categories inversely to their frequency, limiting the largest weight."""
    counts = pd.Series(reference).value_counts().reindex(CATEGORIES).fillna(1).to_numpy(dtype=float)
    weights = counts.sum() / (len(CATEGORIES) * counts)
    return np.minimum(weights, 10.0)


def balanced_training_rows(frame, monitor_per_stockout, seed):
    """Keep every stockout case and sample monitor cases, so training time goes to the rare categories."""
    stockouts = frame[frame.reference != "monitor"]
    monitors = frame[frame.reference == "monitor"]
    sampled = monitors.sample(min(len(monitors), int(len(stockouts) * monitor_per_stockout)), random_state=seed)
    return pd.concat([stockouts, sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class FineTunedClassifier:
    def __init__(self, batch_size, learning_rate, seed, device="cuda"):
        torch.manual_seed(seed)
        self.batch_size, self.learning_rate, self.device, self.seed = batch_size, learning_rate, device, seed
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.model = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=len(CATEGORIES), attn_implementation="sdpa").to(device)
        self.history = []

    def _encode(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(self.device)

    def fit_epoch(self, frame, weights, epoch, log_every=500):
        """Train one pass, with mixed precision and a weighted loss."""
        order = np.random.default_rng(self.seed + epoch).permutation(len(frame))
        texts = case_texts(frame)
        target = np.array([CATEGORIES.index(c) for c in frame.reference])
        weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scaler = torch.amp.GradScaler("cuda")
        self.model.train()
        started, running = time.time(), 0.0
        for number, first in enumerate(range(0, len(order), self.batch_size), 1):
            picks = order[first:first + self.batch_size]
            encoded = self._encode([texts[i] for i in picks])
            labels = torch.tensor(target[picks], device=self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.model(**encoded).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running += float(loss)
            if number % log_every == 0:
                print(f"  epoch {epoch} batch {number}, loss {running / log_every:.3f}, {time.time() - started:.0f}s", flush=True)
                self.history.append({"epoch": epoch, "batch": number, "loss": running / log_every})
                running = 0.0

    def predict(self, frame, batch_size=None):
        """Return probabilities per category, processing shorter texts together to save memory."""
        batch_size = batch_size or self.batch_size * 4
        texts = case_texts(frame)
        order = np.argsort([len(t) for t in texts])
        output = np.zeros((len(texts), len(CATEGORIES)), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for first in range(0, len(order), batch_size):
                picks = order[first:first + batch_size]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**self._encode([texts[i] for i in picks])).logits
                output[picks] = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return output


def zero_shot_probabilities(frame, label_texts, batch_size=64, embedder=None, device="cuda"):
    """Return category similarities from the embedding model, with no training."""
    from sentence_transformers import SentenceTransformer

    embedder = embedder or SentenceTransformer(EMBEDDER, device=device, model_kwargs={"attn_implementation": "sdpa"})
    embedder.max_seq_length = MAX_TOKENS
    queries = embedder.encode(case_texts(frame), prompt=EMBED_PROMPT, batch_size=batch_size, show_progress_bar=False)
    labels = embedder.encode(label_texts, prompt="Label: ", show_progress_bar=False)
    return embedder.similarity(queries, labels).cpu().numpy()
