"""Measure UltraBERT memory and speed on the 4 GB GPU using 2015 cases. Everything runs locally."""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.escalation.cases import CATEGORIES, build_cases
from src.escalation.jev import CRITERIA, case_state

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEED = 42
cases, _ = build_cases()
rows = cases[cases.split == "validation"].sample(1000, random_state=SEED)
texts = [json.dumps(case_state(row)) for row in rows.itertuples(index=False)]
truth = rows.reference.to_numpy(dtype=object)
print("example text:", texts[0][:300], flush=True)
results = {}

torch.cuda.reset_peak_memory_stats()
embedder = SentenceTransformer("dmis-lab/UltraBERT-Embed-150M", device="cuda", model_kwargs={"attn_implementation": "sdpa"})
print("embed: max length", embedder.max_seq_length, "| prompts", list((embedder.prompts or {}).keys()), flush=True)
started = time.time()
queries = embedder.encode(texts, prompt="Instruct: Choose the inventory review outcome\nQuery: ", batch_size=64, show_progress_bar=False)
labels = embedder.encode([f"{name}: {text}" for name, text in CRITERIA.items()], prompt="Label: ", show_progress_bar=False)
similarity = embedder.similarity(queries, labels).cpu().numpy()
elapsed = time.time() - started
choice = np.array(list(CRITERIA), dtype=object)[similarity.argmax(axis=1)]
results["embed"] = {"ms_per_case": round(1000 * elapsed / len(texts), 2), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
                    "macro_f1": round(f1_score(truth, choice, labels=CATEGORIES, average="macro", zero_division=0), 4),
                    "counts": pd.Series(choice).value_counts().to_dict(), "similarity_range": [round(float(similarity.min()), 3), round(float(similarity.max()), 3)]}
print("embed", results["embed"], flush=True)
del embedder
gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/UltraBERT-150M")
lengths = [len(tokenizer(text).input_ids) for text in texts[:200]]
print("tokens per case: mean", round(float(np.mean(lengths)), 1), "max", max(lengths), flush=True)
for batch_size in [32, 16, 8]:
    try:
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/UltraBERT-150M", num_labels=4, attn_implementation="sdpa").to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        started = time.time()
        steps = 10
        for step in range(steps):
            encoded = tokenizer(texts[step * batch_size:(step + 1) * batch_size], padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
            target = torch.tensor([CATEGORIES.index(c) for c in truth[step * batch_size:(step + 1) * batch_size]], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**encoded, labels=target).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_speed = 1000 * (time.time() - started) / (steps * batch_size)
        model.eval()
        started = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            encoded = tokenizer(texts[:128], padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
            model(**encoded)
        results[f"fine_tune_batch_{batch_size}"] = {"train_ms_per_case": round(train_speed, 1), "predict_ms_per_case": round(1000 * (time.time() - started) / 128, 2),
                                                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2), "last_loss": round(float(loss), 3),
                                                    "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
        print(f"batch {batch_size}", results[f"fine_tune_batch_{batch_size}"], flush=True)
        del model, optimizer, encoded, target, loss
        gc.collect()
        torch.cuda.empty_cache()
        break
    except torch.cuda.OutOfMemoryError as error:
        results[f"fine_tune_batch_{batch_size}"] = {"error": str(error)[:120]}
        print(f"batch {batch_size}: out of memory", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
Path("reports/step_07/ultrabert_probe.json").write_text(json.dumps(results, indent=2, default=str))
