"""Run Step 1 smoke checks on a tiny random model and save the results."""
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
results = {"model": MODEL, "device": DEVICE}


def draw(seed):
    set_seed(seed)
    return [np.random.rand(3).tolist(), torch.rand(3).tolist(), torch.rand(3, device=DEVICE).tolist()]


results["seed_same_repeatable"] = draw(42) == draw(42)
results["seed_different_differs"] = draw(42) != draw(43)

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
ids = tokenizer("reorder point", return_tensors="pt").to(DEVICE)
results["load_generate_seconds"] = None
output = model.generate(**ids, max_new_tokens=5, do_sample=False)
results["load_generate_seconds"] = round(time.time() - start, 2)
results["generated_tokens"] = output[0].tolist()

lora = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
results["lora_trainable_params"], results["total_params"] = lora.get_nb_trainable_parameters()
optimizer = torch.optim.AdamW((p for p in lora.parameters() if p.requires_grad), lr=1e-2)
batch = {**ids, "labels": ids["input_ids"]}
losses = []
for _ in range(5):
    loss = lora(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(round(loss.item(), 4))
results["lora_losses"] = losses
results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if DEVICE == "cuda" else None

try:
    from airllm import AutoModel

    air = AutoModel.from_pretrained(MODEL)
    air_ids = air.tokenizer("reorder point", return_tensors="pt").input_ids.to(DEVICE)
    results["airllm_output"] = air.generate(air_ids, max_new_tokens=3, use_cache=True).tolist()
except Exception as exc:
    results["airllm_error"] = f"{type(exc).__name__}: {exc}"[:400]
    results["airllm_trace_tail"] = traceback.format_exc()[-600:]

print(json.dumps(results, indent=2))
Path("reports/step_01/smoke.json").write_text(json.dumps(results, indent=2))
