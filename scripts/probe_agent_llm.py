"""Check that a 4-bit model fits the GPU and can call tools, using a hand-written question and no case data."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.agent.llm import LocalLLM
from src.agent.runner import SYSTEM, parse_tool_calls
from src.agent.tools import tool_schemas

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="smollm")
name = parser.parse_args().model

started = time.time()
llm = LocalLLM(name)
loaded = torch.cuda.memory_allocated() / 2**30
print(f"{llm.repo}: loaded in {time.time() - started:.0f}s, allocated {loaded:.2f} GB, footprint {llm.model.get_memory_footprint() / 2**30:.2f} GB, devices {set(str(p.device) for p in llm.model.parameters())}", flush=True)
torch.cuda.reset_peak_memory_stats()
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Review case_id probe-0001."}]
output = llm.generate(messages, tool_schemas())
print(f"input tokens {output['input_tokens']}, output tokens {output['output_tokens']}, {output['seconds']:.1f}s, {output['output_tokens'] / output['seconds']:.1f} tokens per second")
print("MODEL TEXT\n" + output["text"])
print("PARSED", json.dumps(parse_tool_calls(output["text"])))
again = llm.generate(messages, tool_schemas())
plain = llm.generate(messages, tool_schemas(), use_prefix=False)
print(f"with saved prefix {again['seconds']:.1f}s, without {plain['seconds']:.1f}s, same text {again['text'] == plain['text'] == output['text']}, prefix tokens {len(llm.prefix_ids)}")
peak = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30
print(f"peak allocated {peak:.2f} GB, peak reserved {reserved:.2f} GB of {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GB")
