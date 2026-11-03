"""Load a local instruction model with 4-bit weights and generate one reply at a time. Nothing leaves the machine."""
import copy
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {"qwen": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"), "smollm": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1")}


class LocalLLM:
    def __init__(self, name="smollm", max_new_tokens=400, max_input_tokens=3000):
        self.repo, self.revision = MODELS[name]
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo, revision=self.revision, quantization_config=quantization, device_map={"": 0}, dtype=torch.float16)
        self.model.eval()
        self.prefix_ids, self.prefix_cache = None, None

    def _template(self, messages, tools):
        return self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _prepare_prefix(self, messages, tools):
        """Save the work on the instructions every prompt shares, because re-reading them took most of each turn."""
        first, second = [self.tokenizer(self._template([messages[0], {"role": "user", "content": word}], tools)).input_ids for word in ("first", "second")]
        length = next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)
        self.prefix_ids = first[:length]
        with torch.no_grad():
            self.prefix_cache = self.model(input_ids=torch.tensor([self.prefix_ids], device=self.model.device), use_cache=True).past_key_values

    def generate(self, messages, tools, use_prefix=True):
        """Return greedy output, so a repeat run gives the same text."""
        inputs = self.tokenizer(self._template(messages, tools), return_tensors="pt").to(self.model.device)
        length = inputs.input_ids.shape[1]
        if length > self.max_input_tokens:
            raise ValueError(f"prompt of {length} tokens is over the limit of {self.max_input_tokens}")
        started = time.time()
        extra = {}
        if use_prefix:
            if self.prefix_ids is None:
                self._prepare_prefix(messages, tools)
            if inputs.input_ids[0, :len(self.prefix_ids)].tolist() == self.prefix_ids:
                extra["past_key_values"] = copy.deepcopy(self.prefix_cache)
        with torch.no_grad():
            output = self.model.generate(**inputs, **extra, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id)
        created = output[0][length:]
        return {"text": self.tokenizer.decode(created, skip_special_tokens=True), "input_tokens": int(length), "output_tokens": int(len(created)), "seconds": time.time() - started}
