"""Step 1 tests: imports, compatibility, minimal model loading and seeding."""
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[2]
MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SECRET_NAMES = {"HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "TABPFN_TOKEN", "KUMO_API_KEY", "TYPESAFE_API_KEY"}


def draw(seed):
    set_seed(seed)
    return [np.random.rand(3).tolist(), torch.rand(3).tolist(), torch.rand(3, device=DEVICE).tolist()]


def test_python_matches_pin():
    assert platform.python_version() == (ROOT / ".python-version").read_text().strip()


def test_running_inside_project_venv():
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


@pytest.mark.parametrize("module", ["torch", "numpy", "pandas", "pyarrow", "sklearn", "matplotlib", "transformers",
                                    "peft", "trl", "datasets", "accelerate", "tabpfn", "kumoai", "kaggle", "soup_cli",
                                    "airllm", "dotenv"])
def test_import(module):
    importlib.import_module(module)


def test_dependencies_compatible():
    result = subprocess.run(["uv", "pip", "check", "--python", sys.executable], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_torch_uses_gpu():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).total_memory > 0


def test_same_seed_repeats():
    assert draw(42) == draw(42)


def test_different_seed_differs():
    assert draw(42) != draw(43)


def test_minimal_model_generates_repeatably():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    ids = tokenizer("reorder point", return_tensors="pt").to(DEVICE)
    first = model.generate(**ids, max_new_tokens=5, do_sample=False)
    second = model.generate(**ids, max_new_tokens=5, do_sample=False)
    assert first.shape[1] == ids["input_ids"].shape[1] + 5
    assert torch.equal(first, second)


def test_soup_adapter_reloads():
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    adapter = ROOT / "reports/step_01/soup_tiny/output"
    model = PeftModel.from_pretrained(base, adapter)
    assert any("lora_" in name for name, _ in model.named_parameters())


def test_smoke_results_recorded_airllm_output():
    smoke = json.loads((ROOT / "reports/step_01/smoke.json").read_text())
    assert "airllm_error" not in smoke
    assert len(smoke["airllm_output"][0]) > 0


def test_secrets_file_has_expected_names():
    assert set(dotenv_values(ROOT / ".env.secrets")) == SECRET_NAMES


def test_secrets_file_is_ignored():
    assert ".env.secrets" in (ROOT / ".gitignore").read_text().splitlines()
