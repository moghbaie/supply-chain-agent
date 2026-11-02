"""Print the Step 1 environment report and save it to reports/step_01/."""
import importlib
import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

PACKAGES = ["torch", "numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "transformers", "peft", "trl",
            "datasets", "accelerate", "tabpfn", "kumoai", "kaggle", "soup-cli", "airllm", "pytest", "python-dotenv"]
IMPORTS = ["torch", "numpy", "pandas", "pyarrow", "sklearn", "matplotlib", "transformers", "peft", "trl",
           "datasets", "accelerate", "tabpfn", "kumoai", "soup_cli", "airllm", "pytest", "dotenv"]


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def try_import(name):
    try:
        importlib.import_module(name)
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:160]}"


report = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None,
    "disk_free_gb": round(shutil.disk_usage(".").free / 1024**3, 1),
    "packages": {p: package_version(p) for p in PACKAGES},
    "imports": {m: try_import(m) for m in IMPORTS},
}
print(json.dumps(report, indent=2))
out = Path("reports/step_01")
out.mkdir(parents=True, exist_ok=True)
(out / "environment.json").write_text(json.dumps(report, indent=2))
