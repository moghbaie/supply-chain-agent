"""Download M5 and Olist from Kaggle into separate raw folders and record file checksums."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import dotenv_values

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECRETS = dotenv_values(".env.secrets")
ENV = {**{k: v for k, v in os.environ.items() if not k.startswith("KAGGLE")},
       "KAGGLE_API_TOKEN": SECRETS["KAGGLE_KEY"], "PYTHONIOENCODING": "utf-8"}
KAGGLE = os.path.abspath(".venv/Scripts/kaggle.exe")
SOURCES = {
    "m5": ["competitions", "download", "m5-forecasting-accuracy", "-p", "data/raw/m5"],
    "olist": ["datasets", "download", "olistbr/brazilian-ecommerce", "-p", "data/raw/olist"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {}
for name, args in SOURCES.items():
    folder = Path("data/raw") / name
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    output = (result.stdout + result.stderr).replace(SECRETS["KAGGLE_KEY"], "[REDACTED]").strip().splitlines()
    print(f"{name}: exit {result.returncode} | {output[-1][:170] if output else ''}")
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(folder)
        archive.unlink()
    manifest[name] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(folder.iterdir()) if p.is_file()}
    for file_name, info in manifest[name].items():
        print(f"  {file_name:45} {info['bytes']:>12,} bytes")

Path("reports/step_02/manifest.json").write_text(json.dumps(manifest, indent=2))
sys.exit(0 if all(manifest.values()) else 1)
