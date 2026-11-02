"""Check each key in .env.secrets against its service and print only outcomes, never values.

Each check also runs with a fake key, because some endpoints answer 200 without valid credentials.
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values
from tabpfn.browser_auth import _get_license_name, check_license_accepted, verify_token

SECRETS = {k: v or "" for k, v in dotenv_values(".env.secrets").items()}
FAKE = "KGAT_00000000000000000000000000000000"
TABPFN_API = "https://api.priorlabs.ai"
TABPFN_VERSIONS = ["tabpfn_2_5", "tabpfn_2_6", "tabpfn_3", "tabpfn_3_5"]
M5_FILE = "https://www.kaggle.com/api/v1/competitions/data/download/m5-forecasting-accuracy/calendar.csv"
M5_LIST = "https://www.kaggle.com/api/v1/competitions/data/list/m5-forecasting-accuracy"


def redact(text):
    for value in SECRETS.values():
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def http(url, headers, body=None):
    data = json.dumps(body).encode() if body else None
    request = urllib.request.Request(url, data=data, headers={**headers, "User-Agent": "sca-key-check"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.status, response.read(600).decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(600).decode(errors="replace")
    except Exception as exc:
        return None, type(exc).__name__


def hugging_face(key):
    status, _ = http("https://huggingface.co/api/whoami-v2", {"Authorization": f"Bearer {key}"})
    return status == 200, f"HTTP {status}"


def kaggle(key):
    status, _ = http(M5_LIST, {"Authorization": f"Bearer {key}"})
    return status == 200, f"M5 file list HTTP {status}"


def tabpfn(key):
    valid = verify_token(key, TABPFN_API)
    return valid is True, f"verify_token={valid}"


def kumo(key):
    result = subprocess.run([sys.executable, "-W", "ignore", "scripts/kumo_probe.py"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=400,
                            env={**os.environ, "KUMO_API_KEY": key, "PYTHONIOENCODING": "utf-8"})
    output = redact(result.stdout + result.stderr).strip().splitlines()
    return "PREDICTION_OK" in result.stdout, f"exit {result.returncode}: {output[-1][:200] if output else ''}"


def typesafe(key):
    body = {"model": "jev-latest", "state": "The shipment arrived on time.",
            "questions": {"ontime": {"type": "noul", "instructions": "Did the shipment arrive on time?"}}}
    status, text = http("https://api.typesafe.ai/v1/systemone",
                        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
    return status == 200, f"HTTP {status} {redact(text)[:120] if status == 200 else ''}"


CHECKS = [("HF_TOKEN", hugging_face), ("KAGGLE_KEY", kaggle), ("TABPFN_TOKEN", tabpfn),
          ("KUMO_API_KEY", kumo), ("TYPESAFE_API_KEY", typesafe)]

report = {}
for name, check in CHECKS:
    real_ok, real_detail = check(SECRETS[name])
    fake_ok, fake_detail = check(FAKE)
    same_failure = not real_ok and not fake_ok and real_detail == fake_detail
    verdict = "INCONCLUSIVE" if same_failure or (real_ok and fake_ok) else "WORKING" if real_ok else "NOT WORKING"
    report[name] = {"verdict": verdict, "real": real_detail, "fake": fake_detail}
    print(f"{name:17} {verdict}\n    real: {real_detail}\n    fake: {fake_detail}")

m5_status, m5_body = http(M5_FILE, {"Authorization": f"Bearer {SECRETS['KAGGLE_KEY']}"})
report["kaggle_m5_download"] = f"HTTP {m5_status}"
print(f"Kaggle M5 download permission: HTTP {m5_status} {redact(m5_body)[:110] if m5_status != 200 else ''}")
licenses = {r: check_license_accepted(SECRETS["TABPFN_TOKEN"], TABPFN_API, _get_license_name(r)) for r in TABPFN_VERSIONS}
report["tabpfn_licenses_accepted"] = licenses
print(f"TabPFN licenses accepted: {licenses}")
Path("reports/step_01/api_key_check.json").write_text(redact(json.dumps(report, indent=2)))
