"""Kumo and Jev sending stays off, and the key check contacts neither. No test calls an API."""
import runpy
import urllib.request
from pathlib import Path

import pytest

from src.escalation import jev
from src.supplier import kumo_client

ROOT = Path(__file__).resolve().parents[2]


def no_network(*args, **kwargs):
    raise AssertionError("tried to reach the network")


def test_kumo_sending_is_off():
    assert kumo_client.SENDING_ENABLED is False


def test_jev_sending_is_off():
    assert jev.SENDING_ENABLED is False


@pytest.mark.parametrize("module", [kumo_client, jev])
def test_send_refuses_before_reaching_the_network(module, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(RuntimeError, match="off"):
        module.send({}, "not a key")


def test_kumo_probe_stops_before_contacting_kumo():
    with pytest.raises(SystemExit, match="off"):
        runpy.run_path(str(ROOT / "scripts/kumo_probe.py"))


@pytest.mark.parametrize("term", ["api.typesafe", "TYPESAFE_API_KEY", "KUMO_API_KEY", "kumo_probe", "ai.api.nvidia"])
def test_key_check_does_not_contact_kumo_or_jev(term):
    assert term not in (ROOT / "scripts/check_api_keys.py").read_text(encoding="utf-8")
