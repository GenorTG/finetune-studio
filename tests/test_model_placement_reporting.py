"""The load and status endpoints report where a model really went (autofit), and the UI says so honestly."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

TEMPLATE = (Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
            / "inference.html")
PLACED = {"mode": "partial", "gpu_layers": 14, "total_layers": 48,
          "warnings": ["Only 14/48 layers of m.gguf are on the GPU (the rest run on the CPU)"]}


@pytest.fixture
def engine(monkeypatch):
    eng = MagicMock()
    eng.model = object()
    eng.model_path = "/m/m.gguf"
    eng.vision = False
    eng.is_gguf = True
    eng.idle_seconds = 0
    eng.n_ctx = 32768
    eng.n_gpu_layers = 14
    eng.offload = dict(PLACED)
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine", eng)
    return eng


def test_load_response_reports_context_layers_and_warnings(client, engine, tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    r = client.post("/api/models/load", json={"path": str(model), "n_ctx": 32768})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_ctx"] == 32768 and body["n_gpu_layers"] == 14
    assert body["offload"]["mode"] == "partial" and body["warnings"] == PLACED["warnings"]


def test_status_endpoint_exposes_the_placement(client, engine) -> None:
    body = client.get("/api/inference/status").json()
    assert body["n_ctx"] == 32768 and body["n_gpu_layers"] == 14 and body["offload"]["gpu_layers"] == 14


def test_status_is_empty_placement_when_nothing_is_loaded(client, engine) -> None:
    engine.model = None
    body = client.get("/api/inference/status").json()
    assert body["n_ctx"] is None and body["offload"] == {}


def test_inference_page_no_longer_claims_mixed_offload_is_unsupported() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "not supported by design" not in html and "shrinks context length" not in html
    assert "never reduced" in html and "slider.disabled = false" in html
    assert "placementWarnings" in html   # load warnings reach the user
