"""Registry scan + selector filtering for Inference/Training dropdowns.

After a host reset, selectors must list retained/installed chat models and
must not be poisoned by shared_models embedders, config-only hub leftovers,
or a scan abort on dangling weight symlinks that would omit later models.
"""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.models.registry import (
    ModelInfo,
    _safe_model_name,
    models_for_selectors,
    models_for_training,
    scan_models,
)

_QWEN_CFG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
_E5_CFG = {"model_type": "xlm-roberta", "architectures": ["XLMRobertaModel"]}
_HALF_GIB = 600 * 1024 * 1024


def _sparse_weight(path: Path, size: int) -> None:
    """Create a sparse file so size checks see real byte length without disk fill."""
    with open(path, "wb") as fh:
        if size > 0:
            fh.seek(size - 1)
            fh.write(b"\0")


def _write_model(dir_path: Path, cfg: dict, weight_name: str, weight_bytes: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _sparse_weight(dir_path / weight_name, weight_bytes)


def test_app_hf_cache_org_repo_display_name() -> None:
    name = _safe_model_name(
        "/home/u/.finetune-studio/hf_models/Qwen__Qwen3-4B",
        _QWEN_CFG,
    )
    assert name == "Qwen/Qwen3-4B"


def test_scan_skips_embedding_arch(tmp_path: Path) -> None:
    emb = tmp_path / "shared_models" / "embedders" / "intfloat__e5-large"
    _write_model(emb, _E5_CFG, "model.safetensors", _HALF_GIB)
    assert scan_models([str(tmp_path / "shared_models")]) == []


def test_scan_skips_broken_symlink_weights_and_keeps_sibling(tmp_path: Path) -> None:
    broken = tmp_path / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("a" * 40)
    broken.mkdir(parents=True)
    (broken / "config.json").write_text(json.dumps(_QWEN_CFG), encoding="utf-8")
    (broken / "model.safetensors").symlink_to(broken / "missing-blob")

    good = tmp_path / "hf_models" / "Qwen__Qwen3-4B"
    _write_model(good, _QWEN_CFG, "model.safetensors", _HALF_GIB)

    models = scan_models([str(tmp_path / "hub"), str(tmp_path / "hf_models")])
    paths = [m.path for m in models]
    assert str(good) in paths
    assert not any("Qwen3-0.6B" in p for p in paths)
    assert any(m.name == "Qwen/Qwen3-4B" for m in models)


def test_scan_skips_config_only_incomplete_hub_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "hub" / "models--Qwen--Qwen3-4B" / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(_QWEN_CFG), encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"")
    assert scan_models([str(tmp_path / "hub")]) == []


def test_models_for_selectors_drops_shared_models_keeps_installed() -> None:
    models = [
        ModelInfo(
            name="Qwen/Qwen3-4B",
            path="/home/u/.finetune-studio/hf_models/Qwen__Qwen3-4B",
            format="safetensors",
            size_gb=8.0,
            category="downloaded",
        ),
        ModelInfo(
            name="e5",
            path="/home/u/.finetune-studio/shared_models/embedders/e5",
            format="safetensors",
            size_gb=2.0,
            category="local_helper",
        ),
        ModelInfo(
            name="helper.gguf",
            path="models/gguf/helper.gguf",
            format="gguf",
            size_gb=15.0,
            category="discovered",
        ),
    ]
    selected = models_for_selectors(models)
    paths = [m.path for m in selected]
    assert models[0].path in paths
    assert models[2].path in paths
    assert models[1].path not in paths


def test_models_for_training_excludes_gguf_gptq_keeps_safetensors() -> None:
    models = [
        ModelInfo(
            name="Qwen/Qwen3-4B",
            path="/home/u/.finetune-studio/hf_models/Qwen__Qwen3-4B",
            format="safetensors",
            size_gb=8.0,
            category="downloaded",
        ),
        ModelInfo(
            name="helper.gguf",
            path="models/gguf/helper.gguf",
            format="gguf",
            size_gb=15.0,
            category="local_helper",
        ),
        ModelInfo(
            name="export-gptq",
            path="/tmp/out/run1/gptq",
            format="safetensors",
            size_gb=4.0,
            category="trained_export",
        ),
        ModelInfo(
            name="export-gguf-dir",
            path="/tmp/out/run1/gguf",
            format="safetensors",
            size_gb=4.0,
            category="trained_export",
        ),
        {
            "name": "dict-gptq",
            "path": "/tmp/out/run2/gptq",
            "format": "gptq",
            "category": "trained_export",
        },
    ]
    trainable = models_for_training(models)
    paths = [
        m.path if not isinstance(m, dict) else m["path"]
        for m in trainable
    ]
    assert models[0].path in paths
    assert models[1].path not in paths
    assert models[2].path not in paths
    assert models[3].path not in paths
    assert models[4]["path"] not in paths
    # Inference selector must still keep the helper GGUF.
    infer = models_for_selectors(models)
    infer_paths = [
        m.path if not isinstance(m, dict) else m["path"]
        for m in infer
    ]
    assert models[1].path in infer_paths


def test_training_js_uses_training_query() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "finetune_studio"
        / "webui"
        / "static"
        / "js"
        / "training.js"
    ).read_text(encoding="utf-8")
    assert "for_training=1" in src
    assert "for_selector=1" not in src


def test_training_hf_refresh_skips_incomplete_and_gguf() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "finetune_studio"
        / "webui"
        / "templates"
        / "project_training.html"
    ).read_text(encoding="utf-8")
    assert "if (!hasWeights) continue" in src
    assert "incomplete — no .safetensors" not in src
    assert "if (hasGguf) continue" in src
    assert "p.endsWith('.safetensors') || p.endsWith('.bin');" in src
    assert "No transformer-compatible base models found" in src
    assert "train-base-model-empty" in src
    assert "Only Transformers-compatible bases are listed" in src


def test_training_page_empty_copy_when_no_trainable(client, monkeypatch) -> None:
    from finetune_studio.webui import app as webapp

    monkeypatch.setattr(webapp, "discovered_models", [
        ModelInfo(
            name="helper.gguf",
            path="models/gguf/helper.gguf",
            format="gguf",
            size_gb=15.0,
            category="local_helper",
        ),
    ])
    r = client.post("/api/projects", json={"name": "Train Filter"})
    assert r.status_code in (200, 201), r.text
    pid = r.json()["id"]
    page = client.get(f"/projects/{pid}/training")
    assert page.status_code == 200
    body = page.text
    assert "No transformer-compatible base models found" in body
    assert "train-base-model-empty" in body
    assert "helper.gguf" not in body
    # Inference selector must still list GGUF helpers.
    assert any(
        m.format == "gguf" for m in models_for_selectors(webapp.discovered_models)
    )
    api = client.get("/api/models?for_training=1")
    assert api.status_code == 200
    assert api.json() == []
    api_inf = client.get("/api/models?for_selector=1")
    assert api_inf.status_code == 200
    assert any(m.get("format") == "gguf" for m in api_inf.json())
