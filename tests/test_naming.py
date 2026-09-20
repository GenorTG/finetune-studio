"""Artifact naming scheme (Genor 2026-09-20).

Everything a user sees — model pickers, the running-model label, dataset
registry names — must read as  <Project> · <Base> [· version] · <Kind>
[· QUANT] [· abliterated], never a bare hash dir like ``6f64c46a/merged``.
"""
from __future__ import annotations

from finetune_studio import naming

# ── pure helpers ─────────────────────────────────────────────────────────


def test_detect_quant() -> None:
    assert naming.detect_quant("model-q5_k_m.gguf") == "Q5_K_M"
    assert naming.detect_quant("Qwen3.8-27B-abliterated-Q4_K_M.gguf") == "Q4_K_M"
    assert naming.detect_quant("model-q8_0.gguf") == "Q8_0"
    assert naming.detect_quant("model_f16.gguf") == "F16"
    assert naming.detect_quant("Qwen3-4B") is None
    assert naming.detect_quant("qwen3-4b-abliterated") is None


def test_detect_abliterated() -> None:
    assert naming.detect_abliterated("Qwen3.8-27B-abliterated-Q4_K_M") is True
    assert naming.detect_abliterated("Qwen3-4B") is False


def test_short_base() -> None:
    assert naming.short_base("Qwen/Qwen3-4B") == "Qwen3-4B"
    assert naming.short_base(
        "/home/x/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126"
    ) == "Qwen3-4B"
    assert naming.short_base("/home/g/.finetune-studio/hf_models/Qwen__Qwen3-4B") == "Qwen3-4B"
    assert naming.short_base("unsloth/Qwen3-4B") == "Qwen3-4B"


def test_model_full_name_scheme() -> None:
    s = naming.model_full_name("E2E Smoke", "Qwen3-4B", "v1 e2e-smoke-v1", "merged")
    assert s == "E2E Smoke · Qwen3-4B · v1 e2e-smoke-v1 · merged"
    s2 = naming.model_full_name("Vaelindrath", "Qwen3-4B", kind="GGUF",
                                quant="Q5_K_M", abliterated=True)
    assert s2 == "Vaelindrath · Qwen3-4B · GGUF · Q5_K_M · abliterated"


def test_kind_label() -> None:
    assert naming.kind_label("merged") == "merged"
    assert naming.kind_label("adapter") == "LoRA adapter"
    assert naming.kind_label("gguf") == "GGUF"
    assert naming.kind_label("mycustom") == "mycustom"


# ── DB-aware display ─────────────────────────────────────────────────────


def test_display_for_path_run_export(client) -> None:
    from finetune_studio import db

    proj = db.create_project(name="Naming Verify")
    pid = proj["id"]
    run = db.create_run(pid, "run-x", base_model="Qwen/Qwen3-4B")
    rid8 = run["id"][:8]
    out_rel = f"output/projects/{pid}/runs/{rid8}"
    db.update_run(run["id"], output_path=out_rel)

    disp = naming.display_for_path(f"/home/ops/finetune-studio/{out_rel}/merged")
    assert "Naming Verify" in disp, disp
    assert "Qwen3-4B" in disp, disp
    assert "merged" in disp, disp
    assert rid8 not in disp, disp  # no bare hash dirs in the name anymore


def test_display_for_path_with_pinned_version(client) -> None:
    from finetune_studio import db

    proj = db.create_project(name="Versioned Proj")
    pid = proj["id"]
    run = db.create_run(pid, "run-v", base_model="Qwen/Qwen3-4B")
    rid = run["id"]
    db.update_run(rid, output_path=f"output/projects/{pid}/runs/{rid[:8]}")
    db.create_version(pid, label="velmaris-cut",
                      manifest={"training_runs": [{"run_id": rid, "status": "done"}]})

    disp = naming.display_for_path(f"/abs/output/projects/{pid}/runs/{rid[:8]}/merged")
    assert "v1 velmaris-cut" in disp, disp
    assert "Versioned Proj" in disp, disp


def test_registry_lookup_uses_app_db(client) -> None:
    from finetune_studio import db
    from finetune_studio.models.registry import _lookup_project_name

    proj = db.create_project(name="Lookup Proj")
    pid = proj["id"]
    run = db.create_run(pid, "run-l", base_model="Qwen/Qwen3-4B")
    p2, name = _lookup_project_name(f"/abs/output/projects/{pid}/runs/{run['id'][:8]}/merged")
    assert p2 == pid
    assert name == "Lookup Proj"


def test_status_has_model_display(client, monkeypatch) -> None:
    from types import SimpleNamespace

    import finetune_studio.webui.app as app_mod
    from finetune_studio import db

    proj = db.create_project(name="Status Proj")
    pid = proj["id"]
    run = db.create_run(pid, "run-s", base_model="Qwen/Qwen3-4B")
    rid8 = run["id"][:8]
    db.update_run(run["id"], output_path=f"output/projects/{pid}/runs/{rid8}")

    eng = SimpleNamespace(
        model=object(),
        model_path=f"/srv/app/output/projects/{pid}/runs/{rid8}/merged",
        idle_seconds=1, is_gguf=False, vision=False,
    )
    monkeypatch.setattr(app_mod, "inference_engine", eng)
    r = client.get("/api/inference/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "model_display" in body
    assert "Status Proj" in body["model_display"], body["model_display"]
    assert "Qwen3-4B" in body["model_display"], body["model_display"]
    assert rid8 not in body["model_display"], body["model_display"]


def test_dataset_export_name_is_readable(client) -> None:
    import secrets

    from finetune_studio.data.fs import file_library as fl

    fl._PARSED_CACHE.clear()
    r = client.post("/api/projects", json={"name": "Readable DS"})
    pid = r.json()["id"]
    body = ("# Lore\n\n" + ("Keeper Odo swore on green wax in 1841. " * 60)).encode()
    up = client.post(
        f"/api/projects/{pid}/files/upload",
        files=[("files", (f"ds-{secrets.token_hex(3)}.md", body, "text/markdown"))],
    )
    assert up.status_code == 200, up.text
    ex = client.get(f"/api/projects/{pid}/data-prep/export")
    assert ex.status_code == 200, ex.text
    ds = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert ds, "export must register a dataset"
    name = ds[-1]["name"]
    assert "Readable DS" in name, name
    assert " · " in name, name
    assert pid not in name, name  # no bare pid hashes
