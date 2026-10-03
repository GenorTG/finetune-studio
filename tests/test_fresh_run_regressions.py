"""Base-model benchmark results must appear on the Benchmarks page.

Found in the 2026-10-02 fresh-DB browser run: RUN on the "base model
(untrained)" row stored its scores against the hidden ``__base_model__``
placeholder run, and the page filtered that run out, so the user saw
"No benchmark scores yet" after a successful run.
"""

from __future__ import annotations

from finetune_studio import db


def test_base_model_benchmark_listed_on_page(client) -> None:
    project = db.create_project("Base bench", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    base_run = db.create_run(pid, "__base_model__", base_model="Qwen/Qwen3-0.6B")
    db.update_run(base_run["id"], status="done", output_path="", notes="placeholder")
    db.create_benchmark(
        base_run["id"], "suite-x", {"pass_rate": 9.5, "weighted_score": 9.5}, 100,
        cases=[], model_path="Qwen/Qwen3-0.6B",
    )

    html = client.get(f"/projects/{pid}/benchmarks").text

    assert "No benchmark scores yet" not in html
    assert "base model (untrained)" in html
    assert "suite-x" in html


def test_base_probe_placeholder_hidden_from_run_lists(client) -> None:
    project = db.create_project("Probe hide", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    real = db.create_run(pid, "real run", base_model="Qwen/Qwen3-0.6B")
    probe = db.create_run(pid, "__base_model__", base_model="Qwen/Qwen3-0.6B")

    assert [r["id"] for r in db.list_runs(pid)] == [real["id"]]
    assert probe["id"] in {r["id"] for r in db.list_runs(pid, include_base_probe=True)}
    html = client.get(f"/projects/{pid}/training").text
    assert "__base_model__" not in html


def test_chat_context_offers_tuned_merged_model(client, tmp_path) -> None:
    """Chat must offer the project's fine-tuned (merged) model, not only bases."""
    project = db.create_project("Chat tuned", base_model="Qwen/Qwen3-0.6B")
    pid = project["id"]
    out = tmp_path / "run"
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "config.json").write_text("{}")
    (out / "merged" / "model.safetensors").write_bytes(b"x" * 16)
    run = db.create_run(pid, "tuned", base_model="Qwen/Qwen3-0.6B")
    db.update_run(run["id"], status="done", output_path=str(out))
    db.update_project(pid, production_run=run["id"])

    body = client.get(f"/api/chat-v2/projects/{pid}/context").json()

    assert body["tuned_model_path"] == str(out / "merged")


def test_export_worker_refreshes_model_registry(monkeypatch):
    from finetune_studio.webui.routes import exports, models

    calls = []
    monkeypatch.setattr(models, "refresh_model_registry", lambda: calls.append(1) or 0)
    exports._refresh_registry_quietly()
    assert calls == [1]


def test_rag_settings_before_index_is_not_a_404(client):
    pid = client.post("/api/projects", json={"name": "Rag Fresh"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/rag/settings")
    assert r.status_code == 200
    assert r.json() == {"exists": False}


def test_unknown_page_gets_styled_404_but_api_stays_json(client):
    page = client.get("/projects/nope/files", headers={"accept": "text/html"})
    assert page.status_code == 404
    assert "Page not found" in page.text
    api = client.get("/api/nope", headers={"accept": "text/html"})
    assert api.status_code == 404
    assert api.json() == {"detail": "Not Found"}
    bare = client.get("/projects/nope/files")
    assert bare.json() == {"detail": "Not Found"}


def test_project_overview_links_and_base_model_badge(client):
    pid = client.post("/api/projects", json={"name": "Links"}).json()["id"]
    from finetune_studio import db

    db.update_project(pid, base_model="/x/hf_models/Qwen__Qwen3-0.6B")
    page = client.get(f"/projects/{pid}").text
    assert f'href="/projects/{pid}/data" data-link>\n    <div class="label">Files' in page
    assert f"/projects/{pid}/data-prep#upload" not in page
    assert "base: Qwen/Qwen3-0.6B" in client.get("/projects").text


def test_rag_build_indexes_uploaded_txt_once(client, monkeypatch, tmp_path):
    import numpy as np

    from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo

    def fake(name="fake", device="cpu"):
        def encode(t):
            items = [t] if isinstance(t, str) else list(t)
            out = np.ones((len(items), 8), dtype=np.float32) / (8 ** 0.5)
            return out[0] if isinstance(t, str) else out
        return encode, EmbeddingModelInfo(name="fake", dim=8, normalize=True,
                                          distance="cosine", cached_at="1970-01-01T00:00:00Z")

    monkeypatch.setattr("finetune_studio.data.rag_portable.store.get_embedder", fake)
    root = tmp_path / "fts"
    (root / "projects").mkdir(parents=True)
    monkeypatch.setattr("finetune_studio.data.fs.paths._ROOT", root)
    monkeypatch.setattr("finetune_studio.data.fs.paths._PROJECTS", root / "projects")
    monkeypatch.setattr("finetune_studio.webui.routes.rag._corpus_dir",
                        lambda p: tmp_path / "corpora" / p)
    pid = client.post("/api/projects", json={"name": "Once"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/files/upload",
                    files=[("files", ("people.txt", b"Alice is an engineer. " * 30, "text/plain"))])
    assert r.status_code in (200, 201), r.text
    r = client.post(f"/api/projects/{pid}/rag/build",
                    json={"chunk_size": 200, "overlap": 20, "embedder": "fake", "reset": True})
    assert r.status_code == 200, r.text
    assert r.json().get("documents") == 1, r.json()


def _fake_llama(monkeypatch):
    from finetune_studio.training import gguf_convert as gc

    def fake_run(cmd, *, timeout=3600):
        out = cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[2]
        with open(out, "wb") as fh:
            fh.write(b"gguf")

    monkeypatch.setattr(gc, "find_gguf_convert_script", lambda: "convert.py")
    monkeypatch.setattr(gc, "find_llama_quantize", lambda: "llama-quantize")
    monkeypatch.setattr(gc, "_run_cmd", fake_run)
    return gc


def test_gguf_quant_only_request_drops_f16_intermediate(monkeypatch, tmp_path):
    gc = _fake_llama(monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    out = gc.convert_merged_to_gguf(str(merged), str(tmp_path / "gguf"), ["Q4_K_M"])
    assert out["ok"], out
    names = sorted(p.name for p in (tmp_path / "gguf").iterdir())
    assert len(names) == 1 and "f16" not in names[0].lower(), names
    assert len(out["files"]) == 1 and not out.get("intermediate_path")


def test_gguf_f16_requested_keeps_f16(monkeypatch, tmp_path):
    gc = _fake_llama(monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    out = gc.convert_merged_to_gguf(str(merged), str(tmp_path / "gguf"), ["f16", "Q4_K_M"])
    assert out["ok"], out
    assert any("f16" in p.name.lower() for p in (tmp_path / "gguf").iterdir())


def test_preset_advisor_small_dataset_advice_is_consistent():
    from finetune_studio.training.preset_advisor import propose

    a = propose(tier="precision", base_model_ref="Qwen/Qwen3-4B", pair_count_hint=21)
    assert a.num_epochs == 60 and a.optimizer_steps < a.steps_floor
    assert not any("Epochs raised" in n for n in a.notes)
    assert not any("Raise epochs" in w for w in a.warnings)
    assert any("add more data" in w for w in a.warnings)
    assert a.warmup_steps < a.optimizer_steps // 4


def test_preset_advisor_warmup_scales_with_run_length():
    from finetune_studio.training.preset_advisor import propose

    short = propose(tier="smoke", base_model_ref="Qwen/Qwen3-4B", pair_count_hint=21)
    long_ = propose(tier="precision", base_model_ref="Qwen/Qwen3-4B", pair_count_hint=515)
    assert short.warmup_steps < long_.warmup_steps <= 100


def test_trained_exports_rows_carry_quant(tmp_path):
    from finetune_studio.webui.routes.pages import _scan_run_models

    out = tmp_path / "run"
    (out / "gguf").mkdir(parents=True)
    (out / "gguf" / "model-Q4_K_M.gguf").write_bytes(b"x")
    (out / "gguf" / "model-Q8_0.gguf").write_bytes(b"x")
    (out / "merged").mkdir()
    (out / "merged" / "config.json").write_text("{}")
    rows = _scan_run_models([{"id": "r1", "name": "n", "output_path": str(out)}])
    by_path = {r["path"]: r for r in rows}
    quants = sorted(r["quant"] for r in rows if r["format"] == "gguf")
    assert quants == ["Q4_K_M", "Q8_0"]
    assert by_path[str(out / "merged")]["quant"] == ""


def test_testing_page_empty_state_links_to_export(client):
    pid = client.post("/api/projects", json={"name": "T"}).json()["id"]
    html = client.get(f"/projects/{pid}/testing").text
    assert f'href="/projects/{pid}/export"' in html


def test_rag_page_passphrase_reveal_is_conspicuous(client):
    pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
    html = client.get(f"/projects/{pid}/rag").text
    assert "box.scrollIntoView" in html
    assert html.count("showPassphrase(d.passphrase)") == 2


def test_data_page_upload_empty_click_and_parse_poll(client):
    pid = client.post("/api/projects", json={"name": "U"}).json()["id"]
    html = client.get(f"/projects/{pid}/data").text
    assert "Choose one or more files first" in html
    assert "pollUntilParsed(15)" in html


def test_data_page_hides_raw_parser_codes_for_unreadable_files(client):
    pid = client.post("/api/projects", json={"name": "V"}).json()["id"]
    html = client.get(f"/projects/{pid}/data").text
    assert ">unreadable<" in html
    assert "could not be read" in html


def test_coverage_fill_skips_tabular_junk_and_asks_readable_questions():
    from finetune_studio.data.prep.coverage_fill import _make_pairs_from_chunk

    chunk = (
        "unit_id | class | score | 12 | 34 | 56. "
        "A, B, C, D, E, F, G, H, 1, 2, 3, 4. "
        "The Ledger-Keeper is sworn to record every debt owed to the guild."
    )
    pairs = _make_pairs_from_chunk(chunk, seen_questions=set())
    assert len(pairs) == 1
    q, a = pairs[0]
    assert "Ledger-Keeper" in a and "|" not in q
    assert not q.startswith("According to")


def test_pairs_table_shows_local_time_with_utc_tooltip(client):
    pid = client.post("/api/projects", json={"name": "Z"}).json()["id"]
    html = client.get(f"/projects/{pid}/data-prep").text
    assert "toLocaleString" in html and "UTC' : ''" in html


def test_file_library_table_keeps_name_column_readable_and_actions_wrap(client):
    css = client.get("/static/css/app.css").text
    assert "#fb-files-table .fb-col-name { width: 14em; min-width: 11em; }" in css
    pid = client.post("/api/projects", json={"name": "W"}).json()["id"]
    assert "fb-actions-wrap" in client.get(f"/projects/{pid}/data").text


def test_reupload_same_name_new_bytes_adds_version_not_sql_error(client) -> None:
    pid = db.create_project("Reupload", base_model="Qwen/Qwen3-0.6B")["id"]
    base = f"/api/projects/{pid}/files/upload"
    r1 = client.post(base, files={"files": ("note.txt", b"first version text")}).json()
    assert r1["counts"]["uploaded"] == 1
    r2 = client.post(base, files={"files": ("note.txt", b"second, different bytes")}).json()
    assert r2["counts"]["errors"] == 0, r2
    assert r2["counts"]["uploaded"] == 1
    fid = r1["report"][0]["file_id"]
    versions = client.get(f"/api/projects/{pid}/files/{fid}/versions").json()["versions"]
    assert [v["version"] for v in versions] == [1, 2] or [v["version"] for v in versions] == [2, 1]
    assert len({v["raw_path"] for v in versions}) == 2
