"""Training's "Open in inference" sends a run's output folder: the load opens the model inside it.

Found by the B3 sweep on a project with runs: the button sent ``runs/<id>``, which holds merged/, adapter/ and gguf/ but no
model of its own, and every click ended in a 500 with a tokenizer error. The GGUF row of the export tables 404ed because
its "contents" route only took folders, and a GGUF export is one file.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finetune_studio.models.run_paths import has_model_files, resolve_run_model_dir
from tests.test_project_models import _project, _run_with_export

_PAGE = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_training.html").read_text(
    encoding="utf-8")


def _run_dir(root: Path, *, merged: bool = True, adapter: bool = True) -> Path:
    if merged:
        (root / "merged").mkdir(parents=True)
        (root / "merged" / "config.json").write_text("{}")
    if adapter:
        (root / "adapter").mkdir(parents=True)
        (root / "adapter" / "adapter_config.json").write_text("{}")
    (root / "gguf").mkdir(parents=True)
    return root


def test_run_folder_resolves_to_merged_then_adapter(tmp_path: Path) -> None:
    assert resolve_run_model_dir(str(_run_dir(tmp_path / "a"))) == str(tmp_path / "a" / "merged")
    assert resolve_run_model_dir(str(_run_dir(tmp_path / "b", merged=False))) == str(tmp_path / "b" / "adapter")


def test_a_model_folder_or_repo_id_is_left_alone(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "merged").mkdir()   # a model is a model, whatever sits next to its config
    (model / "merged" / "config.json").write_text("{}")
    assert resolve_run_model_dir(str(model)) == str(model)
    assert resolve_run_model_dir("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"
    assert has_model_files("Qwen/Qwen3.5-9B")


@pytest.mark.parametrize("merged", [True, False])
def test_load_of_a_run_folder_opens_its_model(client, tmp_path: Path, merged: bool) -> None:
    run = _run_dir(tmp_path / "run", merged=merged)
    eng = MagicMock()
    eng.vision = False
    with patch("finetune_studio.webui.app.inference_engine", eng):
        r = client.post("/api/models/load", json={"path": str(run)})
    assert r.status_code == 200, r.text
    want = str(run / ("merged" if merged else "adapter"))
    assert r.json()["model"] == want
    assert eng.load.call_args.args[0] == want


def test_load_of_a_folder_without_a_model_says_so(client, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    (empty / "gguf").mkdir(parents=True)
    eng = MagicMock()
    with patch("finetune_studio.webui.app.inference_engine", eng):
        r = client.post("/api/models/load", json={"path": str(empty)})
    assert r.status_code == 400
    assert "no model in this folder" in r.json()["detail"]
    eng.load.assert_not_called()


def test_gguf_export_row_lists_its_file(client, tmp_path: Path) -> None:
    pid = _project(client)
    _run, merged = _run_with_export(pid, tmp_path)
    gguf = Path(merged).parent / "gguf" / "model-q4_k_m.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"x" * 10)
    r = client.get(f"/api/projects/{pid}/models/{str(gguf).lstrip('/')}/contents")
    assert r.status_code == 200, r.text
    assert r.json()["files"] == [{"name": "model-q4_k_m.gguf", "size_bytes": 10, "is_dir": False}]


def test_run_detail_uses_the_same_load_button() -> None:
    assert "adapter_path" not in _PAGE            # the Inference page never read it: the link loaded nothing
    assert 'class="btn sm primary js-open-inf"' in _PAGE


def test_export_row_shows_what_the_job_concluded(client, tmp_path: Path) -> None:
    from finetune_studio import db

    pid = _project(client)
    run, merged = _run_with_export(pid, tmp_path)
    out = Path(merged).parent / "heretic"
    out.mkdir()
    (out / "config.json").write_text("{}")
    row = db.create_export(pid, run["id"], format="heretic", quant="safetensors")
    db.mark_export_done(row["id"], output_path=str(out), detail="Kept Heretic trial 17: 8% refusals")
    rel = str(out).lstrip("/")
    assert client.get(f"/api/projects/{pid}/models/{rel}/contents").json()["note"] == "Kept Heretic trial 17: 8% refusals"
    assert client.get(f"/api/projects/{pid}/models/{str(merged).lstrip('/')}/contents").json()["note"] == ""


def test_a_heretic_export_is_listed_and_a_cancelled_search_is_not(client, tmp_path: Path) -> None:
    from finetune_studio.webui.routes.pages import _scan_run_models

    run_dir = tmp_path / "r"
    (run_dir / "heretic" / ".study").mkdir(parents=True)   # what a cancelled search leaves
    assert _scan_run_models([{"id": "r1", "output_path": str(run_dir)}]) == []
    (run_dir / "heretic" / "config.json").write_text("{}")
    rows = _scan_run_models([{"id": "r1", "output_path": str(run_dir)}])
    assert [(m["subdir"], m["format"]) for m in rows] == [("heretic", "safetensors")]
