"""What a clean install's first hour found (RESULTS §38)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finetune_studio.data import shared_models as sm

_ROOT = Path(__file__).resolve().parents[1]


def test_download_is_temporary_unless_the_users_cache_has_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path / "hf")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None)
    assert sm._download_cache("o/m", tmp_path, "s1") == tmp_path / "s1-download"
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda _r, _f, cache_dir: "/x/config.json")
    assert sm._download_cache("o/m", tmp_path, "s1") == tmp_path / "hf"


@pytest.mark.parametrize(("kind", "cls"), [("embedder", "SentenceTransformer"), ("reranker", "CrossEncoder")])
def test_registering_leaves_no_download_behind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, cls: str) -> None:
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path / "hf" / "root")
    monkeypatch.setattr(sm, "_dir_for", lambda k: tmp_path / "store" / k)
    monkeypatch.setattr(sm, "_dir_path", lambda k, _n, h: tmp_path / "store" / k / f"m@{h[:8]}")
    monkeypatch.setattr(sm, "_ensure_dirs", lambda: None)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None)
    seen: dict[str, Any] = {}

    class Fake:
        def __init__(self, _name: str, cache_folder: str = "", **_k: Any) -> None:
            seen["cache"] = Path(cache_folder)
            seen["cache"].mkdir(parents=True)
            (seen["cache"] / "blob").write_bytes(b"x")

        def save(self, path: str) -> None:
            (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(f"sentence_transformers.{cls}", Fake)
    ref = sm.register.real("o/m", kind=kind)  # conftest stubs the download branch for every other test
    assert (Path(ref.path) / "config.json").is_file()
    assert seen["cache"].name.endswith("-download") and not seen["cache"].exists()


def test_search_adds_vision_chat_models_to_text_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui.routes import hf_models

    tags: list[Any] = []

    def models(**kw: Any) -> list[SimpleNamespace]:
        tags.append(kw["pipeline_tag"])
        repo, dl = ("re/Qwen3.5-0.8B-GGUF", 5) if kw["pipeline_tag"] == "text-generation" else ("Qwen/Qwen3.5-0.8B", 9)
        return [SimpleNamespace(id=repo, downloads=dl, likes=0, tags=[], last_modified=None, private=False)]

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(list_models=models))
    found = hf_models._search_hf(hf_models.SearchRequest(query="Qwen3.5-0.8B"))
    assert [m["repo_id"] for m in found] == ["Qwen/Qwen3.5-0.8B", "re/Qwen3.5-0.8B-GGUF"]
    assert tags == ["text-generation", "image-text-to-text"]


def test_browsing_without_a_query_stays_on_the_chosen_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui.routes import hf_models

    tags: list[Any] = []

    def models(**kw: Any) -> list[SimpleNamespace]:
        tags.append(kw["pipeline_tag"])
        return [SimpleNamespace(id="a/b", downloads=1, likes=0, tags=[], last_modified=None, private=False)]

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(list_models=models))
    hf_models._search_hf(hf_models.SearchRequest())
    assert tags == ["text-generation"]


def test_quick_work_points_to_the_model_library() -> None:
    page = (_ROOT / "src/finetune_studio/webui/templates/project_wizard.html").read_text(encoding="utf-8")
    assert 'href="/models/explore"' in page and "models folder" not in page


def test_one_slow_status_poll_is_not_an_outage() -> None:
    js = (_ROOT / "src/finetune_studio/webui/static/js/service.js").read_text(encoding="utf-8")
    assert "if (failures >= 2) {" in js and "failures = 0;" in js
    assert "service.js?v=6" in (_ROOT / "src/finetune_studio/webui/templates/base.html").read_text(encoding="utf-8")


def test_install_banner_reads_one_address_line() -> None:
    assert 'server show 2>/dev/null | head -1 | cut' in (_ROOT / "install.sh").read_text(encoding="utf-8")
