"""The Package page shows at once: the knowledge check is cached until the files or index change, the export tool
probe is reused for half a minute, and the readiness checklist loads after the page (GET …/brain/readiness)."""
from __future__ import annotations

from finetune_studio.knowledge import lint as kl
from finetune_studio.knowledge.sources import Source
from finetune_studio.training import export_capabilities as ec

BODY = " ".join(f"Paragraph {i} explains how the warehouse handles pallets during the morning shift." for i in range(30))


def _src(sid: str, name: str, text_sha: str = "t1") -> Source:
    return Source(id=sid, file_id=f"f{sid}", version=1, sha256=sid * 4, filename=name, mime_type="text/plain",
                  raw_path="", size_bytes=1, uploaded_at=0.0, status="ready", text_sha=text_sha, parser="x",
                  char_count=9, chunk_count=3, error="", edited_at=None, deleted=False)


def _patch(monkeypatch, sources: list[Source], calls: list[str]) -> None:
    monkeypatch.setattr(kl, "list_sources", lambda pid: sources)
    monkeypatch.setattr(kl, "ignored_keys", lambda pid: set())

    def read(pid, s):
        calls.append(s.id)
        return BODY if s.id == "a" else "Entirely different words about invoices and shipping terms. " * 20

    monkeypatch.setattr(kl, "parsed_text", read)
    monkeypatch.setattr(kl, "_check_pictures", lambda pid, sources: [])
    monkeypatch.setattr(kl, "_check_index", lambda pid, sources: [])


def test_the_text_checks_are_reused_until_a_file_changes(monkeypatch) -> None:
    calls: list[str] = []
    sources = [_src("a", "guide_draft.txt"), _src("b", "notes.txt")]
    _patch(monkeypatch, sources, calls)
    first = kl.lint("p1")
    assert sorted(calls) == ["a", "b"] and any(f["rule"] for f in first["findings"])
    second = kl.lint("p1")
    assert sorted(calls) == ["a", "b"], "a second check with nothing changed must not read the files again"
    assert second == first
    sources[1] = _src("b", "notes.txt", text_sha="t2")      # the text of one file changed
    kl.lint("p1")
    assert sorted(calls) == ["a", "a", "b", "b"]


def test_an_ignored_finding_is_honoured_through_the_cache(monkeypatch) -> None:
    calls: list[str] = []
    _patch(monkeypatch, [_src("a", "guide_draft.txt"), _src("b", "notes.txt")], calls)
    first = kl.lint("p2")
    key = first["findings"][0]["key"]
    monkeypatch.setattr(kl, "ignored_keys", lambda pid: {key})
    again = kl.lint("p2")
    assert next(f for f in again["findings"] if f["key"] == key)["ignored"] is True


def test_labelled_values_are_read_once_per_passage() -> None:
    kl._labelled_values.cache_clear()
    text = "Monthly fee: 39 EUR for the standard plan."
    kl._labelled_values(text)
    kl._labelled_values(text)
    assert kl._labelled_values.cache_info().hits >= 1


def test_the_export_probe_is_cached_and_forgotten_after_an_install(monkeypatch) -> None:
    ec.forget_cached_capabilities()
    n = {"probes": 0}
    real = ec.ExportCapabilities(gguf=False, gguf_script=None, gguf_hint="x")

    def probe():
        n["probes"] += 1
        return real

    monkeypatch.setattr(ec, "probe_export_capabilities", probe)
    assert ec.cached_export_capabilities() is real and ec.cached_export_capabilities() is real
    assert n["probes"] == 1
    ec.forget_cached_capabilities()
    ec.cached_export_capabilities()
    assert n["probes"] == 2
    ec.forget_cached_capabilities()


def test_the_package_page_does_not_run_the_check_while_rendering(client, monkeypatch) -> None:
    from finetune_studio.knowledge import readiness as rd

    pid = client.post("/api/projects", json={"name": "fast-package"}).json()["id"]
    monkeypatch.setattr(rd, "readiness", lambda pid: (_ for _ in ()).throw(AssertionError("ran while rendering")))
    body = client.get(f"/projects/{pid}/export").text
    assert 'id="package-readiness"' in body and 'id="package-steps"' in body
    assert "/brain/readiness" in body, "the checklist loads from the readiness API"
