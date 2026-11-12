"""Pictures read by OCR only: counted, reported (parse warning, Files flags, Knowledge check), and read by the Vision
reader automatically when one is set (``knowledge.pictures``)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from finetune_studio.knowledge import pictures

OCR_BLOCK = "[Figure 1 · paragraph 3 · picture · read by OCR, not checked]\n31 % 28 %\n[End of figure 1]"
SCAN_BLOCK = "[Figure 2 · page 2 · scanned page · read by OCR, not checked]\nDear Sir\n[End of figure 2]"
READ_BLOCK = "[Figure 3 · page 4 · picture · read by gemma.gguf]\nMonth | Cash\nJul 2025 | £3.2m\n[End of figure 3]"
DOC = "Finance pack FY2025. " * 20 + "\n" + OCR_BLOCK + "\n" + SCAN_BLOCK + "\n" + READ_BLOCK + "\nThe end. " * 10


def test_counts_split_pictures_scans_and_model_readings() -> None:
    assert pictures.counts(DOC) == {"pictures": 2, "pictures_ocr": 1, "scanned": 1, "scanned_ocr": 1, "read": 1}
    assert pictures.counts("no figures here") == {"pictures": 0, "pictures_ocr": 0, "scanned": 0, "scanned_ocr": 0,
                                                  "read": 0}


def test_parse_warning_names_the_fix_when_no_reader_is_set(monkeypatch) -> None:
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "off", "label": "Off (OCR only)"})
    note = pictures.parse_warning(DOC)
    assert note.startswith("2 picture(s) read by OCR only") and "set a vision reader" in note
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "local", "label": "Local · g.gguf"})
    assert "Vision reader reads them next" in pictures.parse_warning(DOC)
    assert pictures.parse_warning("plain text") == ""


def _upload(client, pid: str, name: str, text: str) -> dict:
    r = client.post(f"/api/projects/{pid}/files/upload", files=[("files", (name, text.encode(), "text/markdown"))])
    assert r.status_code == 200, r.text
    return r.json()


def test_files_flags_parse_warning_and_knowledge_check_without_a_reader(client, monkeypatch) -> None:
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "off", "label": "Off (OCR only)",
                                                           "settings": pictures.SETTINGS_LINK})
    pid = client.post("/api/projects", json={"name": "pictures-off"}).json()["id"]
    _upload(client, pid, "fy2025.md", DOC)
    from finetune_studio.knowledge.sources import list_sources, source_dir
    src = list_sources(pid)[0]
    assert src.ready
    parsed = (source_dir(pid, src) / "parsed.json").read_text()
    assert "read by OCR only" in parsed
    from finetune_studio.knowledge.status import pipeline_status
    flags = pipeline_status(pid)[src.file_id]
    assert (flags["pictures_ocr"], flags["scanned_ocr"], flags["figures_read"]) == (1, 1, 1)
    assert pictures._last(pid) is None                      # no reader: nothing started after the parse
    report = client.get(f"/api/projects/{pid}/brain/lint").json()
    pic = [f for f in report["findings"] if f["rule"] == "pictures_ocr"]
    assert len(pic) == 1 and pic[0]["title"].startswith("2 picture(s) read by OCR only")
    assert pic[0]["link"]["href"] == pictures.SETTINGS_LINK and pic[0]["action"] == ""
    st = client.get(f"/api/projects/{pid}/brain/pictures").json()
    assert st["pictures_ocr"] == 1 and st["files"][0]["name"] == "fy2025.md"
    r = client.post(f"/api/projects/{pid}/brain/pictures/read", json={})
    assert r.status_code == 400 and "no Vision reader" in r.text


def test_a_set_reader_starts_reading_after_the_upload_and_the_check_offers_the_button(client, monkeypatch) -> None:
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "local", "label": "Local · g.gguf",
                                                           "settings": pictures.SETTINGS_LINK})
    started = []
    monkeypatch.setattr(pictures, "start", lambda pid, **k: started.append((pid, k)) or {"id": "j"})
    pid = client.post("/api/projects", json={"name": "pictures-on"}).json()["id"]
    _upload(client, pid, "fy2025.md", DOC)
    assert started == [(pid, {"auto": True})]
    _upload(client, pid, "plain.md", "Nothing but words about invoices. " * 30)
    assert len(started) == 1                                 # a file without pictures starts nothing
    pic = [f for f in client.get(f"/api/projects/{pid}/brain/lint").json()["findings"] if f["rule"] == "pictures_ocr"]
    assert pic[0]["action"] == "read_pictures" and "Read pictures" in pic[0]["fix"]


class FakeReader:
    label = "fake-vision"
    closed = False

    def __init__(self, *_a) -> None:
        pass

    def close(self) -> None:
        FakeReader.closed = True


def test_the_job_reads_pending_files_skips_mined_ones_and_closes_the_reader(monkeypatch) -> None:
    from finetune_studio.data.mining import figures as mfig
    from finetune_studio.knowledge import sources as sources_mod
    from finetune_studio.models import vision

    a = SimpleNamespace(id="a", file_id="fa", filename="a.docx", ready=True)
    b = SimpleNamespace(id="b", file_id="fb", filename="b.docx", ready=True)
    calls = {"n": 0}

    def pending(pid, file_ids=None):
        calls["n"] += 1
        return [a, b]

    monkeypatch.setattr(vision, "Reader", FakeReader)
    monkeypatch.setattr(pictures, "pending_files", pending)
    monkeypatch.setattr(pictures, "_mined", lambda pid: {"b"})
    monkeypatch.setattr(pictures, "_mining_busy", lambda pid: set())
    monkeypatch.setattr(sources_mod, "get_source", lambda pid, sid: a if sid == "a" else b)
    monkeypatch.setattr(mfig, "read_file_figures",
                        lambda pid, s, reader, progress: mfig.FigureResult(read=2, changed=True))
    out = pictures.run_in_job("p", None, stop=threading.Event())
    assert out["figures_read"] == 2 and out["files"] == 1
    assert out["skipped"] == ["b.docx: already mined (Reset it in Mining to read its pictures)"]
    assert FakeReader.closed and calls["n"] == 2              # re-checked once for files parsed meanwhile


def test_start_refuses_without_a_reader_and_with_nothing_left(monkeypatch) -> None:
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "off", "label": "Off"})
    with pytest.raises(pictures.PicturesError, match="no Vision reader"):
        pictures.start("p")
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "api", "label": "API"})
    monkeypatch.setattr(pictures, "_running", lambda pid: None)
    monkeypatch.setattr(pictures, "pending_files", lambda pid, ids=None: [])
    with pytest.raises(pictures.PicturesError) as e:
        pictures.start("p")
    assert e.value.status == 409


def test_the_job_kind_is_registered() -> None:
    from finetune_studio.jobs.runner import KINDS
    assert KINDS["pictures"] == "finetune_studio.jobs.kinds.pictures:run"
