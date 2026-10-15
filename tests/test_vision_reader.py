"""Vision reader seat + mining stage 0 (``models/vision.py``, ``data/mining/figures.py``). No real model: a fake reader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from finetune_studio.data import figures as figs
from finetune_studio.data.mining import figures as mfig
from finetune_studio.models import vision


def _parsed(blocks: list[tuple[int, str, str]]) -> str:
    out = ["Intro text."]
    for n, where, body in blocks:
        out.append(figs.block(figs.Figure(n, where, 0, b""), body, mfig.OCR_MARK))
    out.append("Outro.")
    return "\n".join(out)


def test_pending_blocks_and_replace() -> None:
    text = _parsed([(1, "page 3", "84.3 68.8\n2019 2020"), (2, "page 5", "3.44")])
    assert mfig.pending_blocks(text) == [(1, "page 3"), (2, "page 5")]
    new = mfig.replace_block(text, 2, figs.block(figs.Figure(2, "page 5", 5, b""), "Q3-22 | 3.44", "gemma"))
    assert mfig.pending_blocks(new) == [(1, "page 3")]
    assert "[Figure 2 · page 5 · picture · read by gemma]\nQ3-22 | 3.44\n[End of figure 2]" in new and "Outro." in new


class FakeReader:
    label = "fake-vl"

    def __init__(self, fail: set[int] = frozenset()) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def read(self, data: bytes, title: str, where: str) -> str:
        self.calls.append(where)
        if where in self.fail:
            raise vision.VisionError("boom")
        return f"read {where} of {title}"


def _setup(monkeypatch: pytest.MonkeyPatch, text: str, found: list[figs.Figure]) -> tuple[str, Any]:
    """A real source whose parsed text holds ``text``; the figure extractor returns ``found``."""
    from finetune_studio import db
    from tests.knowledge_helpers import add_source

    pid = db.create_project(name="vision", description="")["id"]
    monkeypatch.setattr(figs, "extract", lambda path, page_texts=None: found)
    return pid, add_source(pid, "Annual report.txt", text)


def test_read_file_figures_replaces_only_matching_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.knowledge import sources

    text = _parsed([(1, "page 3", "ocr one"), (2, "page 5", "ocr two"), (3, "page 9", "ocr three")])
    found = [figs.Figure(1, "page 3", 3, b"a"), figs.Figure(2, "page 5", 5, b"b"),
             figs.Figure(3, "page 8", 8, b"c")]              # disagrees with the parse: never read into page 9
    pid, source = _setup(monkeypatch, text, found)
    reader = FakeReader(fail={"page 5"})
    res = mfig.read_file_figures(pid, source, reader)
    assert res.read == 1 and res.changed and reader.calls == ["page 3", "page 5"]
    assert any("figure 2" in f and "boom" in f for f in res.failed)
    assert any("figure 3" in f and "not found" in f for f in res.failed)
    after = sources.get_source(pid, source.id)
    out = sources.parsed_text(pid, after)
    assert "[Figure 1 · page 3 · picture · read by fake-vl]\nread page 3 of Annual report.txt" in out
    assert "ocr two" in out and "ocr three" in out          # failed ones keep their OCR text
    assert after.parser == source.parser + "+vision" and after.edited_at
    assert any("read page 3" in p for p in sources.passages(pid, after.sha256))     # the passages carry the reading
    mfig.read_file_figures(pid, after, FakeReader())                                 # a second pass reads figure 2
    assert sources.get_source(pid, source.id).parser == source.parser + "+vision"   # never "+vision+vision"


def test_read_file_figures_without_pending_blocks_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.knowledge import sources

    pid, source = _setup(monkeypatch, "Plain text only, no pictures at all.", [])
    res = mfig.read_file_figures(pid, source, FakeReader())
    assert res.read == 0 and not res.changed and sources.get_source(pid, source.id).edited_at is None


def test_clean_strips_reasoning_markers() -> None:
    assert vision.clean(">\n<|channel>thought\nhmm\n<channel|>**Title:**\nX | 1") == "**Title:**\nX | 1"
    assert vision.clean("<think>x</think>\nA | 2") == "A | 2"


def test_image_message_is_an_openai_data_url() -> None:
    msg = vision.image_message("read it", b"\x89PNG\r\n\x1a\nrest")
    part = msg[0]["content"][0]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    assert msg[0]["content"][1] == {"type": "text", "text": "read it"}


def test_seat_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.models.manager import ModelManager

    state: dict[str, str] = {}
    monkeypatch.setattr(ModelManager, "get_state", staticmethod(lambda k, d="": state.get(k, d)))
    monkeypatch.setattr(ModelManager, "set_state", staticmethod(lambda k, v: state.__setitem__(k, v)))
    assert vision.get_seat().mode == "off"
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="no mmproj"):
        vision.set_seat(vision.Seat("local", str(gguf)))
    (tmp_path / "mmproj-m-f16.gguf").write_bytes(b"GGUF")
    vision.set_seat(vision.Seat("local", str(gguf)))
    assert vision.get_seat() == vision.Seat("local", str(gguf))
    with pytest.raises(ValueError):
        vision.set_seat(vision.Seat("sideways", ""))
    with pytest.raises(vision.VisionError, match="no Vision reader"):
        vision.Reader(vision.Seat("off"))


def test_scheduler_refuses_a_local_reader_that_would_evict_a_local_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.data.mining.scheduler import MiningError, MiningScheduler

    monkeypatch.setattr(vision, "get_seat", lambda: vision.Seat("local", "/m/vl.gguf"))
    with pytest.raises(MiningError, match="share the one inference worker"):
        MiningScheduler._vision_reader({"kind": "local_gguf", "model": "/m/helper.gguf"})
    monkeypatch.setattr(vision, "get_seat", lambda: vision.Seat("off"))
    assert MiningScheduler._vision_reader({"kind": "local_gguf", "model": "/m/helper.gguf"}) is None


# ── Settings routes + API reading against a fake OpenAI-compatible vision gateway ─────────────────────────────────

import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

VKEY = "sk-vision-test-987654"


@pytest.fixture
def vgateway():
    seen: list[dict[str, Any]] = []
    reply = {"text": "Code: 424242"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"auth": self.headers.get("Authorization"), "body": body})
            code, payload = (200, {"choices": [{"message": {"content": reply["text"]}, "finish_reason": "stop"}]}) \
                if self.headers.get("Authorization") == f"Bearer {VKEY}" else (401, {"error": "bad key"})
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/v1", "seen": seen, "reply": reply}
    srv.shutdown()


def test_vision_settings_default_off_and_api_seat_key_write_only(client: Any, vgateway: dict) -> None:
    st = client.get("/api/settings/vision").json()
    assert st["mode"] == "off" and st["label"] == "Off (OCR only)" and isinstance(st["local_candidates"], list)
    r = client.put("/api/settings/vision", json={"mode": "api"})
    assert r.status_code == 400 and "enter the API vision model" in r.json()["detail"]
    r = client.put("/api/settings/vision", json={"mode": "api", "base_url": vgateway["url"], "model": "vl-1",
                                                 "api_key": VKEY})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["mode"] == "api" and st["api"]["key_set"] and st["label"] == "API · vl-1"
    for url in ("/api/settings/vision", "/api/providers", "/api/settings/helper"):
        assert VKEY not in client.get(url).text
    assert client.put("/api/settings/vision", json={"mode": "nope"}).status_code == 400
    assert client.put("/api/settings/vision", json={"mode": "local", "local_model": "/no/such.gguf"}).status_code == 400


def test_vision_test_route_sends_the_picture_and_checks_the_code(client: Any, vgateway: dict,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.webui.routes import vision_settings

    client.put("/api/settings/vision", json={"mode": "api", "base_url": vgateway["url"], "model": "vl-1",
                                             "api_key": VKEY})
    monkeypatch.setattr(vision_settings.random, "randint", lambda a, b: 424242)
    r = client.post("/api/settings/vision/test")
    assert r.status_code == 200 and r.json()["ok"], r.text
    content = vgateway["seen"][-1]["body"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,") and vgateway["seen"][-1]["body"]["model"] == "vl-1"
    vgateway["reply"]["text"] = "I see a white rectangle."
    r = client.post("/api/settings/vision/test")
    assert r.status_code == 502 and "did not read the code" in r.json()["error"]
    client.put("/api/settings/vision", json={"mode": "off"})
    assert client.post("/api/settings/vision/test").status_code == 400


def test_big_pictures_are_scaled_down_before_reading() -> None:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (1700, 2200), "white").save(buf, "PNG")
    small = vision.fit_image(buf.getvalue())
    with Image.open(io.BytesIO(small)) as im:
        assert max(im.size) == vision.MAX_SIDE_PX and im.size[0] < im.size[1]
    tiny = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(tiny, "PNG")
    assert vision.fit_image(tiny.getvalue()) == tiny.getvalue()


def test_without_a_reader_picture_ocr_is_held_back_but_scanned_pages_are_mined() -> None:
    pic = figs.block(figs.Figure(1, "page 3", 3, b""), "84.3 68.8\n2019 2020", mfig.OCR_MARK)
    scan = figs.block(figs.Figure(2, "page 4", 4, b"", whole_page=True), "Closed by Kowarzik, ext 4419", mfig.OCR_MARK)
    full = f"Revenue grew.\n{pic}\nNext page.\n{scan}\nEnd."
    pictures = mfig.ocr_pictures(full)
    assert pictures == {1}
    held, n = mfig.hold_back_pictures(full, pictures)
    assert n == 1 and "84.3" not in held and mfig.HELD_BACK in held
    assert "Closed by Kowarzik, ext 4419" in held and "Revenue grew." in held and "End." in held
    # a block cut by a chunk boundary: the body is held back on both sides
    cut = full.index("68.8")
    first, second = full[:cut], full[cut:]
    h1, n1 = mfig.hold_back_pictures(first, pictures)
    h2, n2 = mfig.hold_back_pictures(second, pictures)
    assert n1 == n2 == 1 and "84.3" not in h1 and "68.8" not in h2 and "2019" not in h2
    assert "Closed by Kowarzik" in h2
    # a picture already read by the vision model is mined as is
    read = figs.block(figs.Figure(1, "page 3", 3, b""), "2021 | 68.8", "gemma")
    assert mfig.ocr_pictures(read) == set()


def test_helper_sees_neutral_figure_markers() -> None:
    read = figs.block(figs.Figure(1, "rendered page 1", 1, b"", kind_override=figs.PICTURE), "Pump P-12", "Gemma.gguf")
    held = f"[Figure 2 · page 3 · picture · read by OCR, not checked]\n{mfig.HELD_BACK}\n[End of figure 2]"
    out = mfig.neutral_markers(f"Intro\n{read}\n{held}\nEnd")
    assert out == "Intro\n[Figure 1, page 1]\nPump P-12\n[Figure 2, page 3]\n(picture not read)\nEnd"
    assert "Gemma" not in out and "OCR" not in out
