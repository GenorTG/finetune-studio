"""Pictures in documents for the Smart Brain: read by the Vision reader by default, OCR only as the fallback.

A parser leaves every picture (a chart image, a table screenshot, a scanned page) as a ``[Figure N · … · read by OCR,
not checked]`` block (``data/figures.py``). Until now only mining replaced that OCR with the Vision reader's reading, so a
project that was never mined packaged the OCR text: a chart's labels and values came apart (Tarrowfield in-041 answered
£3.6m for £3.2m; in-039 found no 31 % at all).

Now, when a Vision reader is set (Settings -> AI assistant -> Jobs -> Reading pictures), the upload starts the
``pictures`` job as soon as a parsed file holds such blocks; it reads them with ``data.mining.figures.read_file_figures``
and rewrites the file's text (the index marks it changed: Update Smart Brain uses the readings). Without a reader the
figures keep their OCR text, and ``counts`` / ``project_status`` say so: the Files page, the parse warning and the
Knowledge check report "N pictures read by OCR only".

A file that was already mined keeps its OCR text: re-chunking would move the chunks its mined items point at (Reset the
file in Mining to read its pictures).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from finetune_studio.data import figures as figs

log = logging.getLogger(__name__)

KIND = "pictures"
SETTINGS_LINK = "/settings#assistant-card"
OCR_ONLY_NOTE = ("{n} picture(s) read by OCR only — figures in charts may be wrong; set a vision reader "
                 "(Settings → AI assistant → Reading pictures)")

_cache: dict[str, dict[str, int]] = {}
_cache_lock = threading.Lock()


class PicturesError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def counts(text: str) -> dict[str, int]:
    """Figure blocks of a parsed text: ``pictures`` / ``scanned`` (all), ``*_ocr`` (still only OCR or rendered text,
    no model read them), ``read`` (read by a vision model)."""
    out = {"pictures": 0, "pictures_ocr": 0, "scanned": 0, "scanned_ocr": 0, "read": 0}
    for m in figs.BLOCK_RE.finditer(text or ""):
        _where, kind, read_by = figs.header_parts(m.group(2))
        key = "scanned" if kind == figs.SCANNED else "pictures"
        out[key] += 1
        if read_by in figs.UNCHECKED:
            out[f"{key}_ocr"] += 1
        else:
            out["read"] += 1
    return out


def source_counts(pid: str, source: Any) -> dict[str, int]:
    """``counts`` of a source's parsed text, cached on its text hash (the Files page asks for every file)."""
    from finetune_studio.knowledge.sources import parsed_text

    key = f"{pid}:{source.id}:{source.text_sha}"
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    got = counts(parsed_text(pid, source)) if source.ready else counts("")
    with _cache_lock:
        if len(_cache) > 20_000:
            _cache.clear()
        _cache[key] = got
    return got


def parse_warning(text: str) -> str:
    """The parse result's warning about pictures nobody but OCR read ("" when there are none)."""
    c = counts(text)
    n = c["pictures_ocr"] + c["scanned_ocr"]
    if not n:
        return ""
    st = reader_state()
    if st["mode"] == "off":
        return OCR_ONLY_NOTE.format(n=n)
    cost = local_reader_cost()
    where = (f" (it loads {cost['name']} on the GPU, about {cost['gb']:.1f} GB, then unloads it)" if cost
             else f" ({st['label']})")
    return (f"{n} picture(s) read by OCR for now; the Vision reader reads them next{where}: Reading pictures job "
            "in Activity. The Smart Brain then needs an update to use the readings.")


def local_reader_cost() -> dict[str, Any] | None:
    """``{"name", "gb", "path"}`` of a local Vision reader (its GGUF + mmproj on disk), None for an API reader."""
    from pathlib import Path

    from finetune_studio.models import vision

    try:
        seat = vision.resolve_seat(vision.get_seat())
    except Exception:  # noqa: BLE001 - a broken seat: no estimate
        return None
    if seat.mode != "local" or not seat.local_model:
        return None
    size = 0
    for f in (seat.local_model, vision.mmproj_for(seat.local_model)):
        try:
            size += Path(f).stat().st_size if f else 0
        except OSError:
            pass
    return {"name": Path(seat.local_model).name, "gb": size / 1024**3, "path": seat.local_model}


#: Context, compute buffers and the mmproj's own activations on top of the weights (Gemma 4 12B QAT: 10.6 GB
#: resident for 7.5 GB of files).
VRAM_OVERHEAD = 1.35


def vram_problem() -> str:
    """Why a local Vision reader must not load now ("" when it may): not enough free GPU memory. Other programs'
    memory is never freed for it (a game, ComfyUI, another service); the job waits for a later start instead."""
    cost = local_reader_cost()
    if not cost:
        return ""
    try:
        from finetune_studio.webui.gpu_probe import free_and_consumers

        free_mib, users = free_and_consumers()
    except Exception:  # noqa: BLE001 - no probe: the loader's own fit logic decides
        return ""
    if free_mib is None:
        return ""
    need = cost["gb"] * VRAM_OVERHEAD
    free = free_mib / 1024
    if free >= need:
        return ""
    by = ", ".join(f"{u.get('name') or u.get('pid')}" for u in users[:3])
    return (f"not enough free GPU memory to load the Vision reader {cost['name']} (about {need:.1f} GB needed, "
            f"{free:.1f} GB free{'; in use by ' + by if by else ''}). Pictures keep their OCR text for now: start "
            "Reading pictures from the Knowledge check when the GPU is free, or use an API vision model.")


def _worker_holds_other_model() -> str:
    """The model another page loaded into the one inference worker ("" when none): the Vision reader would replace
    it, so the upload's automatic start leaves it alone."""
    from finetune_studio.models.manager import get_manager

    try:
        eng = get_manager().engine
        held = str(getattr(eng, "model_path", "") or "") if eng.model is not None else ""
    except Exception:  # noqa: BLE001 - unknown: assume free
        return ""
    cost = local_reader_cost()
    return held if held and cost and held != cost["path"] else ""


def reader_state() -> dict[str, Any]:
    """The Vision reader seat as the pages show it (``mode`` "off" = OCR only), never a key."""
    from finetune_studio.models import vision

    try:
        d = vision.describe()
    except Exception as e:  # noqa: BLE001 - a broken seat row: report it as off with the reason
        return {"mode": "off", "label": "Off (OCR only)", "error": str(e), "settings": SETTINGS_LINK}
    return {"mode": d["mode"], "label": d["label"], "settings": SETTINGS_LINK}


def _mined(pid: str) -> set[str]:
    from finetune_studio import db
    return {str(it["source_id"]) for it in db.mining.list_items(pid)}


def _mining_busy(pid: str) -> set[str]:
    from finetune_studio import db
    return {str(j["source_id"]) for j in db.mining.list_jobs(pid, active=True)}


def pending_files(pid: str, file_ids: list[str] | None = None) -> list[Any]:
    """Ready sources with figure blocks no model has read yet (optionally only these library file ids)."""
    from finetune_studio.knowledge.sources import list_sources

    want = set(file_ids or [])
    out = []
    for s in list_sources(pid):
        if s.deleted or not s.ready or (want and s.file_id not in want and s.id not in want):
            continue
        c = source_counts(pid, s)
        if c["pictures_ocr"] or c["scanned_ocr"]:
            out.append(s)
    return out


def _running(pid: str) -> dict[str, Any] | None:
    from finetune_studio.jobs import launch
    return next((j for j in launch.list_jobs(KIND, running_only=True) if j.get("project_id") == pid), None)


def _last(pid: str) -> dict[str, Any] | None:
    from finetune_studio.jobs import launch
    jobs = [j for j in launch.list_jobs(KIND) if j.get("project_id") == pid]
    return max(jobs, key=lambda j: j.get("created_at") or 0) if jobs else None


def project_status(pid: str) -> dict[str, Any]:
    """Totals, the files that still hold OCR-only figures, the reader seat and the latest pictures job."""
    from finetune_studio.knowledge.sources import list_sources

    mined = _mined(pid)
    files, tot = [], {"pictures": 0, "pictures_ocr": 0, "scanned": 0, "scanned_ocr": 0, "read": 0}
    for s in list_sources(pid):
        if s.deleted or not s.ready:
            continue
        c = source_counts(pid, s)
        for k in tot:
            tot[k] += c[k]
        if c["pictures_ocr"] or c["scanned_ocr"]:
            files.append({"id": s.id, "file_id": s.file_id, "name": s.filename, **c, "mined": s.id in mined})
    job = _running(pid) or _last(pid)
    return {**tot, "files": files, "reader": reader_state(),
            "job": _job_view(job) if job else None}


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    ex = job.get("exit") or {}
    return {"id": job["id"], "running": bool(job.get("running")), "state": job.get("state") or {},
            "status": "running" if job.get("running") else str(ex.get("status") or ""),
            "result": ex.get("result") or {}, "error": ex.get("error") or ex.get("detail") or ""}


def start(pid: str, *, file_ids: list[str] | None = None, auto: bool = False) -> dict[str, Any]:
    """Launch the pictures job for the project (one at a time: a running one is returned and picks up files that
    were parsed meanwhile). Raises ``PicturesError`` when no Vision reader is set or nothing is left to read."""
    from finetune_studio.jobs import launch, store

    if reader_state()["mode"] == "off":
        raise PicturesError("no Vision reader is set: pictures keep their OCR text. Set one in Settings → AI "
                            "assistant → Reading pictures (a local model with an mmproj, or an API vision model).")
    problem = vram_problem()
    if problem:
        raise PicturesError(problem, 409)
    if auto:
        held = _worker_holds_other_model()
        if held:
            raise PicturesError(f"{held.rsplit('/', 1)[-1]} is loaded (Chat, a test): reading pictures would unload "
                                "it, so it waits; start Reading pictures from the Knowledge check when you are done",
                                409)
    running = _running(pid)
    if running is not None:
        return {**_job_view(running), "already_running": True}
    todo = pending_files(pid, file_ids)
    if not todo:
        raise PicturesError("no picture is left to read: every figure was read by a vision model", 409)
    spec = store.JobSpec(id=store.new_id(KIND), kind=KIND, project_id=pid, stop_grace_s=30.0,
                         title="Reading pictures in documents" + (" (after upload)" if auto else ""),
                         params={"project_id": pid, "file_ids": list(file_ids or []), "auto": auto})
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        raise PicturesError(f"could not start reading pictures: {e}", e.status if e.status >= 400 else 503) from e
    log.info("pictures: started %s for %s (%d file(s))", spec.id, pid, len(todo))
    return {"id": spec.id, "running": True, "files": len(todo), "already_running": False}


def after_parse(pid: str, source: Any) -> None:
    """The upload's background parse ended: read the new file's pictures when a reader is set (never raises)."""
    try:
        if not getattr(source, "ready", False) or reader_state()["mode"] == "off":
            return
        c = source_counts(pid, source)
        if not (c["pictures_ocr"] or c["scanned_ocr"]):
            return
        start(pid, auto=True)
    except PicturesError as e:
        log.info("pictures after parse of %s: %s", getattr(source, "filename", "?"), e)
    except Exception:  # reading pictures is a follow-up; the upload stands
        log.exception("pictures after parse failed for %s", getattr(source, "filename", "?"))


def run_in_job(pid: str, file_ids: list[str] | None, *, stop: threading.Event,
               progress: Callable[[dict[str, Any]], None] = lambda _s: None) -> dict[str, Any]:
    """The job process side: read every pending file's figures with the seated reader; files parsed while it runs
    are picked up before it ends. A mined file (or one a mining job holds) is skipped with the reason."""
    from finetune_studio.data.mining.figures import read_file_figures
    from finetune_studio.knowledge import sources as sources_mod
    from finetune_studio.models.vision import Reader, VisionError

    try:
        reader = Reader()
    except VisionError as e:
        raise RuntimeError(f"Vision reader: {e}") from e
    tried: set[str] = set()
    result: dict[str, Any] = {"files": 0, "figures_read": 0, "failed": [], "skipped": [], "reader": reader.label}
    started = time.time()
    try:
        while not stop.is_set():
            mined, busy = _mined(pid), _mining_busy(pid)
            todo = [s for s in pending_files(pid, file_ids) if s.id not in tried]
            if not todo:
                break
            for i, s in enumerate(todo, 1):
                if stop.is_set():
                    break
                tried.add(s.id)
                if s.id in mined or s.id in busy:
                    result["skipped"].append(f"{s.filename}: already mined (Reset it in Mining to read its pictures)")
                    continue
                progress({"phase": "reading", "file": s.filename, "done": i - 1, "total": len(todo)})
                src = sources_mod.get_source(pid, s.id) or s

                def note(msg: str, _name: str = s.filename) -> None:
                    progress({"phase": "reading", "file": _name, "message": msg})

                res = read_file_figures(pid, src, reader, note)
                result["files"] += int(res.read > 0)
                result["figures_read"] += res.read
                result["failed"].extend(f"{s.filename}: {f}" for f in res.failed[:10])
                if res.skipped:
                    result["skipped"].append(f"{s.filename}: {res.skipped}")
    finally:
        reader.close()
    result["seconds"] = round(time.time() - started, 1)
    result["stopped"] = stop.is_set()
    return result
