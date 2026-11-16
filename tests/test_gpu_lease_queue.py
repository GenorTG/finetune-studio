"""The Vision reader takes the GPU lease like a build, and GPU jobs wait for each other (2026-11-16, Genor).

Before: the ``pictures`` job an upload starts was ``gpu=False`` but loaded a 12 B vision model into the inference
worker on the GPU while a Smart Brain build held the lease (fine on 24 GB, an OOM on a smaller card). Now:

- a local reader on a GPU takes the lease; a CPU-only host takes none (builds there take none either); an API
  reader takes none;
- a GPU job whose spec says ``queue_gpu`` (the pictures job, a Smart Brain build) waits in the queue behind the
  holder instead of being refused, and the pages say behind what ("waiting for the GPU: a Smart Brain build is
  running"); other GPU jobs are still refused with 409;
- the memory check uses the pool the reader really loads from: the card's own memory (discrete), the shared RAM
  (unified-memory GPU: APU / iGPU / Jetson / Apple), the RAM (CPU only) — ``accel.memory_budget``;
- the exported package's runtime loads on the CPU with no GPU and falls back to the CPU when a GPU has no room.

Unified-memory hardware is not on the dev box: those paths are mock-tested only (the accelerator is faked).
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio.accel.device import Accelerator
from finetune_studio.jobs import launch, store
from finetune_studio.jobs.store import JobSpec
from finetune_studio.knowledge import pictures

ROOT = Path(__file__).resolve().parents[1]

DISCRETE = Accelerator(kind="cuda", index=0, name="RTX 3090", total_gb=24.0, free_gb=20.0, compute_capability=(8, 6),
                       supports_bf16=True, supports_flash_attention=True, supports_4bit=True, runtime="CUDA 13.0")
# An APU (Strix Halo class) under ROCm: torch reports a small VRAM carve-out, the GPU really uses the system RAM.
UNIFIED = Accelerator(kind="rocm", index=0, name="Radeon 8060S", total_gb=0.5, free_gb=0.25, compute_capability=(11, 5),
                      supports_bf16=True, supports_flash_attention=False, supports_4bit=True, runtime="ROCm 7.0",
                      unified_memory=True)
CPU = Accelerator(kind="cpu", index=0, name="CPU (x86_64)", total_gb=0.0, free_gb=0.0, compute_capability=(0, 0),
                  supports_bf16=False, supports_flash_attention=False, supports_4bit=False, runtime="CPU")
COST = {"name": "gemma-4-12b.gguf", "gb": 7.8, "path": "/m/gemma-4-12b.gguf"}


def _spec(gpu: bool = False, queue: bool = False, **params) -> JobSpec:
    return JobSpec(id=store.new_id("selftest"), kind="selftest", params=params, gpu=gpu, queue_gpu=queue,
                   stop_grace_s=5, title="t")


def _accel(monkeypatch, acc: Accelerator, *, offload: bool = True) -> None:
    import finetune_studio.accel as accel_pkg
    from finetune_studio.accel import llama

    monkeypatch.setattr(accel_pkg, "get_accelerator", lambda refresh=False: acc)
    monkeypatch.setattr(accel_pkg, "llama_support",
                        lambda refresh=False: llama.LlamaSupport(installed=True, gpu_offload=offload))


def _ram(monkeypatch, available_gb: float, total_gb: float = 128.0) -> None:
    import psutil
    gib = 1024 ** 3
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=available_gb * gib, total=total_gb * gib))


# ── accel: unified memory and the memory budget ─────────────────────────────

def test_unified_memory_comes_from_the_driver_flag_or_the_env(monkeypatch) -> None:
    from finetune_studio.accel import device

    monkeypatch.delenv("FTS_UNIFIED_MEMORY", raising=False)
    assert device._unified(SimpleNamespace(is_integrated=True)) is True
    assert device._unified(SimpleNamespace(is_integrated=False)) is False
    assert device._unified(SimpleNamespace()) is False                 # an older torch: no flag, not unified
    assert device._unified(None, default=True) is True                 # Apple MPS
    monkeypatch.setenv("FTS_UNIFIED_MEMORY", "1")                       # a driver that does not flag its iGPU
    assert device._unified(SimpleNamespace(is_integrated=False)) is True
    monkeypatch.setenv("FTS_UNIFIED_MEMORY", "0")
    assert device._unified(SimpleNamespace(is_integrated=True)) is False


def test_cuda_probe_marks_an_integrated_gpu_unified(monkeypatch) -> None:
    from finetune_studio.accel import device

    monkeypatch.delenv("FTS_UNIFIED_MEMORY", raising=False)
    props = SimpleNamespace(name="Orin", total_memory=8 * 1024 ** 3, major=8, minor=7, is_integrated=True)

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(_i):
            return props

        @staticmethod
        def mem_get_info(_i):
            return (6 * 1024 ** 3, 8 * 1024 ** 3)

        @staticmethod
        def device(_i):
            import contextlib
            return contextlib.nullcontext()

        @staticmethod
        def is_bf16_supported(including_emulation=False):
            return True

    torch = SimpleNamespace(cuda=_Cuda, version=SimpleNamespace(cuda="12.6", hip=None))
    acc = device._cuda_like(torch, None)
    assert acc is not None and acc.unified_memory and acc.is_gpu
    props.is_integrated = False
    assert device._cuda_like(torch, None).unified_memory is False


def test_memory_budget_uses_the_pool_the_device_really_loads_from(monkeypatch) -> None:
    from finetune_studio.accel import memory, ops

    _ram(monkeypatch, 40.0, 128.0)
    monkeypatch.setattr(ops, "mem_info_gb", lambda acc=None: (0.25, 0.5))        # the APU's carve-out
    shared = memory.memory_budget(UNIFIED)
    assert (shared.pool, shared.free_gb, shared.total_gb) == ("shared", 40.0, 128.0)
    system = memory.memory_budget(CPU)
    assert (system.pool, system.free_gb, system.device) == ("system", 40.0, "cpu")
    monkeypatch.setattr(ops, "mem_info_gb", lambda acc=None: (20.0, 24.0))
    dev = memory.memory_budget(DISCRETE)
    assert (dev.pool, dev.free_gb, dev.total_gb, dev.device) == ("device", 20.0, 24.0, "cuda:0")

    def broken(acc=None):
        raise RuntimeError("driver busy")
    monkeypatch.setattr(ops, "mem_info_gb", broken)
    assert memory.memory_budget(DISCRETE).known is False                   # unknown, never a made-up 0 GB


# ── the pictures job: lease, pool, waiting ───────────────────────────────────

def test_the_reader_takes_the_lease_on_a_discrete_or_unified_gpu_only(monkeypatch) -> None:
    monkeypatch.setattr(pictures, "local_reader_cost", lambda: COST)
    _accel(monkeypatch, DISCRETE)
    assert pictures.reader_takes_gpu() is True
    _accel(monkeypatch, UNIFIED)
    assert pictures.reader_takes_gpu() is True                         # an APU shares its RAM with a build too
    _accel(monkeypatch, CPU)
    assert pictures.reader_takes_gpu() is False
    _accel(monkeypatch, DISCRETE, offload=False)                       # a CPU llama.cpp wheel: it loads in RAM
    assert pictures.reader_takes_gpu() is False
    assert pictures.reader_accelerator().kind == "cpu"
    monkeypatch.setattr(pictures, "local_reader_cost", lambda: None)   # an API reader loads nothing here
    _accel(monkeypatch, DISCRETE)
    assert pictures.reader_takes_gpu() is False


def test_memory_check_per_pool(monkeypatch) -> None:
    from finetune_studio.accel import ops
    from finetune_studio.webui import gpu_probe

    monkeypatch.setattr(pictures, "local_reader_cost", lambda: COST)   # needs 7.8 * 1.35 = 10.5 GB
    # Unified: the 0.5 GB "VRAM" carve-out would refuse every load; the 40 GB of shared RAM is what counts.
    _accel(monkeypatch, UNIFIED)
    monkeypatch.setattr(ops, "mem_info_gb", lambda acc=None: (0.25, 0.5))
    monkeypatch.setattr(gpu_probe, "free_and_consumers", lambda: (256, []))   # never asked on unified
    _ram(monkeypatch, 40.0)
    assert pictures.vram_problem() == ""
    _ram(monkeypatch, 6.0)
    problem = pictures.vram_problem()
    assert "free shared memory" in problem and "6.0 GB free" in problem
    # CPU only: the RAM.
    _accel(monkeypatch, CPU)
    assert "not enough free RAM" in pictures.vram_problem()
    _ram(monkeypatch, 64.0)
    assert pictures.vram_problem() == ""
    # Discrete: the card's own memory, with the programs using it named.
    _accel(monkeypatch, DISCRETE)
    monkeypatch.setattr(ops, "mem_info_gb", lambda acc=None: (20.0, 24.0))
    monkeypatch.setattr(gpu_probe, "free_and_consumers", lambda: (4 * 1024, [{"name": "ComfyUI", "pid": 7}]))
    problem = pictures.vram_problem()
    assert "not enough free GPU memory" in problem and "4.0 GB free" in problem and "ComfyUI" in problem


@pytest.fixture
def _reader_on(monkeypatch):
    """A local reader seated, files waiting to be read, no other model loaded."""
    monkeypatch.setattr(pictures, "reader_state", lambda: {"mode": "local", "label": "Local · g.gguf"})
    monkeypatch.setattr(pictures, "local_reader_cost", lambda: COST)
    monkeypatch.setattr(pictures, "_worker_holds_other_model", lambda: "")
    monkeypatch.setattr(pictures, "_running", lambda pid: None)
    monkeypatch.setattr(pictures, "pending_files", lambda pid, ids=None: [SimpleNamespace(id="s1")])
    started: list[JobSpec] = []

    def fake_start(spec):
        started.append(spec)
        busy = launch.gpu_holder.holder  # type: ignore[attr-defined]
        if busy and spec.queue_gpu:
            return {"id": spec.id, "running": True, "queued": True, "waiting_for": busy}
        return {"id": spec.id, "running": True, "queued": False}
    monkeypatch.setattr(launch, "start", fake_start)
    monkeypatch.setattr(launch, "gpu_holder", lambda **k: {"id": "b"} if launch.gpu_holder.holder else None)
    launch.gpu_holder.holder = ""  # type: ignore[attr-defined]
    return started


def test_behind_a_build_the_upload_queues_instead_of_refusing_or_loading(monkeypatch, _reader_on) -> None:
    _accel(monkeypatch, DISCRETE)
    # The build's embedder fills the card: the memory check must NOT refuse now (the memory is ours and frees).
    monkeypatch.setattr(pictures, "vram_problem", lambda *a: "not enough free GPU memory (build running)")
    launch.gpu_holder.holder = "brain_build job brain_build-x (Updating Smart Brain)"  # type: ignore[attr-defined]
    out = pictures.start("p", auto=True)
    spec = _reader_on[-1]
    assert spec.gpu is True and spec.queue_gpu is True
    assert out["waiting_for_gpu"] is True and out["waiting_for_label"] == "a Smart Brain build is running"
    # With the GPU free the memory check still answers at once (a foreign program fills the card).
    launch.gpu_holder.holder = ""  # type: ignore[attr-defined]
    with pytest.raises(pictures.PicturesError) as e:
        pictures.start("p", auto=True)
    assert e.value.status == 409


def test_cpu_only_and_unified_hosts(monkeypatch, _reader_on) -> None:
    monkeypatch.setattr(pictures, "vram_problem", lambda *a: "")
    _accel(monkeypatch, CPU)
    pictures.start("p")
    assert _reader_on[-1].gpu is False and _reader_on[-1].queue_gpu is False      # no lease on a CPU-only host
    _accel(monkeypatch, UNIFIED)
    pictures.start("p")
    assert _reader_on[-1].gpu is True and _reader_on[-1].queue_gpu is True        # an APU: the lease


def test_the_job_checks_memory_once_it_holds_the_lease(monkeypatch) -> None:
    calls = iter(["the build is still exiting", ""])
    monkeypatch.setattr(pictures, "vram_problem", lambda: next(calls))
    seen = []
    pictures.wait_for_memory(threading.Event(), seen.append, settle_s=5, poll_s=0.01)
    assert seen and seen[0]["phase"] == "waiting_for_memory"
    monkeypatch.setattr(pictures, "vram_problem", lambda: "not enough free GPU memory (ComfyUI)")
    with pytest.raises(RuntimeError, match="ComfyUI"):
        pictures.wait_for_memory(threading.Event(), seen.append, settle_s=0.05, poll_s=0.01)


def test_the_job_kind_passes_the_lease_to_the_reader() -> None:
    src = (ROOT / "src/finetune_studio/jobs/kinds/pictures.py").read_text()
    assert "holds_gpu=ctx.spec.gpu" in src


# ── the queue: in the supervisor ─────────────────────────────────────────────

def test_supervisor_queues_a_waiting_gpu_job_and_starts_it_when_the_lease_frees(tmp_path, monkeypatch) -> None:
    from finetune_studio.supervisor import jobs as sup_jobs
    from finetune_studio.supervisor.events import EventLog

    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    monkeypatch.setattr(sup_jobs, "QUEUE_POLL_S", 0.1)

    async def scenario() -> None:
        events = EventLog(tmp_path / "events.jsonl")
        table = sup_jobs.JobTable(events, cwd=os.getcwd())
        build = _spec(gpu=True, queue=True, steps=200, delay_s=0.05)
        reader = _spec(gpu=True, queue=True, steps=1, delay_s=0.0)
        other = _spec(gpu=True, steps=1)
        cpu = _spec(steps=1, delay_s=0.0)
        for s in (build, reader, other, cpu):
            store.write_spec(s)
        await table.start(build.id)
        d = await table.start(reader.id)
        assert d["queued"] is True and d["running"] is True and build.id in d["waiting_for"]
        assert table.gpu_holder().startswith("selftest job " + build.id)   # the waiting job holds nothing
        assert launch.describe(reader.id)["queued"] is True                # the web side reads the same
        with pytest.raises(sup_jobs.JobBusy):
            await table.start(other.id)                                    # no queue_gpu: refused as before
        await table.start(cpu.id)                                          # a CPU job still runs
        assert any(j["id"] == reader.id and j["queued"] for j in table.snapshot())
        await table.stop(build.id)
        await asyncio.wait_for(table._jobs[build.id].waiter, 30)
        for _ in range(300):
            if reader.id in table._jobs:
                break
            await asyncio.sleep(0.05)
        await asyncio.wait_for(table._jobs[reader.id].waiter, 60)
        assert table.describe(reader.id)["exit"]["status"] == "done"
        assert store.read_queued(reader.id) is None
        kinds = [e["kind"] for e in events.since(0, 200)]
        assert "job_queued_for_gpu" in kinds and "job_dequeued_for_gpu" in kinds
        await table.shutdown()
    asyncio.run(scenario())


def test_a_queued_job_can_be_stopped_before_it_ran(tmp_path, monkeypatch) -> None:
    from finetune_studio.supervisor import jobs as sup_jobs
    from finetune_studio.supervisor.events import EventLog

    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)

    async def scenario() -> None:
        table = sup_jobs.JobTable(EventLog(tmp_path / "events.jsonl"), cwd=os.getcwd())
        holder = _spec(gpu=True, steps=200, delay_s=0.05)
        waiter = _spec(gpu=True, queue=True, steps=1)
        store.write_spec(holder)
        store.write_spec(waiter)
        await table.start(holder.id)
        await table.start(waiter.id)
        d = await table.stop(waiter.id)
        assert d["running"] is False and d["exit"]["status"] == "stopped" and d["pid"] is None
        assert launch.describe(waiter.id)["running"] is False
        await table.shutdown()
    asyncio.run(scenario())


def test_a_queue_record_whose_owner_died_reads_as_crashed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    spec = _spec(gpu=True, queue=True)
    store.write_spec(spec)
    store.write_queued(spec.id, {"waiting_for": "x", "owner_pid": 2 ** 22 + 12345})   # no such process
    d = launch.describe(spec.id)
    assert d["running"] is False and d["exit"]["status"] == "crashed"


# ── the queue: without a supervisor (fts webui, the test suite) ──────────────

def test_without_a_supervisor_the_web_process_holds_the_queue(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FTS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("FTS_JOBS_INPROCESS", "1")
    monkeypatch.setattr(launch, "LOCAL_QUEUE_POLL_S", 0.05)
    first = _spec(gpu=True, steps=100, delay_s=0.05)
    launch.start(first)
    second = _spec(gpu=True, queue=True, steps=2, delay_s=0.0)
    d = launch.start(second)
    assert d["queued"] is True and d["running"] is True and first.id in d["waiting_for"]
    assert d["state"]["phase"] == "waiting_for_gpu"
    assert launch.gpu_holder()["id"] == first.id
    with pytest.raises(launch.JobRejected) as e:
        launch.start(_spec(gpu=True, steps=1))
    assert e.value.status == 409
    launch.stop(first.id)
    launch.wait(first.id, timeout=30)
    done = launch.wait(second.id, timeout=30)
    assert done["exit"]["status"] == "done" and store.read_queued(second.id) is None


# ── Smart Brain builds wait too ─────────────────────────────────────────────

def test_a_gpu_build_queues_and_its_progress_says_behind_what(monkeypatch) -> None:
    from finetune_studio.webui import brain_build_jobs

    src = (ROOT / "src/finetune_studio/webui/brain_build_jobs.py").read_text()
    assert "gpu=gpu, queue_gpu=gpu" in src
    monkeypatch.setattr(brain_build_jobs, "_job", lambda bid: {"id": "b", "running": True, "queued": True,
                                                                "waiting_for": "pictures job p-1 (Reading)",
                                                                "state": None, "exit": None})
    monkeypatch.setattr(brain_build_jobs.db, "get_brain_build",
                        lambda bid: {"id": bid, "project_id": "p", "status": "queued"})
    monkeypatch.setattr(brain_build_jobs.store, "read_spec", lambda jid: SimpleNamespace(params={"mode": "build"}))
    st = brain_build_jobs.status("p", "b1")
    assert st["waiting_for_gpu"] is True and st["waiting_for"] == "pictures are being read by the Vision reader"
    route = (ROOT / "src/finetune_studio/webui/routes/brain.py").read_text()
    assert 'phase, message = "queued", f"waiting for the GPU: ' in route


def test_a_build_stopped_while_it_waited_reads_as_stopped(monkeypatch) -> None:
    from finetune_studio.webui import brain_build_jobs

    failed = []
    monkeypatch.setattr(brain_build_jobs.builds_db, "fail_if_active", lambda bid, msg: failed.append(msg) or True)
    monkeypatch.setattr(brain_build_jobs.db, "get_brain_build", lambda bid: {"id": bid, "status": "error"})
    brain_build_jobs._settle({"id": "b1", "status": "queued"},
                             {"running": False, "pid": None, "exit": {"status": "stopped",
                                                                      "detail": "stopped while waiting for the GPU"}})
    assert failed == ["stopped on request (it was still waiting for the GPU)"]    # the page says "Stopped"


# ── the pages say it waits ───────────────────────────────────────────────────

def test_the_pages_show_the_waiting_state() -> None:
    brain = (ROOT / "src/finetune_studio/webui/templates/brain.html").read_text()
    assert "' · waiting for the GPU: ' + (job.waiting_for_label" in brain
    assert "r.waiting_for_gpu ? 'Reading pictures in '" in brain
    assert "p.message ? p.message.charAt(0).toUpperCase()" in brain            # the build's "Waiting for the GPU: …"
    assert "pill: 'Waiting', tone: 'solid-amber', title: 'Waiting for the GPU…'" in brain   # the top card too
    data = (ROOT / "src/finetune_studio/webui/templates/project_data.html").read_text()
    assert "' · waiting for GPU'" in data and "p.pictures_job" in data


def test_pipeline_flags_name_the_waiting_pictures_job(monkeypatch) -> None:
    from finetune_studio.knowledge import status as kstatus

    src = SimpleNamespace(id="s1", file_id="f1", status="ready", error=None, chunk_count=3, parser="pdf",
                          edited_at=None)
    monkeypatch.setattr(kstatus, "list_sources", lambda pid: [src])
    monkeypatch.setattr(kstatus.index, "state", lambda pid: SimpleNamespace(indexed=set(), stale=set()))
    monkeypatch.setattr(kstatus, "source_counts", lambda pid, s: {"pictures_ocr": 2, "scanned_ocr": 0, "read": 0})
    monkeypatch.setattr(pictures, "_running", lambda pid: {"id": "j", "queued": True,
                                                           "waiting_for": "brain_build job b (Updating)"})
    assert kstatus.pipeline_status("p")["f1"]["pictures_job"] == "waiting: a Smart Brain build is running"
    monkeypatch.setattr(pictures, "_running", lambda pid: {"id": "j", "queued": False})
    assert kstatus.pipeline_status("p")["f1"]["pictures_job"] == "reading"


# ── the exported package's runtime: CPU-only and GPU, no CUDA assumed ───────

def _server(monkeypatch, torch_mod):
    from finetune_studio.smart_brain import standalone_server as srv
    if torch_mod is None:
        monkeypatch.setitem(sys.modules, "torch", None)                  # import torch -> ImportError
    else:
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return srv


def _fake_torch(*, cuda: bool = False, mps: bool = False, xpu: bool = False) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.cuda = SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
    t.xpu = SimpleNamespace(is_available=lambda: xpu)
    t.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    t.float16 = "float16"
    return t


def test_package_device_resolution_cpu_only_and_gpu(monkeypatch) -> None:
    srv = _server(monkeypatch, None)
    assert srv._resolve_device("auto") == "cpu"                          # no torch at all
    srv = _server(monkeypatch, _fake_torch())
    assert srv._resolve_device("auto") == "cpu"                          # CPU wheel / CUDA_VISIBLE_DEVICES=""
    assert srv._resolve_device("cuda") == "cpu"                          # an explicit GPU it cannot serve
    srv = _server(monkeypatch, _fake_torch(cuda=True))                   # NVIDIA, ROCm, Jetson, a ROCm APU
    assert srv._resolve_device("auto") == "cuda" and srv._resolve_device("cpu") == "cpu"
    srv = _server(monkeypatch, _fake_torch(mps=True))                    # Apple unified memory
    assert srv._resolve_device("auto") == "mps"
    srv = _server(monkeypatch, _fake_torch(xpu=True))
    assert srv._resolve_device("auto") == "xpu"


def test_package_models_fall_back_to_cpu_when_the_gpu_has_no_room(monkeypatch, capsys) -> None:
    srv = _server(monkeypatch, _fake_torch(cuda=True))

    def load(dev: str):
        if dev != "cpu":
            raise RuntimeError("HIP out of memory. Tried to allocate 1.2 GiB")
        return f"model@{dev}"
    assert srv._load_with_cpu_fallback("bundled embedder", "cuda", load) == ("model@cpu", "cpu")
    assert "loading it on the CPU instead" in capsys.readouterr().err

    def broken(dev: str):
        raise ValueError("config.json is missing")
    with pytest.raises(ValueError):                                     # only an OOM is retried
        srv._load_with_cpu_fallback("bundled embedder", "cuda", broken)
    with pytest.raises(RuntimeError):                                   # on the CPU there is nowhere to go
        srv._load_with_cpu_fallback("x", "cpu", lambda d: (_ for _ in ()).throw(RuntimeError("out of memory")))


def test_package_embedder_reranker_and_image_model_load_through_the_fallback(monkeypatch, tmp_path) -> None:
    srv = _server(monkeypatch, _fake_torch(cuda=True))
    loads: list[tuple[str, str, dict]] = []

    class _ST:
        def __init__(self, path, device="cpu", **kw):
            loads.append(("st", device, kw))
            if device != "cpu":
                raise RuntimeError("CUDA out of memory")
            self.device = device

        def encode(self, items, **_kw):
            import numpy as np
            return np.ones((len(items), 4), dtype=np.float32)

    class _CE:
        def __init__(self, path, device="cpu", **kw):
            loads.append(("ce", device, kw))
            if device != "cpu":
                raise RuntimeError("out of memory")

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _ST
    st_mod.CrossEncoder = _CE
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    enc = srv.ImageEncoder(tmp_path / "image", None, "cuda")
    assert enc.device == "cpu" and enc.dim == 4
    corpus = SimpleNamespace(dir=tmp_path, device="cuda", _local_embedder=None, _local_embedder_failed=False,
                             _reranker=None, _reranker_failed=False, _looks_like_model=lambda d: True)
    emb = srv.Corpus._load_embedder(corpus)
    assert emb is not None and emb.device == "cpu" and corpus._local_embedder_failed is False
    rr = srv.Corpus._load_reranker(corpus)
    assert rr is not None
    ce = [x for x in loads if x[0] == "ce"]
    assert ce[0][1] == "cuda" and "model_kwargs" in ce[0][2]            # fp16 asked only on the GPU
    assert ce[1][1] == "cpu" and "model_kwargs" not in ce[1][2]


def test_package_runtime_has_no_hard_coded_device() -> None:
    """The shipped server picks its device itself: no ``.cuda()``, no ``{"": 0}``, no bf16, no bitsandbytes."""
    src = (ROOT / "src/finetune_studio/smart_brain/standalone_server.py").read_text()
    for bad in (".cuda()", '{"": 0}', "bfloat16", "bitsandbytes", 'device="cuda"', "device='cuda'"):
        assert bad not in src, bad


def test_accelerator_dict_reports_unified_memory() -> None:
    d = UNIFIED.to_dict()
    assert d["unified_memory"] is True and d["is_gpu"] is True
    assert dataclasses.replace(UNIFIED, kind="cpu").is_gpu is False


def test_the_job_process_checks_memory_without_opening_a_gpu_context(monkeypatch) -> None:
    """Live 2026-11-16: the job's own check probed torch and held a 256 MB CUDA context on the card the reader was
    loading on. In the job the device comes from the web process and the vendor tool answers."""
    from finetune_studio.accel import ops
    from finetune_studio.webui import gpu_probe

    monkeypatch.setattr(pictures, "local_reader_cost", lambda: COST)

    def no_torch(*_a, **_k):
        raise AssertionError("the job must not ask torch")
    monkeypatch.setattr(ops, "mem_info_gb", no_torch)
    monkeypatch.setattr(pictures, "reader_accelerator", no_torch)
    asked = {}

    def cli(**kw):
        asked.update(kw)
        return 20 * 1024, []
    monkeypatch.setattr(gpu_probe, "free_and_consumers", cli)
    assert pictures.vram_problem(DISCRETE.to_dict()) == ""
    assert asked == {"cli_only": True, "index": 0, "name": "RTX 3090"}
    _ram(monkeypatch, 3.0)
    assert "free shared memory" in pictures.vram_problem(UNIFIED.to_dict())     # psutil only, no torch
    monkeypatch.setattr(gpu_probe, "_torch_devices", no_torch)
    monkeypatch.setattr(gpu_probe, "_nvidia_smi", lambda: [{"index": 0, "name": "RTX 3090", "used_gb": 1.0,
                                                            "total_gb": 24.0, "pct": 4.0, "source": "nvidia-smi"}])
    monkeypatch.setattr(gpu_probe, "_consumers", lambda index=None: [])
    assert gpu_probe.vram_devices(cli_only=True)[0]["source"] == "nvidia-smi"


def test_the_start_hands_the_device_to_the_job(monkeypatch, _reader_on) -> None:
    monkeypatch.setattr(pictures, "vram_problem", lambda *a: "")
    _accel(monkeypatch, DISCRETE)
    pictures.start("p")
    assert _reader_on[-1].params["reader_device"]["name"] == "RTX 3090"
    src = (ROOT / "src/finetune_studio/jobs/kinds/pictures.py").read_text()
    assert 'reader_device=ctx.params.get("reader_device")' in src
