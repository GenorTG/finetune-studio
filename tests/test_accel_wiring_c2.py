"""GPU-first wiring in webui / RAG / CLI (lane C2). Torch and vendor CLIs are faked."""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finetune_studio import accel
from finetune_studio.accel import device as accel_device
from finetune_studio.data.rag_portable import devices as rag_devices
from finetune_studio.data.rag_portable import embedders, rerankers
from finetune_studio.data.rag_portable.export_config import RagExportConfig
from finetune_studio.webui import gpu_probe
from finetune_studio.webui.routes import system as system_routes

GIB = 1024 ** 3
SERVER = Path(accel_device.__file__).parents[1] / "data" / "rag_portable" / "standalone_server.py"


def _acc(kind: str, index: int = 0, degraded: str = "") -> accel_device.Accelerator:
    gpu = kind != "cpu"
    return accel_device.Accelerator(
        kind=kind, index=index, name=f"fake-{kind}", total_gb=24.0 if gpu else 0.0,
        free_gb=20.0 if gpu else 0.0, compute_capability=(8, 6) if gpu else (0, 0),
        supports_bf16=gpu, supports_flash_attention=False, supports_4bit=gpu,
        runtime=kind.upper(), degraded_reason=degraded)


@pytest.fixture
def fake_acc(monkeypatch: pytest.MonkeyPatch):
    def use(acc: accel_device.Accelerator) -> accel_device.Accelerator:
        monkeypatch.setattr(accel_device, "_cached", acc)
        return acc
    yield use
    accel_device.reset_cache()


# ── Studio device resolution ────────────────────────────────────────────────
@pytest.mark.parametrize("kind,index,want", [
    ("cuda", 1, "cuda:1"), ("rocm", 0, "cuda:0"), ("xpu", 0, "xpu:0"),
    ("mps", 0, "mps"), ("cpu", 0, "cpu"),
])
def test_auto_is_gpu_first_per_vendor_cpu_last(fake_acc: Any, kind: str, index: int, want: str) -> None:
    fake_acc(_acc(kind, index))
    for spelling in ("auto", "", None, " AUTO "):
        assert rag_devices.resolve_device(spelling) == want


def test_explicit_device_is_honoured(fake_acc: Any) -> None:
    fake_acc(_acc("cuda"))
    assert rag_devices.resolve_device("cpu") == "cpu"
    assert rag_devices.resolve_device("cuda:3") == "cuda:3"


def test_batch_size_per_device() -> None:
    assert rag_devices.encode_batch_size("cpu") == rag_devices.CPU_BATCH
    for dev in ("cuda:0", "xpu:0", "mps"):
        assert rag_devices.encode_batch_size(dev) == rag_devices.GPU_BATCH > rag_devices.CPU_BATCH


class _FakeST:
    seen: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, target: str, **kw: Any) -> None:
        _FakeST.seen.append((target, kw))

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **kw: Any) -> np.ndarray:
        _FakeST.kw = kw  # type: ignore[attr-defined]
        return np.ones((len(texts), 4), dtype=np.float32)

    def predict(self, pairs: list, **kw: Any) -> list[float]:
        _FakeST.kw = kw  # type: ignore[attr-defined]
        return [0.5] * len(pairs)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _FakeST.seen = []
    mod = types.SimpleNamespace(SentenceTransformer=_FakeST, CrossEncoder=_FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    import finetune_studio.data.shared_models as sm
    monkeypatch.setattr(sm, "hf_cache_dir", lambda: tmp_path)
    return _FakeST


def test_embedder_defaults_to_auto_and_uses_gpu(fake_acc: Any, fake_st: Any) -> None:
    fake_acc(_acc("xpu"))
    encode, _ = embedders.get_embedder(name="some/model")
    assert fake_st.seen[-1][1]["device"] == "xpu:0"
    encode(["a", "b"])
    assert fake_st.kw["batch_size"] == rag_devices.GPU_BATCH


def test_embedder_cpu_only_without_gpu(fake_acc: Any, fake_st: Any) -> None:
    fake_acc(_acc("cpu"))
    encode, _ = embedders.get_embedder(name="some/model")
    assert fake_st.seen[-1][1]["device"] == "cpu"
    encode(["a"])
    assert fake_st.kw["batch_size"] == rag_devices.CPU_BATCH


def test_reranker_defaults_to_auto(fake_acc: Any, fake_st: Any) -> None:
    fake_acc(_acc("cuda"))
    rerank, _ = rerankers.get_reranker(name="cross/model")
    assert fake_st.seen[-1][1]["device"] == "cuda:0"
    assert rerank("q", ["d1", "d2"]) == [0.5, 0.5]
    assert fake_st.kw["batch_size"] == rag_devices.GPU_BATCH


def test_rag_entrypoints_default_to_auto() -> None:
    import inspect

    from finetune_studio.data.rag_portable.store import PortableRAG
    for fn in (embedders.get_embedder, rerankers.get_reranker,
               PortableRAG.build_from_directory, PortableRAG.rebuild_vectors):
        assert inspect.signature(fn).parameters["device"].default == "auto", fn
    assert RagExportConfig().device == "auto"
    assert RagExportConfig(device="xpu:1").device == "xpu:1"
    with pytest.raises(ValueError):
        RagExportConfig(device="tpu")


# ── shipped standalone package: self-contained resolver ────────────────────
def _load_server() -> types.ModuleType:
    import importlib.util
    spec = importlib.util.spec_from_file_location("standalone_server_under_test", SERVER)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fake_torch(cuda: bool = False, xpu: bool = False, mps: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        xpu=types.SimpleNamespace(is_available=lambda: xpu),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)))


@pytest.mark.parametrize("torch_kw,want", [
    ({"cuda": True}, "cuda"), ({"xpu": True}, "xpu"), ({"mps": True}, "mps"),
    ({"cuda": True, "xpu": True}, "cuda"), ({}, "cpu"),
])
def test_standalone_auto_resolver(monkeypatch: pytest.MonkeyPatch, torch_kw: dict, want: str) -> None:
    srv = _load_server()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**torch_kw))
    assert srv.DEFAULT_CONFIG["device"] == "auto"
    assert srv._resolve_device("auto") == want
    assert srv._resolve_device("") == want
    assert srv._resolve_device("cpu") == "cpu"


def test_standalone_explicit_unavailable_falls_back(monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture) -> None:
    srv = _load_server()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    assert srv._resolve_device("cuda:1") == "cpu"
    assert "unavailable" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert srv._resolve_device("cuda:1") == "cuda:1"
    assert srv._resolve_device("mps") == "cuda"   # asked for the wrong vendor -> best available


def test_standalone_resolver_does_not_need_finetune_studio() -> None:
    code = (
        "import importlib.abc, importlib.util, sys, types\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'finetune_studio': raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.modules['torch'] = types.SimpleNamespace(\n"
        "    cuda=types.SimpleNamespace(is_available=lambda: True), backends=None)\n"
        f"spec = importlib.util.spec_from_file_location('srv', {str(SERVER)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert m._resolve_device('auto') == 'cuda'\n"
        "assert not any(k.split('.')[0] == 'finetune_studio' for k in sys.modules)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_package_install_picks_gpu_wheel_cpu_last() -> None:
    from finetune_studio.data.rag_portable import mcp_package
    sh = mcp_package._INSTALL_SH
    assert sh.index("nvidia-smi") < sh.index("rocm-smi") < sh.index("whl/cpu")
    assert "TORCH_INDEX_URL" in sh and "Darwin" in sh and "whl/xpu" in sh


# ── vendor-neutral VRAM probes ──────────────────────────────────────────────
def test_parse_rocm_smi_json() -> None:
    txt = ('{"card0": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "1073741824",'
           ' "Card Series": "Radeon RX 7900 XTX"}, "system": {"Driver version": "6.8"}}')
    [d] = gpu_probe.parse_rocm_smi(txt)
    assert d["name"] == "Radeon RX 7900 XTX" and d["index"] == 0
    assert d["total_gb"] == 23.98 and d["used_gb"] == 1.0 and d["source"] == "rocm-smi"
    assert gpu_probe.parse_rocm_smi("not json") == []


def test_parse_xpu_smi_and_sycl() -> None:
    disc = '{"device_list":[{"device_id":0,"device_name":"Intel(R) Arc(TM) A770","memory_physical_size_byte":"17179869184"}]}'
    stats = {0: '{"device_level":[{"metrics_type":"XPU_STATS_MEMORY_USED","value":2048}]}'}
    [d] = gpu_probe.parse_xpu_smi(disc, stats)
    assert d["total_gb"] == 16.0 and d["used_gb"] == 2.0 and d["pct"] == 12.5
    [s] = gpu_probe.parse_sycl_ls("[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 1.3\n"
                                  "[opencl:cpu][opencl:0] Intel CPU\n")
    assert s["source"] == "sycl-ls" and "Arc" in s["name"]


def test_parse_rocm_pids() -> None:
    txt = "PID  PROCESS NAME  GPU(s)  VRAM USED  SDMA USED  CU OCCUPANCY\n1234  python3  1  2147483648  0  0\n"
    assert gpu_probe.parse_rocm_pids(txt) == [{"pid": 1234, "vram_mib": 2048}]


def test_missing_nvidia_smi_does_not_blank_an_amd_host(monkeypatch: pytest.MonkeyPatch) -> None:
    amd = [{"index": 0, "name": "AMD", "used_gb": 1.0, "total_gb": 24.0, "pct": 4.2, "source": "rocm-smi"}]
    monkeypatch.setattr(gpu_probe, "_torch_devices", list)       # CPU torch wheel
    monkeypatch.setattr(gpu_probe, "_nvidia_smi", list)          # tool absent
    monkeypatch.setattr(gpu_probe, "_rocm_smi", lambda: amd)
    assert gpu_probe.vram_devices() == amd


def test_a_crashing_probe_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list:
        raise RuntimeError("driver exploded")
    monkeypatch.setattr(gpu_probe, "_torch_devices", boom)
    monkeypatch.setattr(gpu_probe, "_nvidia_smi", lambda: [{"index": 0, "name": "N", "used_gb": 1.0,
                                                           "total_gb": 8.0, "pct": 12.5, "source": "nvidia-smi"}])
    assert gpu_probe.vram_devices()[0]["name"] == "N"


def test_gpu_snapshot_is_vendor_neutral(monkeypatch: pytest.MonkeyPatch, fake_acc: Any) -> None:
    fake_acc(_acc("rocm"))
    monkeypatch.setattr(gpu_probe, "vram_devices", lambda: [
        {"index": 0, "name": "AMD", "used_gb": 4.0, "total_gb": 24.0, "pct": 16.7, "source": "rocm-smi"}])
    monkeypatch.setattr(gpu_probe, "_consumers", lambda: [{"pid": 1, "vram_mib": 4096}])
    free, top = gpu_probe.free_and_consumers()
    assert free == 20 * 1024 and top == [{"pid": 1, "vram_mib": 4096}]


# ── endpoints ───────────────────────────────────────────────────────────────
@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(system_routes.router)
    return TestClient(app)


def test_accelerator_endpoint_shape(client: TestClient, fake_acc: Any) -> None:
    fake_acc(_acc("cuda"))
    d = client.get("/api/system/accelerator").json()
    assert {"accelerator", "torch", "llama_cpp", "hardware", "policy", "degraded_reason"} <= d.keys()
    assert d["accelerator"]["torch_device"] == "cuda:0" and d["accelerator"]["is_gpu"] is True
    assert d["degraded_reason"] == ""
    assert "backends" in d["llama_cpp"] and isinstance(d["hardware"], list)


def test_accelerator_endpoint_reports_degraded(client: TestClient, fake_acc: Any) -> None:
    fake_acc(_acc("cpu", degraded="GPU hardware detected (NVIDIA) but this PyTorch is a CPU-only build"))
    d = client.get("/api/system/accelerator").json()
    assert d["accelerator"]["kind"] == "cpu" and "CPU-only build" in d["degraded_reason"]


def test_resources_carries_accelerator_and_vendor_neutral_vram(
        client: TestClient, fake_acc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_acc(_acc("cpu", degraded="no usable device"))
    monkeypatch.setattr(gpu_probe, "vram_devices", lambda: [
        {"index": 0, "name": "Arc", "used_gb": 1.0, "total_gb": 16.0, "pct": 6.2, "source": "xpu-smi"}])
    d = client.get("/api/system/resources").json()
    assert d["vram"][0]["name"] == "Arc"
    assert d["accelerator"]["degraded_reason"] == "no usable device"
    assert d["accelerator"]["is_gpu"] is False


@pytest.mark.parametrize("path", ["templates/_resources.html", "static/js/settings.js"])
def test_ui_surfaces_degraded_banner(path: str) -> None:
    root = Path(system_routes.__file__).parents[1]
    src = (root / path).read_text()
    assert "degraded" in src and 'role="alert"' in src


# ── CLI ─────────────────────────────────────────────────────────────────────
def _described(degraded: str = "", gpu: bool = True) -> dict:
    acc = _acc("cuda" if gpu else "cpu", degraded=degraded).to_dict()
    return {"accelerator": acc, "torch": {"version": "2.14", "cuda": "13.0", "hip": None, "xpu": None},
            "llama_cpp": {"installed": True, "version": "0.3", "gpu_offload": True, "backends": ["CUDA"]},
            "hardware": ["NVIDIA"], "policy": {}}


def test_cli_accel_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from finetune_studio.cli.commands.accel import cmd_accel
    args = types.SimpleNamespace(json=False)
    monkeypatch.setattr(accel, "llama_gpu_kwargs", lambda *a, **k: ({}, []))
    monkeypatch.setattr(accel, "describe", lambda: _described())
    cmd_accel(args)                                    # healthy GPU host: returns, exit 0
    assert "cuda:0" in capsys.readouterr().out
    monkeypatch.setattr(accel, "describe", lambda: _described(gpu=False))
    cmd_accel(args)                                    # genuinely GPU-less host: exit 0
    monkeypatch.setattr(accel, "describe", lambda: _described(degraded="CPU-only torch build", gpu=False))
    with pytest.raises(SystemExit) as e:
        cmd_accel(args)
    assert e.value.code == 1 and "DEGRADED" in capsys.readouterr().out


def test_cli_accel_is_registered() -> None:
    from finetune_studio.cli._parser import build_parser
    from finetune_studio.cli._registry import COMMANDS
    assert "accel" in COMMANDS
    assert build_parser().parse_args(["accel", "--json"]).command == "accel"
