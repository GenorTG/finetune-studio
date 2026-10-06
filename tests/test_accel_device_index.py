"""Chosen GPU != index 0 (``CUDA_VISIBLE_DEVICES=1,0`` layouts, ``FTS_DEVICE=cuda:1``, best card second).

transformers' ``TrainingArguments._setup_devices`` hard-codes ``cuda:0`` and ``n_gpu = device_count()``,
``torch.cuda.set_device`` is per thread and starts at 0, and PEFT / bitsandbytes follow the current
device. Torch's CUDA namespace is faked, so this runs on any host.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import torch

from finetune_studio import accel
from finetune_studio.accel import device, ops


def _acc(kind: str = "cuda", index: int = 1, count: int = 2) -> device.Accelerator:
    gpu = kind != "cpu"
    return device.Accelerator(
        kind=kind, index=index, name="RTX 3090" if gpu else "CPU", total_gb=24.0 if gpu else 0.0,
        free_gb=20.0 if gpu else 0.0, compute_capability=(8, 6) if gpu else (0, 0),
        supports_bf16=gpu, supports_flash_attention=gpu, supports_4bit=gpu, runtime="CUDA 13.0",
        device_count=count)


class _FakeNs:
    """``torch.cuda`` stand-in that records ``set_device`` (per-thread current device)."""

    def __init__(self) -> None:
        self.current = 0
        self.calls: list[int] = []

    def set_device(self, idx: Any) -> None:
        self.current = int(getattr(idx, "index", idx))
        self.calls.append(self.current)


@pytest.fixture
def ns(monkeypatch: pytest.MonkeyPatch) -> _FakeNs:
    fake = _FakeNs()
    monkeypatch.setattr(ops, "_ns", lambda acc: fake if acc.torch_namespace else None)
    return fake


@pytest.fixture
def use_acc(monkeypatch: pytest.MonkeyPatch):
    def use(acc: device.Accelerator) -> device.Accelerator:
        monkeypatch.setattr(device, "_cached", acc)
        return acc
    yield use
    device.reset_cache()


# ── activate ─────────────────────────────────────────────────────────────────

def test_activate_makes_the_selected_gpu_current(ns: _FakeNs, use_acc: Any) -> None:
    use_acc(_acc("cuda", 1))
    accel.activate()
    assert ns.current == 1


@pytest.mark.parametrize("kind", ["cpu", "mps"])
def test_activate_is_a_noop_without_an_indexed_gpu(ns: _FakeNs, kind: str) -> None:
    accel.activate(_acc(kind, 0))
    assert ns.calls == []


# ── Trainer arguments ────────────────────────────────────────────────────────

class _HardCodedArgs:
    """The relevant slice of ``TrainingArguments``: cached device = cuda:0, n_gpu = device_count."""

    use_cpu = False

    def __init__(self, count: int = 2) -> None:
        self.count = count
        self.setup_runs = 0

    @functools.cached_property
    def _setup_devices(self) -> torch.device:
        self.setup_runs += 1
        self._n_gpu = self.count
        return torch.device("cuda:0")

    @property
    def device(self) -> torch.device:
        return self._setup_devices

    @property
    def n_gpu(self) -> int:
        return self._n_gpu


def test_trainer_args_follow_the_accelerator_not_cuda0(ns: _FakeNs) -> None:
    args = _HardCodedArgs(count=2)
    accel.pin_trainer_args(args, _acc("cuda", 1))
    assert args.device == torch.device("cuda:1")
    assert args.n_gpu == 1                      # 2 visible GPUs would otherwise wrap the model in DataParallel
    assert ns.current == 1                      # transformers' own set_device(cuda:0) is undone
    assert args.setup_runs == 1                 # the original setup still ran (accelerate state exists)


def test_trainer_args_on_index_zero_stay_single_device(ns: _FakeNs) -> None:
    args = _HardCodedArgs(count=2)
    accel.pin_trainer_args(args, _acc("cuda", 0))
    assert args.device == torch.device("cuda:0") and args.n_gpu == 1 and ns.current == 0


def test_trainer_args_untouched_on_cpu(ns: _FakeNs) -> None:
    args = _HardCodedArgs()
    assert accel.pin_trainer_args(args, _acc("cpu", 0)) is args
    assert args.setup_runs == 0 and ns.calls == []


def test_trainer_args_untouched_when_the_plan_forces_cpu(ns: _FakeNs) -> None:
    args = _HardCodedArgs()
    args.use_cpu = True
    accel.pin_trainer_args(args, _acc("cuda", 1))
    assert args.setup_runs == 0 and ns.calls == []


def test_sft_config_is_pinned_to_the_selected_gpu(ns: _FakeNs, use_acc: Any, tmp_path: Path) -> None:
    from finetune_studio.training.sft_args import build_sft_training_args
    use_acc(_acc("cuda", 1))
    args = build_sft_training_args(output_dir=str(tmp_path), num_train_epochs=1, bf16=False, use_cpu=False)
    assert args.device == torch.device("cuda:1")
    assert args.n_gpu == 1
    assert ns.current == 1


# ── loaders ──────────────────────────────────────────────────────────────────

def _fake_auto_model(monkeypatch: pytest.MonkeyPatch, ns: _FakeNs) -> list[tuple[int, Any]]:
    """Record (current device at call time, device_map) for every ``from_pretrained``."""
    import transformers
    seen: list[tuple[int, Any]] = []

    class _Auto:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> str:
            seen.append((ns.current, kw.get("device_map")))
            return "model"

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Auto)
    return seen


def test_load_causal_lm_runs_on_the_selected_gpu(ns: _FakeNs, use_acc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.models import hf_loader
    seen = _fake_auto_model(monkeypatch, ns)
    monkeypatch.setattr(ops, "torch_dtype", lambda acc=None: "bf16")
    hf_loader.load_causal_lm("m", acc=use_acc(_acc("cuda", 1)))
    assert seen == [(1, {"": 1})]


def test_load_merge_base_runs_on_the_selected_gpu(ns: _FakeNs, use_acc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.models import hf_loader
    seen = _fake_auto_model(monkeypatch, ns)
    hf_loader.load_merge_base("m", acc=use_acc(_acc("cuda", 1)))
    assert seen == [(1, {"": 1})]            # PeftModel.from_pretrained then loads adapters on cuda:1


def test_hf_generate_activates_the_loaded_card(ns: _FakeNs, use_acc: Any) -> None:
    from finetune_studio.testing.inference import InferenceEngine
    use_acc(_acc("cuda", 1))
    eng = InferenceEngine()
    tok = type("Tok", (), {
        "pad_token_id": 0,
        "apply_chat_template": staticmethod(lambda *a, **k: "hi"),
        "__call__": lambda self, text, **kw: type("B", (dict,), {"to": lambda s, d: s})(),
    })()
    eng.tokenizer = tok
    eng.model = type("M", (), {"device": "cpu", "generate": lambda self, **kw: (_ for _ in ()).throw(RuntimeError("stop"))})()
    with pytest.raises(RuntimeError, match="stop"):
        eng._generate_hf([{"role": "user", "content": "x"}], 8, 0.0, 0.9, 40, 1.1, None)
    assert ns.current == 1


# ── child-process isolation (the fix for every library that hard-codes device 0) ────────────

@pytest.mark.parametrize("kind,env,index,count,want", [
    # 3090 is visible index 1 (CUDA_VISIBLE_DEVICES=1,0 style): compose, do not replace.
    ("cuda", {"CUDA_VISIBLE_DEVICES": "1,0"}, 1, 2, {"CUDA_VISIBLE_DEVICES": "0"}),
    ("cuda", {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, 1, 2, {"CUDA_VISIBLE_DEVICES": "GPU-bbb"}),
    ("cuda", {}, 1, 2, {"CUDA_VISIBLE_DEVICES": "1"}),
    ("cuda", {}, 0, 2, {"CUDA_VISIBLE_DEVICES": "0"}),            # still hides the other card
    ("cuda", {"CUDA_VISIBLE_DEVICES": "0"}, 0, 1, {}),             # already alone at index 0
    ("cuda", {}, 0, 1, {}),
    ("rocm", {"HIP_VISIBLE_DEVICES": "2,3"}, 1, 2, {"HIP_VISIBLE_DEVICES": "3"}),
    ("rocm", {"CUDA_VISIBLE_DEVICES": "2,3"}, 1, 2, {"CUDA_VISIBLE_DEVICES": "3"}),
    ("rocm", {}, 1, 2, {"HIP_VISIBLE_DEVICES": "1"}),
    ("xpu", {}, 1, 2, {"ZE_AFFINITY_MASK": "1"}),
    ("mps", {}, 0, 1, {}),
    ("cpu", {}, 0, 1, {}),
])
def test_isolated_env_makes_the_chosen_card_device_zero(kind: str, env: dict, index: int, count: int, want: dict) -> None:
    assert accel.isolated_env(_acc(kind, index, count), env) == want


def test_training_worker_applies_device_env_before_importing_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from finetune_studio.training import engine as engine_mod
    from finetune_studio.training import worker

    seen: dict[str, Any] = {}

    class _Eng:
        def __init__(self) -> None:
            self.state = type("S", (), {"status": "idle", "error": "", "message": ""})()

        def on_update(self, _cb: Any) -> None: ...

        def _train(self, _data: Any, _prompt: str) -> None:
            seen["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    class _Q:
        def put(self, _msg: Any) -> None: ...

    monkeypatch.setattr(engine_mod, "TrainingEngine", _Eng)
    monkeypatch.setattr(worker, "_config_from_dict", lambda raw: raw)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    worker.training_worker({}, [], "", _Q(), type("E", (), {"is_set": lambda s: False})(),
                           {"CUDA_VISIBLE_DEVICES": "0"})
    assert seen["visible"] == "0"
    monkeypatch.undo()


def test_engine_start_hands_the_child_its_isolated_device_env(monkeypatch: pytest.MonkeyPatch, use_acc: Any) -> None:
    import multiprocessing as mp

    from finetune_studio.training.engine import TrainingConfig, TrainingEngine
    use_acc(_acc("cuda", 1, 2))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    started: dict[str, Any] = {}

    class _Proc:
        def __init__(self, **kw: Any) -> None:
            started.update(kw)

        def start(self) -> None: ...
        def is_alive(self) -> bool: return False

    ctx = type("Ctx", (), {"Queue": lambda s: None, "Event": lambda s: type("Ev", (), {"clear": lambda s: None})(),
                           "Process": lambda s, **kw: _Proc(**kw)})()
    monkeypatch.setattr(mp, "get_context", lambda _m: ctx)
    eng = TrainingEngine()
    eng.start(TrainingConfig(model_path="m"), [{"messages": []}])
    assert started["args"][-1] == {"CUDA_VISIBLE_DEVICES": "0"}


def test_adapter_is_read_onto_the_base_models_device_not_bare_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # PEFT defaults to torch_device="cuda"; safetensors maps that string to cuda:0 (the wrong card).
    import peft

    from finetune_studio.models.hf_loader import load_peft_adapter
    seen: dict[str, Any] = {}

    def fake(base: Any, path: str, **kw: Any) -> str:
        seen.update(kw, path=path)
        return "peft"

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", staticmethod(fake))
    base = type("M", (), {"device": torch.device("cuda:1")})()
    assert load_peft_adapter(base, "/a") == "peft"
    assert seen == {"torch_device": "cuda:1", "path": "/a"}
    assert load_peft_adapter(type("M", (), {"device": torch.device("cpu")})(), "/a") == "peft"
    assert seen["torch_device"] == "cpu"
