"""The merge after training starts from a free GPU and never pushes the host into OOM (found 2026-11-04, RESULTS §34).

R4 (Qwen3.5-9B QLoRA): ``_do_merge`` did ``del model``, which only dropped its own name; the training function's
``model`` and ``trainer.model`` kept the 4-bit model (~8 GB) on the GPU. The 16-bit base (16.6 GiB) then did not fit
(13.6 GiB free), the merge fell back to CPU RAM with 15 GB free for a 19 GB base, and the kernel's OOM killer killed
another program on the host. Now every training path releases the trainer before merge/export, and a CPU-RAM merge that
cannot fit is refused with a reason (the adapter stays; merge again from Export).
"""
from __future__ import annotations

import gc
import re
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio.models import hf_loader
from finetune_studio.training import engine

ENGINE = Path(engine.__file__).read_text(encoding="utf-8")


class _Big:
    """Stands in for the training model / optimizer state."""


def test_release_trainer_lets_go_of_the_model_and_optimizer() -> None:
    model, opt = _Big(), _Big()
    freed: list[bool] = []
    trainer = SimpleNamespace(model=model, model_wrapped=model, optimizer=opt, lr_scheduler=object(),
                              accelerator=SimpleNamespace(free_memory=lambda: freed.append(True)))
    model_ref, opt_ref = weakref.ref(model), weakref.ref(opt)
    del model, opt
    engine._release_trainer(trainer)
    gc.collect()
    assert model_ref() is None and opt_ref() is None
    assert freed == [True]


def test_every_training_path_frees_the_trainer_before_merge_and_export() -> None:
    merges = [m.start() for m in re.finditer(r"self\._maybe_merge\(None, tokenizer, cfg\.output_dir\)", ENGINE)]
    assert len(merges) == 3  # preference, Unsloth, standard
    for pos in merges:
        before = ENGINE[max(0, pos - 400):pos]
        assert "_release_trainer(trainer)" in before and "del trainer, model" in before
    assert "self._maybe_merge(model, tokenizer, cfg.output_dir)" not in ENGINE
    assert "self._export_gguf_after_train(model, tokenizer)" not in ENGINE


def _gib(n: float) -> int:
    return int(n * 1024**3)


def test_a_cpu_merge_that_cannot_fit_is_refused_with_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_loader, "_weights_bytes", lambda _p: _gib(19))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=_gib(15)))
    with pytest.raises(hf_loader.MergeMemoryError, match=r"needs about 26 GB of RAM, but only 15 GB is free"):
        hf_loader._check_cpu_ram_for_merge("/m/base")


def test_a_cpu_merge_that_fits_goes_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_loader, "_weights_bytes", lambda _p: _gib(19))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=_gib(40)))
    hf_loader._check_cpu_ram_for_merge("/m/base")
    monkeypatch.setattr(hf_loader, "_weights_bytes", lambda _p: 0)   # unknown size (no local weights): no guess
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=_gib(1)))
    hf_loader._check_cpu_ram_for_merge("org/remote-id")


def test_load_merge_base_checks_ram_before_loading_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[object] = []
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained",
                        lambda *a, **k: loads.append(k.get("device_map")))
    monkeypatch.setattr(hf_loader.accel, "activate", lambda _acc: None)
    monkeypatch.setattr(hf_loader, "merge_dtype", lambda _acc=None: None)
    monkeypatch.setattr(hf_loader, "_weights_bytes", lambda _p: _gib(19))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=_gib(15)))
    with pytest.raises(hf_loader.MergeMemoryError):
        hf_loader.load_merge_base("/m/base", acc=SimpleNamespace(is_gpu=False))
    assert loads == []
