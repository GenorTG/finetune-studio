"""llama.cpp's native log is the only place a load failure explains itself (models/llama_native_log.py)."""
from __future__ import annotations

import logging
import sys
import types

from finetune_studio.models import llama_native_log as nl

ERR, WARN, INFO, CONT = 4, 3, 2, 5   # ggml.h enum ggml_log_level


def _feed(level: int, text: str) -> None:
    nl._on_native_log(level, text.encode(), None)


def test_real_load_oom_wording_is_detected_since_the_mark() -> None:
    pos = nl.mark()
    _feed(INFO, "llama_model_loader: loaded meta data with 45 key-value pairs\n")
    assert not nl.oom_since(pos)
    _feed(ERR, "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 7024.41 MiB on device 0: cudaMalloc failed: out of memory\n")
    _feed(ERR, "alloc_tensor_range: failed to allocate CUDA0 buffer of size 7365624064\n")
    assert nl.oom_since(pos)
    assert "cudaMalloc failed" in nl.failure_detail(pos)
    assert not nl.oom_since(nl.mark())   # a later mark does not inherit the old failure


def test_chunks_are_assembled_into_lines_and_continuations_keep_their_level() -> None:
    pos = nl.mark()
    _feed(ERR, "CUDA error: an illegal memory ")
    _feed(CONT, "access was encountered\n")
    _feed(CONT, "  current device: 0, in function launch_mul_mat_q\n")
    lines = nl.lines_since(pos, min_level=ERR)
    assert lines == ["CUDA error: an illegal memory access was encountered",
                     "  current device: 0, in function launch_mul_mat_q"]


def test_info_lines_are_kept_in_the_ring_but_not_counted_as_errors() -> None:
    pos = nl.mark()
    _feed(INFO, "load: model has 28 layers\n")
    assert nl.lines_since(pos) == ["load: model has 28 layers"]
    assert nl.lines_since(pos, min_level=ERR) == []


def test_a_bad_callback_argument_never_raises_into_c() -> None:
    nl._on_native_log(ERR, None, None)   # type: ignore[arg-type]
    nl._on_native_log(ERR, b"\xff\xfe broken utf8\n", None)


def test_install_registers_once_and_survives_a_missing_llama_cpp(monkeypatch) -> None:
    calls: list[object] = []
    stub = types.ModuleType("llama_cpp")
    stub.llama_log_callback = lambda fn: ("trampoline", fn)    # type: ignore[attr-defined]
    stub.llama_log_set = lambda cb, ud: calls.append(cb)       # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    monkeypatch.setattr(nl, "_CALLBACK", None)
    assert nl.install() is True and nl.install() is True
    assert len(calls) == 1                                      # idempotent

    monkeypatch.setattr(nl, "_CALLBACK", None)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)         # import raises ImportError
    assert nl.install() is False


def test_install_tolerates_an_incomplete_llama_cpp(monkeypatch) -> None:
    """A stub/partial module (no llama_log_callback) must not break the loader that calls install()."""
    monkeypatch.setattr(nl, "_CALLBACK", None)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    assert nl.install() is False


def test_levels_follow_the_ggml_enum_when_forwarded_to_python_logging(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llama.cpp")
    pos = nl.mark()
    _feed(INFO, "llama_context: n_ctx = 512\n")
    _feed(WARN, "llama_context: n_ctx_seq (512) > n_ctx_train (0) -- possible training context overflow\n")
    _feed(ERR, "llama_model_load: error loading model\n")
    forwarded = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "llama.cpp"]
    assert forwarded == [(logging.WARNING, "llama_context: n_ctx_seq (512) > n_ctx_train (0) -- possible training context overflow"),
                         (logging.ERROR, "llama_model_load: error loading model")]
    assert nl.failure_detail(pos) == "llama_model_load: error loading model"   # warnings are not the failure reason
