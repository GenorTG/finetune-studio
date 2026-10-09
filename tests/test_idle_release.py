"""Idle auto-unload: busy guard, live timeout, comparator purge, memory release."""
import threading
import time

from finetune_studio.testing import inference as inf


def _loaded_engine():
    e = inf.InferenceEngine()
    e.model = object()
    e.model_path = "x"
    e._last_used = time.time() - 10_000
    return e


def test_timeout_is_live_and_defaults_to_five_minutes(monkeypatch):
    monkeypatch.delenv("FTS_IDLE_TIMEOUT", raising=False)
    assert inf.idle_timeout() == 300
    monkeypatch.setenv("FTS_IDLE_TIMEOUT", "42")
    assert inf.idle_timeout() == 42
    monkeypatch.setenv("FTS_IDLE_TIMEOUT", "junk")
    assert inf.idle_timeout() == 300


def test_auto_unload_when_idle(monkeypatch):
    monkeypatch.setenv("FTS_IDLE_TIMEOUT", "1")
    e = _loaded_engine()
    e._auto_unload()
    assert e.model is None


def test_auto_unload_skips_busy_engine(monkeypatch):
    monkeypatch.setenv("FTS_IDLE_TIMEOUT", "1")
    e = _loaded_engine()
    e._busy = 1
    e._auto_unload()
    assert e.model is not None
    e._idle_timer.cancel()


def test_generate_marks_busy_and_touches_last_used(monkeypatch):
    monkeypatch.setenv("FTS_IDLE_TIMEOUT", "0")
    e = _loaded_engine()
    seen = []
    e._generate_hf = lambda *a, **k: seen.append(e._busy) or "ok"
    assert e.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert seen == [1] and e._busy == 0
    assert time.time() - e._last_used < 5


def test_release_idle_memory_clears_parsed_cache():
    from finetune_studio.data.fs import file_library
    file_library._PARSED_CACHE["k"] = {"x": 1}
    inf.release_idle_memory()
    assert not file_library._PARSED_CACHE


def test_busy_counter_thread_safe():
    e = _loaded_engine()
    e._generate_hf = lambda *a, **k: time.sleep(0.01) or "ok"
    ts = [threading.Thread(target=e.generate, args=([{"role": "user", "content": "x"}],)) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert e._busy == 0
    if e._idle_timer:
        e._idle_timer.cancel()
