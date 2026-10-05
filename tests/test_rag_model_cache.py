"""RAG embedder/reranker cache: hit/miss, device keying, idle expiry, release, concurrency, failures."""
from __future__ import annotations

import threading
import time

import pytest

from finetune_studio.data.rag_portable import model_cache as mc
from finetune_studio.data.rag_portable.model_cache import ModelCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_cache(clock: FakeClock | None = None, timeout: int = 300) -> tuple[ModelCache, FakeClock, list[int]]:
    clock = clock or FakeClock()
    freed: list[int] = []
    cache = ModelCache(
        clock=clock, timeout_fn=lambda: timeout,
        release_memory=lambda: freed.append(1), start_timer=False,
    )
    return cache, clock, freed


def test_hit_returns_same_object_and_loads_once() -> None:
    cache, _, _ = make_cache()
    calls: list[int] = []

    def loader() -> object:
        calls.append(1)
        return object()

    a = cache.get_or_load(("embedder", "m", "cpu"), loader)
    b = cache.get_or_load(("embedder", "m", "cpu"), loader)
    assert a is b
    assert len(calls) == 1


def test_key_includes_kind_name_and_device() -> None:
    cache, _, _ = make_cache()
    seen = [
        cache.get_or_load(("embedder", "m", "cpu"), object),
        cache.get_or_load(("embedder", "m", "cuda:0"), object),
        cache.get_or_load(("embedder", "other", "cpu"), object),
        cache.get_or_load(("reranker", "m", "cpu"), object),
    ]
    assert len({id(v) for v in seen}) == 4
    assert len(cache) == 4


def test_idle_expiry_with_fake_clock_drops_only_idle_entries() -> None:
    cache, clock, freed = make_cache(timeout=300)
    cache.get_or_load(("embedder", "old", "cpu"), object)
    clock.now += 200
    cache.get_or_load(("embedder", "fresh", "cpu"), object)
    clock.now += 150  # old idle 350 s, fresh idle 150 s
    assert cache.expire_idle() == [("embedder", "old", "cpu")]
    assert len(cache) == 1
    assert freed == [1]  # accelerator memory handed back
    clock.now += 200
    assert cache.expire_idle() == [("embedder", "fresh", "cpu")]
    assert len(cache) == 0


def test_use_refreshes_idle_clock() -> None:
    cache, clock, _ = make_cache(timeout=300)
    cache.get_or_load(("embedder", "m", "cpu"), object)
    clock.now += 250
    cache.get_or_load(("embedder", "m", "cpu"), object)  # hit -> touched
    clock.now += 250
    assert cache.expire_idle() == []
    assert len(cache) == 1


def test_timeout_zero_disables_expiry() -> None:
    cache, clock, _ = make_cache(timeout=0)
    cache.get_or_load(("embedder", "m", "cpu"), object)
    clock.now += 10_000
    assert cache.expire_idle() == []
    assert len(cache) == 1


def test_busy_entry_is_not_expired() -> None:
    cache, clock, _ = make_cache(timeout=300)
    inside = threading.Event()
    leave = threading.Event()

    def blocking(x: int) -> int:
        inside.set()
        leave.wait(5)
        return x

    fn, _ = cache.get_or_load(
        ("embedder", "m", "cpu"), lambda: (blocking, "info"),
        lambda v, wrap: (wrap(v[0]), v[1]),
    )
    t = threading.Thread(target=fn, args=(1,))
    t.start()
    assert inside.wait(5)
    clock.now += 1000
    assert cache.expire_idle() == []  # mid-call: never pulled out from under it
    leave.set()
    t.join(5)
    clock.now += 1000
    assert len(cache.expire_idle()) == 1


def test_release_all_empties_and_frees_memory() -> None:
    cache, _, freed = make_cache()
    cache.get_or_load(("embedder", "a", "cpu"), object)
    cache.get_or_load(("reranker", "b", "cpu"), object)
    assert cache.release_all("test") == 2
    assert len(cache) == 0
    assert freed == [1]
    assert cache.release_all() == 0  # nothing left: no pointless memory release
    assert freed == [1]


def test_reload_after_release_calls_loader_again() -> None:
    cache, _, _ = make_cache()
    calls: list[int] = []
    loader = lambda: calls.append(1) or object()  # noqa: E731
    cache.get_or_load(("embedder", "m", "cpu"), loader)
    cache.release_all()
    cache.get_or_load(("embedder", "m", "cpu"), loader)
    assert len(calls) == 2


def test_failed_load_is_not_cached_and_retries() -> None:
    cache, _, _ = make_cache()
    attempts: list[int] = []

    def flaky() -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return "model"

    with pytest.raises(RuntimeError, match="out of memory"):
        cache.get_or_load(("embedder", "m", "cuda:0"), flaky)
    assert len(cache) == 0
    assert cache.get_or_load(("embedder", "m", "cuda:0"), flaky) == "model"
    assert len(attempts) == 2


def test_concurrent_first_requests_load_once() -> None:
    cache, _, _ = make_cache()
    calls: list[int] = []
    gate = threading.Barrier(8)

    def slow_loader() -> object:
        calls.append(1)
        time.sleep(0.15)
        return object()

    out: list[object] = []

    def worker() -> None:
        gate.wait(5)
        out.append(cache.get_or_load(("embedder", "m", "cpu"), slow_loader))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(calls) == 1
    assert len(out) == 8 and len({id(o) for o in out}) == 1


def test_concurrent_waiters_all_see_a_failed_load_retry_once_each() -> None:
    """A failing load wakes the waiters; they retry (nothing was cached) and succeed."""
    cache, _, _ = make_cache()
    attempts: list[int] = []
    lock = threading.Lock()

    def flaky() -> object:
        with lock:
            attempts.append(1)
            n = len(attempts)
        time.sleep(0.05)
        if n == 1:
            raise RuntimeError("boom")
        return "ok"

    results: list[object] = []

    def worker() -> None:
        try:
            results.append(cache.get_or_load(("embedder", "m", "cpu"), flaky))
        except RuntimeError as e:
            results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sum(isinstance(r, RuntimeError) for r in results) == 1
    assert results.count("ok") == 3
    assert len(cache) == 1


def test_load_in_flight_across_release_is_not_cached() -> None:
    cache, _, _ = make_cache()
    started = threading.Event()
    proceed = threading.Event()

    def slow() -> str:
        started.set()
        proceed.wait(5)
        return "late"

    got: list[str] = []
    t = threading.Thread(target=lambda: got.append(cache.get_or_load(("embedder", "m", "cpu"), slow)))
    t.start()
    assert started.wait(5)
    cache.release_all("training start")  # release lands while the load is mid-flight
    proceed.set()
    t.join(5)
    assert got == ["late"]  # the caller still gets its model...
    assert len(cache) == 0  # ...but it does not squat on VRAM after the release


def test_guard_serialises_calls_per_entry() -> None:
    cache, _, _ = make_cache()
    active = 0
    peak = 0
    lock = threading.Lock()

    def encode(x: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x

    fn, _ = cache.get_or_load(
        ("embedder", "m", "cpu"), lambda: (encode, "i"), lambda v, wrap: (wrap(v[0]), v[1])
    )
    threads = [threading.Thread(target=fn, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert peak == 1


def test_timer_expires_entry_in_real_time() -> None:
    freed = threading.Event()
    cache = ModelCache(timeout_fn=lambda: 1, release_memory=freed.set)
    cache.get_or_load(("embedder", "m", "cpu"), object)
    assert len(cache) == 1
    assert freed.wait(5), "idle timer never fired"
    assert len(cache) == 0


def test_stats_reports_entries_and_timeout() -> None:
    cache, clock, _ = make_cache(timeout=300)
    cache.get_or_load(("embedder", "m", "cpu"), object)
    clock.now += 12
    s = cache.stats()
    assert s["loaded"] == 1 and s["idle_timeout"] == 300
    assert s["models"][0] == {"kind": "embedder", "name": "m", "device": "cpu", "idle_seconds": 12.0, "busy": False}


# ── Studio entry points (cached_embedder / cached_reranker) ──


@pytest.fixture
def fake_models(monkeypatch):
    mc.rag_model_cache.release_all()
    calls = {"embedder": [], "reranker": []}

    def fake_get_embedder(name="x", device="auto"):
        calls["embedder"].append((name, device))
        return (lambda t: [0.0]), {"name": name, "device": device}

    def fake_get_reranker(name="y", device="auto"):
        calls["reranker"].append((name, device))
        return (lambda q, d: [0.0] * len(d)), name

    monkeypatch.setattr(mc, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(mc, "get_reranker", fake_get_reranker)
    monkeypatch.setattr(mc, "_release_memory", lambda: None)
    return calls


def test_cached_embedder_second_call_is_warm(fake_models) -> None:
    a = mc.cached_embedder("m", "cpu")
    b = mc.cached_embedder("m", "cpu")
    assert a is b
    assert fake_models["embedder"] == [("m", "cpu")]
    assert a[0]("hi") == [0.0]


def test_auto_device_is_resolved_before_keying(fake_models, monkeypatch) -> None:
    monkeypatch.setattr(mc, "resolve_device", lambda d="auto": "cuda:0" if d in ("", "auto") else d)
    mc.cached_embedder("m", "auto")
    mc.cached_embedder("m", "cuda:0")  # same resolved device -> same entry
    mc.cached_embedder("m", "cpu")  # different device -> new entry
    assert fake_models["embedder"] == [("m", "cuda:0"), ("m", "cpu")]


def test_cached_reranker_and_release(fake_models) -> None:
    mc.cached_reranker("r", "cpu")
    mc.cached_reranker("r", "cpu")
    assert len(fake_models["reranker"]) == 1
    assert mc.release_rag_models("test") == 1
    mc.cached_reranker("r", "cpu")
    assert len(fake_models["reranker"]) == 2


def test_failed_embedder_load_does_not_poison(fake_models, monkeypatch) -> None:
    def boom(name="x", device="auto"):
        raise OSError("weights missing")

    monkeypatch.setattr(mc, "get_embedder", boom)
    with pytest.raises(OSError):
        mc.cached_embedder("m", "cpu")
    assert len(mc.rag_model_cache) == 0
    monkeypatch.setattr(mc, "get_embedder", lambda name="x", device="auto": (lambda t: "enc", "info"))
    encode, info = mc.cached_embedder("m", "cpu")  # the retry loads fresh and succeeds
    assert info == "info" and encode("q") == "enc"
    assert len(mc.rag_model_cache) == 1


def test_release_rag_models_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(mc.rag_model_cache, "release_all", lambda reason="": (_ for _ in ()).throw(RuntimeError("x")))
    assert mc.release_rag_models("t") == 0
