"""A judge call gives up on a hung API connection quickly (models/providers.py, testing/judge.py)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from finetune_studio.models.providers import OpenAICompatProvider, ProviderConfig
from finetune_studio.testing import judge


def _provider(monkeypatch, seen: list[float]) -> OpenAICompatProvider:
    p = OpenAICompatProvider(ProviderConfig(id="api", name="x", kind="openai_compat", model_id="m",
                                            base_url="http://gw.invalid/v1"))

    class Session:
        def post(self, url: str, json: dict, timeout: float) -> Any:
            seen.append(timeout)
            return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "PASS"}}]})

    monkeypatch.setattr(p, "_client", lambda: Session())
    return p


def test_a_call_timeout_overrides_the_provider_default(monkeypatch) -> None:
    seen: list[float] = []
    p = _provider(monkeypatch, seen)
    p.chat([{"role": "user", "content": "q"}], max_tokens=8)
    p.chat([{"role": "user", "content": "q"}], max_tokens=8, timeout=90)
    assert seen == [300.0, 90.0]                     # mining keeps the long default; a judge asks for 90 s


def test_the_judge_asks_for_its_short_timeout() -> None:
    src = (judge.__file__ and open(judge.__file__, encoding="utf-8").read())
    assert "timeout=JUDGE_TIMEOUT_S" in src and judge.JUDGE_TIMEOUT_S <= 120
