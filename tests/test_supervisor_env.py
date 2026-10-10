"""Children re-derive the GPU policy from Settings; the supervisor must not leak what the policy set."""
from __future__ import annotations

from finetune_studio.accel.env import AppliedPolicy
from finetune_studio.accel.saved_choice import SavedChoice
from finetune_studio.supervisor import env as sup_env


def applied(**vars_: str) -> AppliedPolicy:
    return AppliedPolicy("saved", SavedChoice(), (), dict(vars_))


def test_policy_applied_variable_is_stripped(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.setattr(sup_env, "get_applied", lambda: applied(CUDA_VISIBLE_DEVICES="GPU-abc"))
    assert "CUDA_VISIBLE_DEVICES" not in sup_env.child_env()


def test_operator_pinned_variable_passes_through(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(sup_env, "get_applied", lambda: applied())  # the policy set nothing
    assert sup_env.child_env()["CUDA_VISIBLE_DEVICES"] == "1"


def test_value_changed_after_policy_ran_is_kept_and_extra_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "operator-value")
    monkeypatch.setattr(sup_env, "get_applied", lambda: applied(CUDA_VISIBLE_DEVICES="GPU-abc"))
    env = sup_env.child_env({"X": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "operator-value" and env["X"] == "1"
