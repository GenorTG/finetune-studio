"""Installs get the versions that were tested: pins/ is complete, and every install path uses it.

Found 2026-11-05 (RESULTS §37): with no lockfile a Repair moved 52 packages to untested releases. A fresh install would also
have cloned whatever llama.cpp was newest and resolved Heretic's packages anew.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import accel_plan as ap
import pin_versions as pv

_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def _deps() -> set[str]:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = data["optional-dependencies"]
    wanted = [*data["dependencies"], *extras["parsers"], *extras["dev"], *extras["bitsandbytes"], *extras["abliteration"]]
    return {pv.normalize(_NAME.match(d).group(0)) for d in wanted}


def test_every_dependency_has_a_pin() -> None:
    pinned = {**pv.read_pins(pv.PINS / "app.txt"), **pv.read_pins(pv.PINS / "torch.txt")}
    assert not _deps() - set(pinned), "add it to the venv and run: make pins"


def test_pins_are_exact_and_leave_gpu_wheels_to_the_plan() -> None:
    app = pv.read_pins(pv.PINS / "app.txt")
    assert app and not [n for n in app if n in pv.TORCH_FAMILY or n.startswith(("nvidia-", "cuda-"))]
    assert not [v for v in app.values() if "+" in v]
    assert "+" not in pv.read_pins(pv.PINS / "torch.txt")["torch"]
    assert re.fullmatch(r"[0-9a-f]{40}", ap.pinned_lines("llama-cpp")[0])
    assert pv.read_pins(pv.PINS / "heretic.txt").get("optuna")


def test_app_pins_filter_and_build_tags() -> None:
    installed = {"torch": "2.15.0+cu132", "nvidia-cublas": "1", "cuda-toolkit": "2", "numpy": "2.5.4"}
    assert pv.app_pins(installed) == {"numpy": "2.5.4"}
    assert pv.torch_pins(installed) == {"torch": "2.15.0"}


def test_drift_ignores_build_tags_and_absent_packages() -> None:
    pins = {"torch": "2.15.0", "numpy": "2.5.4", "absent": "1"}
    assert pv.drift(pins, {"torch": "2.15.0+cu132", "numpy": "2.5.4"}) == []
    assert pv.drift(pins, {"torch": "2.14.1+cu130", "numpy": "2.5.4"}) == ["torch 2.14.1+cu130 -> 2.15.0"]


def _plan(tag: str = "cu132", fallback: tuple[str, ...] = ("cu130",)) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(torch_pin="", torch_index=f"https://x/{tag}", torch_tag=tag, torch_fallback_tags=list(fallback))


def test_torch_tries_the_tested_version_then_each_indexs_newest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FTS_UNPINNED", raising=False)
    monkeypatch.delenv("FTS_UNSLOTH", raising=False)
    monkeypatch.setattr(ap, "pinned_lines", lambda n: ["torch==2.15.0", "torchvision==0.29.2"] if n == "torch" else [])
    cmds = ap.torch_commands(_plan(), "py")
    assert [t for t, _ in cmds] == ["cu132", "cu132", "cu130", "cu130"]
    assert cmds[0][1][-2:] == ["torch==2.15.0", "torchvision==0.29.2"]
    assert cmds[1][1][-2:] == ["torch", "torchvision"]


def test_the_unsloth_profile_is_not_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTS_UNSLOTH", "1")
    assert ap.pinned_lines("app") == []
    assert "torch<2.13" in ap.torch_commands(_plan(), "py", unsloth_profile=True)[0][1]


def test_fts_unpinned_tries_the_newest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FTS_UNPINNED", "1")
    assert ap.pinned_lines("torch") == []
    assert ap.llama_clone_commands(tmp_path / "l")[0][:2] == ["git", "clone"]
    from finetune_studio.training import heretic_tool

    assert heretic_tool._pin_args() == []


def test_llama_cpp_fetches_the_tested_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FTS_UNPINNED", raising=False)
    monkeypatch.delenv("FTS_UNSLOTH", raising=False)
    cmds = ap.llama_clone_commands(tmp_path / "l")
    assert ap.pinned_lines("llama-cpp")[0] in cmds[1]
    assert cmds[2][-1] == "FETCH_HEAD"


def test_constraints_file_carries_torch_and_the_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FTS_UNPINNED", raising=False)
    monkeypatch.delenv("FTS_UNSLOTH", raising=False)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(ap, "_run", lambda *_a, **_k: (0, '{"torch": "2.15.0+cu132"}'))
    text = ap.write_constraints(venv).read_text(encoding="utf-8")
    assert "torch==2.15.0+cu132\n" in text and "\ntransformers==" in text


def test_heretic_install_uses_its_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FTS_UNPINNED", raising=False)
    from finetune_studio.training import heretic_tool

    args = heretic_tool._pin_args()
    assert args[0] == "-c" and args[1].endswith("pins/heretic.txt")


def test_update_refreshes_the_constraints_and_reports_drift() -> None:
    update = (_ROOT / "update.sh").read_text(encoding="utf-8")
    assert '[ -s "$f" ] || accel constraints' not in update
    assert "scripts/pin_versions.py check" in update
    assert "pin_versions.py check" in (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_the_tool_needs_no_third_party_packages() -> None:
    r = subprocess.run([sys.executable, "-I", "-S", str(_ROOT / "scripts/pin_versions.py"), "--help"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
