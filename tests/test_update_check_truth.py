"""Settings → Updates says what is true (2026-10-28, beta work order B2.2).

Live, Check on the dev box answered "error" with two false alarms: "no working NVIDIA driver, install it and reboot"
(the kernel driver was loaded; only nvidia-smi was missing from the service's PATH: a dangling alternatives link) and
"missing or broken: peft" (installed; `import peft` took longer than the 15 s probe limit under load). And Check never
asked GitHub whether a newer version exists, although its button promised "report drift".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import accel_plan as ap
import install_diagnose as diag

from tests.test_accel_plan import fake_hw

PCI_3090 = "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA102 [GeForce RTX 3090] [10de:2204] (rev a1)\n"
KMOD = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul  7 12:38:21 UTC 2026\n"


def test_a_loaded_kernel_driver_without_nvidia_smi_is_a_working_gpu(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={"lspci -nn": PCI_3090},
            files={"/proc/driver/nvidia/version": KMOD})
    gpu = ap.detect()
    assert gpu.vendor == "nvidia" and gpu.driver_ready is True and gpu.driver_version == "580.178.04"
    assert gpu.hint == "" and "install the vendor driver" not in gpu.hint.lower()
    assert gpu.compute_cap == "8.6"                       # from the GA102 codename
    assert any("nvidia-smi is missing or not working" in n and "580.178.04" in n for n in gpu.notes)
    assert ap.build_plan(gpu).torch_tag.startswith("cu")


def test_no_kernel_driver_still_says_install_the_driver(monkeypatch, tmp_path) -> None:
    fake_hw(monkeypatch, tmp_path, which=["lspci"], commands={"lspci -nn": PCI_3090})
    gpu = ap.detect()
    assert gpu.driver_ready is False and "no working driver" in gpu.hint


def test_a_slow_import_is_not_checked_rather_than_missing(tmp_path) -> None:
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    (venv / "bin" / "python").chmod(0o755)

    def fake_run(cmd, timeout=30):
        code = cmd[-1] if cmd else ""
        if "import peft" in code or code == diag._TORCH_PROBE:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if "'executable'" in code:
            return subprocess.CompletedProcess(cmd, 0, '{"py": "3.12.13", "executable": "x"}\n', "")
        if "__version__" in code:
            return subprocess.CompletedProcess(cmd, 0, "1.0\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch.object(diag, "_run", side_effect=fake_run):
        info = diag.inspect_venv(venv)
    assert info.unchecked == ["torch", "peft"] and info.peft_version is None and info.torch_version is None
    assert diag.HEAVY_IMPORT_TIMEOUT_S >= 60


def test_diagnose_reports_unchecked_imports_without_calling_them_missing(tmp_path) -> None:
    info = diag.VenvInfo(path=tmp_path, python=tmp_path / "python", py_version="3.12", torch_version="2.14",
                         torch_has_cuda=True, torchaudio_version=None, torchaudio_importable=None,
                         transformers_version="5", llama_cpp_version="0.3", peft_version=None, trl_version="1",
                         has_fastapi=True, has_jinja2=True, has_datasets=True, has_accelerate=True,
                         has_safetensors=True, has_huggingface_hub=True, unchecked=["peft"])
    with patch.object(diag, "inspect_venv", return_value=info), \
         patch.object(diag.VenvInfo, "exists", property(lambda self: True)):
        issues = diag.diagnose(tmp_path, tmp_path / "llama.cpp", check_service=False)
    texts = [i.detail for i in issues]
    assert not any("missing or broken" in t and "peft" in t for t in texts), texts
    slow = [i for i in issues if i.code == "slow-import"]
    assert slow and slow[0].severity == 1 and "could not check peft" in slow[0].detail


@pytest.mark.parametrize("needle", [
    'git_env git fetch --quiet origin "$_branch"',
    'log "Update available: $_new new change(s) on $_branch',
    'log "Up to date: $_branch is at $_before, the same as GitHub."',
    'log "Code: updated $_before -> $_after',
])
def test_update_sh_check_asks_github_and_apply_reports_what_moved(needle: str) -> None:
    src = (ROOT / "update.sh").read_text(encoding="utf-8")
    assert needle in src
    assert subprocess.run(["bash", "-n", str(ROOT / "update.sh")], check=False).returncode == 0


def test_check_on_a_branch_behind_github_lists_the_new_changes(tmp_path: Path) -> None:
    """The fetch/compare part of update.sh, run for real on a throwaway pair of git repositories."""
    def git(*a: str, cwd: Path) -> str:
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True,
                              env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "GIT_AUTHOR_NAME": "t",
                                   "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                                   "GIT_COMMITTER_EMAIL": "t@t"}).stdout
    origin, clone = tmp_path / "origin.git", tmp_path / "clone"
    work = tmp_path / "work"
    git("init", "-q", "-b", "dev", str(work), cwd=tmp_path)
    (work / "update.sh").write_text((ROOT / "update.sh").read_text(encoding="utf-8"))
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "first", cwd=work)
    git("clone", "-q", "--bare", str(work), str(origin), cwd=tmp_path)
    git("clone", "-q", "-b", "dev", str(origin), str(clone), cwd=tmp_path)
    (work / "x").write_text("1")
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "fix: the new thing", cwd=work)
    git("push", "-q", str(origin), "dev", cwd=work)
    # Run only update.sh's step 1 (stop before the venv step) in check mode.
    script = (clone / "update.sh").read_text(encoding="utf-8")
    cut = script.index("# ── Step 2")
    (clone / "step1.sh").write_text(script[:cut])
    r = subprocess.run(["bash", "step1.sh", "--check"], cwd=clone, capture_output=True, text=True, check=False,
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert "Update available: 1 new change(s) on dev" in r.stdout, r.stdout + r.stderr
    assert "    - fix: the new thing" in r.stdout
    assert git("rev-list", "--count", "HEAD", cwd=clone).strip() == "1"   # a check changes nothing


def test_settings_card_summarises_the_run_in_words() -> None:
    js = (ROOT / "src/finetune_studio/webui/static/js/settings.js").read_text(encoding="utf-8")
    assert "function updSummary(u)" in js and "update(s) available: Apply update installs them." in js
    assert "Pulls main" not in js and "the new version starts at the next restart" in js
    html = (ROOT / "src/finetune_studio/webui/templates/settings.html").read_text(encoding="utf-8")
    assert "report drift" not in html and "Pull main" not in html
    assert "10–20 minutes" in html                       # Repair says what it costs before anyone presses it
