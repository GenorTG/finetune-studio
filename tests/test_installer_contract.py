"""Pin installer support ranges and parser dependency coverage."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_windows_batch_installer_selects_only_supported_python_and_parser_extra() -> None:
    source = (ROOT / "install.bat").read_text(encoding="utf-8")
    assert "(3,12) <= sys.version_info[:2] < (3,14)" in source
    # `uv sync` re-resolves torch to the default (CPU on Windows) build; the GPU pin must survive.
    assert "uv sync" not in source
    assert 'uv pip install %CONSTRAINTS% -e ".[parsers]"' in source
    assert "accel_plan.py install torch" in source
    assert "Python 3.12 or 3.13 not found" in source


def test_windows_powershell_installer_uses_supported_range_and_parser_extra() -> None:
    source = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert '$version -ge [version]"3.12" -and $version -lt [version]"3.14"' in source
    assert 'uv pip install @constraints -e ".[parsers]"' in source
    assert 'accel_plan.py $Sub' in source and '"torch"' in source


def test_updater_defaults_to_installer_project_local_llama_cpp() -> None:
    source = (ROOT / "update.sh").read_text(encoding="utf-8")
    assert 'LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PWD/.llama.cpp}"' in source


def test_legacy_launcher_uses_the_checkout_venv_not_a_fixed_home_path() -> None:
    source = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
    assert 'REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' in source
    assert 'PYTHON="$REPO_DIR/.venv/bin/python"' in source
    assert "$HOME/finetune-studio/.venv" not in source


def test_pip_fallback_contains_version_bounds_and_parser_packages() -> None:
    source = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "transformers>=5.0.0,<6.0.0" in source
    assert "sentence-transformers>=3.0.0,<6.0.0" in source
    for package in ("pypdf", "python-docx", "openpyxl", "xlrd", "python-pptx", "beautifulsoup4", "striprtf", "Pillow"):
        assert any(
            line.lower().startswith(package.lower() + ">=")
            for line in source.splitlines()
        )


SHELL_SCRIPTS = ["install.sh", "update.sh", "run.sh", "install-service.sh", "scripts/run.sh"]


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_shell_scripts_parse(script: str) -> None:
    r = subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_shell_wrappers_forward_every_argument() -> None:
    assert '$argv' in (ROOT / "install.fish").read_text(encoding="utf-8")
    assert '"$@"' in (ROOT / "install.zsh").read_text(encoding="utf-8")


def test_no_installer_hard_codes_a_gpu_wheel_tag() -> None:
    """Index/tag choices live ONLY in scripts/accel_plan.py (one source for every installer)."""
    tag = re.compile(r"\bcu1[0-9]{2}\b|rocm[0-9]\.[0-9]|whl/xpu|whl/cpu")
    for name in ("install.sh", "update.sh", "install.ps1", "install.bat"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith(("#", "REM")):
                continue
            assert not tag.search(line), f"{name} hard-codes a wheel tag: {line.strip()}"


def test_update_sh_syncs_every_extra_install_sh_installs() -> None:
    """AGENTS gotcha: update.sh must keep updated hosts identical to fresh installs."""
    install = (ROOT / "install.sh").read_text(encoding="utf-8")
    update = (ROOT / "update.sh").read_text(encoding="utf-8")
    for what in re.findall(r"accel install ([a-z-]+)", install):
        if what in ("torch", "llama-cpp-python"):
            continue   # torch is pinned by constraints; llama-cpp-python is verified/repaired by install_diagnose
        assert f"accel install {what}" in update, f"update.sh does not sync {what}"
    assert "numpy>=1.24.0" in update and "scipy>=1.10.0" in update
    assert ".[parsers]" in update and "accel build-llama-cli" in update


def test_windows_installers_install_every_extra_the_unix_installer_does() -> None:
    for name in ("install.ps1", "install.bat"):
        source = (ROOT / name).read_text(encoding="utf-8")
        for what in ("torch", "llama-cpp-python", "bitsandbytes", "unsloth", "build-llama-cli"):
            assert what in source, f"{name} is missing {what}"


def test_pyproject_pins_match_the_tested_stack() -> None:
    py = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"transformers>=5.0.0,<6.0.0"' in py          # live stack is transformers 5.x; the <5 cap was GPTQ-only
    assert "transformers>=4.40.0,<5.0.0" not in py
