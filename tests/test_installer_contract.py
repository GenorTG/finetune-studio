"""Pin installer support ranges and parser dependency coverage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_batch_installer_selects_only_supported_python_and_parser_extra() -> None:
    source = (ROOT / "install.bat").read_text(encoding="utf-8")
    assert "(3,12) <= sys.version_info[:2] < (3,14)" in source
    assert source.count("uv sync --extra parsers") == 2
    assert "Python 3.12 or 3.13 not found" in source


def test_windows_powershell_installer_uses_supported_range_and_parser_extra() -> None:
    source = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert '$version -ge [version]"3.12" -and $version -lt [version]"3.14"' in source
    assert 'uv pip install -e ".[parsers]"' in source


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
    assert "transformers>=4.40.0,<5.0.0" in source
    assert "sentence-transformers>=3.0.0,<6.0.0" in source
    for package in ("pypdf", "python-docx", "openpyxl", "xlrd", "python-pptx", "beautifulsoup4", "striprtf", "Pillow"):
        assert any(
            line.lower().startswith(package.lower() + ">=")
            for line in source.splitlines()
        )
