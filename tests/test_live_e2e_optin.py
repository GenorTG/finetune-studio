"""Live E2E entry points must be explicitly opted into."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE_SCRIPTS = (
    "e2e_functional.py",
    "e2e_training_flow.py",
    "e2e_ui_qa.py",
    "test_phase_bd.py",
    "test_phase_bd_api.py",
)


def test_live_e2e_python_drivers_require_opt_in_before_side_effects() -> None:
    for name in LIVE_SCRIPTS:
        source = (ROOT / name).read_text(encoding="utf-8")
        guard_at = source.find('os.environ.get("FTS_ALLOW_LIVE_E2E") != "1"')
        assert guard_at >= 0, f"{name} has no explicit live-run guard"
        main_at = source.find("async def main(")
        if main_at < 0:
            main_at = source.find("def main(")
        assert main_at >= 0 and main_at < guard_at, f"{name} guard is outside main()"
        assert source.find("OUT.mkdir(") < 0 or source.find("OUT.mkdir(") > guard_at
        assert source.find("SHOTS.mkdir(") < 0 or source.find("SHOTS.mkdir(") > guard_at


def test_nightly_runner_requires_opt_in_before_contacting_services() -> None:
    source = (ROOT / "run_qa.sh").read_text(encoding="utf-8")
    guard_at = source.find('FTS_ALLOW_LIVE_E2E:-0')
    assert guard_at >= 0
    assert guard_at < source.find("curl -fsS")
    assert guard_at < source.find("openclaw message send")
