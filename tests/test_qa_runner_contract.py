"""The nightly runner must report this run, not cumulative history."""
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_qa.sh").read_text(encoding="utf-8")


def test_qa_counts_are_scoped_to_unique_run_log() -> None:
    assert 'RUN_LOG="$SHOTS/run-$(date +%Y%m%d-%H%M%S)-$$.log"' in SCRIPT
    assert "awk '/\\[PASS\\]/" in SCRIPT
    assert "awk '/\\[FAIL\\]/" in SCRIPT
    assert 'grep -c' not in SCRIPT


def test_qa_runner_preserves_nonzero_e2e_status() -> None:
    assert "QA_STATUS=$?" in SCRIPT
    assert 'exit "$QA_STATUS"' in SCRIPT
    assert "[ \"$FAIL\" -eq 0 ]" in SCRIPT
