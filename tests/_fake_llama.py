"""Test helper: a fake llama.cpp toolchain that runs as REAL child processes.

``convert_hf_to_gguf.py`` is a python script and ``llama-quantize`` a shell
script, both honouring env knobs so a test can make them slow, noisy or
failing while the real subprocess/process-group code path runs end to end.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

CONVERT = '''\
import os, pathlib, sys, time
args = sys.argv
out = args[args.index("--outfile") + 1]
sys.stderr.write("convert: starting\\n")
sys.stderr.flush()
time.sleep(float(os.environ.get("FAKE_CONVERT_SECONDS", "0")))
if os.environ.get("FAKE_CONVERT_FAIL"):
    sys.stderr.write(os.environ["FAKE_CONVERT_FAIL"] + "\\n")
    sys.exit(3)
pathlib.Path(out).write_bytes(b"gguf-data")
'''

QUANTIZE = '''\
#!/bin/sh
# usage: llama-quantize [--imatrix FILE] IN OUT TYPE
if [ "$1" = "--imatrix" ]; then shift 2; fi
if [ -n "$FAKE_QUANT_GRANDCHILD_PIDFILE" ]; then
  sleep 300 &
  echo $! > "$FAKE_QUANT_GRANDCHILD_PIDFILE"
  wait
fi
sleep "${FAKE_QUANT_SECONDS:-0}"
if [ -n "$FAKE_QUANT_FAIL" ]; then
  echo "$FAKE_QUANT_FAIL" >&2
  exit 5
fi
echo quantized > "$2"
'''


def install_fake_llama(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create the fake toolchain under ``root`` and make discovery find it."""
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text(CONVERT)
    quant = root / "build" / "bin" / "llama-quantize"
    quant.write_text(QUANTIZE)
    quant.chmod(quant.stat().st_mode | stat.S_IXUSR)
    monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
    monkeypatch.setattr(
        "finetune_studio.training.gguf_convert.llama_cpp_search_paths", lambda: [str(root)],
    )
    monkeypatch.setenv("PATH", os.environ.get("PATH", "") + os.pathsep + "/usr/bin:/bin")
    return root


def make_merged_run(db, project_id: str, base: Path) -> dict:
    """A finished run whose ``merged/`` dir already exists (no GPU merge needed)."""
    out = base / "run-out"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"w" * 16)
    (merged / "config.json").write_text("{}")
    run = db.create_run(project_id=project_id, name="r", base_model="Qwen/Qwen3-0.6B",
                        data_path="/d")
    db.update_run(run["id"], status="done", output_path=str(out))
    return db.get_run(run["id"])


class EngineStub:
    """Stands in for the MagicMock'd ``TrainingEngine`` in ``client``-fixture tests.

    ``_do_export_gguf`` mirrors the real method (merged dir -> shared
    ``convert_merged_to_gguf``) so the fake toolchain runs for real.
    """

    def __init__(self) -> None:
        self.config = None

    def _do_export_gguf(self, output_dir: str, force: bool = False) -> dict:
        from finetune_studio.training.gguf_convert import convert_merged_to_gguf

        quants = list(self.config.gguf_quants) or ["f16", "q8_0"]
        return convert_merged_to_gguf(
            os.path.join(output_dir, "merged"), os.path.join(output_dir, "gguf"),
            quants, force=force,
        )
