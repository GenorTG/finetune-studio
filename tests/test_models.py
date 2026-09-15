import os
import tempfile

from finetune_studio.models.registry import scan_models


def test_scan_empty():
    with tempfile.TemporaryDirectory() as d:
        assert scan_models([d]) == []


def test_scan_gguf():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.gguf")
        with open(path, "wb") as fh:
            fh.write(b"GGUF" + b"\0" * 64)
        models = scan_models([d])
        assert len(models) == 1
        assert models[0].format == "gguf"
