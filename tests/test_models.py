import tempfile, os
from finetune_studio.models.registry import scan_models

def test_scan_empty():
    with tempfile.TemporaryDirectory() as d:
        assert scan_models([d]) == []

def test_scan_gguf():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "test.gguf"), "w").close()
        models = scan_models([d])
        assert len(models) == 1
        assert models[0].format == "gguf"
