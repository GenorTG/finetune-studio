"""Benchmarks must target merged/ or the LoRA adapter, never a bare run dir."""

from finetune_studio.webui.routes.benchmarks import _resolve_trained_target


def test_prefers_merged(tmp_path):
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "config.json").write_text("{}")
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
    assert _resolve_trained_target({"output_path": str(tmp_path)}) == str(tmp_path / "merged")


def test_unmerged_run_falls_back_to_adapter(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
    assert _resolve_trained_target({"output_path": str(tmp_path)}) == str(tmp_path / "adapter")


def test_bare_output_path_when_nothing_else(tmp_path):
    assert _resolve_trained_target({"output_path": str(tmp_path)}) == str(tmp_path)
