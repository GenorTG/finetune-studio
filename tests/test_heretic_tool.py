"""The optional Heretic tool: headless command, output verification, wiring.

Heretic is an external program (AGPL-3.0) installed beside this app, so these
tests pin the parts that are ours: the exact argv that keeps it unattended, the
refusal when it is absent or cannot run headless, and the check that a finished
run really wrote a loadable model. No test here downloads or runs Heretic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.training import heretic_tool


def _model_dir(tmp_path: Path, *, config: bool = True, tokenizer: bool = True,
               weights: bool = True) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    if weights:
        (d / "model.safetensors").write_bytes(b"\x00" * 16)
    if config:
        (d / "config.json").write_text("{}", encoding="utf-8")
    if tokenizer:
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    return d


# ── the headless contract ────────────────────────────────────────────────────

def test_command_closes_every_prompt_with_a_flag() -> None:
    """A missing flag is a hang, not a nuisance: every menu must be pre-answered."""
    cmd = heretic_tool.build_command(
        heretic_tool.AbliterationRun(model_path="/m/merged", output_dir="/m/heretic", n_trials=7)
    )
    joined = " ".join(cmd)
    assert "--model-action save" in joined
    assert "--export-strategy merge" in joined
    assert "--save-directory /m/heretic" in joined
    assert "--checkpoint-action restart" in joined
    assert "--trial-index 0" in joined
    assert "--n-trials 7" in joined
    assert "--model /m/merged" in joined


def test_command_never_asks_and_never_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unattended means: no menu action other than save, and no Hub upload flag."""
    cmd = heretic_tool.build_command(
        heretic_tool.AbliterationRun(model_path="/m", output_dir="/o")
    )
    assert "upload" not in cmd
    assert not any("upload" in c for c in cmd)
    # The only interactive actions Heretic knows are save/upload; save is pinned.
    assert cmd[cmd.index("--model-action") + 1] == "save"


def test_study_dir_is_passed_only_when_given() -> None:
    plain = heretic_tool.build_command(heretic_tool.AbliterationRun(model_path="/m", output_dir="/o"))
    assert "--study-checkpoint-dir" not in plain
    kept = heretic_tool.build_command(
        heretic_tool.AbliterationRun(model_path="/m", output_dir="/o", checkpoint_dir="/o/.study")
    )
    assert "--study-checkpoint-dir" in kept


def test_required_flags_are_all_in_the_command() -> None:
    cmd = heretic_tool.build_command(heretic_tool.AbliterationRun(model_path="/m", output_dir="/o"))
    for flag in heretic_tool._REQUIRED_FLAGS:
        assert flag in cmd, flag


# ── what a finished run must have written ────────────────────────────────────

def test_inspect_output_accepts_a_real_model(tmp_path: Path) -> None:
    d = _model_dir(tmp_path)
    got = heretic_tool.inspect_output(str(d))
    assert got["size_bytes"] > 0
    assert "config.json" in got["files"]


def test_inspect_output_rejects_weights_without_config(tmp_path: Path) -> None:
    """The old export's exact failure: shards on disk, nothing that can load them."""
    d = _model_dir(tmp_path, config=False)
    with pytest.raises(heretic_tool.HereticError) as e:
        heretic_tool.inspect_output(str(d))
    assert "config.json" in str(e.value)


def test_inspect_output_rejects_a_model_without_tokenizer(tmp_path: Path) -> None:
    d = _model_dir(tmp_path, tokenizer=False)
    with pytest.raises(heretic_tool.HereticError) as e:
        heretic_tool.inspect_output(str(d))
    assert "tokenizer" in str(e.value)


def test_inspect_output_rejects_an_empty_directory(tmp_path: Path) -> None:
    d = _model_dir(tmp_path, weights=False)
    with pytest.raises(heretic_tool.HereticError) as e:
        heretic_tool.inspect_output(str(d))
    assert "weight" in str(e.value)


# ── the tool is optional: absence is a clear answer, never a crash ───────────

def test_probe_reports_not_installed_with_a_remedy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTS_HERETIC_ROOT", str(tmp_path / "heretic"))
    status = heretic_tool.probe()
    assert status.installed is False
    assert status.usable is False
    assert "fts heretic install" in status.hint
    assert tmp_path.as_posix() in status.root


def test_run_refuses_when_the_model_is_not_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare adapter is not something Heretic can ablate; say so before spawning."""
    monkeypatch.setattr(heretic_tool, "probe", lambda: heretic_tool.HereticStatus(installed=True, version="2"))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(heretic_tool.HereticError) as e:
        heretic_tool.run_abliteration(
            heretic_tool.AbliterationRun(model_path=str(adapter), output_dir=str(tmp_path / "out"))
        )
    assert "config.json" in str(e.value)


def test_run_refuses_a_build_without_headless_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        heretic_tool, "probe",
        lambda: heretic_tool.HereticStatus(
            installed=True, version="1.4.0", missing_flags=("--model-action",), hint="cannot run unattended",
        ),
    )
    with pytest.raises(heretic_tool.HereticError) as e:
        heretic_tool.run_abliteration(
            heretic_tool.AbliterationRun(model_path=str(_model_dir(tmp_path)), output_dir=str(tmp_path / "out"))
        )
    assert "unattended" in str(e.value)


def test_env_does_not_enable_python_bytecode() -> None:
    """Heretic must not write into this app's package tree."""
    assert heretic_tool.run_env()["PYTHONDONTWRITEBYTECODE"] == "1"


# ── choosing which trial to keep (the bug that shipped) ─────────────────────

# The measured front from a real 24-trial run (Qwen3-0.6B, 2026-10-28). Position 0
# had zero refusals and KL 10.0: the model answered in token soup. Position 2 had
# 0.03 refusals at KL 0.008 and answered coherently when tested.
MEASURED_FRONT = [
    {"pos": 0, "trial": 17, "refusals": 0.00, "kl": 10.0254},
    {"pos": 1, "trial": 11, "refusals": 0.02, "kl": 0.6102},
    {"pos": 2, "trial": 9, "refusals": 0.03, "kl": 0.0079},
]


def test_pick_trial_never_takes_the_destroyed_extreme() -> None:
    """Regression: position 0 is fewest refusals at ANY cost, and it wrecked the model."""
    picked = heretic_tool.pick_trial(MEASURED_FRONT, kl_budget=0.5)
    assert picked["chosen"]["pos"] == 2, picked["chosen"]
    assert picked["budget_met"] is True


def test_pick_trial_respects_the_budget_it_was_given() -> None:
    """A looser budget may accept the middle trial, still never the KL-10 one."""
    picked = heretic_tool.pick_trial(MEASURED_FRONT, kl_budget=1.0)
    assert picked["chosen"]["pos"] == 1
    assert picked["chosen"]["kl"] == 0.6102


def test_pick_trial_falls_back_to_least_damage_and_says_so() -> None:
    """No trial inside the budget: keep the least-damaged and report budget_met=False."""
    picked = heretic_tool.pick_trial(MEASURED_FRONT, kl_budget=0.001)
    assert picked["chosen"]["pos"] == 2
    assert picked["budget_met"] is False


def test_pick_trial_prefers_fewer_refusals_inside_the_budget() -> None:
    front = [
        {"pos": 0, "trial": 1, "refusals": 0.0, "kl": 9.0},
        {"pos": 1, "trial": 2, "refusals": 0.30, "kl": 0.20},
        {"pos": 2, "trial": 3, "refusals": 0.10, "kl": 0.40},
    ]
    picked = heretic_tool.pick_trial(front, kl_budget=0.5)
    assert picked["chosen"]["trial"] == 3, picked["chosen"]


def test_pick_trial_rejects_an_empty_front() -> None:
    with pytest.raises(heretic_tool.HereticError):
        heretic_tool.pick_trial([], kl_budget=0.5)


def test_two_pass_flags_differ_where_it_matters() -> None:
    """Pass 1 searches and discards an adapter; pass 2 merges the chosen trial."""
    run = heretic_tool.AbliterationRun(model_path="/m", output_dir="/o", checkpoint_dir="/o/.study")
    search = heretic_tool.build_command(
        run, save_dir="/o.search", export_strategy="adapter",
        checkpoint_action="restart", trial_index=0,
    )
    save = heretic_tool.build_command(
        run, save_dir="/o", export_strategy="merge",
        checkpoint_action="continue", trial_index=2,
    )
    assert search[search.index("--export-strategy") + 1] == "adapter"
    assert save[save.index("--export-strategy") + 1] == "merge"
    assert search[search.index("--checkpoint-action") + 1] == "restart"
    assert save[save.index("--checkpoint-action") + 1] == "continue"
    assert save[save.index("--trial-index") + 1] == "2"
    assert save[save.index("--save-directory") + 1] == "/o"


# ── the overlay must share one CUDA stack, not carry its own ────────────────

def test_shared_packages_exclude_light_dependencies() -> None:
    """Only the CUDA/ML stack is guarded.

    ``lm-eval`` resolves ``datasets`` and ``numpy`` itself; refusing those would
    fight the resolver for no benefit (they are a few MB, not gigabytes).
    """
    assert "torch" in heretic_tool.SHARED_PACKAGES
    assert "transformers" in heretic_tool.SHARED_PACKAGES
    assert "datasets" not in heretic_tool.SHARED_PACKAGES
    assert "numpy" not in heretic_tool.SHARED_PACKAGES


def test_lm_eval_is_installed_without_its_hf_extra() -> None:
    """``lm-eval[hf]`` pins torch/transformers/accelerate/peft and would install
    a second PyTorch into the overlay (5.8 GB instead of ~0.65 GB)."""
    lm = [d for d in heretic_tool.EXTRA_DEPS if d.startswith("lm-eval")]
    assert lm, "lm-eval must be installed: Heretic imports lm_eval.models.huggingface"
    assert all("[" not in d for d in lm), lm


def test_overlay_duplicates_reports_a_copied_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard reads the filesystem: a copied torch is reported, not assumed away."""
    monkeypatch.setenv("FTS_HERETIC_ROOT", str(tmp_path / "heretic"))
    site = heretic_tool.overlay_site_packages()
    assert heretic_tool.overlay_duplicates() == ()
    (site / "torch").mkdir(parents=True)
    (site / "transformers").mkdir()
    dupes = heretic_tool.overlay_duplicates()
    assert "torch" in dupes and "transformers" in dupes


def test_overlay_duplicates_ignores_light_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTS_HERETIC_ROOT", str(tmp_path / "heretic"))
    site = heretic_tool.overlay_site_packages()
    (site / "datasets").mkdir(parents=True)
    (site / "numpy").mkdir()
    assert heretic_tool.overlay_duplicates() == ()


# ── wiring: preflight, capabilities, routes, UI ──────────────────────────────

def test_export_preflight_refuses_heretic_without_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.training.run_export import validate_export_request

    monkeypatch.setattr(
        heretic_tool, "probe",
        lambda: heretic_tool.HereticStatus(installed=False, hint="Heretic is not installed."),
    )
    refused = validate_export_request({"output_path": "/tmp/run"}, fmt="heretic")
    assert refused is not None
    assert refused["ok"] is False
    assert refused["format"] == "heretic"
    assert refused["install_command"] == "fts heretic install"


def test_heretic_is_a_supported_export_format() -> None:
    from finetune_studio.training.run_export import SUPPORTED_EXPORT_FORMATS

    assert "heretic" in SUPPORTED_EXPORT_FORMATS
    # The pre-existing formats stay exactly as they were: this is additive.
    assert {"gguf", "merged", "abliterated"} <= SUPPORTED_EXPORT_FORMATS


def test_capabilities_carry_heretic_and_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from finetune_studio.training import export_capabilities

    monkeypatch.setattr(
        heretic_tool, "probe",
        lambda: heretic_tool.HereticStatus(installed=False, hint="not installed here"),
    )
    caps = export_capabilities.probe_export_capabilities()
    assert caps.heretic is False
    assert "not installed" in caps.heretic_hint
    assert caps.as_dict()["heretic"] is False


def test_export_page_offers_heretic_and_explains_when_it_cannot_run() -> None:
    from finetune_studio.guide.registry import PAGES

    tpl = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
    html = (tpl / PAGES["export"].template).read_text(encoding="utf-8")
    assert 'value="heretic"' in html
    assert 'id="export-heretic-hint"' in html
    assert "data-heretic=" in html


def test_settings_page_has_the_heretic_card() -> None:
    tpl = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
    html = (tpl / "settings.html").read_text(encoding="utf-8")
    for control in ("heretic-card", "btn-heretic-install", "btn-heretic-check", "btn-heretic-remove"):
        assert f'id="{control}"' in html, control


def test_routes_are_registered() -> None:
    # ``app.routes`` holds included-router wrappers whose ``path`` is empty: the
    # real table is the OpenAPI schema (the same source the API docs serve).
    from finetune_studio.webui.app import app

    paths = set(app.openapi().get("paths", {}))
    assert "/api/system/heretic" in paths
    assert "/api/system/heretic/install" in paths
    assert "/api/system/heretic/upstream" in paths


def test_cli_exposes_the_tool() -> None:
    from finetune_studio.cli._parser import build_parser

    args = build_parser().parse_args(["heretic", "status"])
    assert args.command == "heretic"
    assert args.heretic_action == "status"
