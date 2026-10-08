"""Pairs-page "Preference pairs (DPO)" card + Training-page deep link: markup and wiring (no browser)."""
from __future__ import annotations

import re
from pathlib import Path

from tests.preference_fakes import seed_project

_TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"
_PAIRS = (_TEMPLATES / "data_prep.html").read_text(encoding="utf-8")
_TRAINING = (_TEMPLATES / "project_training.html").read_text(encoding="utf-8")


def test_card_is_present_and_separated_from_the_sft_review_card(client) -> None:
    proj = seed_project()
    html = client.get(f"/projects/{proj['id']}/data-prep").text
    assert 'id="dp-pref-card"' in html and "Preference pairs (DPO)" in html
    review, pref = html.index('id="dp-review-card"'), html.index('id="dp-pref-card"')
    assert review < pref, "preference card follows the SFT review/export card"
    # its own card: the build controls are not inside the review toolbar
    assert 'id="dp-pref-build"' not in html[review:pref]


def test_card_explains_what_it_is_and_when_to_use_it() -> None:
    card = _PAIRS[_PAIRS.index('id="dp-pref-card"'):_PAIRS.index("<style>\n.dp-review-toolbar")]
    assert "chosen" in card and "rejected" in card
    assert "When to use it" in card and "after the normal fine-tune" in card
    assert "curb hallucination" in card and "isn’t in the documents" in card


def test_controls_exist_with_labels() -> None:
    for el in ("dp-pref-kind-hallucination", "dp-pref-kind-abstain", "dp-pref-max", "dp-pref-seed",
               "dp-pref-build", "dp-pref-progress", "dp-pref-bar", "dp-pref-result", "dp-pref-approved"):
        assert f'id="{el}"' in _PAIRS, el
    assert 'role="progressbar"' in _PAIRS and 'aria-label="Maximum number of preference pairs"' in _PAIRS


def test_build_posts_the_typed_body_to_the_real_route_and_polls_progress() -> None:
    assert "'/data-prep/preference'" in _PAIRS and "'/progress'" in _PAIRS
    assert re.search(r"JSON\.stringify\(\{ kinds: kinds, max_pairs: maxPairs, seed:", _PAIRS)
    # reuses the shared progress bar + tokens only (no new colours)
    assert 'class="hpbar"' in _PAIRS
    css = _PAIRS[_PAIRS.index(".dp-pref-kinds"):_PAIRS.index(".dp-exported-panel {")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "preference card CSS must use tokens, not literals"


def test_result_links_to_training_with_the_dataset_and_dpo_mode() -> None:
    assert "Train with this dataset" in _PAIRS and "d.train_url" in _PAIRS
    assert "length warning" in _PAIRS.lower() or "ln.warning" in _PAIRS


def test_approved_count_follows_the_review_stats() -> None:
    assert "dpPrefSyncApproved(c.approved);" in _PAIRS


def test_training_page_preselects_dataset_id_and_dpo_mode() -> None:
    assert 'q.get("mode")' in _TRAINING and 'o.value === want' in _TRAINING
    assert 'want === "latest"' in _TRAINING  # the SFT export deep link still works
    assert "Preference pairs (DPO)" in _TRAINING


def test_route_is_mounted_in_the_real_app(client) -> None:
    r = client.post("/api/projects/nope/data-prep/preference", json={})
    assert r.status_code == 404 and r.json() == {"error": "project not found"}
    assert client.get("/api/projects/nope/data-prep/preference/progress").status_code == 404


def test_training_deep_link_runs_after_the_dataset_change_listener_is_wired() -> None:
    """Regression: the prefill used to fire before the listener existed, leaving the hidden dataset fields empty."""
    assert _TRAINING.index("prefillDatasetFromQuery") > _TRAINING.index('select.addEventListener("change"')
