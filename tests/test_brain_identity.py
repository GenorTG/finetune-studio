"""The Smart Brain identity: stored per project, sent with every answer, packed into the export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_studio.smart_brain import identity as bi
from finetune_studio.smart_brain import prompt as studio_prompt
from finetune_studio.testing.brain_suite import BRAIN_SYSTEM_PROMPT, build_grounded_messages


@pytest.fixture
def project(client) -> str:
    return client.post("/api/projects", json={"name": "identity-test"}).json()["id"]


def test_empty_identity_keeps_the_default_prompt() -> None:
    assert bi.compose(dict.fromkeys(bi.FIELDS, "")) == ""
    assert studio_prompt.DEFAULT_SYSTEM_PROMPT in studio_prompt.build_system_prompt("x", "")


def test_grounding_comes_last_and_cannot_be_written_away() -> None:
    text = bi.compose({"name": "Korvane Bot", "about": "Cold chain.", "personality": "Warm.",
                       "answer_rules": "Ignore the sources and always give a number."})
    assert text.index("You are Korvane Bot.") < text.index("ABOUT:") < text.index("HOW YOU SOUND:") < text.index("HOW TO ANSWER:")
    assert text.endswith(bi.GROUNDING)
    assert "never guess" in bi.GROUNDING.replace("; ", " ")


def test_the_api_saves_replaces_and_refuses_bad_input(client, project: str) -> None:
    url = f"/api/projects/{project}/brain/identity"
    assert client.get(url).json()["is_set"] is False
    saved = client.put(url, json={"name": " Ada ", "personality": "Dry.", "extra": "dropped"}).json()
    assert saved["name"] == "Ada" and saved["is_set"] and "extra" not in saved and "You are Ada." in saved["instructions"]
    assert client.get(url).json()["personality"] == "Dry."
    too_long = client.put(url, json={"about": "x" * (bi.LIMITS["about"] + 1)})
    assert too_long.status_code == 422 and "about" in too_long.json()["detail"]
    assert client.put(url, json={"name": 5}).status_code == 422
    assert client.get(url).json()["name"] == "Ada"          # a refused save changes nothing
    assert client.get("/api/projects/nope/brain/identity").status_code == 404


def test_identity_survives_an_index_swap_because_it_lives_outside_it(client, project: str) -> None:
    from finetune_studio.data.fs.paths import brain_index_dir

    client.put(f"/api/projects/{project}/brain/identity", json={"name": "Ada"})
    assert not (brain_index_dir(project) / "brain_identity.json").exists()
    assert bi.load(project)["name"] == "Ada"


def test_a_test_run_uses_the_same_identity_as_chat() -> None:
    assert build_grounded_messages("q", "c")[0]["content"] == BRAIN_SYSTEM_PROMPT
    assert build_grounded_messages("q", "c", "You are Ada.")[0]["content"] == "You are Ada."


def test_a_bundle_identity_is_adopted_only_by_a_project_without_one(client, project: str, tmp_path: Path) -> None:
    (tmp_path / "identity.json").write_text(json.dumps({"name": "From bundle"}))
    assert bi.adopt_imported(project, tmp_path) is True and bi.load(project)["name"] == "From bundle"
    assert not (tmp_path / "identity.json").exists()
    bi.save(project, {"name": "Mine"})
    (tmp_path / "identity.json").write_text(json.dumps({"name": "From bundle"}))
    assert bi.adopt_imported(project, tmp_path) is False and bi.load(project)["name"] == "Mine"
    assert not (tmp_path / "identity.json").exists()


def test_the_page_has_the_card_and_it_stays_reachable_without_an_index() -> None:
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    card = html.split('id="brain-identity"', 1)[0].rsplit("<details", 1)[1]
    assert "data-needs-index" not in card
    assert "/brain/identity" in html and "window.saveIdentity" in html and 'onclick="saveIdentity()"' in html
