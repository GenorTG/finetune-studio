"""Every page that runs an assistant job offers to load the local assistant after a 409 no_model, then retries."""
from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui"


def _read(rel: str) -> str:
    return (WEB / rel).read_text(encoding="utf-8")


def test_the_shared_helper_loads_only_a_local_connection() -> None:
    app = _read("static/js/app.js")
    assert "async function loadAssistant(job)" in app and "function offerAssistantLoad(box, job, retry)" in app
    assert "j.kind !== 'openai_compat'" in app                        # an API connection is never "loaded"
    assert "'/api/providers/' + encodeURIComponent(id) + '/load'" in app
    assert "loadAssistant, offerAssistantLoad," in app                 # exported on window.fts


def test_each_job_page_offers_it_with_its_own_job_and_retry() -> None:
    pages = {
        "static/js/usecases.js": "offerAssistantLoad(status, 'drafting', startDraft)",
        "templates/project_testing.html": "offerAssistantLoad(msg, 'quiz', buildRewordedQuiz)",
        "templates/data_prep.html": "offerAssistantLoad(document.getElementById('dp-pref-status'), 'preference', dpPrefBuild)",
    }
    for rel, needle in pages.items():
        assert needle in _read(rel), rel
    assert "'study', dpStudyBuild)" in _read("templates/data_prep.html")


def test_providers_reports_each_jobs_connection_kind(client) -> None:
    jobs = client.get("/api/providers").json()["jobs"]
    for job in ("drafting", "quiz", "preference", "study"):
        assert {"provider_id", "kind", "label"} <= set(jobs[job]), job
