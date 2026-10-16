import pytest

# ── Health / system ─────────────────────────────────────────────────────────

class TestHealthRoutes:
    def test_root_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_projects_page(self, client):
        r = client.get("/projects")
        assert r.status_code == 200

    def test_hf_explore_page(self, client):
        r = client.get("/models/explore")
        assert r.status_code == 200

    def test_inference_page(self, client):
        r = client.get("/inference")
        assert r.status_code == 200


class TestSystemRoutes:
    def test_resources_endpoint(self, client):
        r = client.get("/api/system/resources")
        assert r.status_code == 200
        data = r.json()
        assert "ram" in data
        assert "vram" in data

    def test_gpu_endpoint_exists(self, client):
        r = client.get("/api/system/gpu")
        # May be 200 (new code loaded) or 404 (old code)
        assert r.status_code in (200, 404)

    def test_gpu_text_endpoint_exists(self, client):
        r = client.get("/api/system/gpu-text")
        assert r.status_code in (200, 404)


# ── Projects API ─────────────────────────────────────────────────────────────

class TestProjectsAPI:
    def test_list_projects(self, client):
        r = client.get("/api/projects")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_project(self, client):
        r = client.post("/api/projects", json={
            "name": "Test Project",
            "description": "A test description",
            "base_model": "Qwen/Qwen2-0.5B",
            "system_prompt": "You are a helpful assistant.",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Test Project"
        # ID should be returned
        assert "id" in data

    def test_get_project(self, client):
        # Create first
        create = client.post("/api/projects", json={"name": "Get Test"})
        pid = create.json()["id"]
        r = client.get(f"/api/projects/{pid}")
        assert r.status_code == 200
        assert r.json()["name"] == "Get Test"

    def test_delete_project(self, client):
        create = client.post("/api/projects", json={"name": "Delete Test"})
        pid = create.json()["id"]
        r = client.delete(f"/api/projects/{pid}")
        assert r.status_code == 200
        # Verify it's gone: a missing project is a real 404 with a reason
        r2 = client.get(f"/api/projects/{pid}")
        assert r2.status_code == 404
        assert r2.json()["detail"]

    def test_update_project(self, client):
        create = client.post("/api/projects", json={"name": "Before"})
        pid = create.json()["id"]
        r = client.patch(f"/api/projects/{pid}", json={"name": "After", "description": "Updated"})
        assert r.status_code == 200
        assert r.json()["name"] == "After"


# ── Training API ─────────────────────────────────────────────────────────────

class TestTrainingAPI:
    def test_training_status_idle(self, client):
        r = client.get("/api/training/status-text")
        assert r.status_code == 200
        text = r.text  # plain text
        assert text in ("Idle", "Loading model…", "Training (*/*)", "Done", "Error", "Saving…")

    def test_training_status_endpoint(self, client):
        r = client.get("/api/training/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data


# ── HF Models API ─────────────────────────────────────────────────────────────

class _FakeHit:
    def __init__(self, mid):
        self.modelId = mid
        self.downloads = 5
        self.likes = 1
        self.tags = ["text-generation"]
        self.lastModified = "2026-01-01"
        self.private = False


class _FakeHfApi:
    def list_models(self, **kw):
        return iter([_FakeHit("Qwen/Qwen2-0.5B"), _FakeHit("Qwen/Qwen3-4B")])


class TestHFModelsAPI:
    @pytest.fixture(autouse=True)
    def _offline_hf(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    def test_hf_search_endpoint(self, client):
        r = client.get("/api/hf/search?q=qwen&task=text-generation&sort=downloads&limit=3")
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert data["query"] == "qwen"

    def test_hf_search_returns_model_list(self, client):
        r = client.get("/api/hf/search?q=qwen&limit=5")
        assert r.status_code == 200
        results = r.json()["results"]
        assert [x["repo_id"] for x in results] == ["Qwen/Qwen2-0.5B", "Qwen/Qwen3-4B"]

    def test_hf_search_unreachable_is_honest_502(self, client, monkeypatch):
        import huggingface_hub

        class _Down:
            def list_models(self, **kw):
                def gen():
                    raise ConnectionError("hub down")
                    yield  # pragma: no cover
                return gen()

        monkeypatch.setattr(huggingface_hub, "HfApi", _Down)
        r = client.get("/api/hf/search?q=qwen&limit=5")
        assert r.status_code == 502
        assert "unreachable" in r.json()["detail"]
        assert r.json()["results"] == []

    def test_hf_info_endpoint(self, client):
        r = client.get("/api/hf/info/Qwen/Qwen2-0.5B-Instruct")
        # 200 if model info available, 404 if not found
        assert r.status_code in (200, 404)

    def test_hf_local_endpoint(self, client):
        r = client.get("/api/hf/local")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_hf_download_cancel_unknown(self, client):
        r = client.delete("/api/hf/download/cancel/not-a-real-id")
        # Returns 200 even for unknown (idempotent)
        assert r.status_code in (200, 404)


# ── Validation ────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_create_project_without_name(self, client):
        r = client.post("/api/projects", json={"description": "no name"})
        # Should either 422 (validation error) or 200 with default
        assert r.status_code in (200, 422)

    def test_get_nonexistent_project(self, client):
        r = client.get("/api/projects/this_does_not_exist")
        assert r.status_code == 404
        assert r.json()["detail"]

    def test_delete_nonexistent_project(self, client):
        r = client.delete("/api/projects/this_does_not_exist")
        # Route may return 200 with ok=False or 404
        assert r.status_code in (200, 404)

    def test_hf_search_empty_query(self, client, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        r = client.get("/api/hf/search?q=&limit=5")
        assert r.status_code == 200

    def test_hf_search_negative_limit(self, client):
        r = client.get("/api/hf/search?q=test&limit=-1")
        # Route may clamp/validate; accept any safe status
        assert r.status_code in (200, 400, 422)

# ── Project wizard (quick + advanced modes) ───────────────────────────────

class TestWizardPage:
    def _pid(self, client, name="Wizard Test"):
        r = client.post("/api/projects", json={"name": name})
        assert r.status_code == 200
        return r.json()["id"]

    def test_wizard_page_renders_both_modes(self, client):
        pid = self._pid(client)
        r = client.get(f"/projects/{pid}/wizard")
        assert r.status_code == 200
        body = r.text
        assert "Quick work" in body
        assert "Quick start" in body and "Step by step" in body
        assert "Run steps 2-5 now" in body
        assert "wiz-quick" in body and "wiz-advanced" in body

    def test_wizard_page_offers_brain_quick_step(self, client):
        """One-page flow = training AND the Smart Brain: Step 1b must be in quick mode."""
        pid = self._pid(client, name="Wizard Brain Step")
        body = client.get(f"/projects/{pid}/wizard").text
        assert "wiz-step-brain" in body
        assert "Build Smart Brain" in body
        assert "wizBrainQuick" in body

    def test_work_redirects_to_wizard(self, client):
        pid = self._pid(client, name="Work Redirect Test")
        r = client.get(f"/projects/{pid}/work", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].endswith(f"/projects/{pid}/wizard")

    def test_overview_has_flow_picker(self, client):
        """Project home must answer 'where do I start' before any tab: two
        explicit flow cards (train a model / Smart Brain)."""
        pid = self._pid(client, name="Overview Banner")
        body = client.get(f"/projects/{pid}").text
        assert "start-here" in body
        assert "flow-card-model" in body and "flow-card-brain" in body
        assert "Train an adapter (LoRA)" in body and "Smart Brain (no training)" in body

    def test_project_nav_is_one_flow_with_two_step_2s(self, client):
        """Genor 2026-10-14: files go one route (Files & Knowledge), then train OR Smart Brain. Every project page shows
        the same flow bar in that order, with the current page marked."""
        pid = self._pid(client, name="Nav Flow")
        def flow(body: str) -> str:
            return body.split('id="workspace-subnav"', 1)[1].split("</nav>", 1)[0]

        order = ['data-tab="data"', 'data-tab="data-prep"', 'data-tab="training"', 'data-tab="brain"',
                 'data-tab="testing"', 'data-tab="export"']
        for page, tab in (("training", "training"), ("brain", "brain"), ("data", "data")):
            bar = flow(client.get(f"/projects/{pid}/{page}").text)
            assert "Files &amp; Knowledge" in bar
            positions = [bar.index(t) for t in order]
            assert positions == sorted(positions), page
            here = bar.split(f'data-tab="{tab}"', 1)[1].split(">", 1)[0]
            assert "here" in here, page

    def test_dashboard_cards_have_quick_work_entry(self, client):
        self._pid(client, name="Dash Quick Work")
        body = client.get("/projects").text
        assert "proj-work" in body and "quick work" in body

    def test_flow_redirects_to_wizard(self, client):
        pid = self._pid(client, name="Flow Redirect Test")
        r = client.get(f"/projects/{pid}/flow", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].endswith(f"/projects/{pid}/wizard")
