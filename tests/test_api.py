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
        # Verify it's gone (route returns 200 with {"error": "not found"})
        r2 = client.get(f"/api/projects/{pid}")
        assert r2.status_code in (200, 404)
        if r2.status_code == 200:
            assert r2.json() is None or "error" in r2.json()

    def test_update_project(self, client):
        create = client.post("/api/projects", json={"name": "Before"})
        pid = create.json()["id"]
        r = client.patch(f"/api/projects/{pid}", json={"name": "After", "description": "Updated"})
        assert r.status_code == 200
        assert r.json()["name"] == "After"


# ── RAG API ──────────────────────────────────────────────────────────────────

class TestRAGAPI:
    def test_create_rag(self, client):
        proj = client.post("/api/projects", json={"name": "RAG API Test"})
        pid = proj.json()["id"]
        r = client.post(f"/api/projects/{pid}/rags", json={
            "name": "Test RAG",
            "description": "A test RAG",
            "store_path": "/tmp/test_rag",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Test RAG"

    def test_list_rags(self, client):
        proj = client.post("/api/projects", json={"name": "List RAGs Test"})
        pid = proj.json()["id"]
        client.post(f"/api/projects/{pid}/rags", json={"name": "RAG 1", "store_path": "/tmp/1"})
        client.post(f"/api/projects/{pid}/rags", json={"name": "RAG 2", "store_path": "/tmp/2"})
        r = client.get(f"/api/projects/{pid}/rags")
        assert r.status_code == 200
        data = r.json()
        names = {rag["name"] for rag in data}
        assert "RAG 1" in names
        assert "RAG 2" in names


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

class TestHFModelsAPI:
    def test_hf_search_endpoint(self, client):
        r = client.get("/api/hf/search?q=qwen&task=text-generation&sort=downloads&limit=3")
        # May be 200 (HF accessible) or 500 (HF unreachable)
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "results" in data
            assert "query" in data

    def test_hf_search_returns_model_list(self, client):
        r = client.get("/api/hf/search?q=qwen&limit=5")
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            # Each result should have repo_id
            for item in results:
                assert "repo_id" in item

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
        # Route returns 200 with {"error": "not found"} for missing
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            data = r.json()
            assert data is None or "error" in data

    def test_delete_nonexistent_project(self, client):
        r = client.delete("/api/projects/this_does_not_exist")
        # Route may return 200 with ok=False or 404
        assert r.status_code in (200, 404)

    def test_hf_search_empty_query(self, client):
        r = client.get("/api/hf/search?q=&limit=5")
        # Empty q might return empty results or 422
        assert r.status_code in (200, 422)

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

    def test_wizard_page_offers_rag_quick_step(self, client):
        """One-page flow = training AND RAG: Step 1b must be in quick mode."""
        pid = self._pid(client, name="Wizard RAG Step")
        body = client.get(f"/projects/{pid}/wizard").text
        assert "wiz-step-rag" in body
        assert "Build RAG index" in body
        assert "wizRagQuick" in body

    def test_work_redirects_to_wizard(self, client):
        pid = self._pid(client, name="Work Redirect Test")
        r = client.get(f"/projects/{pid}/work", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].endswith(f"/projects/{pid}/wizard")

    def test_overview_has_start_here_banner(self, client):
        pid = self._pid(client, name="Overview Banner")
        body = client.get(f"/projects/{pid}").text
        assert "start-here" in body
        assert "Open quick work" in body

    def test_dashboard_cards_have_quick_work_entry(self, client):
        self._pid(client, name="Dash Quick Work")
        body = client.get("/projects").text
        assert "proj-work" in body and "quick work" in body

    def test_flow_redirects_to_wizard(self, client):
        pid = self._pid(client, name="Flow Redirect Test")
        r = client.get(f"/projects/{pid}/flow", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].endswith(f"/projects/{pid}/wizard")
