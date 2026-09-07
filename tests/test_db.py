# ── Projects ─────────────────────────────────────────────────────────────────

class TestProjects:
    def test_create_and_get_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Test Project", description="A test")
        assert pid is not None
        p = db.get_project(pid)
        assert p is not None
        assert p["name"] == "Test Project"
        assert p["description"] == "A test"

    def test_create_project_with_fields(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(
            name="My Project",
            description="Description here",
            base_model="Qwen/Qwen2-7B",
            system_prompt="You are helpful.",
        )
        p = db.get_project(pid)
        assert p["base_model"] == "Qwen/Qwen2-7B"
        assert p["system_prompt"] == "You are helpful."

    def test_update_project(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Before", description="old")
        db.update_project(pid, name="After", description="new")
        p = db.get_project(pid)
        assert p["name"] == "After"
        assert p["description"] == "new"

    def test_delete_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="To Delete", description="")
        db.delete_project(pid)
        assert db.get_project(pid) is None

    def test_list_projects(self, mock_settings):
        import finetune_studio.db as db
        id1 = db.create_project(name="Project A", description="")
        id2 = db.create_project(name="Project B", description="")
        projects = db.list_projects()
        names = {p["name"] for p in projects}
        assert "Project A" in names
        assert "Project B" in names

    def test_get_nonexistent_project(self, mock_settings):
        import finetune_studio.db as db
        assert db.get_project("does_not_exist") is None


# ── RAGs ────────────────────────────────────────────────────────────────────

class TestRags:
    def test_create_and_get_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Test", description="")
        rid = db.create_rag(
            project_id=pid, name="My RAG", description="A RAG",
            store_path="data/rag_store/my_rag",
        )
        assert rid is not None
        rag = db.get_rag(rid)
        assert rag["name"] == "My RAG"
        assert rag["project_id"] == pid

    def test_update_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Update", description="")
        rid = db.create_rag(project_id=pid, name="Old Name", store_path="/tmp/r")
        db.update_rag(rid, name="New Name", doc_count=10)
        rag = db.get_rag(rid)
        assert rag["name"] == "New Name"
        assert rag["doc_count"] == 10

    def test_delete_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Delete", description="")
        rid = db.create_rag(project_id=pid, name="To Delete", store_path="/tmp/r")
        db.delete_rag(rid)
        assert db.get_rag(rid) is None

    def test_list_rags_for_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG List", description="")
        r1 = db.create_rag(project_id=pid, name="RAG 1", store_path="/tmp/1")
        r2 = db.create_rag(project_id=pid, name="RAG 2", store_path="/tmp/2")
        rags = db.list_rags(pid)
        names = {r["name"] for r in rags}
        assert names == {"RAG 1", "RAG 2"}


# ── Runs ─────────────────────────────────────────────────────────────────────

class TestRuns:
    def test_create_and_get_run(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Run Test", description="")
        rid = db.create_run(
            project_id=pid, name="Run 1",
            config_json={"lr": 1e-4, "epochs": 3},
        )
        assert rid is not None
        run = db.get_run(rid)
        assert run["name"] == "Run 1"
        assert run["status"] == "pending"

    def test_update_run_status(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Status Test", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        db.update_run(rid, status="training", started_at=1234567890.0)
        run = db.get_run(rid)
        assert run["status"] == "training"
        assert run["started_at"] == 1234567890.0

    def test_delete_run(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Delete Run", description="")
        rid = db.create_run(project_id=pid, name="To Delete", config_json={})
        db.delete_run(rid)
        assert db.get_run(rid) is None

    def test_list_runs(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Runs", description="")
        r1 = db.create_run(project_id=pid, name="Run 1", config_json={})
        r2 = db.create_run(project_id=pid, name="Run 2", config_json={})
        runs = db.list_runs(pid)
        assert len(runs) >= 2

    def test_list_runs_with_status_filter(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Filter Runs", description="")
        r1 = db.create_run(project_id=pid, name="Pending Run", config_json={})
        r2 = db.create_run(project_id=pid, name="Done Run", config_json={})
        db.update_run(r2, status="done")
        pending = db.list_runs(pid, status="pending")
        done = db.list_runs(pid, status="done")
        pending_names = {r["name"] for r in pending}
        done_names = {r["name"] for r in done}
        assert "Pending Run" in pending_names
        assert "Done Run" in done_names
        assert "Done Run" not in pending_names


# ── Benchmarks ───────────────────────────────────────────────────────────────

class TestBenchmarks:
    def test_create_and_get_benchmark(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Benchmark Test", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        bid = db.create_benchmark(
            run_id=rid, name="Benchmark 1", score=0.85,
            details='{"loss": 0.15}',
        )
        assert bid is not None
        b = db.get_benchmark(bid)
        assert b["score"] == 0.85
        assert b["run_id"] == rid

    def test_list_benchmarks(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Bench", description="")
        rid = db.create_run(project_id=pid, name="Run", config_json={})
        db.create_benchmark(run_id=rid, name="B1", score=0.8)
        db.create_benchmark(run_id=rid, name="B2", score=0.9)
        benchmarks = db.list_benchmarks(rid)
        assert len(benchmarks) == 2


# ── Reviews ─────────────────────────────────────────────────────────────────

class TestReviews:
    def test_record_and_list_review(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Review Test", description="")
        review_id = db.record_review(
            project_id=pid, file_hash="abc123",
            decision="approved", note="Looks good",
        )
        assert review_id is not None
        reviews = db.list_review(pid)
        assert len(reviews) >= 1
        assert reviews[0]["file_hash"] == "abc123"
        assert reviews[0]["decision"] == "approved"
