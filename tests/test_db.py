# ── Projects ─────────────────────────────────────────────────────────────────

class TestProjects:
    def test_create_and_get_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Test Project", description="A test")["id"]
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
        )["id"]
        p = db.get_project(pid)
        assert p["base_model"] == "Qwen/Qwen2-7B"
        assert p["system_prompt"] == "You are helpful."

    def test_update_project(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Before", description="old")["id"]
        db.update_project(pid, name="After", description="new")
        p = db.get_project(pid)
        assert p["name"] == "After"
        assert p["description"] == "new"

    def test_delete_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="To Delete", description="")["id"]
        db.delete_project(pid)
        assert db.get_project(pid) is None

    def test_list_projects(self, mock_settings):
        import finetune_studio.db as db
        id1 = db.create_project(name="Project A", description="")["id"]
        id2 = db.create_project(name="Project B", description="")["id"]
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
        pid = db.create_project(name="RAG Test", description="")["id"]
        rid = db.create_rag(
            project_id=pid, name="My RAG", description="A RAG",
            store_path="data/rag_store/my_rag",
        )["id"]
        assert rid is not None
        rag = db.get_rag(rid)
        assert rag["name"] == "My RAG"
        assert rag["project_id"] == pid

    def test_update_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Update", description="")["id"]
        rid = db.create_rag(project_id=pid, name="Old Name", store_path="/tmp/r")["id"]
        db.update_rag(rid, name="New Name", doc_count=10)
        rag = db.get_rag(rid)
        assert rag["name"] == "New Name"
        assert rag["doc_count"] == 10

    def test_delete_rag(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG Delete", description="")["id"]
        rid = db.create_rag(project_id=pid, name="To Delete", store_path="/tmp/r")["id"]
        db.delete_rag(rid)
        assert db.get_rag(rid) is None

    def test_list_rags_for_project(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="RAG List", description="")["id"]
        r1 = db.create_rag(project_id=pid, name="RAG 1", store_path="/tmp/1")["id"]
        r2 = db.create_rag(project_id=pid, name="RAG 2", store_path="/tmp/2")["id"]
        rags = db.list_rags(pid)
        names = {r["name"] for r in rags}
        assert names == {"RAG 1", "RAG 2"}


# ── Runs ─────────────────────────────────────────────────────────────────────

class TestRuns:
    def test_create_and_get_run(self, mock_settings):
        import finetune_studio.db as db
        import time
        pid = db.create_project(name="Run Test", description="")["id"]
        rid = db.create_run(
            project_id=pid, name="Run 1",
            settings_obj={"lr": 1e-4, "epochs": 3},
        )["id"]
        assert rid is not None
        run = db.get_run(rid)
        assert run["name"] == "Run 1"
        assert run["status"] == "created"

    def test_update_run_status(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Status Test", description="")["id"]
        rid = db.create_run(project_id=pid, name="Run", settings_obj={})["id"]
        db.update_run(rid, status="training", started_at=1234567890.0)
        run = db.get_run(rid)
        assert run["status"] == "training"
        assert run["started_at"] == 1234567890.0

    def test_delete_run(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Delete Run", description="")["id"]
        rid = db.create_run(project_id=pid, name="To Delete", settings_obj={})["id"]
        db.delete_run(rid)
        assert db.get_run(rid) is None

    def test_list_runs(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Runs", description="")["id"]
        r1 = db.create_run(project_id=pid, name="Run 1", settings_obj={})["id"]
        r2 = db.create_run(project_id=pid, name="Run 2", settings_obj={})["id"]
        runs = db.list_runs(pid)
        assert len(runs) >= 2

    def test_list_runs_with_status_filter(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Filter Runs", description="")["id"]
        r1 = db.create_run(project_id=pid, name="Pending Run", settings_obj={})["id"]
        r2 = db.create_run(project_id=pid, name="Done Run", settings_obj={})["id"]
        db.update_run(r2, status="done")
        all_runs = db.list_runs(pid)
        pending_names = {r["name"] for r in all_runs if r["status"] == "created"}
        done_names = {r["name"] for r in all_runs if r["status"] == "done"}
        assert "Pending Run" in pending_names
        assert "Done Run" in done_names
        assert "Done Run" not in pending_names


# ── Benchmarks ───────────────────────────────────────────────────────────────

class TestBenchmarks:
    def test_create_and_get_benchmark(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Benchmark Test", description="")["id"]
        rid = db.create_run(project_id=pid, name="Run", settings_obj={})["id"]
        bid = db.create_benchmark(
            run_id=rid, suite_name="loss",
            scores={"loss": 0.15},
        )["id"]
        assert bid is not None
        b = db.get_benchmark(bid)
        assert b["scores"]["loss"] == 0.15
        assert b["run_id"] == rid

    def test_list_benchmarks(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="List Bench", description="")["id"]
        rid = db.create_run(project_id=pid, name="Run", settings_obj={})["id"]
        db.create_benchmark(run_id=rid, suite_name="B1", scores={"acc": 0.8})
        db.create_benchmark(run_id=rid, suite_name="B2", scores={"acc": 0.9})
        benchmarks = db.list_benchmarks(run_id=rid)
        assert len(benchmarks) == 2


# ── Reviews ─────────────────────────────────────────────────────────────────

class TestReviews:
    def test_record_and_list_review(self, mock_settings):
        import finetune_studio.db as db
        pid = db.create_project(name="Review Test", description="")["id"]
        review_id = db.record_review(
            project_id=pid, dataset="test.jsonl", row_index=0,
            decision="approved", edited_json=None,
        )
        assert review_id is not None
        reviews = db.list_review(project_id=pid, dataset="test.jsonl")
        assert len(reviews) >= 1


